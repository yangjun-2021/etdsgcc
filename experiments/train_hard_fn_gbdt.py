"""Hard-FN booster: a high-recall GBDT trained on sequence-level behaviour features.

Target: the 616 false negatives that the current meta ensemble misses.  These
samples tend to have low missing-ratio, high consumption, strong weekly
autocorrelation and an upward late-period trend.  This model is intentionally
biased toward recall so the meta-learner can use it to recover those hard
positives.
"""
import os
import sys
import time
import warnings

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything

# Redirect stdout/stderr to a log file for background monitoring
_log_path = os.path.join(OUTPUT_DIR, 'hard_fn_gbdt.log')
_log_fh = open(_log_path, 'w', buffering=1, encoding='utf-8')
sys.stdout = _log_fh
sys.stderr = _log_fh
print(f'Logging to {_log_path}')

warnings.filterwarnings('ignore')
seed_everything(SEED)


def safe_autocorr(x, mask, lag):
    """Per-sample autocorr at lag using observed values only."""
    n = x.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        valid = np.where(~mask[i])[0]
        if len(valid) < lag + 5:
            out[i] = 0.0
            continue
        y = x[i, valid]
        y0 = y[:-lag]
        y1 = y[lag:]
        s0 = y0.std()
        s1 = y1.std()
        if s0 == 0 or s1 == 0:
            out[i] = 0.0
        else:
            out[i] = np.corrcoef(y0, y1)[0, 1]
    return out


def build_hard_fn_features():
    print('Loading preprocessed SGCC...')
    pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
    X_seq = pre['X_seq']          # [N, 5, T]
    mask = pre['impute_mask']     # [N, T]
    stat = pre['stat_features']   # [N, 353]
    residuals = pre['residuals']  # [N, T]
    flags = pre['flags']
    n, t = X_seq.shape[0], X_seq.shape[2]

    # Channel 0 is log1p-transformed consumption
    val = X_seq[:, 0, :].astype(np.float64)
    obs_mask = ~mask
    valid = obs_mask.sum(axis=1)

    feats = {}
    feats['miss_ratio'] = mask.mean(axis=1)
    feats['zero_ratio'] = ((val <= 0) & obs_mask).sum(axis=1) / np.maximum(valid, 1)

    # Basic stats on observed log-consumption
    clean = np.where(obs_mask, val, np.nan)
    feats['mean'] = np.nan_to_num(np.nanmean(clean, axis=1))
    feats['std'] = np.nan_to_num(np.nanstd(clean, axis=1))
    feats['median'] = np.nan_to_num(np.nanmedian(clean, axis=1))
    feats['min'] = np.nan_to_num(np.nanmin(clean, axis=1))
    feats['max'] = np.nan_to_num(np.nanmax(clean, axis=1))
    feats['range'] = feats['max'] - feats['min']
    feats['skew'] = np.nan_to_num(skew(clean, axis=1, nan_policy='omit'))
    feats['kurtosis'] = np.nan_to_num(kurtosis(clean, axis=1, nan_policy='omit'))
    feats['cv'] = np.where(feats['mean'] > 1e-6, feats['std'] / feats['mean'], 0)

    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        feats[f'q{q}'] = np.nan_to_num(np.nanpercentile(clean, q, axis=1))

    # Trends (using observed positions)
    def window_mean(x, m, w):
        valid = ~m
        out = np.zeros(len(x))
        for i in range(len(x)):
            idx = np.where(valid[i])[0]
            if len(idx) < w:
                out[i] = np.nan
            else:
                out[i] = x[i, idx[:w]].mean() if w > 0 else x[i, idx[-w:]].mean()
        return out

    for w in [30, 60, 90, 180]:
        first = np.zeros(n)
        last = np.zeros(n)
        for i in range(n):
            idx = np.where(obs_mask[i])[0]
            if len(idx) < 2 * w:
                first[i] = last[i] = np.nan
            else:
                first[i] = val[i, idx[:w]].mean()
                last[i] = val[i, idx[-w:]].mean()
        first = np.nan_to_num(first)
        last = np.nan_to_num(last)
        feats[f'trend_firstlast_{w}'] = (last - first) / (np.abs(first) + 1e-6)

    # Autocorrelation at weekly / bi-weekly / monthly lags
    for lag in [7, 14, 30]:
        feats[f'autocorr_lag{lag}'] = safe_autocorr(val, mask, lag)

    # Weekly / monthly profile strength
    dow = np.arange(t) % 7
    monthly = np.arange(t) % 365 // 30
    dow_means = np.zeros((n, 7))
    month_means = np.zeros((n, 12))
    for d in range(7):
        day_mask = obs_mask & (dow == d)
        sums = (val * day_mask).sum(axis=1)
        cnts = day_mask.sum(axis=1)
        dow_means[:, d] = np.where(cnts > 0, sums / cnts, 0)
    for m in range(12):
        day_mask = obs_mask & (monthly == m)
        sums = (val * day_mask).sum(axis=1)
        cnts = day_mask.sum(axis=1)
        month_means[:, m] = np.where(cnts > 0, sums / cnts, 0)
    feats['dow_profile_std'] = dow_means.std(axis=1)
    feats['month_profile_std'] = month_means.std(axis=1)

    # Residual-based behaviour
    feats['res_mean'] = np.nan_to_num(np.nanmean(residuals, axis=1))
    feats['res_std'] = np.nan_to_num(np.nanstd(residuals, axis=1))
    feats['res_absmean'] = np.nan_to_num(np.nanmean(np.abs(residuals), axis=1))
    feats['res_maxabs'] = np.nan_to_num(np.nanmax(np.abs(residuals), axis=1))

    # Last-period residual activity
    for w in [30, 60, 90]:
        feats[f'res_abs_last{w}'] = np.nan_to_num(np.nanmean(np.abs(residuals[:, -w:]), axis=1))

    # Flat / zero-streak features
    zero_runs = []
    for i in range(n):
        zero_mask = (val[i] <= 0) & obs_mask[i]
        if not zero_mask.any():
            zero_runs.append(0)
        else:
            diff = np.diff(zero_mask.astype(int))
            starts = np.where(diff == 1)[0] + 1
            if zero_mask[0]:
                starts = np.concatenate([[0], starts])
            zero_runs.append(len(starts))
    feats['zero_streaks'] = np.array(zero_runs, dtype=np.float32)

    # Stack all handcrafted features
    df = pd.DataFrame(feats)
    X_hand = df.values.astype(np.float32)
    X_hand = np.nan_to_num(X_hand, nan=0.0, posinf=0.0, neginf=0.0)

    # Add existing statistical features as a base
    stat = np.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X = np.hstack([stat, X_hand]).astype(np.float32)
    print(f'  Feature matrix: {X.shape}')
    return X, flags


def train_cv(X, y, n_folds=N_FOLDS):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    n = len(y)
    oof_lgb = np.zeros(n)
    oof_xgb = np.zeros(n)

    for fi, (ti, vi) in enumerate(skf.split(X, y)):
        print(f'\n  Fold {fi+1}/{n_folds}')
        X_train, y_train = X[ti], y[ti]
        X_val, y_val = X[vi], y[vi]
        pw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        # High recall weight to target false negatives
        lgb_model = lgb.LGBMClassifier(
            n_estimators=2000, max_depth=7, learning_rate=0.03,
            num_leaves=63, min_child_samples=50, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=0.05,
            scale_pos_weight=pw * 2.0, random_state=SEED + fi, verbose=-1)
        lgb_model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(100, verbose=False),
                                 lgb.log_evaluation(0)])
        oof_lgb[vi] = lgb_model.predict_proba(X_val)[:, 1]

        xgb_model = xgb.XGBClassifier(
            n_estimators=1500, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pw * 2.0, tree_method='hist',
            random_state=SEED + fi, verbosity=0)
        xgb_model.fit(X_train, y_train)
        oof_xgb[vi] = xgb_model.predict_proba(X_val)[:, 1]

        for name, oof_v in [('LGB', oof_lgb[vi]), ('XGB', oof_xgb[vi])]:
            best = (0, 0.5)
            for th in np.arange(0.05, 0.95, 0.005):
                pred = (oof_v > th).astype(int)
                if pred.sum() == 0:
                    continue
                f = f1_score(y_val, pred, zero_division=0)
                if f > best[0]:
                    best = (f, th)
            f, th = best
            pred = (oof_v > th).astype(int)
            print(f'    {name}: F1={f:.4f}, Rec={recall_score(y_val, pred):.4f}, '
                  f'Prec={precision_score(y_val, pred, zero_division=0):.4f}, th={th:.3f}')

    # Ensemble: average of LGB and XGB
    oof_ens = 0.5 * oof_lgb + 0.5 * oof_xgb
    return oof_lgb, oof_xgb, oof_ens


def main():
    t0 = time.time()
    X, y = build_hard_fn_features()
    oof_lgb, oof_xgb, oof_ens = train_cv(X, y)

    for name, oof in [('LGB', oof_lgb), ('XGB', oof_xgb), ('Ens', oof_ens)]:
        best = (0, 0.5)
        for th in np.arange(0.05, 0.95, 0.005):
            pred = (oof > th).astype(int)
            if pred.sum() == 0:
                continue
            f = f1_score(y, pred, zero_division=0)
            if f > best[0]:
                best = (f, th)
        f, th = best
        pred = (oof > th).astype(int)
        print(f'\nOverall {name}: F1={f:.4f}, Rec={recall_score(y, pred):.4f}, '
              f'Prec={precision_score(y, pred, zero_division=0):.4f}, AUC={roc_auc_score(y, oof):.4f}, th={th:.3f}')

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'hard_fn_gbdt_oof.npz'),
        oof_hard_fn_gbdt=oof_ens,
        oof_hard_fn_gbdt_lgb=oof_lgb,
        oof_hard_fn_gbdt_xgb=oof_xgb,
        flags=y,
    )
    print(f'\nSaved to {os.path.join(OUTPUT_DIR, "hard_fn_gbdt_oof.npz")}')
    print(f'Total time: {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
