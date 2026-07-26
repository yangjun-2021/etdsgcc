"""Train a model to predict where the current meta ensemble is wrong.

Target: 1 if meta prediction differs from true label, else 0.
Features: statistical features + behaviour features used by hard-FN GBDT.
The OOF error probability is fed back into the meta-learner.
"""
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything

seed_everything(SEED)


def safe_autocorr(x, mask, lag):
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
        s0, s1 = y0.std(), y1.std()
        if s0 == 0 or s1 == 0:
            out[i] = 0.0
        else:
            out[i] = np.corrcoef(y0, y1)[0, 1]
    return out


def build_features():
    pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
    X_seq = pre['X_seq']
    mask = pre['impute_mask']
    stat = pre['stat_features']
    residuals = pre['residuals']
    flags = pre['flags']
    n, t = X_seq.shape[0], X_seq.shape[2]

    val = X_seq[:, 0, :].astype(np.float64)
    obs_mask = ~mask
    valid = obs_mask.sum(axis=1)

    feats = {}
    feats['miss_ratio'] = mask.mean(axis=1)
    feats['zero_ratio'] = ((val <= 0) & obs_mask).sum(axis=1) / np.maximum(valid, 1)

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

    for w in [30, 60, 90, 180]:
        first, last = np.zeros(n), np.zeros(n)
        for i in range(n):
            idx = np.where(obs_mask[i])[0]
            if len(idx) < 2 * w:
                first[i] = last[i] = np.nan
            else:
                first[i] = val[i, idx[:w]].mean()
                last[i] = val[i, idx[-w:]].mean()
        first, last = np.nan_to_num(first), np.nan_to_num(last)
        feats[f'trend_firstlast_{w}'] = (last - first) / (np.abs(first) + 1e-6)

    for lag in [7, 14, 30]:
        feats[f'autocorr_lag{lag}'] = safe_autocorr(val, mask, lag)

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

    feats['res_mean'] = np.nan_to_num(np.nanmean(residuals, axis=1))
    feats['res_std'] = np.nan_to_num(np.nanstd(residuals, axis=1))
    feats['res_absmean'] = np.nan_to_num(np.nanmean(np.abs(residuals), axis=1))
    feats['res_maxabs'] = np.nan_to_num(np.nanmax(np.abs(residuals), axis=1))
    for w in [30, 60, 90]:
        feats[f'res_abs_last{w}'] = np.nan_to_num(np.nanmean(np.abs(residuals[:, -w:]), axis=1))

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

    df = pd.DataFrame(feats)
    X_hand = df.values.astype(np.float32)
    X_hand = np.nan_to_num(X_hand, nan=0.0, posinf=0.0, neginf=0.0)
    stat = np.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.hstack([stat, X_hand]).astype(np.float32), flags


def main():
    X, y = build_features()
    meta = np.load(os.path.join(OUTPUT_DIR, 'sgcc_mega_meta.npz'))['oof_final']
    # Error target at current best meta threshold
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (meta > th).astype(int)
        if pred.sum() == 0: continue
        f = f1_score(y, pred, zero_division=0)
        if f > best_f1: best_f1, best_th = f, th
    meta_pred = (meta > best_th).astype(int)
    err_target = (meta_pred != y).astype(int)
    print(f'Meta best th={best_th:.3f} F1={best_f1:.4f}, error rate={err_target.mean()*100:.2f}%')

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_xgb = np.zeros(len(y))
    oof_lgb = np.zeros(len(y))
    for fi, (ti, vi) in enumerate(skf.split(X, err_target)):
        pw = (err_target[ti] == 0).sum() / max((err_target[ti] == 1).sum(), 1)
        xg = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05,
                               scale_pos_weight=pw, tree_method='hist',
                               random_state=SEED + fi, verbosity=0)
        xg.fit(X[ti], err_target[ti])
        oof_xgb[vi] = xg.predict_proba(X[vi])[:, 1]

        lg = lgb.LGBMClassifier(n_estimators=500, max_depth=7, learning_rate=0.05,
                                num_leaves=63, scale_pos_weight=pw,
                                random_state=SEED + fi, verbose=-1)
        lg.fit(X[ti], err_target[ti])
        oof_lgb[vi] = lg.predict_proba(X[vi])[:, 1]

        print(f'  Fold {fi+1}: XGB AUC={roc_auc_score(err_target[vi], oof_xgb[vi]):.4f}, '
              f'LGB AUC={roc_auc_score(err_target[vi], oof_lgb[vi]):.4f}')

    oof_ens = 0.5 * oof_xgb + 0.5 * oof_lgb
    print(f'Overall error predictor AUC={roc_auc_score(err_target, oof_ens):.4f}')

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'meta_error_predictor_oof.npz'),
        oof_meta_error_predictor=oof_ens,
        flags=y,
    )
    print(f'Saved to {os.path.join(OUTPUT_DIR, "meta_error_predictor_oof.npz")}')


if __name__ == '__main__':
    main()
