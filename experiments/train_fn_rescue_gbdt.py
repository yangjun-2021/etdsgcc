"""Train a GBDT to rescue current meta false negatives.

Features: rich hand-crafted features.
Target: 1 if sample is FN under current meta, else 0.
Use OOF probabilities to rescue low-confidence meta negatives.
"""
import os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
from scipy.stats import skew, kurtosis

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SEED, N_FOLDS, SGCC_CONFIG
from src.utils.utils import seed_everything

seed_everything(SEED)


def build_features():
    pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
    X_seq = pre['X_seq']
    mask = pre['impute_mask']
    stat = pre['stat_features']
    residuals = pre['residuals']
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
        out = np.zeros(n)
        for i in range(n):
            idx = np.where(obs_mask[i])[0]
            if len(idx) < lag + 5:
                out[i] = 0.0
            else:
                y0 = val[i, idx[:-lag]]
                y1 = val[i, idx[lag:]]
                s0, s1 = y0.std(), y1.std()
                out[i] = 0.0 if s0 == 0 or s1 == 0 else np.corrcoef(y0, y1)[0, 1]
        feats[f'autocorr_lag{lag}'] = out

    feats['res_mean'] = np.nan_to_num(np.nanmean(residuals, axis=1))
    feats['res_std'] = np.nan_to_num(np.nanstd(residuals, axis=1))
    feats['res_absmean'] = np.nan_to_num(np.nanmean(np.abs(residuals), axis=1))
    feats['res_maxabs'] = np.nan_to_num(np.nanmax(np.abs(residuals), axis=1))
    for w in [30, 60, 90]:
        feats[f'res_abs_last{w}'] = np.nan_to_num(np.nanmean(np.abs(residuals[:, -w:]), axis=1))

    # Long zero/negative runs
    zero_runs = []
    for i in range(n):
        zm = (val[i] <= 0) & obs_mask[i]
        if not zm.any():
            zero_runs.append(0)
        else:
            diff = np.diff(zm.astype(int))
            starts = np.where(diff == 1)[0] + 1
            if zm[0]:
                starts = np.concatenate([[0], starts])
            zero_runs.append(len(starts))
    feats['zero_streaks'] = np.array(zero_runs, dtype=np.float32)

    df = pd.DataFrame(feats)
    X_hand = df.values.astype(np.float32)
    X_hand = np.nan_to_num(X_hand, nan=0.0, posinf=0.0, neginf=0.0)
    stat = np.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.hstack([stat, X_hand]).astype(np.float32)


def train_fn_rescue(X, y, meta, skf):
    # Build FN target at current best meta threshold
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (meta > th).astype(int)
        if pred.sum() == 0: continue
        f = f1_score(y, pred, zero_division=0)
        if f > best_f1: best_f1, best_th = f, th
    meta_pred = (meta > best_th).astype(int)
    fn_target = ((meta_pred == 0) & (y == 1)).astype(int)
    print(f'Meta best th={best_th:.3f} F1={best_f1:.4f}, FN target rate={fn_target.mean()*100:.2f}%')

    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    oof_cb = np.zeros(len(y))
    for fi, (ti, vi) in enumerate(skf.split(X, fn_target)):
        pw = (fn_target[ti] == 0).sum() / max((fn_target[ti] == 1).sum(), 1)

        cfg = SGCC_CONFIG['gbdt_params']['lgb'].copy()
        cfg['scale_pos_weight'] = pw
        cfg['random_state'] = SEED + fi
        cfg['verbose'] = -1
        m = lgb.LGBMClassifier(**cfg)
        m.fit(X[ti], fn_target[ti], eval_set=[(X[vi], fn_target[vi])],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        oof_lgb[vi] = m.predict_proba(X[vi])[:, 1]

        cfg = SGCC_CONFIG['gbdt_params']['xgb'].copy()
        cfg['scale_pos_weight'] = pw
        cfg['random_state'] = SEED + fi
        cfg['verbosity'] = 0
        cfg['tree_method'] = 'hist'
        m = xgb.XGBClassifier(**cfg)
        m.fit(X[ti], fn_target[ti])
        oof_xgb[vi] = m.predict_proba(X[vi])[:, 1]

        cfg = SGCC_CONFIG['gbdt_params']['catboost'].copy()
        cfg['random_seed'] = SEED + fi
        m = CatBoostClassifier(**cfg)
        m.fit(X[ti], fn_target[ti], eval_set=(X[vi], fn_target[vi]), early_stopping_rounds=80, verbose=False)
        oof_cb[vi] = m.predict_proba(X[vi])[:, 1]

    # Blend
    best_f1_blend, best_w = 0, (0.4, 0.3, 0.3)
    for wl in np.arange(0.0, 1.01, 0.1):
        for wx in np.arange(0.0, 1.0 - wl + 0.001, 0.1):
            wc = 1.0 - wl - wx
            ens = wl * oof_lgb + wx * oof_xgb + wc * oof_cb
            f = max(f1_score(fn_target, (ens > th).astype(int), zero_division=0) for th in np.arange(0.05, 0.95, 0.01))
            if f > best_f1_blend:
                best_f1_blend, best_w = f, (wl, wx, wc)
    fn_prob = best_w[0]*oof_lgb + best_w[1]*oof_xgb + best_w[2]*oof_cb
    print(f'FN predictor F1={best_f1_blend:.4f} AUC={roc_auc_score(fn_target,fn_prob):.4f} weights={best_w}')
    return fn_prob


def main():
    y = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags']
    meta = np.load(os.path.join(OUTPUT_DIR, 'sgcc_mega_meta.npz'))['oof_final']

    print('Building features...')
    X = build_features()
    print(f'X: {X.shape}')

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fn_prob = train_fn_rescue(X, y, meta, skf)

    # Try to rescue meta predictions
    print('\n=== Rescue experiments ===')
    best_overall = {'f1': 0, 'th_meta_low': 0, 'th_fn': 0}
    for th_meta_low in np.arange(0.05, 0.70, 0.05):
        for th_fn in np.arange(0.05, 0.95, 0.05):
            pred = (meta > 0.5).astype(int)
            rescue = (meta < th_meta_low) & (fn_prob > th_fn)
            pred[rescue] = 1
            if pred.sum() == 0: continue
            f = f1_score(y, pred, zero_division=0)
            if f > best_overall['f1']:
                best_overall = {'f1': f, 'th_meta_low': th_meta_low, 'th_fn': th_fn, 'n_rescue': rescue.sum()}
    print('Best rescue:', best_overall)

    # Also try adding fn_prob as feature to meta retraining (simplified)
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'fn_rescue_gbdt_oof.npz'),
        oof_fn_rescue_gbdt=fn_prob,
        flags=y,
    )


if __name__ == '__main__':
    main()
