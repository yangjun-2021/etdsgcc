"""Train a feature-rich meta-learner using OOFs + hand-crafted features."""
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
from src.training.meta_learner import _load_internal_oofs, _load_external_oofs

seed_everything(SEED)


def load_hand_features():
    pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
    X_seq = pre['X_seq']
    mask = pre['impute_mask']
    stat = pre['stat_features']
    residuals = pre['residuals']
    n = X_seq.shape[0]

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

    feats['res_mean'] = np.nan_to_num(np.nanmean(residuals, axis=1))
    feats['res_std'] = np.nan_to_num(np.nanstd(residuals, axis=1))
    feats['res_absmean'] = np.nan_to_num(np.nanmean(np.abs(residuals), axis=1))
    feats['res_maxabs'] = np.nan_to_num(np.nanmax(np.abs(residuals), axis=1))
    for w in [30, 60, 90]:
        feats[f'res_abs_last{w}'] = np.nan_to_num(np.nanmean(np.abs(residuals[:, -w:]), axis=1))

    df = pd.DataFrame(feats)
    X_hand = df.values.astype(np.float32)
    X_hand = np.nan_to_num(X_hand, nan=0.0, posinf=0.0, neginf=0.0)
    stat = np.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.hstack([stat, X_hand]).astype(np.float32)


def load_oofs(y):
    oofs = {}
    oofs.update(_load_internal_oofs(y))
    oofs.update(_load_external_oofs(y))
    try:
        oofs['Expert-A(GBDT)'] = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_a.npz'))['oof_proba']
    except Exception:
        pass
    try:
        oofs['Expert-B(TCN)'] = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_b.npz'))['oof_proba']
    except Exception:
        pass
    return {k: v for k, v in oofs.items() if len(v) == len(y)}


def correlation_prune(oofs):
    names = sorted(oofs.keys())
    P = np.column_stack([oofs[n] for n in names])
    P = np.nan_to_num(P, nan=0.5, posinf=1.0, neginf=0.0)
    corrs = np.corrcoef(P.T)
    kept = []
    for i, nm in enumerate(names):
        drop = False
        for j in kept:
            if abs(corrs[i, names.index(j)]) > 0.999:
                drop = True
                break
        if not drop:
            kept.append(nm)
    return {k: oofs[k] for k in kept}


def train_meta_gbdt(P, y, skf):
    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    oof_cb = np.zeros(len(y))

    for fi, (ti, vi) in enumerate(skf.split(P, y)):
        pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)

        cfg = {'n_estimators': 800, 'max_depth': 5, 'learning_rate': 0.05,
               'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8,
               'reg_alpha': 0.1, 'reg_lambda': 0.1, 'random_state': SEED+fi, 'verbose': -1}
        m = lgb.LGBMClassifier(**cfg, scale_pos_weight=pw)
        m.fit(P[ti], y[ti], eval_set=[(P[vi], y[vi])],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        oof_lgb[vi] = m.predict_proba(P[vi])[:, 1]

        cfg = {'n_estimators': 600, 'max_depth': 4, 'learning_rate': 0.05,
               'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
               'min_child_weight': 5, 'random_state': SEED+fi, 'verbosity': 0, 'tree_method': 'hist'}
        m = xgb.XGBClassifier(**cfg, scale_pos_weight=pw)
        m.fit(P[ti], y[ti])
        oof_xgb[vi] = m.predict_proba(P[vi])[:, 1]

        cfg = {'iterations': 600, 'depth': 5, 'learning_rate': 0.05, 'l2_leaf_reg': 3.0,
               'subsample': 0.8, 'random_seed': SEED+fi}
        m = CatBoostClassifier(**cfg, auto_class_weights='Balanced', verbose=0)
        m.fit(P[ti], y[ti], eval_set=(P[vi], y[vi]), early_stopping_rounds=80, verbose=False)
        oof_cb[vi] = m.predict_proba(P[vi])[:, 1]

    # Blend
    best_f1, best_w = 0, (0.4, 0.3, 0.3)
    for wl in np.arange(0.0, 1.01, 0.1):
        for wx in np.arange(0.0, 1.0 - wl + 0.001, 0.1):
            wc = 1.0 - wl - wx
            ens = wl * oof_lgb + wx * oof_xgb + wc * oof_cb
            f = max(f1_score(y, (ens > th).astype(int), zero_division=0) for th in np.arange(0.05, 0.95, 0.01))
            if f > best_f1:
                best_f1, best_w = f, (wl, wx, wc)
    ens = best_w[0]*oof_lgb + best_w[1]*oof_xgb + best_w[2]*oof_cb
    return ens, best_w


def main():
    y = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags']

    print('Loading OOFs...')
    oofs = load_oofs(y)
    print(f'Loaded {len(oofs)} OOFs')
    oofs = correlation_prune(oofs)
    print(f'After pruning: {len(oofs)}')

    print('Loading hand-crafted features...')
    hand = load_hand_features()
    print(f'Hand features: {hand.shape}')

    P_oof = np.column_stack([oofs[k] for k in sorted(oofs.keys())])
    P_oof = np.nan_to_num(P_oof, nan=0.5, posinf=1.0, neginf=0.0)
    P = np.hstack([P_oof, hand]).astype(np.float32)
    print(f'Full meta-feature matrix: {P.shape}')

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof, w = train_meta_gbdt(P, y, skf)
    print(f'Blend weights: LGB={w[0]:.2f} XGB={w[1]:.2f} Cat={w[2]:.2f}')

    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (oof > th).astype(int)
        if pred.sum() == 0: continue
        f = f1_score(y, pred, zero_division=0)
        if f > best_f1: best_f1, best_th = f, th
    pred = (oof > best_th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    print(f'\n=== Feature-Rich Meta ===')
    print(f'F1={best_f1:.4f}, Rec={recall_score(y,pred):.4f}, '
          f'Prec={precision_score(y,pred,zero_division=0):.4f}, '
          f'AUC={roc_auc_score(y,oof):.4f}, th={best_th:.3f}')
    print(f'TP={tp} FP={fp} FN={fn}')

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'feature_rich_meta_oof.npz'),
        oof_feature_rich_meta=oof,
        flags=y,
    )


if __name__ == '__main__':
    main()
