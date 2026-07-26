"""Train diverse GBDT experts on rich features and ensemble for recall/F1.

No external data dependencies. Uses stat + novel + dengine + PAA + residual + mask features.
"""
import os, sys, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
from scipy.optimize import nnls
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

seed_everything(SEED)


def load_features():
    base = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
    y = base['flags'].astype(int)
    stat = np.nan_to_num(base['stat_features'], nan=0.0, posinf=0.0, neginf=0.0)
    residuals = base['residuals']
    impute_mask = base['impute_mask']
    n_days = residuals.shape[1]
    n = len(y)

    # novel / dengine
    parts = [stat]
    for fn, key in [('novel_features.npz', 'features'), ('dengine_features.npz', 'X')]:
        try:
            d = np.load(os.path.join(OUTPUT_DIR, fn))
            parts.append(np.nan_to_num(d[key], nan=0.0, posinf=0.0, neginf=0.0))
        except Exception as e:
            print(f'  warning: could not load {fn}: {e}')

    # PAA
    raw_df = pd.read_csv(os.path.join('data', 'raw_data.csv'))
    date_cols = [c for c in raw_df.columns if '/' in str(c) and len(str(c)) <= 10]
    raw = raw_df[date_cols].values.astype(float)
    del raw_df
    N_PAA = 50
    seg = n_days / N_PAA
    paa = np.zeros((n, N_PAA), dtype=np.float32)
    for i in range(N_PAA):
        s, e = int(round(i * seg)), int(round((i + 1) * seg))
        if e > s:
            paa[:, i] = np.nanmean(raw[:, s:e], axis=1)
    parts.append(np.nan_to_num(paa, nan=0.0))

    # residual aggregates
    half = n_days // 2
    res_list = [
        np.nanmean(residuals, axis=1).reshape(-1, 1),
        np.nanstd(residuals, axis=1).reshape(-1, 1),
        np.nanmean(np.abs(residuals), axis=1).reshape(-1, 1),
        np.nanmax(np.abs(residuals), axis=1).reshape(-1, 1),
        np.nanpercentile(residuals, 25, axis=1).reshape(-1, 1),
        np.nanpercentile(residuals, 75, axis=1).reshape(-1, 1),
        np.nanpercentile(residuals, 90, axis=1).reshape(-1, 1),
        np.nanpercentile(residuals, 95, axis=1).reshape(-1, 1),
    ]
    r1 = np.nanmean(residuals[:, :half], axis=1).reshape(-1, 1)
    r2 = np.nanmean(residuals[:, half:], axis=1).reshape(-1, 1)
    res_list.append(((r2 - r1) / (np.maximum(np.abs(r1), 1e-6))).reshape(-1, 1))
    for w in [30, 60, 90, 180]:
        if n_days >= w:
            res_list.append(np.nanmean(np.abs(residuals[:, -w:]), axis=1).reshape(-1, 1))
    parts.append(np.nan_to_num(np.column_stack(res_list), nan=0.0, posinf=0.0, neginf=0.0))

    # mask aggregates
    mask_list = [impute_mask.astype(float).mean(axis=1).reshape(-1, 1)]
    for s, e in [(0, n_days//4), (n_days//4, n_days//2), (n_days//2, 3*n_days//4), (3*n_days//4, n_days),
                 (0, half), (half, n_days)]:
        mask_list.append(impute_mask[:, s:e].astype(float).mean(axis=1).reshape(-1, 1))
    missing_runs = np.zeros(n)
    for i in range(n):
        runs = []
        cr = 0
        for m in impute_mask[i]:
            if m: cr += 1
            else:
                if cr > 0: runs.append(cr)
                cr = 0
        if cr > 0: runs.append(cr)
        missing_runs[i] = max(runs) if runs else 0
    mask_list.append(missing_runs.reshape(-1, 1))
    mask_list.append((~impute_mask).sum(axis=1).reshape(-1, 1) / n_days)
    parts.append(np.nan_to_num(np.column_stack(mask_list), nan=0.0))

    X = np.nan_to_num(np.column_stack(parts), nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e4, 1e4).astype(np.float32)
    return X, y


def train_expert(X, y, name, cfg, tag, skf):
    n = len(y)
    oof = np.zeros(n)
    for fi, (ti, vi) in enumerate(skf.split(X, y)):
        pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
        if tag == 'lgb':
            m = lgb.LGBMClassifier(**cfg, scale_pos_weight=pw, random_state=SEED+fi, verbose=-1)
            m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])],
                  callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        elif tag == 'xgb':
            m = xgb.XGBClassifier(**cfg, scale_pos_weight=pw, random_state=SEED+fi, tree_method='hist', verbosity=0)
            m.fit(X[ti], y[ti])
        elif tag == 'cat':
            m = CatBoostClassifier(**cfg, auto_class_weights='Balanced', random_seed=SEED+fi, verbose=0)
            m.fit(X[ti], y[ti], eval_set=(X[vi], y[vi]), early_stopping_rounds=80, verbose=False)
        oof[vi] = m.predict_proba(X[vi])[:, 1]
    oof = np.nan_to_num(oof, nan=0.5)
    best = max(f1_score(y, (oof > th).astype(int), zero_division=0) for th in np.arange(0.05, 0.95, 0.001))
    print(f'  {name:25s}: F1={best:.4f} AUC={roc_auc_score(y,oof):.4f}')
    return oof


def ensemble_nnls(P, y):
    """Non-negative least squares ensemble weights."""
    # Use clipped probabilities to avoid extreme values
    Pc = np.clip(P, 1e-6, 1 - 1e-6)
    # Transform to log-odds
    Z = np.log(Pc / (1 - Pc))
    # NNLS wants y ~ Z*w; we use {0,1} target via least squares approximation
    w, _ = nnls(Z, y.astype(float))
    if w.sum() == 0:
        w = np.ones(len(w)) / len(w)
    else:
        w = w / w.sum()
    ens = P.dot(w)
    best = max(f1_score(y, (ens > th).astype(int), zero_division=0) for th in np.arange(0.05, 0.95, 0.001))
    print(f'  NNLS ensemble: F1={best:.4f} weights={w.round(3)}')
    return ens, w


def ensemble_lr(P, y):
    from sklearn.linear_model import LogisticRegression
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for fi, (ti, vi) in enumerate(skf.split(P, y)):
        m = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=SEED, C=1.0)
        m.fit(P[ti], y[ti])
        oof[vi] = m.predict_proba(P[vi])[:, 1]
    best = max(f1_score(y, (oof > th).astype(int), zero_division=0) for th in np.arange(0.05, 0.95, 0.001))
    print(f'  LR ensemble: F1={best:.4f}')
    return oof


def main():
    print('Loading features...')
    X, y = load_features()
    print(f'Feature matrix: {X.shape}')

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    experts = [
        # Standard strong models
        ('LGB-d7', {'n_estimators': 1200, 'max_depth': 7, 'learning_rate': 0.05, 'num_leaves': 63,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1}, 'lgb'),
        ('LGB-d8', {'n_estimators': 1200, 'max_depth': 8, 'learning_rate': 0.04, 'num_leaves': 127,
                    'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 0.05, 'reg_lambda': 0.05}, 'lgb'),
        ('XGB-d6', {'n_estimators': 800, 'max_depth': 6, 'learning_rate': 0.04,
                    'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
                    'min_child_weight': 5}, 'xgb'),
        ('XGB-d7', {'n_estimators': 800, 'max_depth': 7, 'learning_rate': 0.03,
                    'subsample': 0.85, 'colsample_bytree': 0.75, 'reg_alpha': 0.05, 'reg_lambda': 0.05,
                    'min_child_weight': 3}, 'xgb'),
        ('Cat-d8', {'iterations': 800, 'depth': 8, 'learning_rate': 0.05, 'l2_leaf_reg': 3.0, 'subsample': 0.8}, 'cat'),
        # Recall-oriented: higher scale_pos_weight will be applied automatically via pw
        ('LGB-recall', {'n_estimators': 1500, 'max_depth': 6, 'learning_rate': 0.03, 'num_leaves': 63,
                        'subsample': 0.9, 'colsample_bytree': 0.8, 'reg_alpha': 0.01, 'reg_lambda': 0.01}, 'lgb'),
    ]

    print('\nTraining experts...')
    oofs = {}
    for name, cfg, tag in experts:
        oofs[name] = train_expert(X, y, name, cfg, tag, skf)

    # Add existing strong internal OOFs
    for fn, key in [
        ('autoresearch_best.npz', 'oof_final'),
        ('mega_boost_enhanced.npz', 'oof_final'),
        ('strong_gbdt_prior_oof.npz', 'oof_strong_gbdt_prior'),
    ]:
        try:
            oofs[fn.replace('.npz', '')] = np.load(os.path.join(OUTPUT_DIR, fn))[key]
            print(f'  Loaded {fn}')
        except Exception:
            pass

    P = np.column_stack(list(oofs.values()))
    P = np.nan_to_num(P, nan=0.5, posinf=1.0, neginf=0.0)

    print('\nEnsembling...')
    ens_lr = ensemble_lr(P, y)
    ens_nnls, w = ensemble_nnls(P, y)

    # Pick best
    best_oof = ens_lr
    best_name = 'LR'
    best_f1 = max(f1_score(y, (best_oof > th).astype(int), zero_division=0) for th in np.arange(0.05, 0.95, 0.001))
    nnls_f1 = max(f1_score(y, (ens_nnls > th).astype(int), zero_division=0) for th in np.arange(0.05, 0.95, 0.001))
    if nnls_f1 > best_f1:
        best_oof = ens_nnls
        best_name = 'NNLS'
        best_f1 = nnls_f1

    # Final metrics
    best_th = 0.5
    best_f1 = 0
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (best_oof > th).astype(int)
        if pred.sum() == 0: continue
        f = f1_score(y, pred, zero_division=0)
        if f > best_f1: best_f1, best_th = f, th
    pred = (best_oof > best_th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    print(f'\n=== Mega GBDT Recall ===')
    print(f'Best ensemble: {best_name}, F1={best_f1:.4f}, Rec={recall_score(y,pred):.4f}, '
          f'Prec={precision_score(y,pred,zero_division=0):.4f}, AUC={roc_auc_score(y,best_oof):.4f}, th={best_th:.3f}')
    print(f'TP={tp} FP={fp} FN={fn}')

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'mega_gbdt_recall_oof.npz'),
        oof_mega_gbdt_recall=best_oof,
        flags=y,
        names=np.array(list(oofs.keys())),
        weights=w if best_name == 'NNLS' else None,
    )


if __name__ == '__main__':
    main()
