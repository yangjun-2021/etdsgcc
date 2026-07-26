"""Train diverse GBDT experts on cleaned labels and ensemble."""
import os, sys, time

# Force line-buffered output on Windows so logs update in real time
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED, N_FOLDS, SGCC_CONFIG
from src.utils.utils import seed_everything
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

seed_everything(SEED)


def load_features():
    base = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
    y = np.load(os.path.join(OUTPUT_DIR, 'cleaned_labels_v1.npz'))['y_clean'].astype(int)
    y_orig = np.load(os.path.join(OUTPUT_DIR, 'cleaned_labels_v1.npz'))['y_orig'].astype(int)
    stat = np.nan_to_num(base['stat_features'], nan=0.0, posinf=0.0, neginf=0.0)
    residuals = base['residuals']
    impute_mask = base['impute_mask']
    n_days = residuals.shape[1]
    n = len(y)

    parts = {}
    parts['stat'] = stat

    for fn, key in [('novel_features.npz', 'features'), ('dengine_features.npz', 'X')]:
        try:
            d = np.load(os.path.join(OUTPUT_DIR, fn))
            parts[key] = np.nan_to_num(d[key], nan=0.0, posinf=0.0, neginf=0.0)
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
    parts['paa'] = np.nan_to_num(paa, nan=0.0)

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
    parts['res'] = np.nan_to_num(np.column_stack(res_list), nan=0.0, posinf=0.0, neginf=0.0)

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
    parts['mask'] = np.nan_to_num(np.column_stack(mask_list), nan=0.0)

    # Normalize each part
    for k in parts:
        m = parts[k].mean(axis=0)
        s = parts[k].std(axis=0).clip(min=1e-6)
        parts[k] = ((parts[k] - m) / s).astype(np.float32)
        parts[k] = np.nan_to_num(parts[k], nan=0.0, posinf=0.0, neginf=0.0)
        parts[k] = np.clip(parts[k], -1e4, 1e4)
    return parts, y, y_orig


def build_X(parts, groups):
    X = np.column_stack([parts[g] for g in groups])
    return X.astype(np.float32)


def best_f1_score(y_true, y_prob):
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (y_prob > th).astype(int)
        if pred.sum() == 0: continue
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_th = f, th
    return best_f1, best_th


def train_triple_ensemble(X, y, name, skf, lgb_cfg, xgb_cfg, cat_cfg):
    n = len(y)
    oof_lgb = np.zeros(n); oof_xgb = np.zeros(n); oof_cat = np.zeros(n)
    for fi, (ti, vi) in enumerate(skf.split(X, y)):
        pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
        m = lgb.LGBMClassifier(**lgb_cfg, scale_pos_weight=pw, random_state=SEED+fi, verbose=-1)
        m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        oof_lgb[vi] = m.predict_proba(X[vi])[:, 1]

        m = xgb.XGBClassifier(**xgb_cfg, scale_pos_weight=pw, random_state=SEED+fi, tree_method='hist', verbosity=0)
        m.fit(X[ti], y[ti])
        oof_xgb[vi] = m.predict_proba(X[vi])[:, 1]

        m = CatBoostClassifier(**cat_cfg, auto_class_weights='Balanced', random_seed=SEED+fi, verbose=0)
        m.fit(X[ti], y[ti], eval_set=(X[vi], y[vi]), early_stopping_rounds=80, verbose=False)
        oof_cat[vi] = m.predict_proba(X[vi])[:, 1]

    # Blend by grid search
    best_f1, best_w = 0, (0.4, 0.3, 0.3)
    for wl in np.arange(0.0, 1.01, 0.1):
        for wx in np.arange(0.0, 1.0 - wl + 0.001, 0.1):
            wc = 1.0 - wl - wx
            ens = wl * oof_lgb + wx * oof_xgb + wc * oof_cat
            f = max(f1_score(y, (ens > th).astype(int), zero_division=0) for th in np.arange(0.05, 0.95, 0.01))
            if f > best_f1:
                best_f1, best_w = f, (wl, wx, wc)
    ens = best_w[0]*oof_lgb + best_w[1]*oof_xgb + best_w[2]*oof_cat
    ens = np.nan_to_num(ens, nan=0.5)
    print(f'  {name:25s}: cleaned-F1={best_f1:.4f} AUC={roc_auc_score(y,ens):.4f} weights={best_w}')
    return ens


def ensemble_lr(P, y):
    from sklearn.linear_model import LogisticRegression
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for fi, (ti, vi) in enumerate(skf.split(P, y)):
        m = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=SEED, C=1.0)
        m.fit(P[ti], y[ti])
        oof[vi] = m.predict_proba(P[vi])[:, 1]
    return oof


def main():
    print('Loading features...')
    parts, y, y_orig = load_features()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # LGB/XGB/Cat base configs
    lgb_base = {'n_estimators': 1200, 'max_depth': 7, 'learning_rate': 0.05, 'num_leaves': 63,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1}
    xgb_base = {'n_estimators': 800, 'max_depth': 6, 'learning_rate': 0.04,
                'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
                'min_child_weight': 5}
    cat_base = {'iterations': 800, 'depth': 8, 'learning_rate': 0.05, 'l2_leaf_reg': 3.0, 'subsample': 0.8}

    # Feature group variants
    variants = [
        ('all', ['stat', 'features', 'X', 'paa', 'res', 'mask']),
        ('no-dengine', ['stat', 'features', 'paa', 'res', 'mask']),
        ('no-novel', ['stat', 'X', 'paa', 'res', 'mask']),
        ('stat-paa-res', ['stat', 'paa', 'res']),
        ('stat-mask-res', ['stat', 'mask', 'res']),
    ]

    oofs = {}
    print('\nTraining triple-ensemble variants on cleaned labels...')
    for vname, groups in variants:
        try:
            X = build_X(parts, groups)
            print(f'  Variant {vname}: {X.shape[1]} dims')
            oofs[vname] = train_triple_ensemble(X, y, vname, skf, lgb_base, xgb_base, cat_base)
        except Exception as e:
            print(f'  Variant {vname} failed: {e}')
            import traceback
            traceback.print_exc()

    P = np.column_stack(list(oofs.values()))
    P = np.nan_to_num(P, nan=0.5, posinf=1.0, neginf=0.0)

    print('\nEnsembling with LR...')
    ens_lr = ensemble_lr(P, y)

    # Evaluate on cleaned labels
    best_f1, best_th = best_f1_score(y, ens_lr)
    pred = (ens_lr > best_th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(f'\n=== Mega GBDT Cleaned (cleaned labels) ===')
    print(f'F1={best_f1:.4f}, Rec={recall_score(y,pred):.4f}, '
          f'Prec={precision_score(y,pred,zero_division=0):.4f}, '
          f'AUC={roc_auc_score(y,ens_lr):.4f}, th={best_th:.3f}')
    print(f'TP={tp} FP={fp} FN={fn}')

    # Evaluate on original labels
    best_f1, best_th = best_f1_score(y_orig, ens_lr)
    pred = (ens_lr > best_th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_orig, pred).ravel()
    print(f'\n=== Mega GBDT Cleaned (original labels) ===')
    print(f'F1={best_f1:.4f}, Rec={recall_score(y_orig,pred):.4f}, '
          f'Prec={precision_score(y_orig,pred,zero_division=0):.4f}, '
          f'AUC={roc_auc_score(y_orig,ens_lr):.4f}, th={best_th:.3f}')
    print(f'TP={tp} FP={fp} FN={fn}')

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'mega_gbdt_cleaned_oof.npz'),
        oof_mega_gbdt_cleaned=ens_lr,
        y_clean=y,
        y_orig=y_orig,
        names=np.array(list(oofs.keys())),
    )


if __name__ == '__main__':
    main()
