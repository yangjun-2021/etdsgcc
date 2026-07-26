"""Fast GBDT recall experiment: Expert-A style ensemble on extended features."""
import os, sys, time

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
    y = base['flags'].astype(int)
    stat = np.nan_to_num(base['stat_features'], nan=0.0, posinf=0.0, neginf=0.0)
    mask = base['impute_mask']
    residuals = base['residuals']
    n_days = residuals.shape[1]
    n = len(y)

    parts = [stat, mask.astype(float).mean(axis=1).reshape(-1, 1)]

    # Add novel / dengine if available
    for fn, key in [('novel_features.npz', 'features'), ('dengine_features.npz', 'X')]:
        try:
            d = np.load(os.path.join(OUTPUT_DIR, fn))
            parts.append(np.nan_to_num(d[key], nan=0.0, posinf=0.0, neginf=0.0))
        except Exception:
            pass

    # PAA-50
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

    # Residual aggregates
    half = n_days // 2
    res_list = [
        np.nanmean(residuals, axis=1).reshape(-1, 1),
        np.nanstd(residuals, axis=1).reshape(-1, 1),
        np.nanmean(np.abs(residuals), axis=1).reshape(-1, 1),
        np.nanmax(np.abs(residuals), axis=1).reshape(-1, 1),
        np.nanpercentile(residuals, 90, axis=1).reshape(-1, 1),
    ]
    r1 = np.nanmean(residuals[:, :half], axis=1).reshape(-1, 1)
    r2 = np.nanmean(residuals[:, half:], axis=1).reshape(-1, 1)
    res_list.append(((r2 - r1) / (np.maximum(np.abs(r1), 1e-6))).reshape(-1, 1))
    for w in [30, 60, 90, 180]:
        res_list.append(np.nanmean(np.abs(residuals[:, -w:]), axis=1).reshape(-1, 1))
    parts.append(np.nan_to_num(np.column_stack(res_list), nan=0.0, posinf=0.0, neginf=0.0))

    X = np.nan_to_num(np.column_stack(parts), nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e4, 1e4).astype(np.float32)
    return X, y


def train_expert_a_style(X, y, skf):
    """Train LGB+XGB+CatBoost ensemble exactly like Expert A."""
    n = len(y)
    oof_lgb = np.zeros(n)
    oof_xgb = np.zeros(n)
    oof_cb = np.zeros(n)

    for fi, (ti, vi) in enumerate(skf.split(X, y)):
        X_train, X_val = X[ti], X[vi]
        y_train, y_val = y[ti], y[vi]
        pw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        # LGB
        cfg = SGCC_CONFIG['gbdt_params']['lgb'].copy()
        cfg['scale_pos_weight'] = pw
        m = lgb.LGBMClassifier(**cfg)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof_lgb[vi] = m.predict_proba(X_val)[:, 1]

        # XGB
        cfg = SGCC_CONFIG['gbdt_params']['xgb'].copy()
        cfg['scale_pos_weight'] = pw
        cfg['eval_metric'] = 'auc'
        m = xgb.XGBClassifier(**cfg)
        m.early_stopping_rounds = 50
        m.callbacks = [xgb.callback.EarlyStopping(rounds=50, save_best=True)]
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[vi] = m.predict_proba(X_val)[:, 1]

        # CatBoost
        cfg = SGCC_CONFIG['gbdt_params']['catboost'].copy()
        m = CatBoostClassifier(**cfg)
        m.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)
        oof_cb[vi] = m.predict_proba(X_val)[:, 1]

    # Grid search weights
    best_f1, best_w = 0, (0.4, 0.3, 0.3)
    for wl in np.arange(0.0, 1.01, 0.05):
        for wx in np.arange(0.0, 1.0 - wl + 0.001, 0.05):
            wc = 1.0 - wl - wx
            ens = wl * oof_lgb + wx * oof_xgb + wc * oof_cb
            f = max(f1_score(y, (ens > th).astype(int), zero_division=0) for th in np.arange(0.05, 0.95, 0.01))
            if f > best_f1:
                best_f1, best_w = f, (wl, wx, wc)
    ens = best_w[0]*oof_lgb + best_w[1]*oof_xgb + best_w[2]*oof_cb
    ens = np.nan_to_num(ens, nan=0.5)
    return ens, best_w


def main():
    print('Loading features...')
    X, y = load_features()
    print(f'X: {X.shape}')

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    t0 = time.time()
    oof, w = train_expert_a_style(X, y, skf)
    print(f'Trained in {(time.time()-t0)/60:.1f} min')

    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (oof > th).astype(int)
        if pred.sum() == 0: continue
        f = f1_score(y, pred, zero_division=0)
        if f > best_f1: best_f1, best_th = f, th
    pred = (oof > best_th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    print(f'\n=== Fast GBDT Recall ===')
    print(f'Weights: LGB={w[0]:.2f} XGB={w[1]:.2f} Cat={w[2]:.2f}')
    print(f'F1={best_f1:.4f}, Rec={recall_score(y,pred):.4f}, '
          f'Prec={precision_score(y,pred,zero_division=0):.4f}, '
          f'AUC={roc_auc_score(y,oof):.4f}, th={best_th:.3f}')
    print(f'TP={tp} FP={fp} FN={fn}')

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'fast_gbdt_recall_oof.npz'),
        oof_fast_gbdt_recall=oof,
        flags=y,
        weights=np.array(w),
    )


if __name__ == '__main__':
    main()
