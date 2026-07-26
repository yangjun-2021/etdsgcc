"""Usage-stratified GBDT: train a separate LightGBM per usage quintile.

Hypothesis: low-usage and high-usage theft patterns differ enough that
per-group models can improve recall on the low-usage subtle cases.
"""
import os
import sys
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SEED, N_FOLDS


def main():
    pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
    X = pre['stat_features'].astype(np.float32)
    miss = pre['impute_mask'].mean(axis=1).reshape(-1, 1)
    X = np.nan_to_num(np.column_stack([X, miss]), nan=0, posinf=0, neginf=0)
    y = pre['flags'].astype(int)

    usage = np.load(os.path.join(OUTPUT_DIR, 'usage_features.npz'))
    log_max = usage['log_max_usage']

    n_groups = 5
    quintiles = np.percentile(log_max, np.linspace(0, 100, n_groups + 1))
    group_ids = np.digitize(log_max, quintiles[1:-1], right=True)

    oof = np.zeros(len(y), dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for g in range(n_groups):
        mask = group_ids == g
        X_g = X[mask]
        y_g = y[mask]
        idx_g = np.where(mask)[0]
        print(f'\nGroup {g}: n={mask.sum()}, positives={y_g.sum()}, rate={y_g.mean()*100:.2f}%')
        for fi, (ti, vi) in enumerate(skf.split(X_g, y_g)):
            pw = (y_g[ti] == 0).sum() / max((y_g[ti] == 1).sum(), 1)
            model = lgb.LGBMClassifier(
                n_estimators=2000,
                max_depth=7,
                learning_rate=0.03,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.05,
                reg_lambda=0.05,
                min_child_samples=50,
                scale_pos_weight=pw,
                random_state=SEED + fi + g * 100,
                verbose=-1,
                n_jobs=-1,
            )
            model.fit(
                X_g[ti], y_g[ti],
                eval_set=[(X_g[vi], y_g[vi])],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
            )
            oof[idx_g[vi]] = model.predict_proba(X_g[vi])[:, 1]

    # Overall metrics
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (oof > th).astype(int)
        if pred.sum() == 0:
            continue
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    pred = (oof > best_th).astype(int)
    print(f'\nUsage-stratified LGB: F1={f1_score(y, pred):.4f}, '
          f'Rec={recall_score(y, pred):.4f}, '
          f'Prec={precision_score(y, pred, zero_division=0):.4f}, '
          f'AUC={roc_auc_score(y, oof):.4f}, th={best_th:.3f}')

    # Per-group metrics
    for g in range(n_groups):
        mask = group_ids == g
        pred_g = pred[mask]
        y_g = y[mask]
        if pred_g.sum() == 0:
            f1 = rec = prec = 0.0
        else:
            f1 = f1_score(y_g, pred_g, zero_division=0)
            rec = recall_score(y_g, pred_g, zero_division=0)
            prec = precision_score(y_g, pred_g, zero_division=0)
        print(f'  G{g}: F1={f1:.4f}, Rec={rec:.4f}, Prec={prec:.4f}')

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'usage_stratified_lgb_oof.npz'),
        oof_usage_stratified_lgb=oof,
        flags=y,
        group_ids=group_ids,
    )
    print(f'Saved to {os.path.join(OUTPUT_DIR, "usage_stratified_lgb_oof.npz")}')


if __name__ == '__main__':
    main()
