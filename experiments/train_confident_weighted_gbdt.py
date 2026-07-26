"""Train GBDT with confident-learning sample weights on original labels.

Use cleaned_labels_v1 consensus as per-sample weight: high consensus =>
high confidence in the (original or cleaned) label; low consensus => likely
noisy sample.  The model still predicts original labels, but focuses on
samples where the label is trustworthy.
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

    cl = np.load(os.path.join(OUTPUT_DIR, 'cleaned_labels_v1.npz'))
    consensus = cl['consensus'].astype(np.float64)
    # Map consensus to sample weights: uncertain (low consensus) gets lower weight
    # but never exactly 0 to keep gradient signal.
    sample_weight = 0.1 + 0.9 * consensus

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)

    for fi, (ti, vi) in enumerate(skf.split(X, y)):
        pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
        model = lgb.LGBMClassifier(
            n_estimators=3000,
            max_depth=8,
            learning_rate=0.02,
            num_leaves=127,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            scale_pos_weight=pw,
            random_state=SEED + fi,
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(
            X[ti], y[ti],
            sample_weight=sample_weight[ti],
            eval_set=[(X[vi], y[vi])],
            eval_sample_weight=[sample_weight[vi]],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof[vi] = model.predict_proba(X[vi])[:, 1]
        fold_auc = roc_auc_score(y[vi], oof[vi])
        print(f'  Fold {fi+1}: AUC={fold_auc:.4f}')

    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (oof > th).astype(int)
        if pred.sum() == 0:
            continue
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    pred = (oof > best_th).astype(int)
    print(f'\nConfident-weighted LGB: F1={f1_score(y, pred):.4f}, '
          f'Rec={recall_score(y, pred):.4f}, '
          f'Prec={precision_score(y, pred, zero_division=0):.4f}, '
          f'AUC={roc_auc_score(y, oof):.4f}, th={best_th:.3f}')

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'confident_weighted_lgb_oof.npz'),
        oof_confident_weighted_lgb=oof,
        flags=y,
    )
    print(f'Saved to {os.path.join(OUTPUT_DIR, "confident_weighted_lgb_oof.npz")}')


if __name__ == '__main__':
    main()
