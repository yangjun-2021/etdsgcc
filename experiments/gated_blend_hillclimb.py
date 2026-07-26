"""Gated blend: predict when hillclimb is likely wrong, then use max of other signals.

Hypothesis: if a gating model can identify hillclimb's mistakes, switching to
more aggressive signals for those samples may improve overall F1.
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
    y = np.load(os.path.join(OUTPUT_DIR, 'hillclimb_best_oof.npz'))['flags']
    hill = np.load(os.path.join(OUTPUT_DIR, 'hillclimb_best_oof.npz'))['oof_hillclimb']
    pred_hill = (hill > 0.52).astype(int)
    error = (pred_hill != y).astype(int)

    pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
    X = pre['stat_features'].astype(np.float32)
    miss = pre['impute_mask'].mean(axis=1).reshape(-1, 1)
    X = np.nan_to_num(np.column_stack([X, miss, hill.reshape(-1, 1)]), nan=0, posinf=0, neginf=0)

    # Gating model: predict hillclimb error
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    gate_proba = np.zeros(len(y))
    for fi, (ti, vi) in enumerate(skf.split(X, error)):
        m = lgb.LGBMClassifier(
            n_estimators=1000, max_depth=6, learning_rate=0.03,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(error[ti]==0).sum()/max((error[ti]==1).sum(),1),
            random_state=SEED+fi, verbose=-1, n_jobs=-1,
        )
        m.fit(X[ti], error[ti], eval_set=[(X[vi], error[vi])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        gate_proba[vi] = m.predict_proba(X[vi])[:, 1]
    print(f'Gating AUC={roc_auc_score(error, gate_proba):.4f}')

    # Alternative signals
    sigs = {
        'v3': np.load(os.path.join(OUTPUT_DIR, 'amst_3ch_synthetic_mixed_ls_v3_oof.npz'))['oof_amst_3ch_synthetic_mixed_ls_v3'],
        'mega': np.load(os.path.join(OUTPUT_DIR, 'sgcc_mega_meta.npz'))['oof_final'],
        'auto': np.load(os.path.join(OUTPUT_DIR, 'autoresearch_best.npz'))['oof_final'],
        'boost': np.load(os.path.join(OUTPUT_DIR, 'mega_boost_enhanced.npz'))['oof_final'],
    }
    alt_max = np.column_stack(list(sigs.values())).max(axis=1)

    # Blend: if gate says error likely, use alt_max, else hill
    best = (0,0,0,0,0)
    for gate_th in np.arange(0.1, 0.91, 0.02):
        blended = np.where(gate_proba > gate_th, alt_max, hill)
        for th in np.arange(0.1, 0.95, 0.01):
            pred = (blended > th).astype(int)
            if pred.sum() == 0:
                continue
            f1 = f1_score(y, pred, zero_division=0)
            if f1 > best[0]:
                best = (f1, gate_th, th, recall_score(y, pred, zero_division=0), precision_score(y, pred, zero_division=0))
    print(f'Best gated blend: F1={best[0]:.4f}, gate_th={best[1]:.2f}, th={best[2]:.2f}, Rec={best[3]:.4f}, Prec={best[4]:.4f}')

    # Also try always using alt_max for high gate, but hill for low
    # baseline hillclimb
    raw_f1 = f1_score(y, pred_hill)
    print(f'Raw hillclimb: F1={raw_f1:.4f}, Rec={recall_score(y,pred_hill):.4f}, Prec={precision_score(y,pred_hill):.4f}')


if __name__ == '__main__':
    main()
