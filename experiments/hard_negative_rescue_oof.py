"""Train a hard-negative rescue model on original SGCC labels.

The rescue model predicts which positives are missed by the current best OOF
(`final-blend-best-oof-final-blend-best`). Its OOF probabilities are saved so
that ImprovedMetaLearner v2 can discover and blend them.
"""
import os
import sys
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything
from src.training.meta_learner_v2 import _best_f1_score, _load_internal_oofs

seed_everything(SEED)


def main():
    # Original labels
    cl = np.load(os.path.join(OUTPUT_DIR, 'cleaned_labels_v1.npz'))
    y_orig = cl['y_orig'].astype(int)

    # Load all internal OOFs and evaluate on original labels
    internal = _load_internal_oofs(y_orig)
    oof_f1s = []
    for nm, oof in internal.items():
        try:
            f1, _, _, _, _ = _best_f1_score(y_orig, oof)
            oof_f1s.append((nm, f1, oof))
        except Exception:
            pass
    oof_f1s.sort(key=lambda x: x[1], reverse=True)
    print(f'Loaded {len(internal)} OOFs')
    for nm, f1, _ in oof_f1s[:10]:
        print(f'  {nm:50s}: F1={f1:.4f}')

    best_nm, best_f1, best_oof = oof_f1s[0]
    print(f'\nBest OOF: {best_nm} F1={best_f1:.4f}')

    # False-negative mask for the best OOF at its optimal threshold
    _, best_th, _, _, _ = _best_f1_score(y_orig, best_oof)
    best_pred = (best_oof > best_th).astype(int)
    fn_mask = (best_pred == 0) & (y_orig == 1)
    print(f'Best OOF FN count: {fn_mask.sum()} / {(y_orig == 1).sum()} positives')

    # Feature matrix: top-50 OOFs on original labels
    top_n = 50
    top_oofs = [oof for _, _, oof in oof_f1s[:top_n]]
    P = np.column_stack(top_oofs)
    P = np.nan_to_num(P, nan=0.5, posinf=1.0, neginf=0.0)

    # Train a rescue model to predict FN mask
    print('\nTraining hard-negative rescue model...')
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rescue_oof = np.zeros(len(y_orig))
    for fi, (ti, vi) in enumerate(skf.split(P, fn_mask)):
        pw = (fn_mask[ti] == 0).sum() / max((fn_mask[ti] == 1).sum(), 1)
        model = xgb.XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            scale_pos_weight=pw, tree_method='hist',
            verbosity=0, random_state=SEED)
        model.fit(P[ti], fn_mask[ti].astype(int))
        rescue_oof[vi] = model.predict_proba(P[vi])[:, 1]
        fold_f1 = f1_score(fn_mask[vi], (rescue_oof[vi] > 0.5).astype(int), zero_division=0)
        fold_auc = roc_auc_score(fn_mask[vi], rescue_oof[vi])
        print(f'  Fold {fi+1}: rescue AUC={fold_auc:.4f}, F1={fold_f1:.4f}')

    rescue_auc = roc_auc_score(fn_mask, rescue_oof)
    rescue_f1 = f1_score(fn_mask, (rescue_oof > 0.5).astype(int), zero_division=0)
    print(f'Rescue OOF overall: AUC={rescue_auc:.4f}, F1={rescue_f1:.4f}')

    # Save rescue OOF
    save_path = os.path.join(OUTPUT_DIR, 'hard_negative_rescue_oof.npz')
    np.savez_compressed(save_path, oof_hard_negative_rescue=rescue_oof)
    print(f'Saved rescue OOF to {save_path}')

    # Quick test: combine best OOF with rescue signal
    print('\nCombining best OOF with rescue signal...')
    best_alpha = 0
    best_combined_f1 = best_f1
    for alpha in np.arange(0.0, 0.51, 0.05):
        combined = best_oof + alpha * rescue_oof
        combined = np.clip(combined, 0, 1)
        f1, _, _, _, _ = _best_f1_score(y_orig, combined)
        if f1 > best_combined_f1:
            best_combined_f1 = f1
            best_alpha = alpha
        print(f'  alpha={alpha:.2f}: combined F1={f1:.4f}')

    combined = np.clip(best_oof + best_alpha * rescue_oof, 0, 1)
    combined_path = os.path.join(OUTPUT_DIR, 'hard_negative_rescue_combined_oof.npz')
    np.savez_compressed(combined_path, oof_rescue_combined=combined)
    print(f'Saved best combined OOF (alpha={best_alpha}) to {combined_path}')


if __name__ == '__main__':
    main()
