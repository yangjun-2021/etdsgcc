"""Retrain Expert A (GBDT ensemble) on original SGCC labels.

This breaks the circular dependence on cleaned-label OOFs and provides a
fresh, complementary OOF for the meta-learner.
"""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SEED
from src.utils.utils import seed_everything
from src.data.preprocess_sgcc import preprocess_sgcc
from src.training.expert_a import ExpertATrainer
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

seed_everything(SEED)


def best_f1(y_true, y_prob):
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (y_prob > th).astype(int)
        if pred.sum() == 0:
            continue
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_th = f, th
    return best_f1, best_th


def main():
    pre_path = os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz')
    if not os.path.exists(pre_path):
        print('Preprocessed features not found; running preprocessing (advanced features skipped)...')
        preprocess_sgcc(use_advanced_features=False)

    print('Loading preprocessed features...')
    pre = np.load(pre_path)
    stat_features = pre['stat_features']
    impute_mask = pre['impute_mask']

    cl = np.load(os.path.join(OUTPUT_DIR, 'cleaned_labels_v1.npz'))
    y_orig = cl['y_orig'].astype(int)
    y_clean = cl['y_clean'].astype(int)

    print(f'Training Expert A on ORIGINAL labels: pos={y_orig.sum()}, neg={(y_orig == 0).sum()}')

    trainer = ExpertATrainer(dataset='sgcc')
    oof_proba, combined_leaf, fold_models = trainer.train(stat_features, y_orig, impute_mask=impute_mask)

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'sgcc_expert_a_original.npz'),
        oof_proba=oof_proba,
        leaf_indices_lgb=combined_leaf[:, :100],
        leaf_indices_xgb=combined_leaf[:, 100:],
        y_orig=y_orig,
        y_clean=y_clean,
    )

    for label_name, y in [('cleaned', y_clean), ('original', y_orig)]:
        f1, th = best_f1(y, oof_proba)
        pred = (oof_proba > th).astype(int)
        print(f'Expert A original-label OOF on {label_name}: '
              f'F1={f1:.4f}, Rec={recall_score(y, pred):.4f}, '
              f'Prec={precision_score(y, pred, zero_division=0):.4f}, '
              f'AUC={roc_auc_score(y, oof_proba):.4f}, th={th:.3f}')


if __name__ == '__main__':
    main()
