"""Retrain Expert A (GBDT) on cleaned labels."""
import os, sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SEED
from src.utils.utils import seed_everything
from src.training.expert_a import ExpertATrainer

seed_everything(SEED)

pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
stat_features = pre['stat_features']
impute_mask = pre['impute_mask']

cl = np.load(os.path.join(OUTPUT_DIR, 'cleaned_labels_v1.npz'))
y_clean = cl['y_clean'].astype(int)
y_orig = cl['y_orig'].astype(int)

print(f'Training Expert A on cleaned labels: pos={y_clean.sum()}, neg={(y_clean==0).sum()}')

trainer = ExpertATrainer(dataset='sgcc')
oof_proba, combined_leaf, fold_models = trainer.train(stat_features, y_clean, impute_mask=impute_mask)

# Save with both cleaned and original labels
np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'sgcc_expert_a_cleaned.npz'),
    oof_proba=oof_proba,
    leaf_indices_lgb=combined_leaf[:, :100],
    leaf_indices_xgb=combined_leaf[:, 100:],
    y_clean=y_clean,
    y_orig=y_orig,
)

# Evaluate on cleaned labels
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
best_f1, best_th = 0, 0.5
for th in np.arange(0.05, 0.95, 0.005):
    pred = (oof_proba > th).astype(int)
    if pred.sum() == 0: continue
    f = f1_score(y_clean, pred, zero_division=0)
    if f > best_f1: best_f1, best_th = f, th
pred = (oof_proba > best_th).astype(int)
print(f'On cleaned labels: F1={best_f1:.4f}, Rec={recall_score(y_clean,pred):.4f}, '
      f'Prec={precision_score(y_clean,pred,zero_division=0):.4f}, AUC={roc_auc_score(y_clean,oof_proba):.4f}, th={best_th:.3f}')

# Evaluate on original labels
best_f1, best_th = 0, 0.5
for th in np.arange(0.05, 0.95, 0.005):
    pred = (oof_proba > th).astype(int)
    if pred.sum() == 0: continue
    f = f1_score(y_orig, pred, zero_division=0)
    if f > best_f1: best_f1, best_th = f, th
pred = (oof_proba > best_th).astype(int)
print(f'On original labels: F1={best_f1:.4f}, Rec={recall_score(y_orig,pred):.4f}, '
      f'Prec={precision_score(y_orig,pred,zero_division=0):.4f}, AUC={roc_auc_score(y_orig,oof_proba):.4f}, th={best_th:.3f}')
