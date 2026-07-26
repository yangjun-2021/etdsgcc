"""Train Foundation Encoder on fold 5 only (full data) to verify potential."""
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything
from src.training.foundation_trainer import FoundationTrainer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
import numpy as np
import torch

seed_everything(SEED)

print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
X_seq = pre['X_seq']
stat = pre['stat_features']
flags = pre['flags']
print(f'X_seq: {X_seq.shape}, stat: {stat.shape}')

# Create fold assignment where fold 5 is validation, others are train (fold 0)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_assignments = np.zeros(len(flags), dtype=int)
for fi, (ti, vi) in enumerate(skf.split(X_seq, flags)):
    fold_assignments[vi] = fi

# Only train/evaluate fold 5
mask = (fold_assignments == 0) | (fold_assignments == 4)
X_seq_sub = X_seq[mask]
stat_sub = stat[mask]
flags_sub = flags[mask]
fa_sub = np.where(fold_assignments[mask] == 4, 1, 0).astype(int)

# Smaller config to fit 8.5GB GPU
trainer = FoundationTrainer(
    dataset='sgcc_foundation_fold5',
    pretrain_epochs=15,
    finetune_epochs=80,
    batch_size=48,
    lr=1e-4,
    weight_decay=1e-4,
    patience=15,
    patch_len=30,
    stride=15,
    d_model=64,
    n_layers=2,
    n_heads=4,
    dropout=0.2,
    mask_ratio=0.35,
    lambda_recon=0.3,
    lambda_consistency=0.1,
    focal_alpha=0.75,
    gamma_pos=1.0,
    gamma_neg=3.0,
    use_weighted_sampler=True,
    device='cuda',
)

results = trainer.train(X_seq_sub, flags_sub, stat_features=stat_sub, fold_assignments=fa_sub)

# Evaluate
oof = results['oof_proba_meta']
y = flags_sub
best_f1, best_th = 0, 0.5
for th in np.arange(0.05, 0.95, 0.005):
    pred = (oof > th).astype(int)
    if pred.sum() == 0: continue
    f = f1_score(y, pred, zero_division=0)
    if f > best_f1: best_f1, best_th = f, th
pred = (oof > best_th).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
print(f'\n=== Final Fold 5 ===')
print(f'F1={best_f1:.4f}, Rec={recall_score(y,pred):.4f}, '
      f'Prec={precision_score(y,pred,zero_division=0):.4f}, '
      f'AUC={roc_auc_score(y,oof):.4f}, th={best_th:.3f}')
print(f'TP={tp} FP={fp} FN={fn}')
