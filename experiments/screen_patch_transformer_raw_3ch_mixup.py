"""Screen PatchTransformer on raw 3ch with Mixup augmentation."""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED
from src.utils.utils import seed_everything
from src.models.patch_transformer import train_patch_transformer, predict_patch_transformer
from src.data.ts_augment import mixup_augment
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import numpy as np
import torch

seed_everything(SEED)

pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed_raw_3ch.npz'))
X_seq = pre['X_seq']
flags = pre['flags']

prior_data = np.load(os.path.join(OUTPUT_DIR, 'strong_gbdt_prior.npz'))
oof_prior = prior_data['prior']

train_idx, val_idx = train_test_split(
    np.arange(len(flags)), test_size=0.2, random_state=SEED, stratify=flags)

rng = np.random.RandomState(SEED)
X_train_aug, y_train_aug = mixup_augment(X_seq[train_idx], flags[train_idx], alpha=0.2, rng=rng)
# prior for mixed samples: average of the two original priors? Simpler: tile original prior for mixed copies.
# mixup_augment returns original + mixed, with original first.
n_orig = len(train_idx)
n_mixed = len(y_train_aug) - n_orig
prior_train_aug = np.concatenate([oof_prior[train_idx], np.full(n_mixed, oof_prior[train_idx].mean(), dtype=np.float32)])
print(f'After Mixup: X_train_aug {X_train_aug.shape}, theft rate {y_train_aug.mean()*100:.2f}%', flush=True)

CFG = {
    'patch_len': 30,
    'stride': 15,
    'd_model': 64,
    'n_layers': 2,
    'n_heads': 4,
    'dropout': 0.2,
    'epochs': 15,
    'batch_size': 128,
    'lr': 3e-4,
}

t0 = time.time()
model = train_patch_transformer(
    X_train_aug, y_train_aug,
    oof_prior=prior_train_aug,
    device='cuda', seed=SEED, verbose=True,
    **CFG
)
val_probs = predict_patch_transformer(model, X_seq[val_idx], oof_prior[val_idx], device='cuda')
val_probs = np.nan_to_num(val_probs, nan=0.5)
elapsed = (time.time() - t0) / 60

best_f1, best_th = 0, 0.5
for th in np.arange(0.05, 0.95, 0.005):
    pred = (val_probs > th).astype(int)
    if pred.sum() == 0: continue
    f = f1_score(flags[val_idx], pred, zero_division=0)
    if f > best_f1: best_f1, best_th = f, th
pred = (val_probs > best_th).astype(int)
print(f'Val: F1={f1_score(flags[val_idx], pred):.4f}, Rec={recall_score(flags[val_idx], pred):.4f}, '
      f'Prec={precision_score(flags[val_idx], pred, zero_division=0):.4f}, '
      f'AUC={roc_auc_score(flags[val_idx], val_probs):.4f}, th={best_th:.3f}, time={elapsed:.1f}min',
      flush=True)
