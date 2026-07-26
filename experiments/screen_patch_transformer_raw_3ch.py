"""Quick single-split screen of Patch Transformer on raw-scale 3ch input."""
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import numpy as np
import torch

seed_everything(SEED)

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed_raw_3ch.npz'))
X_seq = pre['X_seq']
flags = pre['flags']
print(f'X_seq shape: {X_seq.shape}, theft rate: {flags.mean()*100:.2f}%')

prior_data = np.load(os.path.join(OUTPUT_DIR, 'strong_gbdt_prior.npz'))
oof_prior = prior_data['prior']

train_idx, val_idx = train_test_split(
    np.arange(len(flags)), test_size=0.2, random_state=SEED, stratify=flags)

configs = [
    {'patch_len': 30, 'stride': 15, 'd_model': 128, 'n_layers': 4, 'n_heads': 8, 'dropout': 0.2, 'epochs': 30, 'batch_size': 128, 'lr': 3e-4},
    {'patch_len': 24, 'stride': 12, 'd_model': 128, 'n_layers': 4, 'n_heads': 8, 'dropout': 0.2, 'epochs': 30, 'batch_size': 128, 'lr': 3e-4},
    {'patch_len': 30, 'stride': 15, 'd_model': 256, 'n_layers': 4, 'n_heads': 8, 'dropout': 0.2, 'epochs': 30, 'batch_size': 64, 'lr': 3e-4},
]

for cfg in configs:
    print('\n' + '='*60)
    print(f'Config: {cfg}')
    print('='*60)
    t0 = time.time()
    model = train_patch_transformer(
        X_seq[train_idx], flags[train_idx],
        oof_prior=oof_prior[train_idx],
        device='cuda', seed=SEED, verbose=True,
        **cfg
    )
    val_probs = predict_patch_transformer(model, X_seq[val_idx], oof_prior[val_idx], device='cuda')
    val_probs = np.nan_to_num(val_probs, nan=0.5)
    elapsed = (time.time() - t0) / 60

    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (val_probs > th).astype(int)
        if pred.sum() == 0:
            continue
        f = f1_score(flags[val_idx], pred, zero_division=0)
        if f > best_f1:
            best_f1, best_th = f, th
    pred = (val_probs > best_th).astype(int)
    print(f'  Val: F1={f1_score(flags[val_idx], pred):.4f}, Rec={recall_score(flags[val_idx], pred):.4f}, '
          f'Prec={precision_score(flags[val_idx], pred, zero_division=0):.4f}, '
          f'AUC={roc_auc_score(flags[val_idx], val_probs):.4f}, th={best_th:.3f}, time={elapsed:.1f}min')
    del model
    torch.cuda.empty_cache()
