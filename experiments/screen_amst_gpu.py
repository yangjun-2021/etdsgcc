"""Quick single-split screen of AMST-Net on GPU."""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED
from src.utils.utils import seed_everything
from src.training.amst_trainer import AMSTTrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import numpy as np
import torch

seed_everything(SEED)

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
X_seq = pre['X_seq']
flags = pre['flags']
impute_mask = pre['impute_mask']

a = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_a.npz'))
oof_proba_a = a['oof_proba']

# Single stratified split: fold 1 = validation, fold 0 = train
train_idx, val_idx = train_test_split(
    np.arange(len(flags)), test_size=0.2, random_state=SEED, stratify=flags)
fold_assignments = np.zeros(len(flags), dtype=int)
fold_assignments[val_idx] = 1

configs = [
    {'d_mamba': 64, 'd_trans': 128, 'n_mamba_layers': 2, 'n_trans_layers': 2, 'n_heads': 4, 'epochs': 30, 'batch_size': 64, 'dropout': 0.2},
    {'d_mamba': 64, 'd_trans': 128, 'n_mamba_layers': 2, 'n_trans_layers': 2, 'n_heads': 4, 'epochs': 30, 'batch_size': 64, 'dropout': 0.2, 'use_supcon': True},
]

for cfg in configs:
    print('\n' + '='*60)
    print(f'AMST Config: {cfg}')
    print('='*60)
    t0 = time.time()
    trainer = AMSTTrainer(
        dataset='sgcc',
        use_diffaug=False,
        use_supcon=cfg.pop('use_supcon', False),
        use_coteaching=False,
        use_prior=True,
        patience=10,
        **cfg
    )
    oof = trainer.train(X_seq, flags, impute_mask=impute_mask, oof_proba_a=oof_proba_a, fold_assignments=fold_assignments)
    elapsed = (time.time() - t0) / 60
    
    val_probs = oof[val_idx]
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
    torch.cuda.empty_cache()
