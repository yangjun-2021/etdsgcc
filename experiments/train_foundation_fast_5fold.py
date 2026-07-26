"""Train Foundation Encoder full 5-fold with faster config."""
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED
from src.utils.utils import seed_everything
from src.training.foundation_trainer import FoundationTrainer
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

# Faster config: fewer epochs, larger batch
trainer = FoundationTrainer(
    dataset='sgcc_foundation_fast',
    pretrain_epochs=5,
    finetune_epochs=30,
    batch_size=128,
    lr=1e-4,
    weight_decay=1e-4,
    patience=10,
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

results = trainer.train(X_seq, flags, stat_features=stat)

print('\nSaved to', os.path.join(OUTPUT_DIR, 'sgcc_foundation_fast.npz'))
print('Best hybrid F1:', results['best_f1'])
