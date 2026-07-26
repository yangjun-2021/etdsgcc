"""Speed test Foundation encoder with minimal config."""
import os, sys, time

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

pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
X_seq = pre['X_seq'][:5000]  # subset for speed test
flags = pre['flags'][:5000]
stat = pre['stat_features'][:5000]
print(f'Subset: {X_seq.shape}')

# Test with pretrain=0, small epochs
trainer = FoundationTrainer(
    dataset='sgcc_speed_test',
    pretrain_epochs=0,
    finetune_epochs=2,
    batch_size=128,
    lr=1e-4,
    d_model=64,
    n_layers=2,
    n_heads=4,
    patience=12,
    device='cuda',
)

fa = np.zeros(len(flags), dtype=int)
fa[-1000:] = 1  # last 1000 as val

t0 = time.time()
results = trainer.train(X_seq, flags, stat_features=stat, fold_assignments=fa)
print(f'Total time: {(time.time()-t0)/60:.1f} min')
