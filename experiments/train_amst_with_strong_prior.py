"""Train AMST-Net with strong GBDT prior on GPU."""
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

prior_data = np.load(os.path.join(OUTPUT_DIR, 'strong_gbdt_prior.npz'))
oof_prior = prior_data['prior']

t0 = time.time()
trainer = AMSTTrainer(
    dataset='sgcc',
    use_diffaug=False,
    use_supcon=False,
    use_coteaching=False,
    use_prior=True,
    d_mamba=64,
    d_trans=128,
    d_freq=64,
    proj_dim=128,
    n_mamba_layers=2,
    n_trans_layers=2,
    n_heads=4,
    dropout=0.2,
    epochs=50,
    batch_size=64,
    lr=1e-4,
    patience=15,
    recall_weight=5.0,  # optimize for recall
)
oof = trainer.train(X_seq, flags, impute_mask=impute_mask, oof_proba_a=oof_prior)

np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'amst_strong_prior_oof.npz'),
    oof_amst_strong_prior=oof,
    flags=flags,
)
print(f'\nSaved to {os.path.join(OUTPUT_DIR, "amst_strong_prior_oof.npz")}')
print(f'Total time: {(time.time()-t0)/60:.1f} minutes')
