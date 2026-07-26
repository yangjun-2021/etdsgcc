"""Train Expert C (Informer) on GPU with full config."""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED
from src.utils.utils import seed_everything
from src.training.expert_c import ExpertCTrainer
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

a = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_a.npz'))
oof_proba_a = a['oof_proba']

t0 = time.time()
trainer = ExpertCTrainer(
    dataset='sgcc',
    d_model=128,
    n_heads=8,
    num_layers=3,
    dropout=0.3,
    epochs=40,
    batch_size=64,
    lr=3e-4,
)
oof_proba_c = trainer.train(X_seq, flags, oof_proba_a=oof_proba_a)
elapsed = (time.time() - t0) / 60
print(f'\nTotal time: {elapsed:.1f} minutes')
