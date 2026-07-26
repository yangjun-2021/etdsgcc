"""Train AMST-Net with strong GBDT prior on GPU (resumable per-fold)."""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything
from src.training.amst_trainer import AMSTTrainer
from sklearn.model_selection import StratifiedKFold
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

prior_data = np.load(os.path.join(OUTPUT_DIR, 'strong_gbdt_prior.npz'))
oof_prior = prior_data['prior']

t0 = time.time()
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(flags), dtype=np.float32)

# Resume: load any already-computed fold OOFs
for fi in range(N_FOLDS):
    fpath = os.path.join(OUTPUT_DIR, f'amst_strong_prior_fold{fi}.npz')
    if os.path.exists(fpath):
        d = np.load(fpath)
        oof[d['vi']] = d['oof']
        print(f'Loaded fold {fi+1} OOF from {fpath}')

for fi, (ti, vi) in enumerate(skf.split(X_seq, flags)):
    if oof[vi].sum() != 0 and (oof[vi] != 0.5).any():
        print(f'\nFold {fi+1}/{N_FOLDS} already computed, skipping')
        continue
    
    print(f'\n=== Fold {fi+1}/{N_FOLDS} ===')
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
        recall_weight=5.0,
    )
    fold_oof = trainer.train(X_seq[ti], flags[ti], impute_mask=impute_mask[ti], oof_proba_a=oof_prior[ti])
    # The trainer returns OOF for the training fold only (vi indices are validation)
    # Actually, AMSTTrainer.train does full 5-fold CV internally if fold_assignments is None.
    # But here we pass only train data, so it does 5-fold on train data and returns OOF for train data.
    # That's not what we want. We want to train on ti and predict on vi.
    # The AMST trainer doesn't have a direct train/predict interface.
    # So we need to use fold_assignments with a single fold.
PY