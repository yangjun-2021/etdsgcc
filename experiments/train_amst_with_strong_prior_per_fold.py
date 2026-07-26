"""Train AMST-Net fold-by-fold with strong prior, saving each fold OOF."""
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

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
splits = list(skf.split(X_seq, flags))

# Build or resume OOF
oof = np.zeros(len(flags), dtype=np.float32)
completed_folds = []
for fi in range(N_FOLDS):
    fpath = os.path.join(OUTPUT_DIR, f'amst_strong_prior_fold{fi}.npz')
    if os.path.exists(fpath):
        d = np.load(fpath)
        oof[d['vi']] = d['oof']
        completed_folds.append(fi)
        print(f'Resumed fold {fi+1} from {fpath}')

for fi, (ti, vi) in enumerate(splits):
    if fi in completed_folds:
        print(f'Fold {fi+1}/{N_FOLDS} already done, skipping')
        continue

    print(f'\n=== Fold {fi+1}/{N_FOLDS} ===')
    t0 = time.time()
    fold_assignments = np.zeros(len(flags), dtype=int)
    fold_assignments[vi] = 1

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
    # train() will do a single split because fold_assignments has 2 unique values
    fold_oof = trainer.train(X_seq, flags, impute_mask=impute_mask, oof_proba_a=oof_prior, fold_assignments=fold_assignments)
    oof[vi] = fold_oof[vi]

    pred = (oof[vi] > 0.5).astype(int)
    print(f'  Fold {fi+1}: F1={f1_score(flags[vi], pred):.4f}, '
          f'Rec={recall_score(flags[vi], pred):.4f}, '
          f'Prec={precision_score(flags[vi], pred, zero_division=0):.4f}, '
          f'AUC={roc_auc_score(flags[vi], oof[vi]):.4f}')

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, f'amst_strong_prior_fold{fi}.npz'),
        oof=fold_oof[vi],
        vi=vi,
        ti=ti,
    )
    print(f'  Saved fold {fi+1} OOF, time={(time.time()-t0)/60:.1f}min')
    torch.cuda.empty_cache()

# Final evaluation
best_f1, best_th = 0, 0.5
for th in np.arange(0.05, 0.95, 0.005):
    pred = (oof > th).astype(int)
    if pred.sum() == 0:
        continue
    f = f1_score(flags, pred, zero_division=0)
    if f > best_f1:
        best_f1, best_th = f, th
pred = (oof > best_th).astype(int)
print(f'\n=== Overall AMST strong prior ===')
print(f'F1={f1_score(flags, pred):.4f}, Rec={recall_score(flags, pred):.4f}, '
      f'Prec={precision_score(flags, pred, zero_division=0):.4f}, '
      f'AUC={roc_auc_score(flags, oof):.4f}, th={best_th:.3f}')

np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'amst_strong_prior_oof.npz'),
    oof_amst_strong_prior=oof,
    flags=flags,
)
print(f'Saved to {os.path.join(OUTPUT_DIR, "amst_strong_prior_oof.npz")}')
