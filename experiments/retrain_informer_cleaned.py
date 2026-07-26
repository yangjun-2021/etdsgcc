"""Train Informer on cleaned labels with fast config."""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything
from src.models.informer_model import train_informer, predict_informer
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
flags = np.load(os.path.join(OUTPUT_DIR, 'cleaned_labels_v1.npz'))['y_clean'].astype(int)
y_orig = np.load(os.path.join(OUTPUT_DIR, 'cleaned_labels_v1.npz'))['y_orig'].astype(int)

a = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_a_cleaned.npz'))
oof_proba_a = a['oof_proba']

# Fast config
CONFIG = {
    'd_model': 64,
    'n_heads': 4,
    'num_layers': 2,
    'dropout': 0.3,
    'epochs': 40,
    'batch_size': 32,
    'lr': 3e-4,
}

t0 = time.time()
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(flags), dtype=np.float32)

completed_folds = []
for fi in range(N_FOLDS):
    fpath = os.path.join(OUTPUT_DIR, f'informer_cleaned_fold{fi}.npz')
    if os.path.exists(fpath):
        d = np.load(fpath)
        oof[d['vi']] = d['oof']
        completed_folds.append(fi)
        print(f'Resumed fold {fi+1} from {fpath}')

for fi, (ti, vi) in enumerate(skf.split(X_seq, flags)):
    if fi in completed_folds:
        print(f'Fold {fi+1}/{N_FOLDS} already done, skipping')
        continue

    print(f'\n=== Fold {fi+1}/{N_FOLDS} ===')
    tf = time.time()
    model = train_informer(
        X_seq[ti], flags[ti],
        oof_prior=oof_proba_a[ti],
        device='cuda', seed=SEED + fi, verbose=True,
        **CONFIG
    )
    probs = predict_informer(model, X_seq[vi], oof_proba_a[vi], device='cuda')
    oof[vi] = np.nan_to_num(probs, nan=0.5)

    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (oof[vi] > th).astype(int)
        if pred.sum() == 0:
            continue
        f = f1_score(flags[vi], pred, zero_division=0)
        if f > best_f1:
            best_f1, best_th = f, th
    pred = (oof[vi] > best_th).astype(int)
    print(f'  Fold {fi+1}: F1={f1_score(flags[vi], pred):.4f}, '
          f'Rec={recall_score(flags[vi], pred):.4f}, '
          f'Prec={precision_score(flags[vi], pred, zero_division=0):.4f}, '
          f'AUC={roc_auc_score(flags[vi], oof[vi]):.4f}, th={best_th:.3f}')

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, f'informer_cleaned_fold{fi}.npz'),
        oof=oof[vi],
        vi=vi,
        ti=ti,
    )
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'informer_cleaned_fold{fi}.pt'))
    del model
    torch.cuda.empty_cache()
    print(f'  Fold {fi+1} time: {(time.time()-tf)/60:.1f}min')

overall_f1, overall_th = 0, 0.5
for th in np.arange(0.05, 0.95, 0.005):
    pred = (oof > th).astype(int)
    if pred.sum() == 0:
        continue
    f = f1_score(flags, pred, zero_division=0)
    if f > overall_f1:
        overall_f1, overall_th = f, th
pred = (oof > overall_th).astype(int)
print(f'\n=== Overall Informer Cleaned (cleaned labels) ===')
print(f'F1={f1_score(flags, pred):.4f}, Rec={recall_score(flags, pred):.4f}, '
      f'Prec={precision_score(flags, pred, zero_division=0):.4f}, '
      f'AUC={roc_auc_score(flags, oof):.4f}, th={overall_th:.3f}')

overall_f1, overall_th = 0, 0.5
for th in np.arange(0.05, 0.95, 0.005):
    pred = (oof > th).astype(int)
    if pred.sum() == 0:
        continue
    f = f1_score(y_orig, pred, zero_division=0)
    if f > overall_f1:
        overall_f1, overall_th = f, th
pred = (oof > overall_th).astype(int)
print(f'\n=== Overall Informer Cleaned (original labels) ===')
print(f'F1={f1_score(y_orig, pred):.4f}, Rec={recall_score(y_orig, pred):.4f}, '
      f'Prec={precision_score(y_orig, pred, zero_division=0):.4f}, '
      f'AUC={roc_auc_score(y_orig, oof):.4f}, th={overall_th:.3f}')

np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'informer_cleaned_oof.npz'),
    oof_informer_cleaned=oof,
    y_clean=flags,
    y_orig=y_orig,
)
print(f'\nSaved to {os.path.join(OUTPUT_DIR, "informer_cleaned_oof.npz")}')
print(f'Total time: {(time.time()-t0)/60:.1f} minutes')
