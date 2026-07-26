"""Train Informer 3ch with PeerJ-style synthetic anomaly injection.

Mirrors the AMST synthetic-anomaly experiment for architecture diversity.
Synthetic anomalies are injected into each training fold (no val leakage).
"""
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
from src.data.synthetic_anomalies import SyntheticAnomalyAugmenter
from sklearn.model_selection import StratifiedKFold
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

prior_data = np.load(os.path.join(OUTPUT_DIR, 'strong_gbdt_prior.npz'))
oof_prior = prior_data['prior']

# Strong config (matches current best Informer 3ch strong prior)
CONFIG = {
    'd_model': 64,
    'n_heads': 4,
    'num_layers': 2,
    'dropout': 0.3,
    'epochs': 40,
    'batch_size': 32,
    'lr': 3e-4,
    'use_amp': True,
}

# PeerJ-style synthetic anomaly kwargs
SYN_KWARGS = dict(
    anomaly_types=['point', 'contextual', 'collective'],
    point_lambda=0.5,
    contextual_lambda=1.0,
    contextual_k=7,
    collective_lambda=0.5,
    n_synthetic=int((flags == 1).sum()) // N_FOLDS,  # per fold
)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(flags), dtype=np.float32)

for fi, (ti, vi) in enumerate(skf.split(X_seq, flags)):
    print(f'\n=== Fold {fi+1}/{N_FOLDS} ===')
    t0 = time.time()

    X_train, y_train = X_seq[ti], flags[ti]
    X_val, y_val = X_seq[vi], flags[vi]
    prior_train, prior_val = oof_prior[ti], oof_prior[vi]

    # Inject synthetic anomalies into training fold only
    aug = SyntheticAnomalyAugmenter(seed=SEED + fi, **SYN_KWARGS)
    X_syn, y_syn = aug.fit_transform(X_train, y_train)
    X_train_aug = np.concatenate([X_train, X_syn], axis=0)
    y_train_aug = np.concatenate([y_train, y_syn], axis=0)
    prior_train_aug = np.concatenate([
        prior_train,
        np.full(len(y_syn), prior_train.mean(), dtype=np.float32)
    ])
    print(f'  After synthetic aug: {X_train_aug.shape}, theft rate={y_train_aug.mean()*100:.2f}%')

    model = train_informer(
        X_train_aug, y_train_aug,
        oof_prior=prior_train_aug,
        device='cuda', seed=SEED + fi, verbose=True,
        **CONFIG
    )
    probs = predict_informer(model, X_val, prior_val, device='cuda')
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
          f'AUC={roc_auc_score(flags[vi], oof[vi]):.4f}, th={best_th:.3f}, '
          f'time={(time.time()-t0)/60:.1f}min')

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'informer_3ch_synthetic_fold{fi}.pt'))
    del model
    torch.cuda.empty_cache()

overall_f1, overall_th = 0, 0.5
for th in np.arange(0.05, 0.95, 0.005):
    pred = (oof > th).astype(int)
    if pred.sum() == 0:
        continue
    f = f1_score(flags, pred, zero_division=0)
    if f > overall_f1:
        overall_f1, overall_th = f, th
pred = (oof > overall_th).astype(int)
print(f'\n=== Overall Informer 3ch synthetic anomalies ===')
print(f'F1={f1_score(flags, pred):.4f}, Rec={recall_score(flags, pred):.4f}, '
      f'Prec={precision_score(flags, pred, zero_division=0):.4f}, '
      f'AUC={roc_auc_score(flags, oof):.4f}, th={overall_th:.3f}')

np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'informer_3ch_synthetic_oof.npz'),
    oof_informer_3ch_synthetic=oof,
    flags=flags,
)
print(f'Saved to {os.path.join(OUTPUT_DIR, "informer_3ch_synthetic_oof.npz")}')
