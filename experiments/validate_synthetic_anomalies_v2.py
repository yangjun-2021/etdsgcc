"""Fast validation of PeerJ-style synthetic anomaly injection (v2).

Uses AMSTTrainer's built-in synthetic anomaly option to avoid data-handling issues.
"""
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
from src.models.amst_net import AMSTNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import numpy as np
import torch

seed_everything(SEED)

pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed_raw_3ch.npz'))
X_seq = pre['X_seq']
flags = pre['flags']

prior_data = np.load(os.path.join(OUTPUT_DIR, 'strong_gbdt_prior.npz'))
oof_prior = prior_data['prior']

train_idx, val_idx = train_test_split(
    np.arange(len(flags)), test_size=0.2, random_state=SEED, stratify=flags)

base_kwargs = dict(
    dataset='sgcc', use_diffaug=False, use_supcon=False, use_coteaching=False, use_prior=True,
    d_mamba=32, d_trans=64, d_freq=32, proj_dim=64,
    n_mamba_layers=1, n_trans_layers=1, n_heads=2, dropout=0.2,
    epochs=10, batch_size=128, lr=1e-4, patience=10, recall_weight=5.0, use_amp=True,
)


def run(name, use_syn):
    print(f'\n=== {name} ===')
    t0 = time.time()
    kwargs = base_kwargs.copy()
    if use_syn:
        kwargs['use_synthetic_anomalies'] = True
        kwargs['synthetic_anomaly_kwargs'] = dict(
            anomaly_types=['point', 'contextual', 'collective'],
            point_lambda=0.5, contextual_lambda=1.0, contextual_k=7,
            collective_lambda=0.5, seed=SEED,
        )
    trainer = AMSTTrainer(device='cuda', **kwargs)
    X_train, y_train = X_seq[train_idx], flags[train_idx]
    X_val, y_val = X_seq[val_idx], flags[val_idx]
    X_tr_aug, y_tr_aug = trainer._augment(X_train, y_train)
    prior_aug = np.concatenate([
        oof_prior[train_idx],
        np.full(len(y_tr_aug) - len(y_train), oof_prior[train_idx].mean(), dtype=np.float32)
    ])
    train_loader = trainer._build_loaders(X_tr_aug, y_tr_aug, prior=prior_aug, shuffle=True)
    val_loader = trainer._build_loaders(X_val, y_val, prior=oof_prior[val_idx], shuffle=False)
    model = AMSTNet(
        in_channels=X_seq.shape[1], seq_len=X_seq.shape[2],
        d_mamba=32, d_trans=64, d_freq=32, proj_dim=64,
        n_mamba_layers=1, n_trans_layers=1, n_heads=2, dropout=0.2,
        use_freq=True, use_supcon=False, prior_dim=1,
    )
    model = trainer._train_single_network(model, train_loader, val_loader, y_val,
        epochs=trainer.epochs, lr=trainer.lr, weight_decay=trainer.weight_decay,
        patience=trainer.patience, fold_idx=0)
    val_proba = trainer._predict_proba(model, val_loader)
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (val_proba > th).astype(int)
        if pred.sum() == 0: continue
        f = f1_score(y_val, pred, zero_division=0)
        if f > best_f1: best_f1, best_th = f, th
    pred = (val_proba > best_th).astype(int)
    print(f'Val: F1={f1_score(y_val, pred):.4f}, Rec={recall_score(y_val, pred):.4f}, '
          f'Prec={precision_score(y_val, pred, zero_division=0):.4f}, '
          f'AUC={roc_auc_score(y_val, val_proba):.4f}, th={best_th:.3f}, time={(time.time()-t0)/60:.1f}min')
    return best_f1


f1_base = run('Baseline', use_syn=False)
f1_syn = run('Synthetic anomalies', use_syn=True)
print(f'\nBaseline={f1_base:.4f}  Synthetic={f1_syn:.4f}  Delta={f1_syn-f1_base:+.4f}')
