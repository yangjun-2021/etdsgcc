"""Fast validation of PeerJ-style synthetic anomaly injection.

Trains a small AMST on raw 3ch SGCC with and without synthetic anomalies
and compares validation F1/recall on a single stratified split.
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
from src.data.synthetic_anomalies import SyntheticAnomalyAugmenter
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

base_trainer_kwargs = dict(
    dataset='sgcc', use_diffaug=False, use_supcon=False, use_coteaching=False, use_prior=True,
    d_mamba=32, d_trans=64, d_freq=32, proj_dim=64,
    n_mamba_layers=1, n_trans_layers=1, n_heads=2, dropout=0.2,
    epochs=15, batch_size=128, lr=1e-4, patience=10, recall_weight=5.0, use_amp=True,
)


def run_condition(name, X_train, y_train, prior_train, use_syn=False):
    print(f'\n=== Condition: {name} ===')
    t0 = time.time()
    trainer = AMSTTrainer(device='cuda', **base_trainer_kwargs)
    X_tr_aug, y_tr_aug = trainer._augment(X_train, y_train)

    if use_syn:
        syn_aug = SyntheticAnomalyAugmenter(
            n_synthetic=int((y_train == 1).sum()),
            anomaly_types=['point', 'contextual', 'collective'],
            point_lambda=0.5,
            contextual_lambda=1.0,
            contextual_k=7,
            collective_lambda=0.5,
            seed=SEED,
        )
        X_syn, y_syn = syn_aug.fit_transform(X_tr_aug, y_tr_aug)
        X_tr_aug = np.concatenate([X_tr_aug, X_syn], axis=0)
        y_tr_aug = np.concatenate([y_tr_aug, y_syn], axis=0)
        print(f'  Added {len(X_syn)} synthetic anomalies, theft rate={y_tr_aug.mean()*100:.1f}%')

    if prior_train is not None:
        prior_aug = np.concatenate([
            prior_train,
            np.full(len(y_tr_aug) - len(prior_train), prior_train.mean(), dtype=np.float32)
        ])
    else:
        prior_aug = None

    train_loader = trainer._build_loaders(X_tr_aug, y_tr_aug, prior=prior_aug, shuffle=True)
    val_loader = trainer._build_loaders(X_seq[val_idx], flags[val_idx], prior=oof_prior[val_idx], shuffle=False)

    model = AMSTNet(
        in_channels=X_seq.shape[1], seq_len=X_seq.shape[2],
        d_mamba=trainer.d_mamba, d_trans=trainer.d_trans, d_freq=trainer.d_freq, proj_dim=trainer.proj_dim,
        n_mamba_layers=trainer.n_mamba_layers, n_trans_layers=trainer.n_trans_layers, n_heads=trainer.n_heads,
        dropout=trainer.dropout, use_freq=True, use_supcon=trainer.use_supcon, prior_dim=1,
    )
    print(f'  Model params: {sum(p.numel() for p in model.parameters()):,}', flush=True)

    model = trainer._train_single_network(model, train_loader, val_loader, flags[val_idx],
        epochs=trainer.epochs, lr=trainer.lr, weight_decay=trainer.weight_decay,
        patience=trainer.patience, fold_idx=0)
    val_proba = trainer._predict_proba(model, val_loader)

    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (val_proba > th).astype(int)
        if pred.sum() == 0:
            continue
        f = f1_score(flags[val_idx], pred, zero_division=0)
        if f > best_f1:
            best_f1, best_th = f, th
    pred = (val_proba > best_th).astype(int)
    print(f'  Val: F1={f1_score(flags[val_idx], pred):.4f}, Rec={recall_score(flags[val_idx], pred):.4f}, '
          f'Prec={precision_score(flags[val_idx], pred, zero_division=0):.4f}, '
          f'AUC={roc_auc_score(flags[val_idx], val_proba):.4f}, th={best_th:.3f}, time={(time.time()-t0)/60:.1f}min',
          flush=True)
    return best_f1


if __name__ == '__main__':
    X_train, y_train = X_seq[train_idx], flags[train_idx]
    prior_train = oof_prior[train_idx]

    f1_baseline = run_condition('Baseline (no synthetic)', X_train, y_train, prior_train, use_syn=False)
    f1_syn = run_condition('Synthetic anomalies', X_train, y_train, prior_train, use_syn=True)

    print('\n=== Summary ===')
    print(f'Baseline F1: {f1_baseline:.4f}')
    print(f'Synthetic  F1: {f1_syn:.4f}')
    print(f'Delta: {f1_syn - f1_baseline:+.4f}')
