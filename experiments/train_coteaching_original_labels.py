"""Co-teaching on raw 3-channel sequences using ORIGINAL SGCC labels.

This is the first concrete step of the original-label retraining plan.
It trains two TCN-like networks that cross-supervise each other on small-loss
samples, filtering out the ~744 label-noise samples in y_orig.

Usage:
    python experiments/train_coteaching_original_labels.py

Requires: GPU (set device='cuda'). CPU training is prohibitively slow.
"""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.training.coteaching import train_coteaching_cv
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score


def main():
    print('[CoTeaching-Original] Loading preprocessed raw 3ch data...')
    pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed_raw_3ch.npz'))
    X_seq = pre['X_seq'].astype(np.float32)

    # Use ORIGINAL labels, not the cleaned flags stored in the npz
    cl = np.load(os.path.join(OUTPUT_DIR, 'cleaned_labels_v1.npz'))
    y_orig = cl['y_orig'].astype(int)
    y_clean = cl['y_clean'].astype(int)

    print(f'X_seq shape: {X_seq.shape}, y_orig: {y_orig.shape}')
    print(f'Original positives: {y_orig.sum()} / {len(y_orig)} '
          f'({y_orig.mean()*100:.2f}%)')
    print(f'Cleaned vs original diff: {(y_clean != y_orig).sum()} samples')

    # Use the existing strong prior (trained on cleaned labels) as a starting point.
    # Ideally this should be replaced with a prior retrained on y_orig (see plan).
    prior_path = os.path.join(OUTPUT_DIR, 'stronger_gbdt_prior_v3.npz')
    prior = np.load(prior_path)['prior'].astype(np.float32)
    print(f'Prior shape: {prior.shape}')

    oof = train_coteaching_cv(
        X_seq, y_orig,            # <-- key change: original labels
        leaf_indices=None,
        oof_prior=prior,
        n_folds=N_FOLDS,
        seed=SEED,
        device='cuda',            # <-- requires GPU
        tcn_channels=[32, 32, 32, 16],
        kernel_size=5,
        dropout=0.3,
        proj_dim=64,
        epochs=50,
        batch_size=128,
        lr=3e-4,
        supcon_weight=0.3,
        supcon_temp=0.07,
        sce_alpha=1.0,
        sce_beta=0.5,
        forget_rate=0.15,         # filter ~15% highest-loss samples per epoch
        warmup_epochs=10,
    )

    # Evaluate on original labels
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (oof > th).astype(int)
        if pred.sum() == 0:
            continue
        f = f1_score(y_orig, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_th = f, th
    pred = (oof > best_th).astype(int)
    print(f'\nCo-teaching raw 3ch on ORIGINAL labels: '
          f'F1={f1_score(y_orig, pred):.4f}, '
          f'Rec={recall_score(y_orig, pred):.4f}, '
          f'Prec={precision_score(y_orig, pred, zero_division=0):.4f}, '
          f'AUC={roc_auc_score(y_orig, oof):.4f}, th={best_th:.3f}')

    save_path = os.path.join(OUTPUT_DIR, 'coteaching_original_oof.npz')
    np.savez_compressed(
        save_path,
        oof_coteaching_original=oof,
        y_orig=y_orig,
    )
    print(f'Saved to {save_path}')


if __name__ == '__main__':
    main()
