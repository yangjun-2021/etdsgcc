"""SupCon + SCE noise-robust classifier on raw 3ch sequences with prior v3.

A new OOF signal targeting label-noise robustness and hard mimicking-theft cases.
"""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.models.supcon_model import train_supcon_cv
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score


def main():
    pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed_raw_3ch.npz'))
    X_seq = pre['X_seq'].astype(np.float32)
    flags = pre['flags'].astype(int)

    prior = np.load(os.path.join(OUTPUT_DIR, 'stronger_gbdt_prior_v3.npz'))['prior'].astype(np.float32)

    print(f'X_seq shape: {X_seq.shape}, flags: {flags.shape}, prior: {prior.shape}')

    oof = train_supcon_cv(
        X_seq, flags,
        leaf_indices=None,
        oof_prior=prior,
        n_folds=N_FOLDS,
        seed=SEED,
        device='cuda',
        tcn_channels=[32, 32, 32, 16],
        kernel_size=5,
        dropout=0.3,
        proj_dim=64,
        epochs=3,
        batch_size=128,
        lr=3e-4,
        supcon_weight=0.3,
        supcon_temp=0.07,
        sce_alpha=1.0,
        sce_beta=0.5,
    )

    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (oof > th).astype(int)
        if pred.sum() == 0:
            continue
        f1 = f1_score(flags, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    pred = (oof > best_th).astype(int)
    print(f'\nSupCon raw 3ch + prior v3: F1={f1_score(flags, pred):.4f}, '
          f'Rec={recall_score(flags, pred):.4f}, '
          f'Prec={precision_score(flags, pred, zero_division=0):.4f}, '
          f'AUC={roc_auc_score(flags, oof):.4f}, th={best_th:.3f}')

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'supcon_raw_3ch_v3_oof.npz'),
        oof_supcon_raw_3ch_v3=oof,
        flags=flags,
    )
    print(f'Saved to {os.path.join(OUTPUT_DIR, "supcon_raw_3ch_v3_oof.npz")}')


if __name__ == '__main__':
    main()
