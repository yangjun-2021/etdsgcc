"""V3 voter: Co-teaching (dual TCN) on raw 3ch sequences, ORIGINAL labels, NO prior feature.

Untainted voter for the v3 consensus label cleaning: unlike
train_coteaching_original_labels.py, this script passes oof_prior=None so no
cleaned-label-derived prior enters the model.

Usage:
    conda run -n ml python experiments/v3_tcn_ct_orig.py

Requires: GPU.
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
    print('[V3Voter-CoTeaching] Loading preprocessed raw 3ch data...')
    pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed_raw_3ch.npz'))
    X_seq = pre['X_seq'].astype(np.float32)

    cl = np.load(os.path.join(OUTPUT_DIR, 'cleaned_labels_v1.npz'))
    y_orig = cl['y_orig'].astype(int)

    print(f'X_seq shape: {X_seq.shape}, y_orig positives: {y_orig.sum()} '
          f'({y_orig.mean()*100:.2f}%)')

    oof = train_coteaching_cv(
        X_seq, y_orig,
        leaf_indices=None,
        oof_prior=None,           # untainted: no cleaned-label-derived prior
        n_folds=N_FOLDS,
        seed=SEED,
        device='cuda',
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
        forget_rate=0.15,
        warmup_epochs=10,
    )

    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (oof > th).astype(int)
        if pred.sum() == 0:
            continue
        f = f1_score(y_orig, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_th = f, th
    pred = (oof > best_th).astype(int)
    print(f'\nV3-voter co-teaching (original labels, no prior): '
          f'F1={f1_score(y_orig, pred):.4f}, '
          f'Rec={recall_score(y_orig, pred):.4f}, '
          f'Prec={precision_score(y_orig, pred, zero_division=0):.4f}, '
          f'AUC={roc_auc_score(y_orig, oof):.4f}, th={best_th:.3f}')

    save_path = os.path.join(OUTPUT_DIR, 'v3voter_tcn_ct_oof.npz')
    np.savez_compressed(
        save_path,
        oof_v3voter_tcn_ct=oof,
        y_orig=y_orig,
    )
    print(f'Saved to {save_path}')


if __name__ == '__main__':
    main()
