"""
AMST-Net Smoke Test: tiny model + tiny data + short training.

This is a pure algorithm sanity check on CPU. It does NOT:
- Load Expert-A prior
- Use DiffAug
- Use advanced features

It only verifies the pipeline runs and loss decreases.

Usage:
    python experiments/amst_smoke_test.py
"""
import argparse
import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import SEED, OUTPUT_DIR
from src.data.preprocess_sgcc import preprocess_sgcc
from src.training.amst_trainer import AMSTTrainer
from src.utils.utils import seed_everything


def stratified_subsample(X_seq, flags, impute_mask, n_samples, n_theft, seed=SEED):
    rng = np.random.RandomState(seed)
    pos_idx = np.where(flags == 1)[0]
    neg_idx = np.where(flags == 0)[0]
    sel_pos = rng.choice(pos_idx, size=n_theft, replace=False)
    sel_neg = rng.choice(neg_idx, size=n_samples - n_theft, replace=False)
    sel = np.concatenate([sel_pos, sel_neg])
    rng.shuffle(sel)
    return X_seq[sel], flags[sel], impute_mask[sel]


def make_fold_assignments(flags, n_folds=2, seed=SEED):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_assignments = np.zeros(len(flags), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf.split(np.zeros(len(flags)), flags)):
        fold_assignments[val_idx] = fold_idx
    return fold_assignments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--n_theft', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seq_stride', type=int, default=4, help='Subsample time dimension: T=1034/stride')
    parser.add_argument('--load-cache', action='store_true', help='Load cached sgcc_preprocessed.npz')
    args = parser.parse_args()

    seed_everything(SEED)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    print("=" * 70)
    print("AMST-Net Smoke Test")
    print(f"  n_samples={args.n_samples}, n_theft={args.n_theft}, epochs={args.epochs}, stride={args.seq_stride}")
    print("=" * 70)

    if args.load_cache and os.path.exists(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz')):
        data = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
        X_seq_full, flags_full, impute_mask_full = data['X_seq'], data['flags'], data['impute_mask']
    else:
        X_seq_full, _, flags_full, impute_mask_full = preprocess_sgcc(use_advanced_features=False)

    X_seq, flags, impute_mask = stratified_subsample(
        X_seq_full, flags_full, impute_mask_full,
        n_samples=args.n_samples, n_theft=args.n_theft
    )

    # Subsample time dimension for speed
    X_seq = X_seq[:, :, ::args.seq_stride]
    impute_mask = impute_mask[:, ::args.seq_stride]
    print(f"Subset shape: {X_seq.shape}, theft rate={flags.mean()*100:.2f}%")

    fold_assignments = make_fold_assignments(flags, n_folds=2, seed=SEED)

    trainer = AMSTTrainer(
        dataset='sgcc',
        use_diffaug=False,
        use_supcon=True,
        use_coteaching=False,
        use_prior=False,  # Do not use previous Expert-A results
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=1e-3,
        patience=10,
        d_mamba=16,
        d_trans=32,
        d_freq=16,
        proj_dim=32,
        n_mamba_layers=1,
        n_trans_layers=1,
        n_heads=2,
        dropout=0.1,
    )

    t0 = time.time()
    oof_proba = trainer.train(X_seq, flags, impute_mask=impute_mask, fold_assignments=fold_assignments)
    elapsed = time.time() - t0
    print(f"\nSmoke test completed in {elapsed:.1f}s")
    print(f"OOF proba shape: {oof_proba.shape}, mean: {oof_proba.mean():.4f}")
    print(f"Flag mean: {flags.mean():.4f}")


if __name__ == '__main__':
    main()
