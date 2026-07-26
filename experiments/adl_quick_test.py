"""
ADL-Net Quick Verification: train on a small stratified subset.

This is for rapid algorithm debugging before running the full dataset.
It does NOT use any external model prior.

Usage:
    python experiments/adl_quick_test.py --n_samples 4000 --n_theft 400 --epochs 20
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

_log_path = os.path.join(OUTPUT_DIR, 'adl_quick_test.log')
_log_fh = open(_log_path, 'w', buffering=1, encoding='utf-8')
sys.stdout = _log_fh
sys.stderr = _log_fh
print(f'Logging to {_log_path}')

from src.data.preprocess_sgcc import preprocess_sgcc
from src.training.adl_trainer import ADLTrainer
from src.utils.utils import seed_everything
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score


def load_preprocessed_from_cache(npz_path):
    """Load cached preprocessed SGCC data."""
    data = np.load(npz_path)
    return data['X_seq'], data['stat_features'], data['flags'], data['impute_mask']


def stratified_subsample(X_seq, flags, impute_mask, n_samples, n_theft, seed=SEED):
    """Sample n_theft positives and n_samples-n_theft negatives."""
    rng = np.random.RandomState(seed)
    pos_idx = np.where(flags == 1)[0]
    neg_idx = np.where(flags == 0)[0]
    assert n_theft <= len(pos_idx), f"Requested {n_theft} theft but only {len(pos_idx)} available"
    assert (n_samples - n_theft) <= len(neg_idx), f"Requested {n_samples - n_theft} normal but only {len(neg_idx)} available"

    sel_pos = rng.choice(pos_idx, size=n_theft, replace=False)
    sel_neg = rng.choice(neg_idx, size=n_samples - n_theft, replace=False)
    sel = np.concatenate([sel_pos, sel_neg])
    rng.shuffle(sel)
    return X_seq[sel], flags[sel], impute_mask[sel]


def make_fold_assignments(flags, n_folds=3, seed=SEED):
    """Create fold assignment array for the trainer."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_assignments = np.zeros(len(flags), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf.split(np.zeros(len(flags)), flags)):
        fold_assignments[val_idx] = fold_idx
    return fold_assignments


def main():
    parser = argparse.ArgumentParser(description='ADL-Net quick subset verification')
    parser.add_argument('--n_samples', type=int, default=4000, help='Total subset size')
    parser.add_argument('--n_theft', type=int, default=400, help='Number of positive samples in subset')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs per fold')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--no-advanced', action='store_true', help='Skip advanced features for faster iteration')
    parser.add_argument('--folds', type=int, default=3, help='Number of CV folds')
    parser.add_argument('--load-cache', action='store_true', help='Load cached sgcc_preprocessed.npz directly')
    parser.add_argument('--seq-stride', type=int, default=1, help='Subsample time dimension by this factor')
    # ADL model flags
    parser.add_argument('--n_atoms', type=int, default=256)
    parser.add_argument('--sparsity', type=float, default=0.1)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lambda_recon', type=float, default=1.0)
    parser.add_argument('--lambda_contrast', type=float, default=0.5)
    parser.add_argument('--queue_size', type=int, default=4096)
    # Optimization flags
    parser.add_argument('--focal-alpha', type=float, default=0.75)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--weighted-sampler', action='store_true')
    parser.add_argument('--no-tsa', action='store_true', help='Disable time-series augmentation')
    args = parser.parse_args()

    seed_everything(SEED)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    print("=" * 70)
    print("ADL-Net Quick Subset Verification")
    print(f"  n_samples={args.n_samples}, n_theft={args.n_theft}, epochs={args.epochs}, folds={args.folds}")
    print(f"  n_atoms={args.n_atoms}, d_model={args.d_model}, n_layers={args.n_layers}, queue_size={args.queue_size}")
    print(f"  lambda_recon={args.lambda_recon}, lambda_contrast={args.lambda_contrast}")
    print("=" * 70)

    print("\nPreprocessing SGCC (if not cached)...")
    if args.load_cache and os.path.exists(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz')):
        X_seq_full, _, flags_full, impute_mask_full = load_preprocessed_from_cache(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
    else:
        X_seq_full, _, flags_full, impute_mask_full = preprocess_sgcc(use_advanced_features=not args.no_advanced)

    X_seq, flags, impute_mask = stratified_subsample(
        X_seq_full, flags_full, impute_mask_full,
        n_samples=args.n_samples, n_theft=args.n_theft
    )
    if args.seq_stride > 1:
        X_seq = X_seq[:, :, ::args.seq_stride]
        impute_mask = impute_mask[:, ::args.seq_stride]
    print(f"Subset shape: {X_seq.shape}, theft rate={flags.mean()*100:.2f}%")

    fold_assignments = make_fold_assignments(flags, n_folds=args.folds, seed=SEED)

    trainer = ADLTrainer(
        dataset='sgcc',
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        n_atoms=args.n_atoms,
        sparsity=args.sparsity,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        lambda_recon=args.lambda_recon,
        lambda_contrast=args.lambda_contrast,
        queue_size=args.queue_size,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        use_weighted_sampler=args.weighted_sampler,
        use_tsa=not args.no_tsa,
    )

    t0 = time.time()
    oof_proba = trainer.train(X_seq, flags, fold_assignments=fold_assignments)
    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")

    overall_f1 = f1_score(flags, (oof_proba > 0.5).astype(int), zero_division=0)
    overall_rec = recall_score(flags, (oof_proba > 0.5).astype(int), zero_division=0)
    overall_prec = precision_score(flags, (oof_proba > 0.5).astype(int), zero_division=0)
    overall_auc = roc_auc_score(flags, oof_proba)
    print(f"\n[Quick Test] F1={overall_f1:.4f}, Recall={overall_rec:.4f}, Precision={overall_prec:.4f}, AUC={overall_auc:.4f}")

    out_path = os.path.join(OUTPUT_DIR, f'adl_quick_test_{args.n_samples}_{args.epochs}.npz')
    np.savez_compressed(out_path, oof_proba=oof_proba, flags=flags)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
