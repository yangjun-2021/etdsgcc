"""
AMST-Net Quick Verification: train on a small stratified subset.

This is for rapid algorithm debugging before running the full dataset.
It does NOT load Expert-A prior and uses reduced epochs.

Usage:
    python experiments/amst_quick_test.py --n_samples 4000 --n_theft 400 --epochs 20 --batch_size 64
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
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_assignments = np.zeros(len(flags), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf.split(np.zeros(len(flags)), flags)):
        fold_assignments[val_idx] = fold_idx
    return fold_assignments


def main():
    parser = argparse.ArgumentParser(description='AMST-Net quick subset verification')
    parser.add_argument('--n_samples', type=int, default=4000, help='Total subset size')
    parser.add_argument('--n_theft', type=int, default=400, help='Number of positive samples in subset')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs per fold')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--no-diffaug', action='store_true')
    parser.add_argument('--supcon', action='store_true', help='Enable supervised contrastive loss (often hurts on tiny data)')
    parser.add_argument('--no-advanced', action='store_true', help='Skip advanced features for faster iteration')
    parser.add_argument('--folds', type=int, default=3, help='Number of CV folds')
    parser.add_argument('--load-cache', action='store_true', help='Load cached sgcc_preprocessed.npz directly')
    parser.add_argument('--seq-stride', type=int, default=1, help='Subsample time dimension by this factor (e.g. 4 -> T=258, much faster on CPU)')
    # Small model flags for quick CPU verification
    parser.add_argument('--d_trans', type=int, default=128)
    parser.add_argument('--d_mamba', type=int, default=64)
    parser.add_argument('--n_trans_layers', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=4)
    # Optimization / class-balance flags
    parser.add_argument('--synthetic-ratio', type=float, default=1.0, help='DiffAug synthetic samples per real theft (default 1.0 -> balanced)')
    parser.add_argument('--pos-weight', type=float, default=None, help='Positive class weight in focal loss')
    parser.add_argument('--label-smoothing', type=float, default=0.05)
    parser.add_argument('--recall-weight', type=float, default=1.5)
    parser.add_argument('--focal-alpha', type=float, default=0.65)
    parser.add_argument('--focal-gamma', type=float, default=1.5)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--weighted-sampler', action='store_true', help='Use class-balanced weighted random sampler')
    # Lightweight time-series augmentation options
    parser.add_argument('--tsa', action='store_true', help='Use lightweight time-series augmentation')
    parser.add_argument('--tsa-methods', type=str, default=None, help='Comma-separated list: jitter,scale,permutation,timewarp')
    parser.add_argument('--tsa-prob', type=float, default=0.5)
    parser.add_argument('--tsa-severity', type=float, default=0.1)
    parser.add_argument('--tsa-copies', type=int, default=1)
    parser.add_argument('--mixup', action='store_true', help='Use same-label mixup augmentation')
    parser.add_argument('--mixup-alpha', type=float, default=0.2)
    parser.add_argument('--use-prior', action='store_true', help='Load Expert-A OOF probabilities as prior for AMST')
    parser.add_argument('--no-branch-attention', action='store_true', help='Disable branch cross-attention fusion (use legacy concat+FC)')
    parser.add_argument('--d-fusion', type=int, default=128, help='Common dim for branch cross-attention')
    parser.add_argument('--branch-attn-heads', type=int, default=4, help='Number of attention heads for branch cross-attention')
    args = parser.parse_args()

    seed_everything(SEED)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    print("=" * 70)
    print("AMST-Net Quick Subset Verification")
    print(f"  n_samples={args.n_samples}, n_theft={args.n_theft}, epochs={args.epochs}, folds={args.folds}")
    print(f"  DiffAug={not args.no_diffaug}, TSA={args.tsa}, Mixup={args.mixup}, SupCon={args.supcon}, Prior={args.use_prior}")
    print(f"  Model: d_mamba={args.d_mamba}, d_trans={args.d_trans}, n_layers={args.n_trans_layers}, n_heads={args.n_heads}")
    print(f"  BranchAttention={not args.no_branch_attention}, d_fusion={args.d_fusion}, branch_heads={args.branch_attn_heads}")
    print(f"  focal(alpha={args.focal_alpha}, gamma={args.focal_gamma}, recall_w={args.recall_weight})")
    print(f"  synthetic_ratio={args.synthetic_ratio}, pos_weight={args.pos_weight}, label_smoothing={args.label_smoothing}, seq_stride={args.seq_stride}")
    print("=" * 70)

    print("\nPreprocessing SGCC (if not cached)...")
    if args.load_cache and os.path.exists(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz')):
        X_seq_full, _, flags_full, impute_mask_full = load_preprocessed_from_cache(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
    else:
        X_seq_full, stat_features, flags_full, impute_mask_full = preprocess_sgcc(use_advanced_features=not args.no_advanced)

    X_seq, flags, impute_mask = stratified_subsample(
        X_seq_full, flags_full, impute_mask_full,
        n_samples=args.n_samples, n_theft=args.n_theft
    )
    if args.seq_stride > 1:
        X_seq = X_seq[:, :, ::args.seq_stride]
        impute_mask = impute_mask[:, ::args.seq_stride]
    print(f"Subset shape: {X_seq.shape}, theft rate={flags.mean()*100:.2f}%")

    fold_assignments = make_fold_assignments(flags, n_folds=args.folds, seed=SEED)

    # Load Expert-A prior if requested (must subset to match sampled indices)
    oof_proba_a = None
    if args.use_prior:
        prior_path = os.path.join(OUTPUT_DIR, 'sgcc_expert_a.npz')
        if os.path.exists(prior_path):
            try:
                expert_data = np.load(prior_path, allow_pickle=True)
                oof_full = None
                for key in ['oof_proba_a', 'proba', 'y_pred', 'oof_proba']:
                    if key in expert_data.files:
                        oof_full = expert_data[key]
                        break
                if oof_full is None:
                    raise ValueError(f"No OOF probability array found in {prior_path}; keys={expert_data.files}")
                # Subset to the same indices used by stratified_subsample
                pos_idx = np.where(flags_full == 1)[0]
                neg_idx = np.where(flags_full == 0)[0]
                rng = np.random.RandomState(SEED)
                sel_pos = rng.choice(pos_idx, size=args.n_theft, replace=False)
                sel_neg = rng.choice(neg_idx, size=args.n_samples - args.n_theft, replace=False)
                sel = np.concatenate([sel_pos, sel_neg])
                rng.shuffle(sel)
                oof_proba_a = oof_full[sel]
                print(f"[QuickTest] Loaded Expert-A prior from {prior_path}, subset shape={oof_proba_a.shape}, mean={oof_proba_a.mean():.4f}")
            except Exception as e:
                print(f"[QuickTest] Failed to load Expert-A prior: {e}")
                args.use_prior = False
        else:
            print(f"[QuickTest] Expert-A prior not found at {prior_path}, disabling prior.")
            args.use_prior = False

    trainer = AMSTTrainer(
        dataset='sgcc',
        use_diffaug=not args.no_diffaug,
        use_supcon=args.supcon,
        use_coteaching=False,
        use_prior=args.use_prior,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        d_mamba=args.d_mamba,
        d_trans=args.d_trans,
        n_trans_layers=args.n_trans_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        synthetic_ratio=args.synthetic_ratio,
        pos_weight=args.pos_weight,
        label_smoothing=args.label_smoothing,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        recall_weight=args.recall_weight,
        use_weighted_sampler=args.weighted_sampler,
        use_tsa=args.tsa,
        tsa_methods=args.tsa_methods.split(',') if args.tsa_methods else None,
        tsa_prob=args.tsa_prob,
        tsa_severity=args.tsa_severity,
        tsa_copies=args.tsa_copies,
        use_mixup=args.mixup,
        mixup_alpha=args.mixup_alpha,
        use_branch_attention=not args.no_branch_attention,
        d_fusion=args.d_fusion,
        branch_attn_heads=args.branch_attn_heads,
    )

    t0 = time.time()
    oof_proba = trainer.train(X_seq, flags, impute_mask=impute_mask, fold_assignments=fold_assignments, oof_proba_a=oof_proba_a)
    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")

    # Final metrics
    overall_f1 = f1_score(flags, (oof_proba > 0.5).astype(int), zero_division=0)
    overall_rec = recall_score(flags, (oof_proba > 0.5).astype(int), zero_division=0)
    overall_prec = precision_score(flags, (oof_proba > 0.5).astype(int), zero_division=0)
    overall_auc = roc_auc_score(flags, oof_proba)
    print(f"\n[Quick Test] F1={overall_f1:.4f}, Recall={overall_rec:.4f}, Precision={overall_prec:.4f}, AUC={overall_auc:.4f}")

    # Save result
    out_path = os.path.join(OUTPUT_DIR, f'amst_quick_test_{args.n_samples}_{args.epochs}.npz')
    np.savez_compressed(out_path, oof_proba=oof_proba, flags=flags)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
