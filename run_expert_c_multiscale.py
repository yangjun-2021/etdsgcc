"""
Standalone runner for Expert C: MultiScaleCNN1D on SGCC.

Loads cached preprocessed SGCC data, trains the multi-scale CNN expert
across 5 folds, and saves out-of-fold probabilities to
output/sgcc_expert_c_multiscale.npz.

Usage:
    python run_expert_c_multiscale.py
    python run_expert_c_multiscale.py --preprocess  # force re-preprocessing
"""
import argparse
import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from config import OUTPUT_DIR, SEED
from src.utils.utils import seed_everything
from src.training.expert_c_multiscale import ExpertCMultiScaleTrainer


def main():
    parser = argparse.ArgumentParser(description='Run Expert C (MultiScaleCNN1D) on SGCC')
    parser.add_argument('--preprocess', action='store_true',
                        help='Force re-run SGCC preprocessing')
    parser.add_argument('--no-advanced', action='store_true',
                        help='Disable advanced features during preprocessing')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Max training epochs per fold')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.25,
                        help='Dropout rate')
    args = parser.parse_args()

    seed_everything(SEED)

    cached_path = os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz')
    fast_path = os.path.join(OUTPUT_DIR, 'sgcc_xseq_fast.npz')
    if args.preprocess or not os.path.exists(cached_path):
        if os.path.exists(fast_path) and not args.preprocess:
            print(f"[Runner] Loading fast X_seq cache from {fast_path}")
            cached = np.load(fast_path)
            X_seq = cached['X_seq']
            flags = cached['flags']
            print(f"  Loaded: X_seq={X_seq.shape}, flags={flags.shape}")
        else:
            print("[Runner] Preprocessing SGCC...")
            from src.data.preprocess_sgcc import preprocess_sgcc
            X_seq, stat_features, flags, impute_mask = preprocess_sgcc(
                use_advanced_features=not args.no_advanced
            )
    else:
        print(f"[Runner] Loading cached preprocessed data from {cached_path}")
        cached = np.load(cached_path)
        X_seq = cached['X_seq']
        flags = cached['flags']
        print(f"  Loaded: X_seq={X_seq.shape}, flags={flags.shape}")

    print("\n[Runner] Training Expert C (MultiScaleCNN1D)...")
    trainer = ExpertCMultiScaleTrainer(
        dataset='sgcc',
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        dropout=args.dropout,
    )
    oof_proba_c = trainer.train(X_seq, flags)

    print("\n[Runner] Done.")
    print(f"  OOF shape: {oof_proba_c.shape}")
    print(f"  OOF path:  {os.path.join(OUTPUT_DIR, 'sgcc_expert_c_multiscale.npz')}")


if __name__ == '__main__':
    main()
