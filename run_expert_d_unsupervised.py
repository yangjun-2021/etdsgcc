"""
Standalone runner for Expert D: unsupervised anomaly detection on SGCC.

Loads the fast X_seq cache and trains CPU-efficient anomaly detectors.
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from config import OUTPUT_DIR, SEED
from src.utils.utils import seed_everything
from src.training.expert_d_unsupervised import ExpertDUnsupervisedTrainer


def main():
    parser = argparse.ArgumentParser(description='Run Expert D (Unsupervised) on SGCC')
    parser.add_argument('--n-neighbors', type=int, default=20,
                        help='Neighbors for LOF and k-NN')
    parser.add_argument('--n-estimators', type=int, default=200,
                        help='Trees for Isolation Forest')
    parser.add_argument('--contamination', type=float, default=0.1,
                        help='Expected contamination fraction')
    args = parser.parse_args()

    seed_everything(SEED)

    fast_path = os.path.join(OUTPUT_DIR, 'sgcc_xseq_fast.npz')
    if not os.path.exists(fast_path):
        raise FileNotFoundError(f"Fast X_seq cache not found: {fast_path}. Run build_xseq_fast.py first.")

    print(f"[Runner] Loading fast X_seq cache from {fast_path}")
    cached = np.load(fast_path)
    X_seq = cached['X_seq']
    flags = cached['flags']
    print(f"  Loaded: X_seq={X_seq.shape}, flags={flags.shape}")

    print("\n[Runner] Training Expert D (Unsupervised Anomaly Detection)...")
    trainer = ExpertDUnsupervisedTrainer(
        dataset='sgcc',
        n_neighbors=args.n_neighbors,
        n_estimators=args.n_estimators,
        contamination=args.contamination,
    )
    oof_proba_d = trainer.train(X_seq, flags)

    print("\n[Runner] Done.")
    print(f"  OOF shape: {oof_proba_d.shape}")
    print(f"  OOF path:  {os.path.join(OUTPUT_DIR, 'sgcc_expert_d_unsupervised.npz')}")


if __name__ == '__main__':
    main()
