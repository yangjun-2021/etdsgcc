"""
AMST-ETD Experiment: Standalone evaluation of AMST-Net on SGCC.

This script bypasses the full pipeline and trains AMST-Net directly on the
preprocessed SGCC sequential data. It is useful for quick ablation studies.

Usage:
    python experiments/amst_experiment.py --epochs 80 --batch_size 64 --lr 1e-4
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import SEED
from src.data.preprocess_sgcc import preprocess_sgcc
from src.training.amst_trainer import AMSTTrainer
from src.utils.utils import seed_everything
from sklearn.metrics import roc_auc_score
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='AMST-ETD standalone experiment')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--no-diffaug', action='store_true')
    parser.add_argument('--no-supcon', action='store_true')
    parser.add_argument('--no-prior', action='store_true', help='Do not use Expert-A OOF prior')
    parser.add_argument('--coteaching', action='store_true')
    args = parser.parse_args()

    seed_everything(SEED)

    print("Preprocessing SGCC (if not cached)...")
    X_seq, stat_features, flags, impute_mask = preprocess_sgcc(use_advanced_features=True)

    # Load Expert-A OOF prior if available
    oof_proba_a = None
    expert_a_path = os.path.join(PROJECT_ROOT, 'output', 'sgcc_expert_a.npz')
    if not args.no_prior and os.path.exists(expert_a_path):
        try:
            expert_data = np.load(expert_a_path, allow_pickle=True)
            oof_proba_a = expert_data['oof_proba'].astype(np.float32)
            # Sanity check ordering
            expert_flags = expert_data.get('flags', None)
            if expert_flags is not None and not np.array_equal(expert_flags, flags):
                print("[Warning] Expert-A flags do not match current flags. Verify preprocessing seed.")
            print(f"[AMST-Experiment] Loaded Expert-A OOF prior from {expert_a_path}, AUC={roc_auc_score(flags, oof_proba_a):.4f}")
        except Exception as e:
            print(f"[AMST-Experiment] Could not load Expert-A prior: {e}")
            oof_proba_a = None

    trainer = AMSTTrainer(
        dataset='sgcc',
        use_diffaug=not args.no_diffaug,
        use_supcon=not args.no_supcon,
        use_coteaching=args.coteaching,
        use_prior=not args.no_prior,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )

    oof_proba = trainer.train(X_seq, flags, impute_mask=impute_mask, oof_proba_a=oof_proba_a)
    print("\nOOF probabilities saved. Run meta-learner stacking to combine with GBDT.")


if __name__ == '__main__':
    main()
