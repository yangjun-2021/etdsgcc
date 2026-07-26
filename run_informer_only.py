"""
Train Informer (Expert C) using cached Expert A/B OOFs and run meta-learner.

This avoids retraining Expert A/B when their cached OOFs are available.
Usage:
    python run_informer_only.py
"""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED
from src.utils.utils import seed_everything
from src.training.expert_c import ExpertCTrainer
from src.training.meta_learner import MetaLearner
from src.evaluation.evaluate import evaluate_dataset


def main():
    seed_everything(SEED)

    print("Loading cached preprocessed data and Expert A/B OOFs...")
    pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
    X_seq = pre['X_seq']
    stat_features = pre['stat_features']
    flags = pre['flags']
    impute_mask = pre['impute_mask']

    a = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_a.npz'))
    oof_proba_a = a['oof_proba']

    b = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_b.npz'))
    oof_proba_b = b['oof_proba']

    print(f"  X_seq={X_seq.shape}, stat_features={stat_features.shape}")
    print(f"  Expert A OOF={oof_proba_a.shape}, Expert B OOF={oof_proba_b.shape}")

    print("\nTraining Expert C (Informer)...")
    trainer_c = ExpertCTrainer(dataset='sgcc')
    oof_proba_c = trainer_c.train(X_seq, flags, oof_proba_a=oof_proba_a)

    print("\nTraining Meta-Learner with A/B/C OOFs...")
    meta_learner = MetaLearner(dataset='sgcc')
    results = meta_learner.train(
        stat_features, flags, impute_mask=impute_mask,
        oof_proba_a=oof_proba_a, oof_proba_b=oof_proba_b, oof_proba_c=oof_proba_c
    )

    print("\nEvaluating...")
    evaluate_dataset('sgcc', results, OUTPUT_DIR)

    print("\nDone.")


if __name__ == '__main__':
    main()
