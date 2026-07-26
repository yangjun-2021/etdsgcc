"""
Run Meta-Learner with cached Expert A/B OOFs plus existing Informer OOF.

This is a fast path to validate whether adding Informer improves the stack.
Usage:
    python run_meta_with_informer.py
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
from src.training.meta_learner import MetaLearner
from src.evaluation.evaluate import evaluate_dataset


def main():
    seed_everything(SEED)

    print("Loading cached OOFs...")
    pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
    stat_features = pre['stat_features']
    flags = pre['flags']
    impute_mask = pre['impute_mask']

    a = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_a.npz'))
    oof_proba_a = a['oof_proba']

    b = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_b.npz'))
    oof_proba_b = b['oof_proba']

    inf = np.load(os.path.join(OUTPUT_DIR, 'informer_oof.npz'))
    oof_proba_c = inf['oof_informer']

    print(f"  A={oof_proba_a.shape}, B={oof_proba_b.shape}, C={oof_proba_c.shape}")

    print("\nTraining Meta-Learner with A/B/C OOFs...")
    meta_learner = MetaLearner(dataset='sgcc')
    results = meta_learner.train(
        stat_features, flags, impute_mask=impute_mask,
        oof_proba_a=oof_proba_a, oof_proba_b=oof_proba_b, oof_proba_c=oof_proba_c,
        skip_new_experts=True,
    )

    print("\nEvaluating...")
    evaluate_dataset('sgcc', results, OUTPUT_DIR)

    print("\nDone.")


if __name__ == '__main__':
    main()
