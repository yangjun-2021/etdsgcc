"""
Run Meta-Learner with cached Expert A/B OOFs plus the retrained Informer OOF (v2).

Usage:
    python run_meta_with_informer_v2.py
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

    print("Loading cached OOFs (including retrained Informer v2)...")
    pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
    stat_features = pre['stat_features']
    flags = pre['flags']
    impute_mask = pre['impute_mask']

    a = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_a.npz'))
    oof_proba_a = a['oof_proba']

    b = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_b.npz'))
    oof_proba_b = b['oof_proba']

    inf_path = os.path.join(OUTPUT_DIR, 'informer_oof_v2.npz')
    if not os.path.exists(inf_path):
        print(f"ERROR: {inf_path} not found. Run retrain_expert_c.py first.")
        sys.exit(1)

    inf = np.load(inf_path)
    oof_proba_c = inf['oof_informer']

    print(f"  A={oof_proba_a.shape}, B={oof_proba_b.shape}, C(v2)={oof_proba_c.shape}")

    print("\nTraining Meta-Learner with A/B/C(v2) OOFs...")
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
