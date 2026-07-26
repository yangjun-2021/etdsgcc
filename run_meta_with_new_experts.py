"""Run MegaMetaLearner including new experts on extended feature pool."""
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

    pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
    stat_features = pre['stat_features']
    flags = pre['flags']
    impute_mask = pre['impute_mask']

    a = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_a.npz'))
    oof_proba_a = a['oof_proba']

    b = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_b.npz'))
    oof_proba_b = b['oof_proba']

    meta_learner = MetaLearner(dataset='sgcc')
    results = meta_learner.train(
        stat_features, flags, impute_mask=impute_mask,
        oof_proba_a=oof_proba_a, oof_proba_b=oof_proba_b,
        skip_new_experts=False,
    )
    evaluate_dataset('sgcc', results, OUTPUT_DIR)


if __name__ == '__main__':
    main()
