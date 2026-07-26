"""Finish the OEDI pipeline from cached expert OOFs: classic MetaLearner + evaluation.

run_pipeline.py --dataset oedi crashed at the MetaLearner step (external bundled
OOFs are SGCC-length); experts' caches are already on disk. This driver completes
the pipeline without retraining.

Usage:
    conda run -n ml python experiments/run_oedi_meta.py
"""
import os
import sys
import pickle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np

from config import OUTPUT_DIR
from src.training.meta_learner import MetaLearner
from src.evaluation.evaluate import evaluate_dataset


def main():
    pre = np.load(os.path.join(OUTPUT_DIR, 'oedi_preprocessed.npz'))
    stat_features, y, fold_assignments = pre['stat_features'], pre['y'], pre['fold_assignments']

    a = np.load(os.path.join(OUTPUT_DIR, 'oedi_expert_a.npz'))
    b = np.load(os.path.join(OUTPUT_DIR, 'oedi_expert_b.npz'))
    oof_a, oof_b = a['oof_proba'], b['oof_proba']

    meta_learner = MetaLearner(dataset='oedi')
    results = meta_learner.train(stat_features, y, fold_assignments=fold_assignments,
                                 oof_proba_a=oof_a, oof_proba_b=oof_b,
                                 skip_new_experts=True)

    print('\n[Step 5/6] Evaluating OEDI...')
    results['y'] = y  # evaluate_dataset expects results['y']
    evaluate_dataset('oedi', results, OUTPUT_DIR)

    # convenience npz for the per-type analysis script
    np.savez_compressed(os.path.join(OUTPUT_DIR, 'oedi_meta_results.npz'),
                        oof_proba_meta=results['oof_proba_meta'], y=y)
    print('Saved oedi_meta_results.npz')


if __name__ == '__main__':
    main()
