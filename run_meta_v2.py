"""
Quick runner for ImprovedMetaLearner v2 using cached OOFs.

Usage:
    python run_meta_v2.py --dataset sgcc
    python run_meta_v2.py --dataset oedi
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from config import OUTPUT_DIR
from src.utils.utils import seed_everything
from config import SEED


def load_cached_oofs(dataset, label_source='cleaned'):
    """Load cached expert OOF probabilities and labels.

    Parameters
    ----------
    label_source : {'cleaned', 'original', 'cleaned_v3'}
        'cleaned' uses the label-cleaned flags stored in the cached OOF files
        (these labels were flipped to match model consensus and inflate metrics).
        'original' uses the raw SGCC labels from cleaned_labels_v1.npz, giving
        a more honest estimate of generalization.
        'cleaned_v3' uses the v3 consensus-cleaned labels from
        cleaned_labels_v3.npz (untainted original-label voters only).
    """
    oof_a = oof_b = oof_c = None
    path_a = os.path.join(OUTPUT_DIR, f'{dataset}_expert_a.npz')
    path_b = os.path.join(OUTPUT_DIR, f'{dataset}_expert_b.npz')
    path_c = os.path.join(OUTPUT_DIR, f'{dataset}_expert_c.npz')

    if os.path.exists(path_a):
        data = np.load(path_a)
        oof_a = data['oof_proba']
        print(f"  Loaded Expert A OOF: {oof_a.shape}")
    if os.path.exists(path_b):
        data = np.load(path_b)
        oof_b = data['oof_proba']
        print(f"  Loaded Expert B OOF: {oof_b.shape}")
    if os.path.exists(path_c):
        data = np.load(path_c)
        oof_c = data['oof_proba']
        print(f"  Loaded Expert C OOF: {oof_c.shape}")

    labels = None
    impute_mask = None

    if label_source == 'original' and dataset == 'sgcc':
        clean_path = os.path.join(OUTPUT_DIR, 'cleaned_labels_v1.npz')
        if os.path.exists(clean_path):
            data = np.load(clean_path)
            labels = data['y_orig'].astype(int)
            print(f"  Loaded ORIGINAL SGCC labels: {labels.shape}, theft rate: {labels.mean():.4f}")
        else:
            print("  WARNING: cleaned_labels_v1.npz not found; falling back to cleaned flags")
    elif label_source == 'cleaned_v3' and dataset == 'sgcc':
        v3_path = os.path.join(OUTPUT_DIR, 'cleaned_labels_v3.npz')
        if os.path.exists(v3_path):
            data = np.load(v3_path)
            labels = data['y_clean'].astype(int)
            print(f"  Loaded V3 CLEANED SGCC labels: {labels.shape}, theft rate: {labels.mean():.4f}")
        else:
            print("  WARNING: cleaned_labels_v3.npz not found; falling back to cleaned flags")

    if labels is None:
        # Try expert_a cache first (its flags match existing OOFs)
        if os.path.exists(path_a):
            data = np.load(path_a)
            key = 'flags' if dataset == 'sgcc' else 'y'
            if key in data.files:
                labels = data[key]

        # Fallback to preprocessed cache
        if labels is None:
            pre_path = os.path.join(OUTPUT_DIR, f'{dataset}_preprocessed.npz')
            if os.path.exists(pre_path):
                data = np.load(pre_path)
                if 'flags' in data.files:
                    labels = data['flags']
                if 'impute_mask' in data.files:
                    impute_mask = data['impute_mask']

        # Fallback to fast X_seq cache
        if labels is None:
            fast_path = os.path.join(OUTPUT_DIR, f'{dataset}_xseq_fast.npz')
            if os.path.exists(fast_path):
                data = np.load(fast_path)
                if 'flags' in data.files:
                    labels = data['flags']
                if 'impute_mask' in data.files:
                    impute_mask = data['impute_mask']

    if labels is not None and label_source == 'cleaned':
        print(f"  Loaded CLEANED labels: {labels.shape}, theft rate: {labels.mean():.4f}")

    return oof_a, oof_b, oof_c, labels, impute_mask


def main():
    parser = argparse.ArgumentParser(description='Run Improved Meta-Learner v2')
    parser.add_argument('--dataset', choices=['sgcc', 'oedi'], default='sgcc',
                        help='Dataset to run (default: sgcc)')
    parser.add_argument('--label-source', choices=['cleaned', 'original', 'cleaned_v3'], default='cleaned',
                        help="Label source: 'cleaned' flags (default), 'original' labels, or 'cleaned_v3' consensus labels")
    parser.add_argument('--pool', default='',
                        help="Comma-separated substrings to restrict the OOF pool "
                             "(e.g. 'v3voter,v3clean' for the untainted v3 family). "
                             "Empty = full auto-discovered pool.")
    args = parser.parse_args()

    seed_everything(SEED)

    print(f"\n{'='*70}")
    print(f"  Running ImprovedMetaLearner v2 on {args.dataset.upper()}")
    print(f"{'='*70}")

    oof_a, oof_b, oof_c, labels, impute_mask = load_cached_oofs(args.dataset, label_source=args.label_source)
    if labels is None:
        raise RuntimeError(f"Could not load labels for {args.dataset}")

    from src.training.meta_learner_v2 import ImprovedMetaLearner
    learner = ImprovedMetaLearner(dataset=args.dataset)
    results = learner.train(
        stat_features=None,
        labels=labels,
        impute_mask=impute_mask,
        oof_proba_a=oof_a,
        oof_proba_b=oof_b,
        oof_proba_c=oof_c,
        skip_new_experts=True,
        pool_include=[p for p in args.pool.split(',') if p] or None,
    )

    print(f"\n  Best F1: {results['best_f1']:.4f}")


if __name__ == '__main__':
    main()
