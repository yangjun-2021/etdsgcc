"""Analyze the 378 hard cases where hillclimb and cleaned labels both disagree with original."""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR


def main():
    y_orig = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags'].astype(int)
    cl = np.load(os.path.join(OUTPUT_DIR, 'cleaned_labels_v1.npz'))
    y_clean = cl['y_clean'].astype(int)

    proba = np.load(os.path.join(OUTPUT_DIR, 'hillclimb_best_oof.npz'))['oof_hillclimb']
    # Best threshold from earlier
    th = 0.52
    pred = (proba > th).astype(int)

    # Hard cases: model prediction != original label AND original label == cleaned label
    hard = (pred != y_orig) & (y_orig == y_clean)
    n_hard = hard.sum()
    print(f'Hard model errors (pred!=orig & orig==clean): {n_hard}')

    usage = np.load(os.path.join(OUTPUT_DIR, 'usage_features.npz'))
    log_max = usage['log_max_usage']
    miss = usage['missing_rate']

    quintiles = np.percentile(log_max, np.linspace(0, 100, 6))
    q = np.digitize(log_max, quintiles[1:-1], right=True)

    print('\nHard case distribution by usage quintile:')
    for qi in range(5):
        mask = hard & (q == qi)
        print(f'  q{qi+1}: {mask.sum()} / {n_hard} ({mask.sum()/n_hard*100:.1f}%)')

    print('\nUsage stats:')
    print(f'  hard log_max mean: {log_max[hard].mean():.3f}, median: {np.median(log_max[hard]):.3f}')
    print(f'  all  log_max mean: {log_max.mean():.3f}, median: {np.median(log_max):.3f}')
    print(f'  hard missing_rate mean: {miss[hard].mean():.3f}')
    print(f'  all  missing_rate mean: {miss.mean():.3f}')

    # How many hard are FN vs FP?
    hard_fn = hard & (y_orig == 1) & (pred == 0)
    hard_fp = hard & (y_orig == 0) & (pred == 1)
    print(f'\nHard FN: {hard_fn.sum()}, Hard FP: {hard_fp.sum()}')

    # Save for inspection
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'hard_case_analysis.npz'),
        hard_idx=np.where(hard)[0],
        hard_fn_idx=np.where(hard_fn)[0],
        hard_fp_idx=np.where(hard_fp)[0],
        log_max=log_max,
        missing_rate=miss,
        usage_quintile=q,
    )
    print(f'\nSaved to {os.path.join(OUTPUT_DIR, "hard_case_analysis.npz")}')


if __name__ == '__main__':
    main()
