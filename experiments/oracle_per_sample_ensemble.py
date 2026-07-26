"""Oracle per-sample ensemble upper bound.

For each sample, choose the model that would give the correct prediction
(if possible) at the optimal threshold. This gives an unrealistic upper bound
on what any ensemble of these signals could achieve.
"""
import os
import sys
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR


def main():
    y = np.load(os.path.join(OUTPUT_DIR, 'hillclimb_best_oof.npz'))['flags']
    signals = {
        'hillclimb': np.load(os.path.join(OUTPUT_DIR, 'hillclimb_best_oof.npz'))['oof_hillclimb'],
        'v3': np.load(os.path.join(OUTPUT_DIR, 'amst_3ch_synthetic_mixed_ls_v3_oof.npz'))['oof_amst_3ch_synthetic_mixed_ls_v3'],
    }
    # Add other decent OOFs if available
    for fname, key in [
        ('sgcc_mega_meta.npz', 'oof_final'),
        ('autoresearch_best.npz', 'oof_final'),
        ('mega_boost_enhanced.npz', 'oof_final'),
    ]:
        p = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(p):
            try:
                signals[fname.replace('.npz', '')] = np.load(p)[key]
            except Exception:
                pass

    P = np.column_stack(list(signals.values()))
    names = list(signals.keys())
    print(f'Signals: {names}')

    # Oracle: for each sample, if y=1 choose max proba, if y=0 choose min proba
    oracle_proba = np.where(y == 1, P.max(axis=1), P.min(axis=1))

    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (oracle_proba > th).astype(int)
        if pred.sum() == 0:
            continue
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    pred = (oracle_proba > best_th).astype(int)
    print(f'Oracle per-sample ensemble: F1={f1_score(y, pred):.4f}, '
          f'Rec={recall_score(y, pred):.4f}, '
          f'Prec={precision_score(y, pred, zero_division=0):.4f}, th={best_th:.3f}')

    # Also compute best possible from any single threshold per signal? Already known.


if __name__ == '__main__':
    main()
