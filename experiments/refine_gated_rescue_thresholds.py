"""Refine gated rescue blend with per-usage-quintile thresholds.

Coordinate descent for per-quintile thresholds on gated_rescue_blend_oof.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import numpy as np


def best_f1(y, p):
    best = (0, 0, 0, 0)
    for th in np.linspace(0.01, 0.99, 199):
        pred = (p >= th).astype(int)
        if pred.sum() == 0:
            continue
        f = f1_score(y, pred, zero_division=0)
        if f > best[0]:
            best = (f, recall_score(y, pred), precision_score(y, pred), th)
    return best


def main():
    y = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags'].astype(int)
    p = np.load(os.path.join(OUTPUT_DIR, 'gated_rescue_blend_oof.npz'))['oof_gated_rescue_blend']
    usage = np.load(os.path.join(OUTPUT_DIR, 'usage_features.npz'))
    log_max = usage['log_max_usage']

    # global baseline
    f1_base, rec_base, prec_base, th_base = best_f1(y, p)
    print(f'Global gated rescue: F1={f1_base:.4f} Rec={rec_base:.4f} Prec={prec_base:.4f} th={th_base:.3f}')

    quintiles = np.percentile(log_max, np.linspace(0, 100, 6))
    q = np.digitize(log_max, quintiles[1:-1], right=True)

    # coordinate descent for per-quintile thresholds
    ths = np.full(5, th_base)
    best_f = f1_base
    improved = True
    grid = np.arange(0.05, 0.96, 0.05)
    while improved:
        improved = False
        for qi in range(5):
            for t in grid:
                new_ths = ths.copy()
                new_ths[qi] = t
                pred = (p >= new_ths[q]).astype(int)
                f = f1_score(y, pred, zero_division=0)
                if f > best_f:
                    best_f = f
                    ths = new_ths
                    improved = True

    pred = (p >= ths[q]).astype(int)
    rec = recall_score(y, pred)
    prec = precision_score(y, pred, zero_division=0)
    auc = roc_auc_score(y, p)
    print(f'\nSubgroup-refined gated rescue: F1={best_f:.4f} Rec={rec:.4f} Prec={prec:.4f} AUC={auc:.4f}')
    print(f'Per-quintile thresholds: {ths}')

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'gated_rescue_refined_oof.npz'),
        flags=y,
        oof_gated_rescue_refined=p,
        thresholds=ths,
        quintiles=quintiles,
    )
    print(f'Saved to {os.path.join(OUTPUT_DIR, "gated_rescue_refined_oof.npz")}')


if __name__ == '__main__':
    main()
