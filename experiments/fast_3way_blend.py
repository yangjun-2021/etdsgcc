"""Fast 3-way blend search over top signals with coarse grid."""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import numpy as np


def best_f1(y, p, n=99):
    best = (0, 0, 0, 0)
    for th in np.linspace(0.01, 0.99, n):
        pred = (p >= th).astype(int)
        if pred.sum() == 0:
            continue
        f = f1_score(y, pred, zero_division=0)
        if f > best[0]:
            best = (f, recall_score(y, pred), precision_score(y, pred), th)
    return best


def main():
    y = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags'].astype(int)
    signals = {
        'hillclimb': np.load(os.path.join(OUTPUT_DIR, 'hillclimb_best_oof.npz'))['oof_hillclimb'],
        'mega_meta': np.load(os.path.join(OUTPUT_DIR, 'mega_meta_all_oofs_oof.npz'))['oof_mega_meta_all_oofs'],
        'gated_rescue': np.load(os.path.join(OUTPUT_DIR, 'gated_rescue_blend_oof.npz'))['oof_gated_rescue_blend'],
        'supcon': np.load(os.path.join(OUTPUT_DIR, 'supcon_raw_3ch_v3_oof.npz'))['oof_supcon_raw_3ch_v3'],
        'coteaching': np.load(os.path.join(OUTPUT_DIR, 'coteaching_raw_3ch_v3_oof.npz'))['oof_coteaching_raw_3ch_v3'],
    }
    names = list(signals.keys())
    P = np.column_stack([signals[n] for n in names])

    best = (0, None)
    weights = np.arange(0.1, 1.0, 0.1)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for k in range(j + 1, len(names)):
                for wi in weights:
                    for wj in weights:
                        wk = 1.0 - wi - wj
                        if wk < 0.1 or wk > 0.9:
                            continue
                        blend = wi * P[:, i] + wj * P[:, j] + wk * P[:, k]
                        f1, rec, prec, th = best_f1(y, blend, n=99)
                        if f1 > best[0]:
                            best = (f1, (wi, wj, wk, names[i], names[j], names[k], th, rec, prec))
    f1, info = best
    if info is None:
        print('No 3-way blend found.')
        return
    wi, wj, wk, ni, nj, nk, th, rec, prec = info
    print(f'Best 3-way: {wi:.2f}*{ni} + {wj:.2f}*{nj} + {wk:.2f}*{nk}')
    print(f'  F1={f1:.4f} Rec={rec:.4f} Prec={prec:.4f} th={th:.3f}')


if __name__ == '__main__':
    main()
