"""Save and optionally refine the best discovered blend: 0.6*hillclimb + 0.4*mega_meta."""
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
    hc = np.load(os.path.join(OUTPUT_DIR, 'hillclimb_best_oof.npz'))['oof_hillclimb']
    mm = np.load(os.path.join(OUTPUT_DIR, 'mega_meta_all_oofs_oof.npz'))['oof_mega_meta_all_oofs']

    # Best discovered blend from final_blend_search_v2
    p = 0.6 * hc + 0.4 * mm
    f1, rec, prec, th = best_f1(y, p)
    auc = roc_auc_score(y, p)
    print(f'0.6*hillclimb + 0.4*mega_meta: F1={f1:.4f} Rec={rec:.4f} Prec={prec:.4f} AUC={auc:.4f} th={th:.3f}')

    # Try subgroup threshold refinement
    usage = np.load(os.path.join(OUTPUT_DIR, 'usage_features.npz'))
    log_max = usage['log_max_usage']
    quintiles = np.percentile(log_max, np.linspace(0, 100, 6))
    q = np.digitize(log_max, quintiles[1:-1], right=True)

    ths = np.full(5, th)
    best_f = f1
    improved = True
    while improved:
        improved = False
        for qi in range(5):
            for t in np.arange(0.05, 0.96, 0.05):
                new_ths = ths.copy()
                new_ths[qi] = t
                pred = (p >= new_ths[q]).astype(int)
                f = f1_score(y, pred, zero_division=0)
                if f > best_f:
                    best_f = f
                    ths = new_ths
                    improved = True

    pred = (p >= ths[q]).astype(int)
    rec2 = recall_score(y, pred)
    prec2 = precision_score(y, pred, zero_division=0)
    print(f'Subgroup refined: F1={best_f:.4f} Rec={rec2:.4f} Prec={prec2:.4f} thresholds={ths}')

    # Save the better of global vs subgroup
    if best_f > f1:
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, 'final_blend_best_oof.npz'),
            flags=y,
            oof_final_blend_best=p,
            thresholds=ths,
            use_subgroup=True,
        )
        print(f'Saved subgroup-refined blend (F1={best_f:.4f}) to output/final_blend_best_oof.npz')
    else:
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, 'final_blend_best_oof.npz'),
            flags=y,
            oof_final_blend_best=p,
            threshold=th,
            use_subgroup=False,
        )
        print(f'Saved global blend (F1={f1:.4f}) to output/final_blend_best_oof.npz')


if __name__ == '__main__':
    main()
