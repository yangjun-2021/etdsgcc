"""Final blend search: try direct weighted blends of top OOFs + subgroup thresholds.

Loads the strongest OOF signals and searches:
1. Global weighted blend + threshold
2. Per-usage-quintile thresholds on the best blend
3. Recall-prioritized blend
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED
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


def best_recall_constrained(y, p, min_recall=0.90):
    best = (0, 0, 0, 0)
    for th in np.linspace(0.01, 0.99, 199):
        pred = (p >= th).astype(int)
        if pred.sum() == 0:
            continue
        rec = recall_score(y, pred)
        if rec < min_recall:
            continue
        f = f1_score(y, pred, zero_division=0)
        if f > best[0]:
            best = (f, rec, precision_score(y, pred, zero_division=0), th)
    return best


def subgroup_threshold_search(y, p, q):
    """Coordinate descent for per-quintile thresholds."""
    # Start with global best threshold
    _, _, _, best_th = best_f1(y, p)
    ths = np.full(5, best_th)
    best_f = f1_score(y, (p >= best_th).astype(int))
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
    return best_f, ths


def main():
    flags = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags']
    usage = np.load(os.path.join(OUTPUT_DIR, 'usage_features.npz'))
    log_max = usage['log_max_usage']
    quintiles = np.percentile(log_max, np.linspace(0, 100, 6))
    q = np.digitize(log_max, quintiles[1:-1], right=True)

    # Candidate OOF files and keys
    candidates = [
        ('hillclimb_best_oof.npz', 'oof_hillclimb'),
        ('amst_3ch_synthetic_subtle_v3_oof.npz', 'oof_amst_3ch_synthetic_subtle_v3'),
        ('amst_3ch_synthetic_mixed_ls_v3_oof.npz', 'oof_amst_3ch_synthetic_mixed_ls_v3'),
        ('amst_3ch_synthetic_mixed_ls_v3_gce_oof.npz', 'oof_amst_3ch_synthetic_mixed_ls_v3_gce'),
        ('supcon_raw_3ch_v3_oof.npz', 'oof_supcon_raw_3ch_v3'),
        ('coteaching_raw_3ch_v3_oof.npz', 'oof_coteaching_raw_3ch_v3'),
        ('mega_meta_all_oofs_oof.npz', 'oof_mega_meta_all_oofs'),
        ('gated_rescue_blend_oof.npz', 'oof_gated_rescue_blend'),
        ('gated_rescue_refined_oof.npz', 'oof_gated_rescue_refined'),
    ]

    signals = {}
    for fname, key in candidates:
        fpath = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(fpath):
            continue
        try:
            arr = np.load(fpath)[key]
            if arr.ndim > 1:
                arr = arr[:, 1] if arr.shape[1] == 2 else arr.ravel()
            signals[fname.replace('_oof.npz', '')] = np.nan_to_num(arr.astype(np.float64), nan=0.5)
        except Exception as e:
            print(f'skip {fname}: {e}')

    print(f'Loaded {len(signals)} signals: {list(signals.keys())}')

    # Print individual metrics
    print('\nIndividual signals:')
    for name, p in signals.items():
        f1, rec, prec, th = best_f1(flags, p)
        auc = roc_auc_score(flags, p)
        print(f'  {name:35s}: F1={f1:.4f} Rec={rec:.4f} Prec={prec:.4f} AUC={auc:.4f} th={th:.3f}')

    if len(signals) < 2:
        print('Need at least 2 signals for blending.')
        return

    names = list(signals.keys())
    P = np.column_stack([signals[n] for n in names])

    # Pairwise and 3-way blends
    print('\nTop pairwise blends (grid search weights):')
    best_blends = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for wi in np.arange(0.1, 1.0, 0.1):
                wj = 1.0 - wi
                blend = wi * P[:, i] + wj * P[:, j]
                f1, rec, prec, th = best_f1(flags, blend)
                best_blends.append((f1, rec, prec, th, wi, names[i], wj, names[j]))
    best_blends.sort(reverse=True)
    for b in best_blends[:5]:
        print(f'  {b[4]:.1f}*{b[5]:20s} + {b[6]:.1f}*{b[7]:20s}: F1={b[0]:.4f} Rec={b[1]:.4f} Prec={b[2]:.4f} th={b[3]:.3f}')

    # 3-way blends
    print('\nTop 3-way blends:')
    best3 = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for k in range(j + 1, len(names)):
                for wi in np.arange(0.1, 0.9, 0.2):
                    for wj in np.arange(0.1, 1.0 - wi - 0.1, 0.2):
                        wk = 1.0 - wi - wj
                        blend = wi * P[:, i] + wj * P[:, j] + wk * P[:, k]
                        f1, rec, prec, th = best_f1(flags, blend)
                        best3.append((f1, rec, prec, th, wi, names[i], wj, names[j], wk, names[k]))
    best3.sort(reverse=True)
    for b in best3[:5]:
        print(f'  {b[4]:.1f}*{b[5]:15s} + {b[6]:.1f}*{b[7]:15s} + {b[8]:.1f}*{b[9]:15s}: F1={b[0]:.4f} Rec={b[1]:.4f} Prec={b[2]:.4f} th={b[3]:.3f}')

    # Best blend + subgroup thresholds
    if best_blends:
        b = best_blends[0]
        i = names.index(b[6])
        j = names.index(b[8])
        blend = b[4] * P[:, i] + b[7] * P[:, j]
        f1_sub, ths = subgroup_threshold_search(flags, blend, q)
        pred = (blend >= ths[q]).astype(int)
        print(f'\nSubgroup threshold on best pair ({b[6]}+{b[8]}): F1={f1_sub:.4f} Rec={recall_score(flags,pred):.4f} Prec={precision_score(flags,pred):.4f}')
        print(f'  thresholds: {ths}')

    # Recall-constrained best
    print('\nRecall>=0.90 constrained best blend:')
    best_rec = (0, 0, 0, 0, '', '')
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for wi in np.arange(0.1, 1.0, 0.1):
                blend = wi * P[:, i] + (1 - wi) * P[:, j]
                f1, rec, prec, th = best_recall_constrained(flags, blend)
                if f1 > best_rec[0]:
                    best_rec = (f1, rec, prec, th, f'{wi:.1f}*{names[i]} + {1-wi:.1f}*{names[j]}', '')
    print(f'  {best_rec[4]}: F1={best_rec[0]:.4f} Rec={best_rec[1]:.4f} Prec={best_rec[2]:.4f} th={best_rec[3]:.3f}')


if __name__ == '__main__':
    main()
