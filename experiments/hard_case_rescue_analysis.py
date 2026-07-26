"""Analyze which OOF signal best rescues hillclimb FN per usage subgroup.

Loads hillclimb and complementary OOFs, splits by log(max_usage) quintiles,
and reports per-subgroup best rescue signal and an oracle upper bound.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import numpy as np


def load_oof(fname, key_guess):
    fpath = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(fpath):
        return None
    d = np.load(fpath)
    # try key_guess or any oof_* key
    for k in [key_guess] + [kk for kk in d.keys() if kk.startswith('oof_')]:
        if k in d:
            arr = d[k]
            if arr.ndim > 1:
                arr = arr[:, 1] if arr.shape[1] == 2 else arr.ravel()
            return np.nan_to_num(arr.astype(np.float64), nan=0.5)
    return None


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
    usage = np.load(os.path.join(OUTPUT_DIR, 'usage_features.npz'))
    log_max = usage['log_max_usage']
    miss = usage['missing_rate']

    quintiles = np.percentile(log_max, np.linspace(0, 100, 6))
    q = np.digitize(log_max, quintiles[1:-1], right=True)

    signals = {
        'hillclimb': load_oof('hillclimb_best_oof.npz', 'oof_hillclimb'),
        'supcon_v3': load_oof('supcon_raw_3ch_v3_oof.npz', 'oof_supcon_raw_3ch_v3'),
        'subtle_v3': load_oof('amst_3ch_synthetic_subtle_v3_oof.npz', 'oof_amst_3ch_synthetic_subtle_v3'),
        'amst_v3': load_oof('amst_3ch_synthetic_mixed_ls_v3_oof.npz', 'oof_amst_3ch_synthetic_mixed_ls_v3'),
        'mega_meta': load_oof('mega_meta_all_oofs_oof.npz', 'oof_mega_meta_all_oofs'),
        'informer_large': load_oof('informer_large_strong_prior_oof.npz', 'oof_informer_large_strong_prior'),
        'patch_recall': load_oof('patch_transformer_raw_3ch_recall_oof.npz', 'oof_patch_transformer_raw_3ch_recall'),
    }
    signals = {k: v for k, v in signals.items() if v is not None}
    print(f'Loaded {len(signals)} signals: {list(signals.keys())}')

    # Hillclimb baseline
    hc = signals['hillclimb']
    f1_hc, rec_hc, prec_hc, th_hc = best_f1(y, hc)
    pred_hc = (hc >= th_hc).astype(int)
    print(f'\nHillclimb baseline: F1={f1_hc:.4f} Rec={rec_hc:.4f} Prec={prec_hc:.4f} th={th_hc:.3f}')

    # Overall signal comparison
    print('\nOverall best per signal:')
    for name, p in signals.items():
        f1, rec, prec, th = best_f1(y, p)
        auc = roc_auc_score(y, p)
        print(f'  {name:15s}: F1={f1:.4f} Rec={rec:.4f} Prec={prec:.4f} AUC={auc:.4f} th={th:.3f}')

    # Per-quintile analysis
    print('\nPer-usage-quintile hillclimb errors and best rescue signal:')
    print('q | n_pos | hc_F1 | hc_Rec | hc_Prec | best_rescue | gain_F1 | rescue_rate | note')
    for qi in range(5):
        mask = q == qi
        yq = y[mask]
        hcq = hc[mask]
        f1_q, rec_q, prec_q, th_q = best_f1(yq, hcq)
        pred_q = (hcq >= th_q).astype(int)
        fn_idx = np.where(mask & (y == 1) & (pred_hc == 0))[0]
        n_fn = len(fn_idx)
        best_gain = -1
        best_name = 'none'
        best_rescue_rate = 0
        best_f1_rescue = f1_q
        for name, p in signals.items():
            if name == 'hillclimb':
                continue
            # per-subgroup threshold for this signal
            pq = p[mask]
            f1_s, rec_s, prec_s, th_s = best_f1(yq, pq)
            gain = f1_s - f1_q
            if gain > best_gain:
                best_gain = gain
                best_name = name
                best_f1_rescue = f1_s
                # rescue rate: among hillclimb FN in this quintile, fraction this signal predicts positive
                if n_fn > 0:
                    best_rescue_rate = (p[fn_idx] >= th_s).mean()
        print(f' q{qi+1} | {yq.sum():4d} | {f1_q:.3f} | {rec_q:.3f}  | {prec_q:.3f}   | {best_name:11s} | +{best_gain:.3f} | {best_rescue_rate:.3f}     | n_FN={n_fn}, rescue F1={best_f1_rescue:.3f}')

    # Oracle: per-sample choose signal that maximizes F1 (cheating upper bound)
    names = list(signals.keys())
    P = np.column_stack([signals[n] for n in names])
    # Oracle: for each sample, if y=1 take max proba, if y=0 take min proba
    oracle = np.where(y == 1, P.max(axis=1), P.min(axis=1))
    f1_oracle, rec_oracle, prec_oracle, th_oracle = best_f1(y, oracle)
    print(f'\nOracle per-sample choose-best-signal upper bound: F1={f1_oracle:.4f} Rec={rec_oracle:.4f} Prec={prec_oracle:.4f} th={th_oracle:.3f}')

    # Subgroup oracle: within each quintile choose signal with best F1
    oracle_sub = np.zeros_like(hc)
    for qi in range(5):
        mask = q == qi
        yq = y[mask]
        Pq = P[mask]
        best_f1_q = -1
        best_col = 0
        for j in range(Pq.shape[1]):
            f1_s, _, _, _ = best_f1(yq, Pq[:, j])
            if f1_s > best_f1_q:
                best_f1_q = f1_s
                best_col = j
        oracle_sub[mask] = Pq[:, best_col]
    f1_sub, rec_sub, prec_sub, th_sub = best_f1(y, oracle_sub)
    print(f'Subgroup-oracle (per-quintile best signal): F1={f1_sub:.4f} Rec={rec_sub:.4f} Prec={prec_sub:.4f} th={th_sub:.3f}')


if __name__ == '__main__':
    main()
