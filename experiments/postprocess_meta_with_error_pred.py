"""Post-process current meta OOF using error-predictor probabilities.

Try two strategies:
1. Flip predictions where error predictor is confident and meta is uncertain.
2. Add a calibrated correction term to meta probability.
"""
import os, sys
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SEED
from src.utils.utils import seed_everything

seed_everything(SEED)


def main():
    meta = np.load(os.path.join(OUTPUT_DIR, 'sgcc_mega_meta.npz'))['oof_final']
    err = np.load(os.path.join(OUTPUT_DIR, 'meta_error_predictor_oof.npz'))['oof_meta_error_predictor']
    y = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags']

    print(f'Meta OOF shape={meta.shape}, Error pred shape={err.shape}')

    # baseline
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (meta > th).astype(int)
        if pred.sum() == 0: continue
        f = f1_score(y, pred, zero_division=0)
        if f > best_f1: best_f1, best_th = f, th
    base_pred = (meta > best_th).astype(int)
    print(f'Baseline meta th={best_th:.3f} F1={best_f1:.4f} '
          f'Rec={recall_score(y,base_pred):.4f} Prec={precision_score(y,base_pred,zero_division=0):.4f}')

    # Strategy 1: flip predictions when error predictor confident
    best = {'f1': 0, 'rec': 0, 'prec': 0, 'th_err_flip': 0, 'th_meta': 0, 'mode': ''}
    for mode in ['flip_both', 'flip_fn_only', 'flip_fp_only']:
        for th_meta in np.arange(0.30, 0.80, 0.05):
            for th_err in np.arange(0.05, 0.95, 0.05):
                pred = (meta > th_meta).astype(int)
                flip = err > th_err
                if mode == 'flip_both':
                    pred[flip] = 1 - pred[flip]
                elif mode == 'flip_fn_only':
                    pred[flip & (pred == 0)] = 1
                else:
                    pred[flip & (pred == 1)] = 0
                if pred.sum() == 0: continue
                f = f1_score(y, pred, zero_division=0)
                r = recall_score(y, pred, zero_division=0)
                p = precision_score(y, pred, zero_division=0)
                if f > best['f1']:
                    best = {'f1': f, 'rec': r, 'prec': p,
                            'th_err_flip': th_err, 'th_meta': th_meta, 'mode': mode}
    print(f'Best flip: {best}')

    # Strategy 2: blend meta prob with error signal
    best2 = {'f1': 0, 'alpha': 0, 'beta': 0, 'th': 0}
    # Use error prob to up-weight recall-sensitive correction
    for alpha in np.arange(0.0, 1.05, 0.05):
        for beta in np.arange(-1.0, 1.05, 0.05):
            prob = meta + alpha * (err - 0.5) + beta * err * (1 - meta)
            prob = np.clip(prob, 0, 1)
            for th in np.arange(0.05, 0.95, 0.05):
                pred = (prob > th).astype(int)
                if pred.sum() == 0: continue
                f = f1_score(y, pred, zero_division=0)
                if f > best2['f1']:
                    best2 = {'f1': f, 'alpha': alpha, 'beta': beta, 'th': th,
                             'rec': recall_score(y, pred, zero_division=0),
                             'prec': precision_score(y, pred, zero_division=0)}
    print(f'Best blend: {best2}')

    # Strategy 3: only rescue low-confidence negatives with high error prob
    best3 = {'f1': 0, 'th_meta_low': 0, 'th_err': 0, 'th': 0}
    for th_meta_low in np.arange(0.05, 0.60, 0.05):
        for th_err in np.arange(0.10, 0.95, 0.05):
            prob = meta.copy()
            rescue = (meta <= th_meta_low) & (err > th_err)
            prob[rescue] = 0.9
            for th in np.arange(0.05, 0.95, 0.05):
                pred = (prob > th).astype(int)
                if pred.sum() == 0: continue
                f = f1_score(y, pred, zero_division=0)
                if f > best3['f1']:
                    best3 = {'f1': f, 'th_meta_low': th_meta_low, 'th_err': th_err, 'th': th,
                             'rec': recall_score(y, pred, zero_division=0),
                             'prec': precision_score(y, pred, zero_division=0)}
    print(f'Best rescue: {best3}')


if __name__ == '__main__':
    main()
