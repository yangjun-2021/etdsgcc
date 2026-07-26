"""Per-theft-type and per-building evaluation of the OEDI pipeline.

Reads output/oedi_expert_b.npz (or falls back to expert_a) plus oedi_meta.pkl
and reports, at the global best-F1 threshold of the meta OOF if available:
- overall F1 / recall / precision / AUC
- per-theft-type recall and AUC (Theft1..Theft6)
- per-building-type recall

Usage:
    conda run -n ml python experiments/oedi_per_type_eval.py [--oof oedi_expert_b|oedi_expert_a|meta]
"""
import argparse
import os
import pickle
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

from config import OUTPUT_DIR


def best_f1(y, p):
    best, bth = 0.0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        f = f1_score(y, (p > th).astype(int), zero_division=0)
        if f > best:
            best, bth = f, th
    return best, bth


def load_oof(which):
    if which == 'meta':
        for fname, key in [('oedi_meta_results.npz', 'oof_proba_meta')]:
            path = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(path):
                d = np.load(path)
                if key in d.files:
                    return d[key], fname + ':' + key
        # fall through to expert b if no meta cache
        which = 'oedi_expert_b'
    path = os.path.join(OUTPUT_DIR, f'{which}.npz')
    d = np.load(path)
    return d['oof_proba'], f'{which}.npz:oof_proba'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oof', default='meta',
                    help="meta (default) | oedi_expert_b | oedi_expert_a")
    args = ap.parse_args()

    d = np.load(os.path.join(OUTPUT_DIR, 'oedi_preprocessed.npz'))
    y = d['y'].astype(int)
    meta = pickle.load(open(os.path.join(OUTPUT_DIR, 'oedi_meta.pkl'), 'rb'))
    theft_labels = meta['theft_type_labels']
    building_ids = meta['building_ids']

    proba, src = load_oof(args.oof)
    print(f'OOF source: {src}  (n={len(proba)})')
    assert len(proba) == len(y), 'OOF length mismatch'

    f1, th = best_f1(y, proba)
    pred = (proba > th).astype(int)
    print(f'\n=== Overall: F1={f1:.4f} th={th:.3f} '
          f'Rec={recall_score(y, pred):.4f} Prec={precision_score(y, pred, zero_division=0):.4f} '
          f'AUC={roc_auc_score(y, proba):.4f} ===')

    print('\nPer-theft-type (recall at global threshold / AUC / n):')
    for tt in sorted(set(theft_labels)):
        idx = np.array([i for i, t in enumerate(theft_labels) if t == tt])
        if tt == 'Normal':
            # false positive rate on normals
            fpr = pred[idx].mean()
            print(f'  {tt:8s} n={len(idx):4d}  FPR={fpr:.4f}')
        else:
            rec = recall_score(y[idx], pred[idx], zero_division=0)
            auc = roc_auc_score((np.array(theft_labels) == 'Normal').astype(int), proba) if False else None
            print(f'  {tt:8s} n={len(idx):4d}  Rec={rec:.4f}')

    print('\nPer-building-type (n / pos / recall / precision):')
    for b in sorted(set(building_ids)):
        idx = np.array([i for i, x in enumerate(building_ids) if x == b])
        yy, pp = y[idx], pred[idx]
        if yy.sum() == 0:
            print(f'  {b:22s} n={len(idx):4d} pos=0 (all normal)  FPR={pp.mean():.4f}')
        else:
            print(f'  {b:22s} n={len(idx):4d} pos={int(yy.sum()):3d}  '
                  f'Rec={recall_score(yy, pp, zero_division=0):.4f}  '
                  f'Prec={precision_score(yy, pp, zero_division=0):.4f}')

    # per-theft-type AUC (theft vs normal, restricted binary view)
    print('\nPer-theft-type AUC (this type vs Normal):')
    normal_idx = np.array([i for i, t in enumerate(theft_labels) if t == 'Normal'])
    for tt in sorted(set(theft_labels)):
        if tt == 'Normal':
            continue
        idx = np.array([i for i, t in enumerate(theft_labels) if t == tt])
        sel = np.concatenate([idx, normal_idx])
        yy = (np.array(theft_labels)[sel] != 'Normal').astype(int)
        auc = roc_auc_score(yy, proba[sel])
        print(f'  {tt:8s} AUC={auc:.4f}')


if __name__ == '__main__':
    main()
