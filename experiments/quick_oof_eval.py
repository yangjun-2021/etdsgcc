"""Quickly evaluate all saved OOF/prediction arrays in output/."""
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from config import OUTPUT_DIR, SEED
from src.utils.utils import seed_everything

seed_everything(SEED)

flags_path = os.path.join(OUTPUT_DIR, 'sgcc_preprocessed_raw_3ch.npz')
if os.path.exists(flags_path):
    flags = np.load(flags_path)['flags']
else:
    flags = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags']

results = []
npz_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npz')]
for fname in sorted(npz_files):
    fpath = os.path.join(OUTPUT_DIR, fname)
    try:
        data = np.load(fpath, allow_pickle=True)
    except Exception as e:
        continue
    skip_keys = {'flags', 'y', 'y_orig', 'names', 'fold', 'ti', 'vi', 'train_idx', 'val_idx'}
    for key in data.files:
        if key in skip_keys:
            continue
        arr = data[key]
        if not isinstance(arr, np.ndarray) or arr.dtype.kind in {'O', 'U'}:
            continue
        if arr.ndim != 1 or len(arr) != len(flags):
            continue
        # Skip integer labels / binary predictions (not probabilities)
        uniq = np.unique(arr)
        if len(uniq) <= 2:
            continue
        # Only evaluate arrays that look like probabilities (values in [0,1] roughly)
        if arr.min() < -0.01 or arr.max() > 1.01:
            continue
        y_prob = arr.astype(np.float64)
        try:
            auc = roc_auc_score(flags, y_prob)
        except Exception:
            auc = np.nan
        best_f1, best_th = 0, 0.5
        for th in np.arange(0.05, 0.95, 0.005):
            pred = (y_prob > th).astype(int)
            if pred.sum() == 0:
                continue
            f = f1_score(flags, pred, zero_division=0)
            if f > best_f1:
                best_f1, best_th = f, th
        pred = (y_prob > best_th).astype(int)
        rec = recall_score(flags, pred, zero_division=0)
        prec = precision_score(flags, pred, zero_division=0)
        results.append({
            'file': fname,
            'key': key,
            'auc': auc,
            'f1': best_f1,
            'recall': rec,
            'precision': prec,
            'th': best_th,
        })

df = pd.DataFrame(results)
df = df.sort_values('f1', ascending=False)
out_csv = os.path.join(OUTPUT_DIR, 'quick_oof_eval.csv')
df.to_csv(out_csv, index=False)
print(f'Evaluated {len(df)} arrays. Top 10 by F1:')
print(df.head(10).to_string(index=False))
print(f'Saved to {out_csv}')
