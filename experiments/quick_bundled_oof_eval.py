"""Evaluate each column in bundled_oofs.csv / clean_baseline_oofs.csv."""
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from config import OUTPUT_DIR

flags = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags']

results = []
for fname in ['bundled_oofs.csv', 'clean_baseline_oofs.csv']:
    fpath = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(fpath):
        continue
    df = pd.read_csv(fpath)
    for col in df.columns:
        if col in ('FLAG', 'fold', 'id'):
            continue
        arr = df[col].values.astype(np.float64)
        if len(arr) != len(flags):
            continue
        uniq = np.unique(arr)
        if len(uniq) <= 2:
            continue
        try:
            auc = roc_auc_score(flags, arr)
        except Exception:
            auc = np.nan
        best_f1, best_th = 0, 0.5
        for th in np.arange(0.05, 0.95, 0.005):
            pred = (arr > th).astype(int)
            if pred.sum() == 0:
                continue
            f = f1_score(flags, pred, zero_division=0)
            if f > best_f1:
                best_f1, best_th = f, th
        pred = (arr > best_th).astype(int)
        results.append({
            'file': fname,
            'col': col,
            'auc': auc,
            'f1': best_f1,
            'recall': recall_score(flags, pred, zero_division=0),
            'precision': precision_score(flags, pred, zero_division=0),
            'th': best_th,
        })

dfres = pd.DataFrame(results).sort_values('f1', ascending=False)
out_csv = os.path.join(OUTPUT_DIR, 'quick_bundled_oof_eval.csv')
dfres.to_csv(out_csv, index=False)
print(dfres.head(20).to_string(index=False))
print(f'Saved to {out_csv}')
