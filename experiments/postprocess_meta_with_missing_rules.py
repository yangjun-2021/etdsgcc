"""Post-process meta predictions using missing-ratio / consumption rules.

FNs have very high missing ratio (0.907) and low consumption.
FPs also have high missing ratio (0.851) but slightly higher consumption.
Try rules to rescue FN and filter FP.
"""
import os, sys
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR

meta = np.load(os.path.join(OUTPUT_DIR, 'sgcc_mega_meta.npz'))['oof_final']
y = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags']
pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
X_seq = pre['X_seq']
mask = pre['impute_mask']

val = X_seq[:, 0, :].astype(np.float64)
obs_mask = ~mask
miss_ratio = mask.mean(axis=1)
clean = np.where(obs_mask, val, np.nan)
mean_cons = np.nan_to_num(np.nanmean(clean, axis=1), nan=0.0)
std_cons = np.nan_to_num(np.nanstd(clean, axis=1), nan=0.0)
zero_ratio = ((val <= 0) & obs_mask).sum(axis=1) / np.maximum(obs_mask.sum(axis=1), 1)

best = {'f1': 0, 'th': 0.5, 'mr_fn': 0, 'mean_fn': 0, 'mr_fp': 0, 'mean_fp': 0}

# Baseline
for th in np.arange(0.05, 0.95, 0.005):
    pred = (meta > th).astype(int)
    if pred.sum() == 0: continue
    f = f1_score(y, pred, zero_division=0)
    if f > best['f1']:
        best = {'f1': f, 'th': th, 'mr_fn': 0, 'mean_fn': 0, 'mr_fp': 1, 'mean_fp': 1e9}
print(f'Baseline: F1={best["f1"]:.4f} at th={best["th"]:.3f}')

# Search post-process rules
# Rescue FN: if meta_pred==0 but (high missing + low consumption pattern) -> 1
# Filter FP: if meta_pred==1 but (very high missing + low consumption) -> 0
print('\nSearching rules...')
best_f1 = best['f1']
for th in np.arange(0.50, 0.85, 0.05):
    pred = (meta > th).astype(int)
    for mr_fn in np.arange(0.70, 1.00, 0.05):
        for mean_fn in np.arange(0.0, 3.0, 0.2):
            for mr_fp in np.arange(0.70, 1.00, 0.05):
                for mean_fp in np.arange(0.0, 3.0, 0.2):
                    p = pred.copy()
                    # Rescue potential FN
                    rescue = (p == 0) & (miss_ratio > mr_fn) & (mean_cons < mean_fn)
                    # Filter potential FP
                    filtr = (p == 1) & (miss_ratio > mr_fp) & (mean_cons < mean_fp)
                    p[rescue] = 1
                    p[filtr] = 0
                    if p.sum() == 0: continue
                    f = f1_score(y, p, zero_division=0)
                    if f > best_f1:
                        best_f1 = f
                        best = {'f1': f, 'th': th, 'mr_fn': mr_fn, 'mean_fn': mean_fn,
                                'mr_fp': mr_fp, 'mean_fp': mean_fp,
                                'n_rescue': rescue.sum(), 'n_filter': filtr.sum()}

print(f'Best rule: F1={best["f1"]:.4f} th={best["th"]:.3f}')
print(f'  Rescue: miss>{best["mr_fn"]:.2f} & mean<{best["mean_fn"]:.2f} (n={best.get("n_rescue",0)})')
print(f'  Filter: miss>{best["mr_fp"]:.2f} & mean<{best["mean_fp"]:.2f} (n={best.get("n_filter",0)})')

# Also try adding miss_ratio as a direct correction to meta probability
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

print('\n=== Calibrate meta with miss_ratio + mean_cons via LR ===')
X_calib = np.column_stack([meta, miss_ratio, mean_cons, std_cons, zero_ratio])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_calib = np.zeros(len(y))
for ti, vi in skf.split(X_calib, y):
    m = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=42)
    m.fit(X_calib[ti], y[ti])
    oof_calib[vi] = m.predict_proba(X_calib[vi])[:, 1]

best_f1_c = 0
for th in np.arange(0.05, 0.95, 0.005):
    pred = (oof_calib > th).astype(int)
    if pred.sum() == 0: continue
    f = f1_score(y, pred, zero_division=0)
    if f > best_f1_c: best_f1_c = f
print(f'LR calibrated: F1={best_f1_c:.4f}')
