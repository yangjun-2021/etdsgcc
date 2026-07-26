"""Error analysis for current best meta OOF."""
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR

# Load data
meta = np.load(os.path.join(OUTPUT_DIR, 'sgcc_mega_meta.npz'))['oof_final']
y = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags']
pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
X_seq = pre['X_seq']
mask = pre['impute_mask']
stat = pre['stat_features']

best_th = 0.5
best_f1 = 0
for th in np.arange(0.05, 0.95, 0.005):
    pred = (meta > th).astype(int)
    if pred.sum() == 0: continue
    f = f1_score(y, pred, zero_division=0)
    if f > best_f1: best_f1, best_th = f, th
pred = (meta > best_th).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
print(f'Best th={best_th:.3f} F1={best_f1:.4f} TN={tn} FP={fp} FN={fn} TP={tp}')

# Try to load some strong individual OOFs
strong_oofs = {}
for fname, key in [
    ('autoresearch_best.npz', 'oof_final'),
    ('mega_boost_enhanced.npz', 'oof_final'),
    ('informer_oof.npz', 'oof_informer'),
    ('amst_3ch_recall10_oof.npz', 'oof_amst_3ch_recall10'),
    ('patch_transformer_raw_3ch_oof.npz', 'oof_patch_transformer_raw_3ch'),
    ('supcon_raw_3ch_oof.npz', 'oof_supcon_raw_3ch'),
]:
    try:
        d = np.load(os.path.join(OUTPUT_DIR, fname))
        strong_oofs[fname.replace('.npz','')] = d[key]
    except Exception as e:
        print(f'Could not load {fname}: {e}')

# For each error type, compute how many strong models also got it wrong
print('\n=== Error overlap with strong OOFs ===')
for name, oof in strong_oofs.items():
    oof_pred = (oof > 0.5).astype(int)
    fn_overlap = ((pred == 0) & (y == 1) & (oof_pred == 0)).sum()
    fp_overlap = ((pred == 1) & (y == 0) & (oof_pred == 1)).sum()
    print(f'{name:30s}: FN_overlap={fn_overlap}/{fn} ({fn_overlap/fn*100:.1f}%)  '
          f'FP_overlap={fp_overlap}/{fp} ({fp_overlap/fp*100:.1f}%)')

# Analyze FN/FP by statistical features
print('\n=== FN/FP feature stats ===')
fn_idx = np.where((pred == 0) & (y == 1))[0]
fp_idx = np.where((pred == 1) & (y == 0))[0]
tp_idx = np.where((pred == 1) & (y == 1))[0]
tn_idx = np.where((pred == 0) & (y == 0))[0]

val = X_seq[:, 0, :].astype(np.float64)
obs_mask = ~mask

def feats(idx):
    v = val[idx]
    m = obs_mask[idx]
    clean = np.where(m, v, np.nan)
    return {
        'n': len(idx),
        'miss_ratio': m.mean(axis=1).mean(),
        'zero_ratio': ((v <= 0) & m).sum(axis=1).mean() / max(m.sum(axis=1).mean(), 1),
        'mean': np.nanmean(clean),
        'std': np.nanstd(clean),
        'max': np.nanmax(clean),
        'min': np.nanmin(clean),
    }

for label, idx in [('FN', fn_idx), ('FP', fp_idx), ('TP', tp_idx), ('TN', tn_idx)]:
    f = feats(idx)
    print(f'{label:3s}: n={f["n"]:4d} miss={f["miss_ratio"]:.3f} zero={f["zero_ratio"]:.3f} '
          f'mean={f["mean"]:.2f} std={f["std"]:.2f} max={f["max"]:.1f} min={f["min"]:.1f}')

# Check if FNs are mostly low-confidence positives
print('\n=== Meta score distribution by true label ===')
for label, idx in [('FN', fn_idx), ('TP', tp_idx), ('FP', fp_idx), ('TN', tn_idx)]:
    scores = meta[idx]
    print(f'{label:3s}: mean={scores.mean():.3f} median={np.median(scores):.3f} '
          f'q25={np.percentile(scores,25):.3f} q75={np.percentile(scores,75):.3f}')
