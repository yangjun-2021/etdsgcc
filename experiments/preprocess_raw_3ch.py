"""Create a raw-scale 3-channel SGCC input for alternative models.

Channels:
  0: raw imputed consumption (linear scale)
  1: raw first-order difference
  2: missing-value mask

This is intentionally different from the log1p-based sgcc_preprocessed_3ch.npz,
so models trained on it may capture high-consumption theft patterns that are
compressed by the log transform.
"""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SGCC_RAW_PATH
from src.data.preprocess_sgcc import three_layer_imputation
import pandas as pd

print('Loading raw SGCC data...')
df = pd.read_csv(SGCC_RAW_PATH)
date_cols = [c for c in df.columns if '/' in str(c) and len(str(c)) <= 10]
raw = df[date_cols].values.astype(float)
flags = df['FLAG'].values.astype(int)
print(f'  Shape: {raw.shape}, Theft: {flags.sum()}/{len(flags)}')

print('Imputing missing values...')
imputed, impute_mask = three_layer_imputation(raw, flags)

# raw diff (imputed)
diff = np.diff(imputed, axis=1)
diff = np.concatenate([diff[:, :1], diff], axis=1)

# per-sample z-score on raw value for stability
mean = imputed.mean(axis=1, keepdims=True)
std = imputed.std(axis=1, keepdims=True) + 1e-6
val = (imputed - mean) / std

# diff z-score
dmean = diff.mean(axis=1, keepdims=True)
dstd = diff.std(axis=1, keepdims=True) + 1e-6
diff_z = (diff - dmean) / dstd

mask_ch = impute_mask.astype(np.float32)

X_seq = np.stack([val.astype(np.float32), diff_z.astype(np.float32), mask_ch], axis=1)
print(f'X_seq shape: {X_seq.shape}')
print(f'  ch0 range: [{X_seq[:,0].min():.3f}, {X_seq[:,0].max():.3f}]')
print(f'  ch1 range: [{X_seq[:,1].min():.3f}, {X_seq[:,1].max():.3f}]')
print(f'  ch2 range: [{X_seq[:,2].min():.3f}, {X_seq[:,2].max():.3f}]')

np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'sgcc_preprocessed_raw_3ch.npz'),
    X_seq=X_seq,
    flags=flags,
)
print(f'Saved to {os.path.join(OUTPUT_DIR, "sgcc_preprocessed_raw_3ch.npz")}')
