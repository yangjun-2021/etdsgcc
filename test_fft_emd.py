"""
Standalone smoke / correctness test for the new FFT + EMD features.

Run:
    python test_fft_emd.py

It will:
  1. Load a small sample of SGCC raw data.
  2. Apply the same imputation / winsorization / log1p transform used by the
     preprocessing pipeline.
  3. Extract FFT and EMD features.
  4. Check shapes, feature counts, NaN/Inf handling, and class-conditional
     statistics.
"""
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import SGCC_RAW_PATH, SGCC_CONFIG
from src.features.fft_emd_features import extract_fft_features, extract_emd_features


def quick_preprocess_sample(raw, flags, n_sample=500, seed=42):
    """Minimal preprocessing for testing (no 3-layer imputation)."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(flags), size=min(n_sample, len(flags)), replace=False)
    raw = raw[idx]
    flags = flags[idx]

    imputed = np.nan_to_num(raw, nan=0.0)
    upper = np.percentile(imputed, 99.5)
    lower = np.percentile(imputed, 0.5)
    winsorized = np.clip(imputed, lower, upper)
    transformed = np.log1p(np.maximum(winsorized, 0.0))
    return transformed, flags


def main():
    print(f"[Test] Loading SGCC raw data from {SGCC_RAW_PATH}")
    df = pd.read_csv(SGCC_RAW_PATH)
    date_cols = [c for c in df.columns if '/' in str(c) and len(str(c)) <= 10]
    raw = df[date_cols].values.astype(float)
    flags = df[SGCC_CONFIG['label_col']].values.astype(int)

    n_sample = 500
    transformed, flags = quick_preprocess_sample(raw, flags, n_sample=n_sample)

    print("[Test] Extracting FFT features...")
    fft_mat, fft_names = extract_fft_features(transformed, n_bands=8)
    print(f"  Shape: {fft_mat.shape}, count: {len(fft_names)}")
    assert fft_mat.shape == (n_sample, len(fft_names))
    assert len(fft_names) < 30, f"FFT features exceed limit: {len(fft_names)}"
    assert np.isfinite(fft_mat).all(), "FFT features contain NaN/Inf"

    print("[Test] Extracting EMD features...")
    emd_mat, emd_names = extract_emd_features(transformed, n_imfs=4)
    print(f"  Shape: {emd_mat.shape}, count: {len(emd_names)}")
    assert emd_mat.shape == (n_sample, len(emd_names))
    assert len(emd_names) < 40, f"EMD features exceed limit: {len(emd_names)}"
    assert np.isfinite(emd_mat).all(), "EMD features contain NaN/Inf"

    combined = np.column_stack([fft_mat, emd_mat])
    print(f"[Test] Combined FFT+EMD shape: {combined.shape}")

    # Basic class-conditional sanity check.
    pos_mask = flags == 1
    neg_mask = flags == 0
    for name, mat in [('FFT', fft_mat), ('EMD', emd_mat)]:
        pos_mean = np.mean(mat[pos_mask], axis=0)
        neg_mean = np.mean(mat[neg_mask], axis=0)
        diff = np.abs(pos_mean - neg_mean)
        top_idx = np.argsort(diff)[-5:][::-1]
        print(f"\n  Top 5 discriminative {name} features (by mean difference):")
        for idx in top_idx:
            feature_name = (fft_names if name == 'FFT' else emd_names)[idx]
            print(f"    {feature_name:<40s} diff={diff[idx]:.4f}")

    print("\n[Test] All checks passed.")


if __name__ == '__main__':
    main()
