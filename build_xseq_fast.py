"""
Fast X_seq builder for SGCC that skips slow statistical feature computation.
Only reconstructs the multi-channel input needed by deep experts.
"""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR
from src.data.preprocess_sgcc import (
    load_sgcc, three_layer_imputation, winsorize_and_log,
    compute_rolling_volatility, compute_entropy_proxy, compute_stl_residual,
    build_multi_channel_input
)


def main():
    print("[Fast X_seq] Loading SGCC raw data...")
    raw, flags, cons_no, date_cols = load_sgcc()

    print("[Fast X_seq] Imputation...")
    imputed, impute_mask = three_layer_imputation(raw, flags)

    print("[Fast X_seq] Winsorize + log...")
    transformed, upper_clip, lower_clip, winsorized = winsorize_and_log(imputed)

    print("[Fast X_seq] Computing channels...")
    volatility = compute_rolling_volatility(transformed, window=7)
    entropy_proxy = compute_entropy_proxy(transformed, impute_mask, window=30, stride=7)
    residuals = compute_stl_residual(transformed, impute_mask, period=7)

    print("[Fast X_seq] Building X_seq...")
    X_seq = build_multi_channel_input(transformed, volatility, entropy_proxy, impute_mask, residuals)

    save_path = os.path.join(OUTPUT_DIR, 'sgcc_xseq_fast.npz')
    np.savez_compressed(
        save_path,
        X_seq=X_seq,
        flags=flags,
        impute_mask=impute_mask,
    )
    print(f"[Fast X_seq] Saved to {save_path}: X_seq={X_seq.shape}")


if __name__ == '__main__':
    main()
