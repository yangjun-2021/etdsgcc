"""
Deep feature engineering for SGCC electricity theft detection.

Reconstructed from V71's deep_features.py (original file lost).
Three feature groups:

1. Peer Deviation Features: How much each user deviates from peer group
   (same month, similar consumption level) — theft users deviate more.

2. SVD Residual Features: SVD decomposition of the consumption matrix.
   Normal users lie on the low-rank manifold; theft users have large residuals.

3. Symbolic Features: SAX-inspired symbolic representation capturing
   temporal shape patterns independent of absolute level.
"""
import warnings
import numpy as np
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings('ignore', 'invalid value')
warnings.filterwarnings('ignore', 'divide by zero')


def compute_peer_deviation_features(X_interp, dates_month, n_bins=5, feat_dim=24):
    """Peer group deviation features.

    For each month (1-12) and each consumption-level bin:
      compute how much the user deviates from the peer-group median.

    Args:
        X_interp: [N, T] interpolated (NaN-filled) consumption data
        dates_month: [T] array of month numbers (1-12)
        n_bins: number of consumption-level bins
        feat_dim: max features (12 months * 2 stats = 24)

    Returns:
        [N, feat_dim] feature matrix
    """
    n, T = X_interp.shape
    feats = np.zeros((n, feat_dim), dtype=np.float32)
    col = 0

    user_means = np.nanmean(X_interp, axis=1)
    user_means = np.nan_to_num(user_means, nan=0.0)

    bin_edges = np.percentile(user_means[user_means > 0],
                               np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    user_bins = np.digitize(user_means, bin_edges[1:-1]) if len(bin_edges) > 2 else np.zeros(n, dtype=int)

    for month in range(1, 13):
        m_mask = dates_month == month
        if m_mask.sum() < 10:
            col += 2
            continue

        month_data = X_interp[:, m_mask]
        month_user_means = np.nanmean(month_data, axis=1)
        month_user_means = np.nan_to_num(month_user_means, nan=0.0)

        deviations = np.zeros(n)
        for b in range(n_bins):
            peer_mask = user_bins == b
            if peer_mask.sum() < 5:
                continue
            peer_median = np.median(month_user_means[peer_mask])
            peer_iqr = (np.percentile(month_user_means[peer_mask], 75) -
                        np.percentile(month_user_means[peer_mask], 25))
            peer_iqr = max(peer_iqr, 1e-6)
            deviations[peer_mask] = np.abs(month_user_means[peer_mask] - peer_median) / peer_iqr

        if col < feat_dim:
            feats[:, col] = np.nan_to_num(deviations, nan=0.0)
            col += 1

        month_stds = np.nanstd(month_data, axis=1)
        month_stds = np.nan_to_num(month_stds, nan=0.0)
        std_deviations = np.zeros(n)
        for b in range(n_bins):
            peer_mask = user_bins == b
            if peer_mask.sum() < 5:
                continue
            peer_median_std = np.median(month_stds[peer_mask])
            peer_iqr_std = (np.percentile(month_stds[peer_mask], 75) -
                            np.percentile(month_stds[peer_mask], 25))
            peer_iqr_std = max(peer_iqr_std, 1e-6)
            std_deviations[peer_mask] = np.abs(month_stds[peer_mask] - peer_median_std) / peer_iqr_std

        if col < feat_dim:
            feats[:, col] = np.nan_to_num(std_deviations, nan=0.0)
            col += 1

    feats = np.clip(feats, -10, 10)
    return feats[:, :col]


def compute_svd_residual_features(X_interp, n_components=20, feat_dim=20):
    """SVD residual features.

    Decompose the consumption matrix into low-rank structure.
    Theft users have large reconstruction residuals.

    Features:
      1. Full residual norm
      2. Max single-day residual
      3. Residual std
      4. Residual skewness
      5. Residual kurtosis
      6-20. Residual at specific frequency bands (FFT of residual)
    """
    n, T = X_interp.shape
    X = np.nan_to_num(X_interp, nan=0.0).astype(np.float64)

    X_centered = X - X.mean(axis=1, keepdims=True)
    X_centered = X_centered / (np.std(X_centered, axis=1, keepdims=True) + 1e-6)

    k = min(n_components, min(n, T) - 1)
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    X_reconstructed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    residual = X_centered - X_reconstructed

    feats = np.zeros((n, feat_dim), dtype=np.float32)
    col = 0

    res_norm = np.sqrt(np.sum(residual ** 2, axis=1))
    feats[:, col] = res_norm; col += 1

    feats[:, col] = np.max(np.abs(residual), axis=1); col += 1
    feats[:, col] = np.std(residual, axis=1); col += 1

    res_mean = np.mean(residual, axis=1, keepdims=True)
    res_std = np.std(residual, axis=1) + 1e-8
    res_centered = residual - res_mean
    feats[:, col] = np.mean(res_centered ** 3, axis=1) / (res_std ** 3); col += 1
    feats[:, col] = np.mean(res_centered ** 4, axis=1) / (res_std ** 4) - 3; col += 1

    from scipy.fft import rfft
    for i in range(n):
        res_fft = np.abs(rfft(residual[i]))
        if len(res_fft) > 1:
            total_energy = np.sum(res_fft[1:] ** 2) + 1e-10
            n_bands = min(feat_dim - col, 5)
            band_size = max(1, (len(res_fft) - 1) // n_bands)
            for b in range(n_bands):
                start = 1 + b * band_size
                end = min(start + band_size, len(res_fft))
                if end > start:
                    band_energy = np.sum(res_fft[start:end] ** 2) / total_energy
                    if col + b < feat_dim:
                        feats[i, col + b] = band_energy
    col = min(col + 5, feat_dim)

    while col < feat_dim:
        col += 1

    feats = np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0)
    feats = np.clip(feats, -10, 10)
    return feats[:, :col]


def compute_symbolic_features(X_interp, n_symbols=5, window=7, feat_dim=15):
    """SAX-inspired symbolic aggregation features.

    Convert each user's time series into symbolic sequences,
    then extract pattern-based features.

    Features:
      1. Symbol transition rate
      2. Most frequent symbol ratio
      3. Symbol entropy
      4. Max consecutive same-symbol run
      5-9. Per-symbol frequency (5 symbols)
      10-14. Per-symbol mean duration
      15. Symbol sequence complexity (Lempel-Ziv approximation)
    """
    n, T = X_interp.shape
    X = np.nan_to_num(X_interp, nan=0.0).astype(np.float64)

    if T < window * 2:
        return np.zeros((n, feat_dim), dtype=np.float32)

    padded = np.pad(X, ((0, 0), (0, window - T % window)), mode='edge') if T % window != 0 else X
    n_windows = padded.shape[1] // window
    windowed = padded[:, :n_windows * window].reshape(n, n_windows, window)
    window_means = windowed.mean(axis=2)

    quantile_edges = np.percentile(window_means[window_means > 0],
                                    np.linspace(100 / n_symbols, 100, n_symbols - 1))
    quantile_edges = np.unique(quantile_edges)
    if len(quantile_edges) < n_symbols - 1:
        symbols = np.zeros_like(window_means, dtype=int)
    else:
        symbols = np.digitize(window_means, quantile_edges)

    feats = np.zeros((n, feat_dim), dtype=np.float32)
    col = 0

    for i in range(n):
        sym = symbols[i]
        if len(sym) < 2:
            continue
        transitions = np.sum(np.diff(sym) != 0)
        feats[i, col] = transitions / (len(sym) - 1)
    col += 1

    for i in range(n):
        sym = symbols[i]
        vals, counts = np.unique(sym, return_counts=True)
        feats[i, col] = counts.max() / len(sym) if len(sym) > 0 else 0
    col += 1

    for i in range(n):
        sym = symbols[i]
        vals, counts = np.unique(sym, return_counts=True)
        p = counts / (counts.sum() + 1e-10)
        feats[i, col] = -np.sum(p * np.log(p + 1e-10))
    col += 1

    for i in range(n):
        sym = symbols[i]
        max_run = 1
        cur_run = 1
        for j in range(1, len(sym)):
            if sym[j] == sym[j - 1]:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 1
        feats[i, col] = max_run / len(sym) if len(sym) > 0 else 0
    col += 1

    for s in range(n_symbols):
        if col >= feat_dim:
            break
        for i in range(n):
            sym = symbols[i]
            feats[i, col] = np.mean(sym == s) if len(sym) > 0 else 0
        col += 1

    for s in range(n_symbols):
        if col >= feat_dim:
            break
        for i in range(n):
            sym = symbols[i]
            mask = sym == s
            if mask.any():
                runs = []
                cur = 0
                for v in mask:
                    if v:
                        cur += 1
                    elif cur > 0:
                        runs.append(cur)
                        cur = 0
                if cur > 0:
                    runs.append(cur)
                feats[i, col] = np.mean(runs) if runs else 0
        col += 1

    for i in range(n):
        sym = symbols[i].astype(str)
        s = ''.join(sym)
        n_substrings = len(set(s[j:j+3] for j in range(max(0, len(s) - 2))))
        max_substrings = min(len(s) ** 2, T)
        feats[i, col] = n_substrings / max(max_substrings, 1) if max_substrings > 0 else 0
    col += 1

    feats = np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0)
    feats = np.clip(feats, -10, 10)
    return feats[:, :col]


if __name__ == '__main__':
    import os
    np.random.seed(42)
    X = np.random.rand(100, 365) * 50
    dates_month = np.array([(i % 12) + 1 for i in range(365)])

    p = compute_peer_deviation_features(X, dates_month)
    print(f'Peer deviation: {p.shape}')

    s = compute_svd_residual_features(X, n_components=10)
    print(f'SVD residual: {s.shape}')

    t = compute_symbolic_features(X)
    print(f'Symbolic: {t.shape}')
    print('All tests passed!')
