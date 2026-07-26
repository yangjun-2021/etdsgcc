"""
FFT + EMD feature engineering for SGCC electricity theft detection.

Provides two lightweight, pure-NumPy/SciPy feature extractors that operate on the
log1p-transformed SGCC consumption curves:

  - extract_fft_features : frequency-domain statistics (band energy, spectral
                           entropy, dominant frequency, low/high ratio, etc.)
  - extract_emd_features : simple empirical-mode-decomposition statistics
                           (per-IMF mean/std/energy/fuzzy entropy) using a fast
                           smoothing-based IMF approximation.

Both functions return a feature matrix and a matching list of feature names.
All outputs are sanitized for NaN/Inf and clipped to a safe range.
"""
import warnings

import numpy as np
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings('ignore', 'Mean of empty slice')
warnings.filterwarnings('ignore', 'All-NaN slice')
warnings.filterwarnings('ignore', 'Degrees of freedom')
warnings.filterwarnings('ignore', 'invalid value')
warnings.filterwarnings('ignore', 'divide by zero')
warnings.filterwarnings('ignore', 'Precision loss')


def _sanitize(X):
    """Replace NaN/Inf and clip to a safe range."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -1e6, 1e6)


# ---------------------------------------------------------------------------
# FFT features
# ---------------------------------------------------------------------------
def extract_fft_features(transformed, n_bands=8):
    """Extract FFT-based frequency-domain features.

    Parameters
    ----------
    transformed : np.ndarray, shape [N, T]
        log1p-transformed consumption curves (already imputed).
    n_bands : int, default 8
        Number of linear frequency bands to summarize.

    Returns
    -------
    feature_matrix : np.ndarray, shape [N, F_fft]
    feature_names : list[str]
    """
    transformed = np.asarray(transformed, dtype=np.float32)
    transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples, n_days = transformed.shape
    # Remove DC component per user to avoid a dominant zero-frequency peak.
    x = transformed - np.mean(transformed, axis=1, keepdims=True)

    fft_vals = np.fft.rfft(x, axis=1)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n_days, d=1.0).astype(np.float32)

    eps = 1e-12
    total_power = np.sum(power, axis=1) + eps
    power_norm = power / total_power[:, None]

    features = {}

    # Global spectral statistics
    features['fft_total_energy'] = np.log1p(total_power)
    features['fft_spectral_entropy'] = -np.sum(
        power_norm * np.log(power_norm + eps), axis=1
    )

    peak_idx = np.argmax(power, axis=1)
    features['fft_dominant_freq'] = freqs[peak_idx]
    features['fft_dominant_power_ratio'] = (
        power[np.arange(n_samples), peak_idx] / total_power
    )
    features['fft_dominant_period'] = np.where(
        features['fft_dominant_freq'] > 1e-6,
        1.0 / features['fft_dominant_freq'],
        float(n_days),
    )

    low_power = np.sum(power[:, freqs < 0.01], axis=1)
    high_power = np.sum(power[:, freqs >= 0.01], axis=1) + eps
    features['fft_low_high_freq_ratio'] = low_power / high_power

    features['fft_spectral_centroid'] = np.sum(freqs * power_norm, axis=1)

    # Spectral rolloff (80% energy cutoff)
    cum_power = np.cumsum(power_norm, axis=1)
    rolloff_idx = np.argmax(cum_power >= 0.8, axis=1)
    # If the threshold is never reached, use the last bin.
    rolloff_idx = np.where(cum_power[:, -1] >= 0.8, rolloff_idx, len(freqs) - 1)
    features['fft_spectral_rolloff_80'] = freqs[rolloff_idx]

    # Spectral flatness: geometric / arithmetic mean of power distribution
    log_power_norm = np.log(power_norm + eps)
    geo_mean = np.exp(np.mean(log_power_norm, axis=1))
    arith_mean = np.mean(power_norm, axis=1) + eps
    features['fft_spectral_flatness'] = geo_mean / arith_mean

    # Linear frequency bands
    nyquist = freqs[-1] if len(freqs) else 0.5
    band_edges = np.linspace(0.0, nyquist, n_bands + 1)
    band_energies = np.zeros((n_samples, n_bands), dtype=np.float32)
    for b in range(n_bands):
        low, high = band_edges[b], band_edges[b + 1]
        if b == n_bands - 1:
            mask = (freqs >= low) & (freqs <= high)
        else:
            mask = (freqs >= low) & (freqs < high)
        band_energies[:, b] = np.sum(power[:, mask], axis=1) / total_power

    for b in range(n_bands):
        features[f'fft_band_{b}_energy_ratio'] = band_energies[:, b]

    features['fft_band_energy_std'] = np.std(band_energies, axis=1)
    features['fft_band_energy_max'] = np.max(band_energies, axis=1)
    features['fft_band_energy_min'] = np.min(band_energies, axis=1)

    feature_names = sorted(features.keys())
    feature_matrix = np.column_stack([features[k] for k in feature_names])
    feature_matrix = _sanitize(feature_matrix)
    return feature_matrix, feature_names


# ---------------------------------------------------------------------------
# Simple EMD features
# ---------------------------------------------------------------------------
def _simple_emd(X, n_imfs=4):
    """Fast smoothing-based empirical mode decomposition.

    Each IMF is obtained as the residual between the current signal and a
    progressively stronger low-pass smoothing filter. The final residual is the
    leftover trend. This is a pure-NumPy/SciPy approximation of EMD and avoids
    the PyEMD dependency.

    Parameters
    ----------
    X : np.ndarray, shape [N, T]
    n_imfs : int

    Returns
    -------
    imfs : np.ndarray, shape [N, n_imfs + 1, T]
        The first n_imfs slices are IMFs, the last slice is the residual trend.
    """
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples, n_days = X.shape
    windows = [7, 15, 31, 63]
    # Adapt windows to shorter series.
    windows = [max(3, min(w, n_days)) for w in windows[:n_imfs]]
    # Pad with the largest feasible window if we somehow ended up short.
    while len(windows) < n_imfs:
        windows.append(max(3, n_days))

    imfs = np.zeros((n_samples, n_imfs + 1, n_days), dtype=np.float32)
    current = X.copy()

    for i, w in enumerate(windows):
        # Uniform smoothing is fast, edge-safe, and acts as a low-pass filter.
        trend = uniform_filter1d(current, size=w, axis=1, mode='nearest')
        imfs[:, i, :] = current - trend
        current = trend

    imfs[:, -1, :] = current
    return imfs


def _fuzzy_entropy_single(x, m=2, r_factor=0.15):
    """Fuzzy entropy for a single 1-D series (used as a fallback)."""
    x = np.asarray(x, dtype=np.float32)
    x = x - np.mean(x)
    r = r_factor * np.std(x) + 1e-12
    if np.std(x) < 1e-12 or len(x) < m + 2:
        return 0.0

    def _phi(mm):
        L = len(x)
        vec = np.lib.stride_tricks.sliding_window_view(x, window_shape=mm)
        # Exclude the trivial self-match by subtracting 1.0 later.
        sq = np.sum(vec ** 2, axis=1, keepdims=True)
        d2 = sq + sq.T - 2.0 * (vec @ vec.T)
        d2 = np.maximum(d2, 0.0)
        sim = np.exp(-d2 / (r ** 2))
        # Exclude self-similarity and average over the remaining pairs.
        return np.mean((np.sum(sim, axis=1) - 1.0) / (sim.shape[0] - 1))

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    if phi_m <= 0 or phi_m1 <= 0:
        return 0.0
    return float(np.log(phi_m) - np.log(phi_m1))


def _batch_fuzzy_entropy(X, m=2, r_factor=0.15, batch_size=256):
    """Vectorized fuzzy entropy for many 1-D series.

    Downsamples long series to ~200 points to keep the O(L^2) distance matrix
    memory-friendly while preserving the complexity signal.
    """
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples, L_full = X.shape
    # Reduce length to keep the O(L^2) computation bounded.
    target_len = 200
    downsample = max(1, int(np.floor(L_full / target_len)))
    if downsample > 1:
        X = X[:, ::downsample]
    L = X.shape[1]
    if L < m + 2:
        return np.zeros(n_samples, dtype=np.float32)

    # Center each series.
    X = X - np.mean(X, axis=1, keepdims=True)
    r = r_factor * np.std(X, axis=1, keepdims=True) + 1e-12

    def _phi_for_m(mm):
        P = L - mm + 1
        # [N, P, mm] embedding vectors
        vecs = np.lib.stride_tricks.sliding_window_view(X, window_shape=mm, axis=1)
        # Some NumPy versions return [N, P, mm]; ensure contiguous float32.
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)

        phi = np.empty(n_samples, dtype=np.float32)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            A = vecs[start:end]                       # [B, P, mm]
            A_sq = np.sum(A ** 2, axis=2)[:, :, None]  # [B, P, 1]
            # Pairwise squared Euclidean distances: [B, P, P]
            D2 = A_sq + np.transpose(A_sq, (0, 2, 1)) - 2.0 * (A @ np.transpose(A, (0, 2, 1)))
            D2 = np.maximum(D2, 0.0)
            rr = r[start:end].reshape(-1, 1, 1)        # [B, 1, 1]
            sim = np.exp(-D2 / (rr ** 2))
            # Exclude self-matches.
            phi_batch = np.mean(
                (np.sum(sim, axis=2) - 1.0) / (P - 1), axis=1
            )
            phi[start:end] = phi_batch
        return phi

    phi_m = _phi_for_m(m)
    phi_m1 = _phi_for_m(m + 1)

    phi_m = np.maximum(phi_m, 1e-12)
    phi_m1 = np.maximum(phi_m1, 1e-12)
    fuzzy_en = np.log(phi_m) - np.log(phi_m1)
    return _sanitize(fuzzy_en)


def extract_emd_features(transformed, n_imfs=4):
    """Extract simple EMD-based features.

    Parameters
    ----------
    transformed : np.ndarray, shape [N, T]
        log1p-transformed consumption curves (already imputed).
    n_imfs : int, default 4
        Number of intrinsic mode functions to extract.

    Returns
    -------
    feature_matrix : np.ndarray, shape [N, F_emd]
    feature_names : list[str]
    """
    transformed = np.asarray(transformed, dtype=np.float32)
    transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)

    imfs = _simple_emd(transformed, n_imfs=n_imfs)
    n_samples = imfs.shape[0]
    n_components = n_imfs + 1

    eps = 1e-12
    means = np.mean(imfs, axis=2)
    stds = np.std(imfs, axis=2)
    energy = np.sum(imfs ** 2, axis=2)
    total_energy = np.sum(energy, axis=1, keepdims=True) + eps
    rel_energy = energy / total_energy

    # Fuzzy entropy computed across all components at once for efficiency.
    imfs_2d = imfs.reshape(-1, imfs.shape[2])
    fuzzy_en = _batch_fuzzy_entropy(imfs_2d, m=2, r_factor=0.15, batch_size=256)
    fuzzy_en = fuzzy_en.reshape(n_samples, n_components)

    features = {}
    for i in range(n_components):
        prefix = f'emd_c{i}'
        features[f'{prefix}_mean'] = means[:, i]
        features[f'{prefix}_std'] = stds[:, i]
        features[f'{prefix}_energy_log'] = np.log1p(energy[:, i])
        features[f'{prefix}_rel_energy'] = rel_energy[:, i]
        features[f'{prefix}_fuzzy_entropy'] = fuzzy_en[:, i]

    features['emd_total_energy_log'] = np.log1p(total_energy.squeeze())
    features['emd_energy_std_across_components'] = np.std(rel_energy, axis=1)

    feature_names = sorted(features.keys())
    feature_matrix = np.column_stack([features[k] for k in feature_names])
    feature_matrix = _sanitize(feature_matrix)
    return feature_matrix, feature_names


if __name__ == '__main__':
    # Minimal smoke test.
    rng = np.random.RandomState(42)
    X = rng.randn(50, 1035).astype(np.float32)
    X[:10] += np.sin(np.linspace(0, 20 * np.pi, 1035))

    fft_mat, fft_names = extract_fft_features(X, n_bands=8)
    emd_mat, emd_names = extract_emd_features(X, n_imfs=4)

    print(f"FFT features: {fft_mat.shape}, names={fft_names[:5]}...")
    print(f"EMD features: {emd_mat.shape}, names={emd_names[:5]}...")
    assert np.isfinite(fft_mat).all()
    assert np.isfinite(emd_mat).all()
    assert len(fft_names) == fft_mat.shape[1]
    assert len(emd_names) == emd_mat.shape[1]
    print("Smoke test passed.")
