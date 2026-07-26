"""
Top-Conference Time Series Methods Adapted for Classification
==============================================================

1. FilterNet (NIPS 2024): Multi-band frequency filters
   - Apply band-pass filters in frequency domain
   - Extract band-specific energy and statistics
   - Feature: 10 frequency bands × 3 stats = 30d

2. TimeKAN (ICLR 2025): Cascade frequency decomposition
   - Multi-scale moving average to get N frequency levels
   - Per-level statistics + cross-level ratios
   - Feature: 5 levels × 3 stats = 15d

3. PatchTST (ICLR 2023): Patch-based representation
   - Split sequence into overlapping patches
   - Per-patch statistics + aggregation
   - Feature: aggregated patch statistics = 15d
"""
import numpy as np
from scipy.fft import rfft, rfftfreq, irfft
from scipy.stats import skew, kurtosis


def compute_filternet_features(X, n_bands=10, feat_dim=30):
    """
    FilterNet-inspired: Multi-band frequency filtering.
    
    For each user:
      1. Apply FFT
      2. Split frequency spectrum into n_bands (exponentially spaced)
      3. For each band: apply band-pass filter, extract statistics
    
    Features (30d = 10 bands × 3 stats):
      - Band energy ratio (total energy in band / total energy)
      - Band spectral centroid  
      - Band spectral spread
    """
    n, T = X.shape
    feats = np.zeros((n, feat_dim), dtype=np.float32)
    col = 0
    
    # Frequency bands: exponentially spaced (more resolution at low frequencies)
    max_freq = 0.5  # Nyquist
    band_edges = np.logspace(-3, np.log10(max_freq), n_bands + 1)
    # Convert to FFT bin indices
    freqs = rfftfreq(T)
    
    band_bins = []
    for i in range(n_bands):
        lo, hi = band_edges[i], band_edges[i+1]
        mask = (freqs >= lo) & (freqs < hi)
        band_bins.append(np.where(mask)[0])
    
    for i in range(n):
        loc_col = 0
        ts = X[i]
        if np.std(ts) < 1e-6:
            col += 30
            continue
        
        # FFT
        fft_vals = np.abs(rfft(ts - ts.mean()))
        total_energy = np.sum(fft_vals ** 2) + 1e-10
        
        for bi, bins in enumerate(band_bins):
            if len(bins) < 2:
                for _ in range(3):
                    if loc_col < feat_dim: loc_col += 1
                continue
            
            band_vals = fft_vals[bins]
            band_energy = np.sum(band_vals ** 2) / total_energy
            
            # Band spectral centroid
            band_freqs = freqs[bins]
            centroid = np.sum(band_freqs * band_vals) / (np.sum(band_vals) + 1e-10)
            
            # Band spectral spread
            spread = np.sqrt(np.sum((band_freqs - centroid) ** 2 * band_vals) / (np.sum(band_vals) + 1e-10))
            
            feats[i, loc_col] = band_energy
            feats[i, loc_col + 1] = centroid
            feats[i, loc_col + 2] = spread
            loc_col += 3
        
        col += 30
    
    feats = np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=0.0)
    return feats[:, :col]


def compute_timekan_features(X, n_levels=5, window_d=7, feat_dim=15):
    """
    TimeKAN-inspired: Cascade frequency decomposition.
    
    For each user:
      1. Apply moving average at level i to get smoothed series
      2. Subtract smoothed from previous level to get "frequency band"
      3. Extract statistics for each band
    
    Features (15d = 5 levels × 3 stats):
      - Band mean
      - Band std  
      - Band energy ratio
    """
    n, T = X.shape
    feats = np.zeros((n, feat_dim), dtype=np.float32)
    col = 0
    
    for i in range(n):
        loc_col = 0
        ts = X[i]
        if np.std(ts) < 1e-6:
            continue
        
        # Multi-level decomposition
        prev_smooth = ts.copy()
        total_energy = np.sum(ts ** 2) + 1e-10
        
        for level in range(n_levels):
            # Moving average with increasing window
            window = int(window_d * (2 ** level))
            if window >= T // 2:
                break
            
            # Simple moving average
            kernel = np.ones(window) / window
            smooth = np.convolve(prev_smooth, kernel, mode='same')
            # Fix edge effects
            smooth[:window//2] = smooth[window//2]
            smooth[-window//2:] = smooth[-window//2-1]
            
            # Detail = previous - smoothed (this is the frequency band)
            detail = prev_smooth - smooth
            
            # Band statistics
            band_mean = np.mean(np.abs(detail))
            band_std = np.std(detail)
            band_energy = np.sum(detail ** 2) / total_energy
            
            feats[i, loc_col] = band_mean
            feats[i, loc_col + 1] = band_std
            feats[i, loc_col + 2] = band_energy
            loc_col += 3
            
            prev_smooth = smooth
    
    feats = np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=0.0)
    return feats[:, :col]


def compute_patchtst_features(X, patch_len=30, stride=15, feat_dim=15):
    """
    PatchTST-inspired: Patch-based time series representation.
    
    For each user:
      1. Split sequence into overlapping patches of length patch_len
      2. For each patch: extract statistics (mean, std, max, min, range, cv, trend)
      3. Aggregate across patches: mean, std, max of each patch statistic
      4. Also: adjacent patch differences (temporal dynamics)
    
    Features (15d):
      - Mean of patch means
      - Std of patch means
      - Max patch mean / min patch mean (range)
      - Mean of patch stds
      - Max patch std
      - Patch trend (slope of patch means over time)
      - Patch mean CV
      - Max patch max
      - Min patch min
      - Adjacent patch correlation
      - Patch energy distribution
    """
    n, T = X.shape
    feats = np.zeros((n, feat_dim), dtype=np.float32)
    col = 0
    
    for i in range(n):
        loc_col = 0
        ts = X[i]
        
        # Extract patches
        patches_means = []
        patches_stds = []
        patches_maxs = []
        patches_mins = []
        patches_cvs = []
        
        for start in range(0, T - patch_len + 1, stride):
            patch = ts[start:start + patch_len]
            non_zero = patch[patch > 0.01]
            if len(non_zero) < 3:
                continue
            patches_means.append(np.mean(non_zero))
            patches_stds.append(np.std(non_zero))
            patches_maxs.append(np.max(non_zero))
            patches_mins.append(np.min(non_zero))
            patches_cvs.append(np.std(non_zero) / (np.mean(non_zero) + 1e-6))
        
        if len(patches_means) < 3:
            continue
        
        patches_means = np.array(patches_means)
        patches_stds = np.array(patches_stds)
        patches_maxs = np.array(patches_maxs)
        patches_mins = np.array(patches_mins)
        
        # Patch-level aggregation
        feats[i, loc_col] = patches_means.mean()  # avg of patch means
        feats[i, loc_col + 1] = patches_means.std()  # variation across patches
        feats[i, loc_col + 2] = patches_means.max() / (patches_means.min() + 1e-6)  # range ratio
        feats[i, loc_col + 3] = patches_stds.mean()  # avg patch std
        feats[i, loc_col + 4] = patches_stds.max()  # max patch variability
        feats[i, loc_col + 5] = np.polyfit(np.arange(len(patches_means)), patches_means, 1)[0]  # trend
        
        # Normalized by global mean
        global_mean = np.nanmean(ts)
        feats[i, loc_col + 6] = (patches_means.max() - patches_means.min()) / (global_mean + 1e-6)
        
        # Adjacent patch similarity
        if len(patches_means) > 2:
            adj_diff = np.diff(patches_means)
            feats[i, loc_col + 7] = np.std(adj_diff) / (np.abs(adj_diff).mean() + 1e-6)  # normalized variability
            feats[i, loc_col + 8] = (np.abs(adj_diff) > np.std(adj_diff) * 2).sum()  # jump count
            feats[i, loc_col + 9] = np.corrcoef(patches_means[:-1], patches_means[1:])[0, 1] if len(adj_diff) > 1 else 0
        
        # Extreme patch detection
        feats[i, loc_col + 10] = patches_maxs.max() / (global_mean + 1e-6)  # max/global
        feats[i, loc_col + 11] = patches_mins.min() / (global_mean + 1e-6)  # min/global
        feats[i, loc_col + 12] = (patches_maxs - patches_mins).max() / (global_mean + 1e-6)  # max daily range
        
        # Early vs late patches
        mid = len(patches_means) // 2
        if mid > 0:
            early_mean = patches_means[:mid].mean()
            late_mean = patches_means[mid:].mean()
            feats[i, loc_col + 13] = (late_mean - early_mean) / (global_mean + 1e-6)
            feats[i, loc_col + 14] = np.std(patches_means[mid:]) / (np.std(patches_means[:mid]) + 1e-6)  # variability change
    
    feats = np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=0.0)
    return feats
