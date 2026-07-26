"""
Deep Data Engineering: Novel features for detecting "mimicking" theft users.

Focus: The 78.7% of theft users with no clear behavioral signature.
These users have moderate consumption, low missing rate, low CV,
and daily curves correlated with normal users (r=0.47).

New feature categories:
  1. Entropy & Complexity — Sample entropy, Permutation entropy, LZ complexity
  2. Distribution Shape — Best-fit distribution params, Hellinger distance
  3. Extreme Value — GPD tail parameters, exceedance statistics
  4. Burstiness — Concentration of consumption, inter-event times
  5. Time Asymmetry — Forward/backward transition probabilities
  6. Fractal Features — Hurst exponent, detrended fluctuation analysis

All features computable from daily consumption data only.
"""
import os, time, glob
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats, special
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import roc_auc_score

SEED = 42; np.random.seed(SEED)


# ======================================================================
# 1. SAMPLE ENTROPY — measures time series regularity
# ======================================================================
def sample_entropy(u, m=2, r_factor=0.2):
    """Sample entropy: higher = more irregular/complex.
    
    Lower values indicate more self-similarity (repeating patterns).
    Theft users tampering with meters might have LOWER sample entropy
    because their consumption becomes artificially "too regular".
    """
    n = len(u)
    if n < m + 2:
        return 0.0
    
    r = r_factor * np.std(u)
    if r < 1e-10:
        return 0.0
    
    def _count_matches(template_len):
        count = 0
        templates = np.array([u[i:i+template_len] for i in range(n - template_len)])
        for i in range(len(templates) - 1):
            dists = np.max(np.abs(templates[i+1:] - templates[i]), axis=1)
            count += np.sum(dists < r)
        return count
    
    A = _count_matches(m + 1)
    B = _count_matches(m)
    
    if B == 0 or A == 0:
        return 0.0
    return -np.log(A / B) if A > 0 and B > 0 else 0.0


# ======================================================================
# 2. PERMUTATION ENTROPY — captures ordinal patterns
# ======================================================================
def permutation_entropy(u, order=3, delay=1):
    """Permutation entropy: higher = more random ordering.
    
    Counts the frequency of ordinal patterns (e.g., "up-up", "up-down").
    Theft users with regular patterns have low permutation entropy.
    """
    n = len(u)
    if n < order * delay:
        return 0.0
    
    permutations = []
    for i in range(n - (order - 1) * delay):
        window = [u[i + j * delay] for j in range(order)]
        sorted_idx = sorted(range(order), key=lambda k: window[k])
        perm = tuple(np.argsort(sorted_idx))
        permutations.append(perm)
    
    unique_perms = set(permutations)
    counts = {p: permutations.count(p) for p in unique_perms}
    probs = np.array(list(counts.values())) / len(permutations)
    return -np.sum(probs * np.log(probs + 1e-10)) / np.log(len(unique_perms))


# ======================================================================
# 3. LEMPEL-ZIV COMPLEXITY — algorithmic complexity
# ======================================================================
def lz_complexity(u, n_bins=5):
    """Lempel-Ziv complexity of binarized time series.
    
    Lower LZ = more repetitive patterns (potential artificial regularity).
    """
    n = len(u)
    if n < 10:
        return 0.0
    
    # Binarize into n_bins levels
    bins = np.percentile(u, np.linspace(0, 100, n_bins + 1))
    if np.unique(bins).shape[0] < 2:
        return 0.0
    s = np.digitize(u, bins[1:-1])
    
    # LZ76 algorithm
    i, c, l = 0, 1, 1
    while i + l <= n:
        sub = tuple(s[i:i+l])
        prev = tuple(s[:i+l])
        if sub in [prev[j:j+l] for j in range(len(prev) - l + 1)]:
            l += 1
        else:
            c += 1; i += l; l = 1
    return c / (n / np.log2(n)) if n > 1 else 0


# ======================================================================
# 4. EXTREME VALUE DISTRIBUTION FITTING
# ======================================================================
def extreme_value_features(u):
    """Fit Generalized Pareto Distribution to upper tail.
    
    Returns: shape (xi), scale (sigma), tail index
    xi > 0: heavy-tailed (more extreme high-consumption days)
    xi < 0: bounded tail
    xi = 0: exponential tail (normal behavior)
    """
    n = len(u)
    if n < 20:
        return 0.0, 0.0, 0.0, 0.0
    
    # Use top 10% as exceedances
    threshold = np.percentile(u, 90)
    exceedances = u[u > threshold] - threshold
    n_exc = len(exceedances)
    
    if n_exc < 10:
        return 0.0, 0.0, 0.0, 0.0
    
    # Method of moments estimator for GPD
    mean_exc = np.mean(exceedances)
    var_exc = np.var(exceedances)
    
    if var_exc < 1e-10:
        return 0.0, mean_exc, 0.0, 0.0
    
    # GPD parameters
    xi_hat = 0.5 * (1 - mean_exc**2 / var_exc)
    sigma_hat = 0.5 * mean_exc * (1 + mean_exc**2 / var_exc)
    
    sigma_hat = max(sigma_hat, 1e-6)
    
    # Tail heaviness indicator
    p99 = np.percentile(u, 99)
    p95 = np.percentile(u, 95)
    p90 = np.percentile(u, 90)
    tail_ratio = (p99 - p95) / (p95 - p90 + 1e-6)  # >1 = heavy tail
    
    # Exceedance ratio
    exc_ratio = n_exc / n
    
    return xi_hat, sigma_hat, tail_ratio, exc_ratio


# ======================================================================
# 5. DISTRIBUTION SHAPE FEATURES
# ======================================================================
def distribution_shape_features(u):
    """Distribution shape features beyond standard moments.
    
    Returns: gini coefficient, Hellinger distance from normal, 
             bimodality coefficient, spread/centrality ratio
    """
    n = len(u)
    pos = u[u > 0] if np.any(u > 0) else u
    pos = np.sort(pos)
    
    if len(pos) < 10:
        return np.zeros(6)
    
    # Gini coefficient (consumption inequality within user)
    cumsum = np.cumsum(pos)
    gini = (2 * np.sum((np.arange(1, len(pos)+1) * pos)) - 
            (len(pos) + 1) * np.sum(pos)) / (len(pos) * np.sum(pos) + 1e-6)
    
    # Hellinger distance from log-normal
    log_pos = np.log(pos + 1e-6)
    mu_ln = np.mean(log_pos); sigma_ln = np.std(log_pos)
    
    # KDE-based PDF
    hist, edges = np.histogram(u, bins=min(50, n//5), density=True)
    # Normal PDF
    x_mid = (edges[:-1] + edges[1:]) / 2
    normal_pdf = stats.norm.pdf(x_mid, np.mean(u), np.std(u) + 1e-6)
    lognormal_pdf = stats.lognorm.pdf(np.maximum(x_mid, 1e-6), sigma_ln, scale=np.exp(mu_ln))
    # Hellinger distance: sqrt(1 - sum(sqrt(p*q)))
    eps = 1e-10
    h_normal = np.sqrt(max(0, 1 - np.sum(np.sqrt(hist * normal_pdf + eps))))
    h_lognormal = np.sqrt(max(0, 1 - np.sum(np.sqrt(hist * lognormal_pdf + eps))))
    
    # Bimodality coefficient (Sarle's coefficient)
    n_valid = len(u)
    skewness = stats.skew(u) if n_valid > 2 else 0
    kurtosis = stats.kurtosis(u) if n_valid > 3 else 0
    bc = (skewness**2 + 1) / (kurtosis + 3 * (n_valid - 1)**2 / ((n_valid - 2) * (n_valid - 3)) + 1e-6)
    # bc > 0.555 indicates bimodal
    
    # Spread/centrality: IQR / median
    p25, p50, p75 = np.percentile(pos, [25, 50, 75])
    spread_ratio = (p75 - p25) / (p50 + 1e-6)
    
    # Proportion of mass below median
    mass_below_median = np.mean(u <= p50)
    
    return np.array([gini, h_normal, h_lognormal, bc, spread_ratio, mass_below_median])


# ======================================================================
# 6. BURSTINESS & INTER-EVENT PATTERNS
# ======================================================================
def burstiness_features(u):
    """Burstiness: how concentrated is consumption in time?
    
    Burstiness B = (σ/μ - 1) / (σ/μ + 1), where σ/μ is CV of inter-event times.
    Higher B = more bursty (consumption concentrated in fewer days).
    """
    n = len(u)
    if n < 10:
        return np.zeros(3)
    
    p50 = np.median(u)
    
    # Identify "events" as days above median
    above_median = u > p50
    # Inter-event intervals (days between above-median consumption)
    event_indices = np.where(above_median)[0]
    if len(event_indices) < 2:
        return np.zeros(3)
    
    inter_events = np.diff(event_indices)
    tau_mean = np.mean(inter_events)
    tau_std = np.std(inter_events)
    
    # Burstiness measure (Goh & Barabasi, 2008)
    cv_ie = tau_std / (tau_mean + 1e-6)
    B = (cv_ie - 1) / (cv_ie + 1) if cv_ie > 0 else -1.0
    
    # Memory coefficient: correlation between consecutive inter-event times
    if len(inter_events) > 2:
        M = np.corrcoef(inter_events[:-1], inter_events[1:])[0, 1]
        M = 0 if np.isnan(M) else M
    else:
        M = 0
    
    # Event concentration: top 10% of days account for what % of total consumption?
    total = np.sum(u)
    if total > 1e-6:
        top_indices = np.argsort(u)[-max(1, n//10):]
        concentration = np.sum(u[top_indices]) / total
    else:
        concentration = 0.1
    
    return np.array([B, M, concentration])


# ======================================================================
# 7. HURST EXPONENT (Detrended Fluctuation Analysis)
# ======================================================================
def hurst_dfa(u, scales=None):
    """Hurst exponent via Detrended Fluctuation Analysis.
    
    H ~ 0.5: random walk (white noise)
    H > 0.5: persistent (trends continue)
    H < 0.5: anti-persistent (mean-reverting)
    """
    n = len(u)
    if n < 100:
        return 0.5
    
    y = np.cumsum(u - np.mean(u))
    
    if scales is None:
        scales = np.unique(np.logspace(1, np.log10(n//4), 10).astype(int))
    
    flucts = []
    for s in scales:
        if s < 10 or s > n // 4:
            continue
        n_segments = n // s
        rms = 0
        for v in range(n_segments):
            seg = y[v*s:(v+1)*s]
            if len(seg) < 4:
                continue
            x = np.arange(len(seg))
            coeffs = np.polyfit(x, seg, 1)
            trend = np.polyval(coeffs, x)
            rms += np.mean((seg - trend) ** 2)
        if n_segments > 0:
            flucts.append(np.sqrt(rms / n_segments))
    
    if len(flucts) < 3:
        return 0.5
    
    valid_scales = [s for s in scales if s >= 10 and s <= n//4]
    if len(valid_scales) < 3:
        return 0.5
    log_scales = np.log(valid_scales[:len(flucts)])
    log_flucts = np.log(np.array(flucts) + 1e-10)
    H = np.polyfit(log_scales, log_flucts, 1)[0]
    return H


# ======================================================================
# 8. TIME-REVERSAL ASYMMETRY
# ======================================================================
def time_reversal_asymmetry(u, n_bins=10):
    """Time-reversal asymmetry: are transitions forward ≠ backward?
    
    For natural processes, forward and backward transition probabilities
    should be similar. Theft interventions create asymmetric transitions:
    consumption tends to drop suddenly but recover gradually.
    """
    n = len(u)
    if n < 20:
        return np.zeros(3)
    
    # Discretize into bins
    bins = np.percentile(u, np.linspace(0, 100, n_bins + 1))
    if np.unique(bins).shape[0] < 3:
        return np.zeros(3)
    discrete = np.digitize(u, bins[1:-1])
    
    # Forward transition matrix
    P_forward = np.zeros((n_bins, n_bins))
    for i in range(n - 1):
        P_forward[discrete[i], discrete[i+1]] += 1
    P_forward = P_forward / (P_forward.sum(axis=1, keepdims=True) + 1e-10)
    
    # Backward (time-reversed) transition matrix
    P_backward = np.zeros((n_bins, n_bins))
    for i in range(n - 1, 0, -1):
        P_backward[discrete[i], discrete[i-1]] += 1
    P_backward = P_backward / (P_backward.sum(axis=1, keepdims=True) + 1e-10)
    
    # Asymmetry metrics
    asymmetry = np.sum(np.abs(P_forward - P_backward)) / (2 * n_bins)
    
    # Drop asymmetry specifically (transitions to lower bins)
    dropped = np.sum(P_forward * np.triu(np.ones((n_bins, n_bins)), k=1))
    risen = np.sum(P_forward * np.tril(np.ones((n_bins, n_bins)), k=-1))
    drop_rise_ratio = dropped / (risen + 1e-6)
    
    # Max asymmetry in any bin
    bin_asym = np.max(np.abs(P_forward.sum(axis=1) - P_backward.sum(axis=1)))
    
    return np.array([asymmetry, drop_rise_ratio, bin_asym])


# ======================================================================
# MAIN: Compute all novel features
# ======================================================================
def compute_novel_features(filled, sample_size=5000):
    """Compute all novel features for all users.
    
    Due to computational cost (sample entropy is O(n²)), 
    use vectorized operations where possible.
    """
    n_users, n_days = filled.shape
    print(f"[Deep Engineering] Computing novel features for {n_users} users...")
    
    # Features that are fast to compute for all users
    print("  Distribution shape + extreme value (all users)...")
    dsf = np.zeros((n_users, 6))
    evf = np.zeros((n_users, 4))
    bf = np.zeros((n_users, 3))
    tra = np.zeros((n_users, 3))
    
    for i in range(n_users):
        row = filled[i]
        valid = row[row > 0] if np.any(row > 0) else row
        dsf[i] = distribution_shape_features(valid)
        xi, sigma, tr, er = extreme_value_features(valid)
        evf[i] = [xi, sigma, tr, er]
        bf[i] = burstiness_features(valid)
        tra[i] = time_reversal_asymmetry(valid)
    
    # Features that are slow: compute for sample and use as reference
    print(f"  Entropy + Hurst (sample {sample_size} users)...")
    sample_idx = np.random.choice(n_users, min(sample_size, n_users), replace=False)
    
    se = np.zeros(n_users)
    pe = np.zeros(n_users)
    lz = np.zeros(n_users)
    hurst = np.zeros(n_users)
    
    for idx in sample_idx:
        row = filled[idx]
        valid = row[row > 0] if np.any(row > 0) else row
        # Downsample for speed
        if len(valid) > 200:
            step = max(1, len(valid) // 200)
            valid = valid[::step]
        se[idx] = sample_entropy(valid)
        pe[idx] = permutation_entropy(valid, order=4)
        lz[idx] = lz_complexity(valid, n_bins=5)
        hurst[idx] = hurst_dfa(valid)
    
    # Fill remaining with mean
    se_mean = np.mean(se[se != 0]) if np.any(se != 0) else 0
    pe_mean = np.mean(pe[pe != 0]) if np.any(pe != 0) else 0
    lz_mean = np.mean(lz[lz != 0]) if np.any(lz != 0) else 0
    hurst_mean = np.mean(hurst[hurst != 0]) if np.any(hurst != 0) else 0.5
    
    mask = ~np.isin(np.arange(n_users), sample_idx)
    se[mask] = se_mean
    pe[mask] = pe_mean
    lz[mask] = lz_mean
    hurst[mask] = hurst_mean
    
    # Assemble
    X = np.column_stack([
        se, pe, lz,           # Entropy (3)
        dsf,                   # Distribution shape (6)
        evf,                   # Extreme value (4)
        bf,                    # Burstiness (3)
        tra,                   # Time-reversal asymmetry (3)
        hurst,                 # Hurst (1)
    ])
    
    feature_names = [
        'sample_entropy', 'perm_entropy', 'lz_complexity',
        'gini', 'h_normal', 'h_lognormal', 'bimodality_coef', 'spread_ratio', 'mass_below_med',
        'gpd_xi', 'gpd_sigma', 'tail_ratio', 'exc_ratio',
        'burstiness', 'burst_memory', 'concentration',
        't_rev_asym', 'drop_rise_ratio', 'bin_asym',
        'hurst_h',
    ]
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Total novel features: {len(feature_names)}")
    return X.astype(np.float32), feature_names


# ======================================================================
# AUC Evaluation per feature
# ======================================================================
def evaluate_features(features, feature_names, y):
    """Compute AUC for each novel feature."""
    print(f"\n{'Feature':<25s} {'AUC':>7s} {'Theft mean':>11s} {'Normal mean':>11s}")
    print("-" * 58)
    for i, name in enumerate(feature_names):
        vals = features[:, i]
        if np.std(vals) < 1e-10:
            auc = 0.5
        else:
            try:
                auc = roc_auc_score(y, vals)
                if auc < 0.5:
                    auc = roc_auc_score(y, -vals)  # Try flipped
            except:
                auc = 0.5
        t_mean = np.mean(vals[y == 1])
        n_mean = np.mean(vals[y == 0])
        marker = " *" if auc > 0.55 else ""
        print(f"{name:<25s} {auc:>6.4f} {t_mean:>10.4f} {n_mean:>10.4f}{marker}")


# ======================================================================
# RUN
# ======================================================================
if __name__ == '__main__':
    t0 = time.time()
    print("=" * 60)
    print("  Deep Data Engineering: Novel Feature Analysis")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv('data/raw_data.csv')
    dc = [c for c in df.columns if '/' in str(c) and len(str(c)) <= 10]
    raw = df[dc].values.astype(float); y = df['FLAG'].values.astype(np.int32)
    filled = np.nan_to_num(raw, nan=0)
    user_med = np.nan_to_num(np.nanmedian(raw, axis=1), nan=0)
    for i in range(len(raw)):
        miss = np.isnan(raw[i])
        if miss.any(): filled[i, miss] = user_med[i]
    print(f"  Shape: {filled.shape}, theft={y.sum()}")
    
    # Compute novel features
    novel_X, novel_names = compute_novel_features(filled, sample_size=5000)
    
    # Evaluate
    evaluate_features(novel_X, novel_names, y)
    
    print(f"\nTime: {(time.time()-t0)/60:.1f} min")
    
    np.savez('output/novel_features.npz', features=novel_X, 
             feature_names=np.array(novel_names, dtype=object), y=y)
