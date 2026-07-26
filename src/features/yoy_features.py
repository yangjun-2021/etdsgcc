"""
Year-over-Year Temporal Drift Features
=======================================
Innovation: The SGCC dataset spans exactly 3 years (2014-2016).
For each month, compare consumption across years.

Key insight:
  - Normal users: Year N month M ≈ Year N+1 month M (stable seasonal pattern)
  - Theft users: Pattern CHANGES between years (theft starts/evolves)
  - The "drift score" captures how much the user's behavior changes over time
"""
import numpy as np


def compute_yoy_drift_features(X_raw, dates_info, feat_dim=36):
    """
    Compute Year-over-Year temporal drift features.
    
    For each user and each month (1-12):
      1. Extract consumption in that month for each year
      2. Compute per-month statistics across years
      3. Quantify year-to-year changes
    
    Features (36 dims = 12 months × 3):
      - yoy_cv: CV of monthly consumption across years (higher = less stable)
      - yoy_trend: Linear trend of monthly consumption across years (>0 = increasing, <0 = decreasing)
      - yoy_drop: Max drop between any two consecutive years (captures sudden theft start)
    """
    n, T = X_raw.shape
    feats = np.zeros((n, feat_dim), dtype=np.float32)
    col = 0
    
    # Parse dates to get year and month info
    # dates_info format: list of (year, month, day) tuples
    years = np.array([d[0] for d in dates_info])  # (T,)
    months = np.array([d[1] for d in dates_info])  # (T,)
    
    for m in range(1, 13):
        # Find all days in month m across all years
        m_mask = months == m
        if m_mask.sum() < 10:
            col += 3
            continue
        
        m_days = np.where(m_mask)[0]
        
        for i in range(n):
            vals = X_raw[i, m_days]
            # Remove NaN
            valid_vals = vals[~np.isnan(vals)]
            
            if len(valid_vals) < 10:
                continue
            
            # Group by year
            year_vals = {}
            for d_idx, global_d in enumerate(m_days):
                yr = years[global_d]
                if not np.isnan(vals[d_idx]):
                    if yr not in year_vals:
                        year_vals[yr] = []
                    year_vals[yr].append(vals[d_idx])
            
            # Compute per-year means
            year_means = {}
            for yr, yv in year_vals.items():
                if len(yv) > 0:
                    year_means[yr] = np.mean(yv)
            
            if len(year_means) >= 2:
                yr_list = sorted(year_means.keys())
                yr_vals_list = [year_means[yr] for yr in yr_list]
                
                # 1. CV across years (stability metric)
                mean_across_years = np.mean(yr_vals_list)
                if mean_across_years > 0.1:
                    cv = np.std(yr_vals_list) / mean_across_years
                    feats[i, col] = cv
                
                # 2. Linear trend across years
                if len(yr_list) >= 2:
                    x = np.arange(len(yr_list))
                    y = np.array(yr_vals_list)
                    if len(x) > 1 and np.std(y) > 0:
                        slope = np.polyfit(x, y, 1)[0]
                        feats[i, col + 1] = slope / (mean_across_years + 1e-6)
                
                # 3. Max drop between consecutive years
                if len(yr_vals_list) >= 2:
                    max_drop = 0
                    for j in range(1, len(yr_vals_list)):
                        drop = (yr_vals_list[j-1] - yr_vals_list[j]) / (yr_vals_list[j-1] + 1e-6)
                        max_drop = max(max_drop, drop)
                    feats[i, col + 2] = max_drop
        
        col += 3
    
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats[:, :col]


def compute_multiscale_regularity(X_interp, feat_dim=10):
    """
    Multi-Scale Regularity Score.
    
    For each user, compute entropy/predictability at multiple time scales.
    A user that's too regular across ALL scales is suspicious.
    
    Features:
      - Entropy at scales [1, 3, 7, 14, 30, 60, 90, 180, 365] days
      - Multi-scale regularity score (product of entropies)
      - Delta between scales (how much does entropy change with scale)
    """
    n, T = X_interp.shape
    feats = np.zeros((n, feat_dim), dtype=np.float32)
    col = 0
    
    # 1. Daily-level regularity: std(diff)
    diffs = np.diff(X_interp, axis=1)
    feats[:, col] = np.std(diffs, axis=1) / (np.std(X_interp, axis=1) + 1e-6)
    col += 1
    
    # 2. Zero-crossing rate (how often consumption changes direction)
    sign_changes = np.diff(np.sign(diffs), axis=1) != 0
    feats[:, col] = sign_changes.sum(axis=1) / T
    col += 1
    
    # 3-5. Entropy at different quantile levels
    for n_bins in [5, 10, 20]:
        entropies = np.zeros(n)
        for i in range(n):
            ts = X_interp[i]
            non_zero = ts[ts > 0]
            if len(non_zero) < 10:
                continue
            hist, _ = np.histogram(non_zero, bins=n_bins)
            hist = hist / (hist.sum() + 1e-10)
            entropies[i] = -np.sum(hist * np.log(hist + 1e-10))
        feats[:, col] = entropies / np.log(n_bins)  # Normalize
        col += 1
    
    # 6. Auto-correlation at lag-1 (predictability)
    for i in range(n):
        ts = X_interp[i]
        if np.std(ts) > 1e-6:
            acf = np.corrcoef(ts[:-1], ts[1:])[0, 1]
            feats[i, col] = 0 if np.isnan(acf) else acf
    col += 1
    
    # 7. Ratio of weekday variance to overall variance
    # Already covered by holiday features
    
    # 8. Spectral flatness (Wiener entropy of the spectrum)
    from scipy.fft import rfft
    for i in range(n):
        ts = X_interp[i] - X_interp[i].mean()
        if np.std(ts) > 1e-6:
            fv = np.abs(rfft(ts))
            fv = fv[1:]  # Remove DC
            if len(fv) > 0:
                log_spec = np.log(fv + 1e-10)
                feats[i, col] = np.exp(log_spec.mean()) / (fv.mean() + 1e-10)
    col += 1
    
    # 9. Run-length encoding entropy
    for i in range(n):
        ts = X_interp[i]
        if np.std(ts) < 1e-6:
            continue
        # Binarize: above/below median
        above_median = ts > np.median(ts)
        runs = []
        cur_run = 1
        for j in range(1, T):
            if above_median[j] == above_median[j-1]:
                cur_run += 1
            else:
                runs.append(cur_run)
                cur_run = 1
        runs.append(cur_run)
        if runs:
            # Entropy of run lengths
            run_hist, _ = np.histogram(runs, bins=min(10, len(set(runs))))
            run_hist = run_hist / (run_hist.sum() + 1e-10)
            feats[i, col] = -np.sum(run_hist * np.log(run_hist + 1e-10))
    col += 1
    
    # 10. Longest run of above-median consumption
    above = X_interp > np.median(X_interp, axis=1, keepdims=True)
    for i in range(n):
        max_run = 0
        cur_run = 0
        for j in range(T):
            if above[i, j]:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 0
        feats[i, col] = max_run / T
    col += 1
    
    feats = np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=0.0)
    return feats[:, :col]
