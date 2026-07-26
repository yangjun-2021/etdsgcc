"""
Advanced feature engineering for SGCC electricity theft detection.

Ported from D:/Project/ThiefElectricity/data_engine_v4.py (ElectricityTheftDataEngineV4).
Function-based interface for clean integration with preprocess_sgcc.py.

Feature groups:
  1. Basic statistics (mean/std/cv/median/mad/range/iqr + 21 percentiles + ratios)
  2. Extreme & threshold features (above/below ratios, top/bottom 5%)
  3. Temporal pattern features (diff stats, sign changes, peaks/valleys)
  4. Frequency domain features (FFT power at weekly/monthly/quarterly/semi-annual, spectral entropy)
  5. Trend & change features (linear regression slope, segment slopes, rolling mean change)
  6. Zero & missing pattern features (zero runs, missing segments)
  7. Sliding window features (last 7/14/30/60 days stats)
  8. Holiday & monthly features (spring festival, national day, monthly means)
  9. Deep behavior features (monthly CV, weekday/weekend ratio, CUSUM, autocorrelation)
 10. Low-consumption theft features (flatness, sudden changes, zero runs, monthly extremes)
 11. Consumption tier features (quintile one-hot encoding)

Cluster deviation features are SKIPPED here because they require per-fold KMeans fitting
to avoid data leakage. They can be added later via a per-fold wrapper in training scripts.

NOTE: This module does NOT do imputation or winsorization - it expects already-processed
data from preprocess_sgcc.py (imputed + winsorized + log1p transformed).
"""
import datetime
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings('ignore', 'Mean of empty slice')
warnings.filterwarnings('ignore', 'All-NaN slice')
warnings.filterwarnings('ignore', 'Degrees of freedom')
warnings.filterwarnings('ignore', 'invalid value')
warnings.filterwarnings('ignore', 'divide by zero')
warnings.filterwarnings('ignore', 'Precision loss')


def parse_date_columns(date_cols):
    """Parse SGCC date columns (format M/D/YYYY) into date metadata.

    Returns:
        dates: np.array of (year, month, day) tuples
        seasonal_indices: np.array of month numbers (1-12)
        weekday_indices: np.array of weekday numbers (0=Monday)
    """
    dates = []
    seasonal_indices = []
    weekday_indices = []
    for col in date_cols:
        parts = str(col).split('/')
        if len(parts) == 3:
            month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
            dates.append((year, month, day))
            seasonal_indices.append(month)
            try:
                dt = datetime.date(year, month, day)
                weekday_indices.append(dt.weekday())
            except Exception:
                weekday_indices.append(0)
        else:
            dates.append((2000, 1, 1))
            seasonal_indices.append(6)
            weekday_indices.append(0)
    return (np.array(dates),
            np.array(seasonal_indices),
            np.array(weekday_indices))


def _max_consecutive_zeros(arr):
    """Max run of zero values in a 1D array."""
    if not np.any(arr == 0):
        return 0
    binary = (arr == 0).astype(int)
    diff = np.diff(np.concatenate([[0], binary, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) == 0:
        return 0
    return int(np.max(ends - starts))


def _find_continuous_missing(row):
    """Identify continuous missing segments [start, end) in a 1D array."""
    missing = np.isnan(row)
    if not missing.any():
        return []
    diff = np.diff(np.concatenate([[False], missing, [False]]).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def _get_zero_run_lengths(row):
    """Lengths of all consecutive zero-value runs in a 1D array."""
    arr = (row == 0).astype(int)
    diff = np.diff(np.concatenate([[0], arr, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [int(ends[i] - starts[i]) for i in range(len(starts))]


def _safe_percentile(arr, q):
    if len(arr) == 0:
        return 0.0
    return np.percentile(arr, q)


def extract_advanced_features(transformed, winsorized, raw, impute_mask,
                              date_cols, flags=None, verbose=True, ae_epochs=300):
    """Extract advanced features for SGCC electricity theft detection.

    Args:
        transformed: [N, T] log1p-transformed winsorized data (from preprocess_sgcc)
        winsorized: [N, T] winsorized (pre-log) data
        raw: [N, T] original raw data with NaN for missing
        impute_mask: [N, T] bool, True where raw was missing
        date_cols: list of date column name strings (M/D/YYYY)
        flags: [N] labels (optional, used only for consumption tier calc)
        verbose: print progress

    Returns:
        feature_matrix: [N, F] float32
        feature_names: list of str
    """
    n_samples, n_days = transformed.shape
    dates, seasonal_indices, weekday_indices = parse_date_columns(date_cols)

    features = {}

    if verbose:
        print("[Advanced] 1) Basic statistics...")
    for i in range(n_samples):
        row = transformed[i]
        valid = row[~np.isnan(row)] if np.isnan(row).any() else row
        if len(valid) == 0:
            valid = np.array([0.0])
        f = {}
        f['adv_mean'] = np.mean(valid)
        f['adv_std'] = np.std(valid)
        f['adv_var'] = np.var(valid)
        f['adv_cv'] = np.clip(f['adv_std'] / (f['adv_mean'] + 1e-8), 0, 20)
        f['adv_median'] = np.median(valid)
        f['adv_mad'] = np.mean(np.abs(valid - f['adv_median']))
        f['adv_range'] = np.max(valid) - np.min(valid)
        f['adv_iqr'] = np.percentile(valid, 75) - np.percentile(valid, 25)
        for p in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                  50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]:
            f[f'adv_q{p}'] = np.percentile(valid, p)
        f['adv_q95_q05'] = f['adv_q95'] / (f['adv_q5'] + 1e-8)
        f['adv_q90_q10'] = f['adv_q90'] / (f['adv_q10'] + 1e-8)
        f['adv_q50_q25'] = f['adv_q50'] - f['adv_q25']
        f['adv_q75_q50'] = f['adv_q75'] - f['adv_q50']
        f['adv_skew'] = stats.skew(valid) if len(valid) > 2 else 0
        f['adv_kurt'] = stats.kurtosis(valid) if len(valid) > 3 else 0
        f['adv_max'] = np.max(valid)
        f['adv_min'] = np.min(valid)
        f['adv_max_mean_ratio'] = f['adv_max'] / (f['adv_mean'] + 1e-8)
        f['adv_min_mean_ratio'] = f['adv_min'] / (f['adv_mean'] + 1e-8)
        f['adv_max_min_ratio'] = f['adv_max'] / (f['adv_min'] + 1e-8)
        features[i] = f

    if verbose:
        print("[Advanced] 2) Extreme & threshold features...")
    for i in range(n_samples):
        row = transformed[i]
        valid = row[~np.isnan(row)] if np.isnan(row).any() else row
        f = features[i]
        for threshold in [10, 20, 30, 50, 100]:
            f[f'adv_above_{threshold}_ratio'] = np.mean(valid > threshold)
        for threshold in [5, 10, 20]:
            f[f'adv_below_{threshold}_ratio'] = np.mean(valid < threshold)
        p5, p95 = np.percentile(valid, [5, 95])
        f['adv_top5_pct_mean'] = np.mean(valid[valid >= p95]) if np.sum(valid >= p95) > 0 else 0
        f['adv_bottom5_pct_mean'] = np.mean(valid[valid <= p5]) if np.sum(valid <= p5) > 0 else 0

    if verbose:
        print("[Advanced] 3) Temporal pattern features...")
    for i in range(n_samples):
        row = transformed[i]
        valid = row[~np.isnan(row)] if np.isnan(row).any() else row
        f = features[i]
        if len(valid) > 1:
            diff = np.diff(valid)
            f['adv_diff_mean'] = np.mean(diff)
            f['adv_diff_std'] = np.std(diff)
            f['adv_diff_abs_mean'] = np.mean(np.abs(diff))
            f['adv_diff_abs_max'] = np.max(np.abs(diff))
            f['adv_diff_q75'] = np.percentile(np.abs(diff), 75)
            f['adv_diff_q90'] = np.percentile(np.abs(diff), 90)
            sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
            f['adv_sign_change_ratio'] = sign_changes / (len(diff) + 1e-8)
            pos_ratio = np.mean(diff > 0)
            f['adv_positive_diff_ratio'] = pos_ratio
            f['adv_negative_diff_ratio'] = 1 - pos_ratio
            peaks, _ = find_peaks(valid, distance=7)
            f['adv_n_peaks'] = len(peaks)
            f['adv_peak_ratio'] = len(peaks) / (len(valid) + 1e-8)
            valleys, _ = find_peaks(-valid, distance=7)
            f['adv_n_valleys'] = len(valleys)
        else:
            for k in ['adv_diff_mean', 'adv_diff_std', 'adv_diff_abs_mean',
                      'adv_diff_abs_max', 'adv_diff_q75', 'adv_diff_q90',
                      'adv_sign_change_ratio', 'adv_positive_diff_ratio',
                      'adv_negative_diff_ratio', 'adv_n_peaks', 'adv_peak_ratio',
                      'adv_n_valleys']:
                f[k] = 0

    if verbose:
        print("[Advanced] 4) Frequency domain features...")
    for i in range(n_samples):
        row = transformed[i]
        valid = row[~np.isnan(row)] if np.isnan(row).any() else row
        f = features[i]
        if len(valid) > 30:
            fft_vals = fft(valid - np.mean(valid))
            power = np.abs(fft_vals) ** 2
            freqs = fftfreq(len(valid))
            pos_mask = freqs > 0
            total_pos_power = np.sum(power[pos_mask]) + 1e-8
            for period, name in {7: 'weekly', 30: 'monthly',
                                  90: 'quarterly', 180: 'semi_annual'}.items():
                idx = min(int(round(len(valid) / period)), len(power) - 1)
                if idx > 0:
                    f[f'adv_power_{name}'] = power[idx] / total_pos_power
                else:
                    f[f'adv_power_{name}'] = 0
            power_norm = power[pos_mask] / total_pos_power
            f['adv_spectral_entropy'] = -np.sum(power_norm * np.log(power_norm + 1e-8))
            peak_idx = np.argmax(power[pos_mask])
            f['adv_dominant_freq'] = freqs[pos_mask][peak_idx] if len(freqs[pos_mask]) > 0 else 0
            f['adv_dominant_period'] = (1 / (f['adv_dominant_freq'] + 1e-8)
                                         if f['adv_dominant_freq'] > 0 else 0)
            low_power = np.sum(power[(freqs > 0) & (freqs < 0.01)])
            high_power = np.sum(power[(freqs >= 0.01) & (freqs < 0.5)])
            f['adv_low_high_freq_ratio'] = low_power / (high_power + 1e-8)
        else:
            for name in ['weekly', 'monthly', 'quarterly', 'semi_annual']:
                f[f'adv_power_{name}'] = 0
            for k in ['adv_spectral_entropy', 'adv_dominant_freq',
                      'adv_dominant_period', 'adv_low_high_freq_ratio']:
                f[k] = 0

    if verbose:
        print("[Advanced] 5) Trend & change features...")
    for i in range(n_samples):
        row = winsorized[i]
        valid = row[~np.isnan(row)] if np.isnan(row).any() else row
        f = features[i]
        if len(valid) > 30:
            x = np.arange(len(valid))
            slope, _, r_value, _, std_err = stats.linregress(x, valid)
            f['adv_trend_slope'] = slope
            f['adv_trend_r2'] = r_value ** 2
            f['adv_trend_std_err'] = std_err
            n = len(valid)
            segments = [valid[:n // 3], valid[n // 3:2 * n // 3], valid[2 * n // 3:]]
            seg_slopes = []
            for seg in segments:
                if len(seg) > 5:
                    s, _, _, _, _ = stats.linregress(np.arange(len(seg)), seg)
                    seg_slopes.append(s)
            if len(seg_slopes) == 3:
                f['adv_trend_slope_first'] = seg_slopes[0]
                f['adv_trend_slope_mid'] = seg_slopes[1]
                f['adv_trend_slope_last'] = seg_slopes[2]
                f['adv_trend_acceleration'] = seg_slopes[2] - seg_slopes[0]
            else:
                for k in ['adv_trend_slope_first', 'adv_trend_slope_mid',
                          'adv_trend_slope_last', 'adv_trend_acceleration']:
                    f[k] = 0
            f['adv_rolling_mean_last'] = np.mean(valid[-7:])
            f['adv_rolling_mean_first'] = np.mean(valid[:7])
            f['adv_rolling_mean_change'] = f['adv_rolling_mean_last'] - f['adv_rolling_mean_first']
        else:
            for k in ['adv_trend_slope', 'adv_trend_r2', 'adv_trend_std_err',
                      'adv_trend_slope_first', 'adv_trend_slope_mid',
                      'adv_trend_slope_last', 'adv_trend_acceleration',
                      'adv_rolling_mean_last', 'adv_rolling_mean_first',
                      'adv_rolling_mean_change']:
                f[k] = 0

    if verbose:
        print("[Advanced] 6) Zero & missing pattern features...")
    for i in range(n_samples):
        winsor_row = winsorized[i]
        raw_row = raw[i]
        f = features[i]
        zero_mask = (winsor_row == 0)
        f['adv_zero_ratio'] = zero_mask.mean()
        f['adv_max_consecutive_zeros'] = _max_consecutive_zeros(winsor_row)
        below_1 = (winsor_row < 1)
        f['adv_below_1_ratio'] = below_1.mean()
        raw_missing = np.isnan(raw_row)
        f['adv_original_missing_ratio'] = raw_missing.mean()
        missing_segs = _find_continuous_missing(raw_row)
        f['adv_n_missing_segments'] = len(missing_segs)
        f['adv_max_missing_segment'] = max((e - s for s, e in missing_segs), default=0)

    if verbose:
        print("[Advanced] 7) Sliding window features...")
    for window in [7, 14, 30, 60]:
        for i in range(n_samples):
            row = transformed[i]
            window_data = row[-window:] if window <= len(row) else row
            valid = window_data[~np.isnan(window_data)] if np.isnan(window_data).any() else window_data
            f = features[i]
            if len(valid) > 0:
                f[f'adv_window_{window}_mean'] = np.mean(valid)
                f[f'adv_window_{window}_std'] = np.std(valid) if len(valid) > 1 else 0
                f[f'adv_window_{window}_max'] = np.max(valid)
                f[f'adv_window_{window}_min'] = np.min(valid)
                f[f'adv_window_{window}_range'] = np.max(valid) - np.min(valid)
            else:
                for suffix in ['mean', 'std', 'max', 'min', 'range']:
                    f[f'adv_window_{window}_{suffix}'] = 0

    if verbose:
        print("[Advanced] 8) Holiday & monthly features...")
    spring_indices = []
    national_indices = []
    all_holiday_indices = []
    for idx, (year, month, day) in enumerate(dates):
        if (month == 1 and day >= 28) or (month == 2 and day <= 15):
            spring_indices.append(idx)
            all_holiday_indices.append(idx)
        if month == 10 and 1 <= day <= 7:
            national_indices.append(idx)
            all_holiday_indices.append(idx)
    for i in range(n_samples):
        raw_row = raw[i]
        row_valid = np.where(np.isnan(raw_row), 0, raw_row)
        f = features[i]
        if len(spring_indices) > 0:
            spring_vals = row_valid[spring_indices]
            spring_valid = spring_vals[spring_vals > 0]
            f['adv_spring_mean'] = np.mean(spring_valid) if len(spring_valid) > 0 else 0
            f['adv_spring_zero_ratio'] = np.mean(spring_vals == 0)
            pre_spring_idx = [idx for idx in range(len(dates))
                              if dates[idx][1] == 1 and dates[idx][2] <= 25]
            if len(pre_spring_idx) > 0:
                pre_vals = row_valid[pre_spring_idx]
                pre_mean = np.mean(pre_vals[pre_vals > 0]) if np.sum(pre_vals > 0) > 0 else 1
                f['adv_spring_ratio'] = f['adv_spring_mean'] / (pre_mean + 1e-8)
            else:
                f['adv_spring_ratio'] = 1.0
        else:
            f['adv_spring_mean'] = f['adv_spring_zero_ratio'] = f['adv_spring_ratio'] = 0
        if len(national_indices) > 0:
            nat_vals = row_valid[national_indices]
            nat_valid = nat_vals[nat_vals > 0]
            f['adv_national_mean'] = np.mean(nat_valid) if len(nat_valid) > 0 else 0
            f['adv_national_zero_ratio'] = np.mean(nat_vals == 0)
        else:
            f['adv_national_mean'] = f['adv_national_zero_ratio'] = 0
        if len(all_holiday_indices) > 0:
            hol_vals = row_valid[all_holiday_indices]
            f['adv_holiday_zero_ratio'] = np.mean(hol_vals == 0)
            f['adv_holiday_mean'] = np.mean(hol_vals[hol_vals > 0]) if np.sum(hol_vals > 0) > 0 else 0
        else:
            f['adv_holiday_zero_ratio'] = f['adv_holiday_mean'] = 0
        for m in range(1, 13):
            month_idx = np.where(seasonal_indices == m)[0]
            if len(month_idx) > 0:
                month_vals = raw_row[month_idx]
                month_valid = month_vals[~np.isnan(month_vals)]
                f[f'adv_month_{m}_mean'] = np.mean(month_valid) if len(month_valid) > 0 else 0
            else:
                f[f'adv_month_{m}_mean'] = 0
        monthly_means = [f[f'adv_month_{m}_mean'] for m in range(1, 13)]
        overall_mean = np.mean([x for x in monthly_means if x > 0]) if any(x > 0 for x in monthly_means) else 1
        for m in range(1, 13):
            f[f'adv_month_{m}_ratio'] = f[f'adv_month_{m}_mean'] / (overall_mean + 1e-8)

    if verbose:
        print("[Advanced] 9) Deep behavior features...")
    for i in range(n_samples):
        row = transformed[i]
        raw_row = raw[i]
        valid = row[~np.isnan(row)] if np.isnan(row).any() else row
        f = features[i]
        if len(valid) > 10:
            monthly_cvs = []
            for month in range(1, 13):
                month_idx = np.where(seasonal_indices == month)[0]
                if len(month_idx) > 0:
                    m_vals = row[month_idx]
                    m_valid = m_vals[~np.isnan(m_vals)]
                    if len(m_valid) > 1:
                        cv = np.std(m_valid) / (np.mean(m_valid) + 1e-8)
                        monthly_cvs.append(cv)
            f['adv_monthly_cv_mean'] = np.mean(monthly_cvs) if monthly_cvs else 0
            f['adv_monthly_cv_std'] = np.std(monthly_cvs) if len(monthly_cvs) > 1 else 0
            weekday_vals, weekend_vals = [], []
            for idx, val in enumerate(row):
                if idx >= len(weekday_indices):
                    continue
                wd = weekday_indices[idx]
                if not np.isnan(val):
                    if wd < 5:
                        weekday_vals.append(val)
                    else:
                        weekend_vals.append(val)
            if weekday_vals and weekend_vals:
                f['adv_weekday_weekend_ratio'] = np.mean(weekday_vals) / (np.mean(weekend_vals) + 1e-8)
            else:
                f['adv_weekday_weekend_ratio'] = 1.0
            target = np.mean(valid)
            cusum_pos = np.maximum(0, valid - target)
            cusum_neg = np.maximum(0, target - valid)
            f['adv_cusum_max'] = max(np.max(cusum_pos), np.max(cusum_neg))
            f['adv_cusum_mean'] = np.mean(cusum_pos + cusum_neg)
        else:
            f['adv_monthly_cv_mean'] = f['adv_monthly_cv_std'] = 0
            f['adv_weekday_weekend_ratio'] = 1.0
            f['adv_cusum_max'] = f['adv_cusum_mean'] = 0
        if len(valid) > 20:
            try:
                ac_lag1 = np.corrcoef(valid[:-1], valid[1:])[0, 1]
                f['adv_autocorr_lag1'] = ac_lag1 if not np.isnan(ac_lag1) else 0
            except Exception:
                f['adv_autocorr_lag1'] = 0
            if len(valid) > 7:
                ac_lag7 = np.corrcoef(valid[:-7], valid[7:])[0, 1]
                f['adv_autocorr_lag7'] = ac_lag7 if not np.isnan(ac_lag7) else 0
            else:
                f['adv_autocorr_lag7'] = 0
            if len(valid) > 30:
                ac_lag30 = np.corrcoef(valid[:-30], valid[30:])[0, 1]
                f['adv_autocorr_lag30'] = ac_lag30 if not np.isnan(ac_lag30) else 0
            else:
                f['adv_autocorr_lag30'] = 0
        else:
            f['adv_autocorr_lag1'] = f['adv_autocorr_lag7'] = f['adv_autocorr_lag30'] = 0

    if verbose:
        print("[Advanced] 10) Low-consumption theft features...")
    mean_consumptions = np.array([
        np.mean(transformed[i][~np.isnan(transformed[i])])
        if np.isnan(transformed[i]).any() and len(transformed[i][~np.isnan(transformed[i])]) > 0
        else (np.mean(transformed[i]) if len(transformed[i]) > 0 else 0)
        for i in range(n_samples)
    ])
    valid_means = mean_consumptions[mean_consumptions > 0]
    tier_bounds = (np.percentile(valid_means, [20, 40, 60, 80])
                   if len(valid_means) > 10 else np.array([0, 0, 0, 0]))
    for i in range(n_samples):
        row = transformed[i]
        raw_row = raw[i]
        valid = row[~np.isnan(row)] if np.isnan(row).any() else row
        f = features[i]
        if len(valid) > 10:
            diffs = np.diff(valid)
            f['adv_consumption_flatness'] = 1.0 / (np.std(diffs) + 1e-8)
            diff_mean = np.mean(diffs)
            diff_std = np.std(diffs) + 1e-8
            f['adv_sudden_change_ratio'] = np.mean(np.abs(diffs - diff_mean) > 3 * diff_std)
            low_vals = np.sum(valid < np.percentile(valid, 10))
            f['adv_low_consumption_days_ratio'] = low_vals / (len(valid) + 1e-8)
            if 'adv_spectral_entropy' in f:
                f['adv_too_regular'] = 1.0 / (f['adv_spectral_entropy'] + 1e-8)
            else:
                f['adv_too_regular'] = 0
            zero_runs = _get_zero_run_lengths(raw_row)
            f['adv_max_zero_run'] = max(zero_runs) if zero_runs else 0
            f['adv_n_zero_runs'] = len(zero_runs)
            monthly_means = []
            for month in range(1, 13):
                m_idx = np.where(seasonal_indices == month)[0]
                if len(m_idx) > 0:
                    m_vals = raw_row[m_idx]
                    m_valid = m_vals[~np.isnan(m_vals)]
                    monthly_means.append(np.mean(m_valid) if len(m_valid) > 0 else np.nan)
            monthly_means = np.array([v for v in monthly_means if not np.isnan(v)])
            if len(monthly_means) > 3:
                f['adv_monthly_max_min_ratio'] = np.max(monthly_means) / (np.min(monthly_means) + 1e-8)
                f['adv_monthly_std_norm'] = np.std(monthly_means) / (np.mean(monthly_means) + 1e-8)
            else:
                f['adv_monthly_max_min_ratio'] = 1.0
                f['adv_monthly_std_norm'] = 0
        else:
            for k in ['adv_consumption_flatness', 'adv_sudden_change_ratio',
                      'adv_low_consumption_days_ratio', 'adv_too_regular',
                      'adv_max_zero_run', 'adv_n_zero_runs',
                      'adv_monthly_max_min_ratio', 'adv_monthly_std_norm']:
                f[k] = 0

    if verbose:
        print("[Advanced] 11) Consumption tier features...")
    for i in range(n_samples):
        m = mean_consumptions[i]
        f = features[i]
        if m <= 0:
            tier = 0
        elif len(tier_bounds) > 0 and m <= tier_bounds[0]:
            tier = 1
        elif len(tier_bounds) > 1 and m <= tier_bounds[1]:
            tier = 2
        elif len(tier_bounds) > 2 and m <= tier_bounds[2]:
            tier = 3
        elif len(tier_bounds) > 3 and m <= tier_bounds[3]:
            tier = 4
        else:
            tier = 5
        f['adv_consumption_tier'] = tier
        for t in range(1, 6):
            f[f'adv_tier_{t}'] = 1 if tier == t else 0

    features_df = pd.DataFrame.from_dict(features, orient='index')

    const_cols = [c for c in features_df.columns if features_df[c].std() < 1e-10]
    if const_cols:
        if verbose:
            print(f"[Advanced] Removing {len(const_cols)} constant features")
        features_df.drop(columns=const_cols, inplace=True)

    if verbose:
        print("[Advanced] 12) YoY drift features...")
    try:
        from src.features.yoy_features import compute_yoy_drift_features, compute_multiscale_regularity
        dates_info = [(int(d[0]), int(d[1]), int(d[2])) for d in dates]
        yoy_feats = compute_yoy_drift_features(raw, dates_info)
        msr_feats = compute_multiscale_regularity(np.nan_to_num(transformed, nan=0.0))
        for j in range(yoy_feats.shape[1]):
            features_df[f'adv_yoy_{j}'] = yoy_feats[:, j]
        for j in range(msr_feats.shape[1]):
            features_df[f'adv_msr_{j}'] = msr_feats[:, j]
        if verbose:
            print(f"  YoY: {yoy_feats.shape[1]} + MSR: {msr_feats.shape[1]} features")
    except Exception as e:
        if verbose:
            print(f"  YoY features skipped: {e}")

    if verbose:
        print("[Advanced] 13) TopConf features (FilterNet + TimeKAN + PatchTST)...")
    try:
        from src.features.topconf_features import (compute_filternet_features,
                                       compute_timekan_features,
                                       compute_patchtst_features)
        interp = np.nan_to_num(transformed, nan=0.0)
        fn_feats = compute_filternet_features(interp)
        tk_feats = compute_timekan_features(interp)
        pt_feats = compute_patchtst_features(interp)
        for j in range(fn_feats.shape[1]):
            features_df[f'adv_fn_{j}'] = fn_feats[:, j]
        for j in range(tk_feats.shape[1]):
            features_df[f'adv_tk_{j}'] = tk_feats[:, j]
        for j in range(pt_feats.shape[1]):
            features_df[f'adv_pt_{j}'] = pt_feats[:, j]
        if verbose:
            print(f"  FilterNet: {fn_feats.shape[1]} + TimeKAN: {tk_feats.shape[1]} + PatchTST: {pt_feats.shape[1]}")
    except Exception as e:
        if verbose:
            print(f"  TopConf features skipped: {e}")

    if verbose:
        print("[Advanced] 14) Deep features (peer_deviation + SVD + symbolic)...")
    try:
        from src.features.deep_features import (compute_peer_deviation_features,
                                    compute_svd_residual_features,
                                    compute_symbolic_features)
        seasonal_indices = np.array([int(d[1]) for d in dates])
        interp = np.nan_to_num(transformed, nan=0.0)
        peer_feats = compute_peer_deviation_features(interp, seasonal_indices)
        svd_feats = compute_svd_residual_features(interp, n_components=20)
        sym_feats = compute_symbolic_features(interp)
        for j in range(peer_feats.shape[1]):
            features_df[f'adv_peer_{j}'] = peer_feats[:, j]
        for j in range(svd_feats.shape[1]):
            features_df[f'adv_svd_{j}'] = svd_feats[:, j]
        for j in range(sym_feats.shape[1]):
            features_df[f'adv_sym_{j}'] = sym_feats[:, j]
        if verbose:
            print(f"  Peer: {peer_feats.shape[1]} + SVD: {svd_feats.shape[1]} + Symbolic: {sym_feats.shape[1]}")
    except Exception as e:
        if verbose:
            print(f"  Deep features skipped: {e}")

    if verbose:
        print("[Advanced] 15) Autoencoder anomaly features...")
    try:
        import torch
        from src.features.autoencoder_features import compute_autoencoder_features
        ae_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ae_feats = compute_autoencoder_features(
            np.nan_to_num(transformed, nan=0.0),
            flags if flags is not None else np.zeros(n_samples, dtype=int),
            [f'{int(d[1])}/{int(d[2])}/{int(d[0])}' for d in dates],
            epochs=ae_epochs, batch_size=256, device=ae_device, seed=42, verbose=verbose,
        )
        for j in range(ae_feats.shape[1]):
            features_df[f'adv_ae_{j}'] = ae_feats[:, j]
        if verbose:
            print(f"  AE: {ae_feats.shape[1]} features")
    except Exception as e:
        if verbose:
            print(f"  Autoencoder features skipped: {e}")

    feature_matrix = features_df.values.astype(np.float32)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    feature_matrix = np.clip(feature_matrix, -1e6, 1e6)
    feature_names = list(features_df.columns)

    if verbose:
        print(f"[Advanced] Total features: {len(feature_names)}")
    return feature_matrix, feature_names


def remove_high_corr_features(feature_matrix, feature_names, threshold=0.95, verbose=True):
    """Remove highly correlated features (keep the one with higher variance)."""
    if feature_matrix.shape[1] < 2:
        return feature_matrix, feature_names
    corr_matrix = np.corrcoef(feature_matrix.T)
    n_feat = corr_matrix.shape[0]
    to_remove = set()
    for i in range(n_feat):
        if i in to_remove:
            continue
        for j in range(i + 1, n_feat):
            if j in to_remove:
                continue
            if abs(corr_matrix[i, j]) > threshold:
                var_i = np.var(feature_matrix[:, i])
                var_j = np.var(feature_matrix[:, j])
                if var_i < var_j:
                    to_remove.add(i)
                else:
                    to_remove.add(j)
    if to_remove:
        keep_idx = [i for i in range(n_feat) if i not in to_remove]
        feature_matrix = feature_matrix[:, keep_idx]
        feature_names = [feature_names[i] for i in keep_idx]
        if verbose:
            print(f"[Advanced] Removed {len(to_remove)} highly-correlated features, "
                  f"remaining: {len(feature_names)}")
    return feature_matrix, feature_names


if __name__ == '__main__':
    import os
    import pandas as pd
    from config import SGCC_RAW_PATH

    print("Testing advanced_features on SGCC raw data...")
    df = pd.read_csv(SGCC_RAW_PATH)
    date_cols = [c for c in df.columns if '/' in str(c) and len(str(c)) <= 10]
    raw = df[date_cols].values.astype(float)
    flags = df['FLAG'].values.astype(int)

    imputed = np.nan_to_num(raw, nan=0.0)
    upper = np.percentile(imputed, 99)
    lower = np.percentile(imputed, 1)
    winsorized = np.clip(imputed, lower, upper)
    transformed = np.log1p(np.maximum(winsorized, 0))
    impute_mask = np.isnan(raw)

    feat_mat, feat_names = extract_advanced_features(
        transformed, winsorized, raw, impute_mask, date_cols, flags, verbose=True
    )
    print(f"\nFeature matrix: {feat_mat.shape}")
    print(f"Sample names: {feat_names[:10]}")
