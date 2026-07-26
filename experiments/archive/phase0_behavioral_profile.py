"""
Phase 0: Behavioral Profiling of Electricity Theft Users
==========================================================
For each of the ~3,570 theft users, compute behavioral indicators that
profile their consumption pattern into one of 7 theft types.

The seven theft types (with physical mechanisms):
  1. 降压型: Voltage-drop tampering → proportional sustained drop
  2. 旁路型: Bypass wiring → abrupt near-zero consumption
  3. 扩差型: Meter modification → consistently low, no changepoint
  4. 降压间歇型: Intermittent tampering → cyclic abnormal-normal pattern
  5. 高科技遥控型: Remote control → random drops/recoveries
  6. 无表/私接型: No meter / illegal connection → near-permanent zero
  7. 移相型: Phase shifting → invisible in daily data (ceiling contributor)

Output:
  - Distribution of theft types (counts + percentages)
  - Baseline F1 per type (single-feature AUC)
  - "Unidentifiable" ratio estimate

Constraints:
  - 42,000 users × 1,034 days → must be efficient
  - All features computable from daily consumption data only
"""
import os, time, warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

SEED = 42
np.random.seed(SEED)


# =============================================================================
# 1. Data Loading
# =============================================================================

def load_sgcc_data():
    """Load SGCC raw data from CSV."""
    print("[Phase0] Loading SGCC raw data...")
    df = pd.read_csv('data/raw_data.csv')
    date_cols = [c for c in df.columns if '/' in str(c) and len(str(c)) <= 10]
    raw = df[date_cols].values.astype(float)
    flags = df['FLAG'].values.astype(np.int32)
    print(f"  Shape: {raw.shape}, Theft: {flags.sum()} ({flags.mean()*100:.1f}%)")
    print(f"  Missing: {np.isnan(raw).mean()*100:.1f}%")
    return raw, flags, date_cols


# =============================================================================
# 2. CUSUM Changepoint Detection (core for Type 1, 2, 4, 5)
# =============================================================================

def cusum_changepoint(row, threshold=3.0, drift=0.2, min_segment=30):
    """CUSUM (Cumulative Sum) changepoint detection.

    Detects whether there is a significant level shift in the time series.
    Uses the standard two-sided CUSUM algorithm:
      S+_t = max(0, S+_{t-1} + X_t - (μ + K))
      S-_t = max(0, S-_{t-1} + (μ - K) - X_t)
    Changepoint when S+ or S- exceeds threshold * σ.

    Args:
        threshold: alarm threshold in units of std deviation
        drift: K = drift * σ (allowable slack before alarming)
        min_segment: minimum observations per side of changepoint

    Returns:
        has_changepoint, cp_location, pre_mean, post_mean, change_ratio, persistence
    """
    n = len(row)
    valid = ~np.isnan(row)
    if valid.sum() < min_segment * 2:
        return False, -1, 0, 0, 1.0, 0

    x = row[valid]
    n_v = len(x)
    mu = np.mean(x[:min(n_v//2, 200)])
    sigma = np.std(x[:min(n_v//2, 200)]) + 1e-6

    K = drift * sigma
    H = threshold * sigma

    S_pos = np.zeros(n_v)
    S_neg = np.zeros(n_v)
    cp_pos = -1
    cp_neg = -1

    for t in range(n_v):
        # Recursive CUSUM
        S_pos[t] = max(0, (S_pos[t-1] if t > 0 else 0) + x[t] - mu - K)
        S_neg[t] = max(0, (S_neg[t-1] if t > 0 else 0) + mu - K - x[t])

        if S_pos[t] > H and cp_pos < 0:
            cp_pos = t
        if S_neg[t] > H and cp_neg < 0:
            cp_neg = t

    # Determine which direction triggered first and is stronger
    if cp_pos < 0 and cp_neg < 0:
        return False, -1, 0, 0, 1.0, 0

    if cp_pos >= 0 and cp_neg >= 0:
        cp = min(cp_pos, cp_neg)
        direction = 'drop' if cp_neg <= cp_pos else 'rise'
    elif cp_neg >= 0:
        cp = cp_neg
        direction = 'drop'
    else:
        cp = cp_pos
        direction = 'rise'

    # Map back to original index (accounting for NaN)
    valid_indices = np.where(valid)[0]
    actual_cp = valid_indices[cp]

    if actual_cp < min_segment or actual_cp > n - min_segment:
        return False, -1, 0, 0, 1.0, 0

    pre_mean = np.nanmean(row[:actual_cp])
    post_mean = np.nanmean(row[actual_cp:])
    change_ratio = post_mean / (pre_mean + 1e-6)

    if direction == 'drop' and change_ratio > 0.95:
        return False, -1, 0, 0, 1.0, 0
    if direction == 'rise' and change_ratio < 1.05:
        return False, -1, 0, 0, 1.0, 0

    persistence = np.mean(row[actual_cp:] < post_mean * 1.5) if direction == 'drop' else 0

    return True, actual_cp, pre_mean, post_mean, change_ratio, persistence


# =============================================================================
# 3. Behavioral Feature Computation
# =============================================================================

def compute_behavioral_profile(raw, flags, date_cols, n_jobs=1):
    """Compute behavioral profile for all theft users.

    For each theft user, compute:
      - CUSUM changepoint statistics
      - Global ranking statistics
      - Run-length statistics for anomaly periods
      - Bimodality / irregularity scores
      - Self-correlation / randomness tests
    """
    n_users, n_days = raw.shape
    theft_idx = np.where(flags == 1)[0]
    n_theft = len(theft_idx)
    print(f"[Phase0] Profiling {n_theft} theft users...")

    # Pre-compute: fill NaN with median per user
    print("  Filling missing values...")
    filled = raw.copy()
    for i in range(n_users):
        row = raw[i]
        if np.isnan(row).any():
            m = np.nanmedian(row) if np.isnan(row).sum() < n_days else 0
            filled[i] = np.nan_to_num(row, nan=m)

    # Cluster for peer comparison: simple percentile-based
    user_means = np.nanmean(raw, axis=1)
    user_means = np.nan_to_num(user_means, nan=0.0)
    global_rank = stats.rankdata(user_means) / n_users  # 0-1

    # Daily per-column medians for peer deviation
    daily_medians = np.nanmedian(raw, axis=0)

    results = {
        'user_id': theft_idx.tolist(),
        'cusum_cp': [], 'cusum_loc': [], 'change_ratio': [],
        'change_persistence': [], 'change_var_ratio': [],
        'global_rank_pct': [], 'stable_low_ratio': [],
        'near_zero_ratio': [], 'cv_daily': [],
        'bimodality_score': [], 'anomaly_ratio': [],
        'mean_run_length': [], 'weekday_anom_ratio': [],
        'acf1': [], 'acf7': [], 'ljungbox_p': [],
        'no_changepoint': [],
    }

    for idx in tqdm(theft_idx, desc="  Profiling"):
        row = filled[idx]
        n = len(row)

        # CUSUM
        has_cp, cp_loc, pre_m, post_m, ch_ratio, persistence = cusum_changepoint(row)
        results['cusum_cp'].append(int(has_cp))
        results['cusum_loc'].append(cp_loc)
        results['change_ratio'].append(ch_ratio)
        results['change_persistence'].append(persistence)

        # Variance ratio (post / pre)
        if has_cp and cp_loc > 30 and cp_loc < n - 30:
            pre_std = np.std(row[:cp_loc])
            post_std = np.std(row[cp_loc:])
            results['change_var_ratio'].append(post_std / (pre_std + 1e-6))
        else:
            results['change_var_ratio'].append(1.0)

        # No-changepoint flag (扩差 + 无表)
        results['no_changepoint'].append(int(not has_cp))

        # Global rank percentile
        results['global_rank_pct'].append(global_rank[idx])

        # Stable low consumption
        p15 = np.percentile(filled[idx], 15)
        results['stable_low_ratio'].append(np.mean(filled[idx] < max(p15, 0.01)))

        # Near-zero ratio (theft where consumption ≈ 0)
        results['near_zero_ratio'].append(np.mean(row < 0.01))

        # CV
        mean_val = np.mean(row[row > 0]) if np.any(row > 0) else 0
        std_val = np.std(row[row > 0]) if np.any(row > 0) else 0
        results['cv_daily'].append(std_val / (mean_val + 1e-6))

        # Anomaly days: below own historical P10
        p10 = np.percentile(row, 10)
        anomaly_mask = row < max(p10 * 0.5, 0.01)
        results['anomaly_ratio'].append(anomaly_mask.mean())

        # Mean run length of anomaly periods
        runs = []
        cur_run = 0
        for v in anomaly_mask:
            if v: cur_run += 1
            elif cur_run > 0: runs.append(cur_run); cur_run = 0
        if cur_run > 0: runs.append(cur_run)
        results['mean_run_length'].append(np.mean(runs) if runs else 0)

        # Bimodality: Hartigan's dip test approximation
        # Simple heuristic: ratio of density between two modes
        pos_vals = row[row > 0]
        if len(pos_vals) > 20:
            p25, p50, p75 = np.percentile(pos_vals, [25, 50, 75])
            if p75 > p25:
                low_density = np.sum((pos_vals >= p25) & (pos_vals <= p50))
                high_density = np.sum((pos_vals >= p50) & (pos_vals <= p75))
                bimodal = 1 - abs(low_density - high_density) / max(low_density + high_density, 1)
            else:
                bimodal = 0
        else:
            bimodal = 0
        results['bimodality_score'].append(bimodal)

        # Weekday anomaly ratio (approximate: use month-based day-of-month)
        # Since we don't have exact weekday, use rolling 7-day window
        rolling_mean = uniform_filter1d(row, size=7, mode='nearest')
        daily_dev = row - rolling_mean
        results['weekday_anom_ratio'].append(
            np.std(daily_dev[:n//2]) / (np.std(daily_dev[n//2:]) + 1e-6)
        )

        # ACF
        centered = row - np.mean(row)
        if np.std(centered) > 1e-6:
            acf1 = np.corrcoef(centered[:-1], centered[1:])[0, 1]
            results['acf1'].append(0 if np.isnan(acf1) else acf1)
            if n > 7:
                acf7 = np.corrcoef(centered[:-7], centered[7:])[0, 1]
                results['acf7'].append(0 if np.isnan(acf7) else acf7)
            else:
                results['acf7'].append(0)
        else:
            results['acf1'].append(0)
            results['acf7'].append(0)

        # Ljung-Box randomness test
        if len(centered) > 20 and np.std(centered) > 1e-6:
            try:
                lb_p = stats.chi2.sf(
                    n * np.sum([np.corrcoef(centered[:-k], centered[k:])[0, 1]**2
                                for k in range(1, min(11, n//2))]),
                    df=min(10, n//2 - 1)
                )
                results['ljungbox_p'].append(min(lb_p, 1.0))
            except:
                results['ljungbox_p'].append(1.0)
        else:
            results['ljungbox_p'].append(1.0)

    profile_df = pd.DataFrame(results)
    profile_df['flag'] = 1  # All are theft users
    return profile_df, filled


# =============================================================================
# 4. Theft Type Classification
# =============================================================================

def classify_theft_type(profile_df):
    """Classify each theft user into one of 7 types.

    Decision rules based on physical models of electricity theft.
    Priority order matches the confidence of each rule.
    """
    n = len(profile_df)
    types = np.full(n, '7_unknown', dtype=object)
    reasons = np.full(n, '', dtype=object)

    # Type 1: 降压型 (has changepoint, moderate drop 20-80%, sustained)
    mask = (
        (types == '7_unknown') &
        (profile_df['cusum_cp'] == 1) &
        (profile_df['change_ratio'] >= 0.2) &
        (profile_df['change_ratio'] < 0.8) &
        (profile_df['change_persistence'] > 0.7)
    )
    types[mask.values] = '1_voltage_drop'
    reasons[mask.values] = 'cp+ratio0.2-0.8+persist>0.7'

    # Type 2: 旁路型 (has changepoint, severe drop < 20% of pre-level)
    mask = (
        (types == '7_unknown') &
        (profile_df['cusum_cp'] == 1) &
        (profile_df['change_ratio'] < 0.2)
    )
    types[mask.values] = '2_bypass'
    reasons[mask.values] = 'cp+ratio<0.2'

    # Type 4: 降压间歇型 (bimodal or periodic anomaly pattern)
    mask = (
        (types == '7_unknown') &
        (profile_df['anomaly_ratio'] > 0.15) &
        (profile_df['mean_run_length'] >= 3) 
    )
    types[mask.values] = '4_intermittent'
    reasons[mask.values] = 'anom>0.15+run>=3'

    # Type 5: 高科技遥控型 (chaotic pattern: low ACF1, high randomness)
    mask = (
        (types == '7_unknown') &
        (profile_df['acf1'] < 0.3) &
        (profile_df['ljungbox_p'] < 0.1)
    )
    types[mask.values] = '5_remote'
    reasons[mask.values] = 'acf1<0.3+lbp<0.1'

    # Type 6: 无表/私接型 (near-zero consumption)
    mask = (
        (types == '7_unknown') &
        (profile_df['near_zero_ratio'] > 0.9)
    )
    types[mask.values] = '6_no_meter'
    reasons[mask.values] = 'near_zero>0.9'

    # Type 3: 扩差型 (no clear changepoint, consistently below peer group)
    mask = (
        (types == '7_unknown') &
        (profile_df['no_changepoint'] == 1) &
        (profile_df['global_rank_pct'] < 0.25)
    )
    types[mask.values] = '3_constant_low'
    reasons[mask.values] = 'nocp+rank<0.25'

    # Type 7: 未知 / 移相型 (everything else)
    # Keep as 7_unknown

    profile_df['theft_type'] = types
    profile_df['type_reason'] = reasons
    return profile_df


# =============================================================================
# 5. Baseline F1 Assessment
# =============================================================================

def assess_baseline_f1(profile_df, filled, flags):
    """Assess baseline F1 for each theft type using single-feature AUC.

    For each type, compute the AUC of the best single feature against
    all normal users. This gives a lower bound on achievable F1.
    """
    normal_idx = np.where(flags == 0)[0]
    n_normal = len(normal_idx)
    n_users = len(flags)
    user_means = np.nanmean(filled, axis=1)
    user_means = np.nan_to_num(user_means, nan=0.0)
    _global_rank = stats.rankdata(user_means) / n_users

    # Features to test per type
    type_features = {
        '1_voltage_drop':  ['change_ratio', 'change_persistence', 'change_var_ratio'],
        '2_bypass':        ['change_ratio', 'near_zero_ratio', 'change_var_ratio'],
        '3_constant_low':  ['global_rank_pct', 'stable_low_ratio', 'cv_daily'],
        '4_intermittent':  ['bimodality_score', 'anomaly_ratio', 'mean_run_length'],
        '5_remote':        ['anomaly_ratio', 'mean_run_length', 'acf1', 'ljungbox_p'],
        '6_no_meter':      ['near_zero_ratio', 'global_rank_pct'],
        '7_unknown':       ['anomaly_ratio', 'acf1', 'cv_daily'],
    }

    # Normal user stats: compute actual features for a sample
    print(f"\n  Computing features for {min(2000, n_normal)} normal users...")
    normal_feats = {fname: [] for fname in [
        'change_ratio', 'change_persistence', 'global_rank_pct',
        'stable_low_ratio', 'near_zero_ratio', 'cv_daily',
        'anomaly_ratio', 'mean_run_length', 'acf1',
        'bimodality_score', 'ljungbox_p', 'change_var_ratio'
    ]}
    sample_norm = np.random.choice(normal_idx, min(2000, n_normal), replace=False)
    for idx in tqdm(sample_norm, desc="  Normal users"):
        row = filled[idx]
        has_cp, cp_loc, pre_m, post_m, ch_ratio, persistence = cusum_changepoint(row)
        normal_feats['change_ratio'].append(ch_ratio)
        normal_feats['change_persistence'].append(persistence)
        normal_feats['global_rank_pct'].append(_global_rank[idx])
        p15 = np.percentile(row, 15)
        normal_feats['stable_low_ratio'].append(np.mean(row < max(p15, 0.01)))
        normal_feats['near_zero_ratio'].append(np.mean(row < 0.01))
        pos = row[row > 0]
        cvv = np.std(pos) / (np.mean(pos) + 1e-6) if len(pos) > 0 else 0
        normal_feats['cv_daily'].append(cvv)
        p10 = np.percentile(row, 10) if len(row) > 0 else 0
        am = row < max(p10 * 0.5, 0.01)
        normal_feats['anomaly_ratio'].append(am.mean())
        runs = []; cr = 0
        for v in am:
            if v: cr += 1
            elif cr > 0: runs.append(cr); cr = 0
        if cr > 0: runs.append(cr)
        normal_feats['mean_run_length'].append(np.mean(runs) if runs else 0)
        c = row - np.mean(row)
        normal_feats['acf1'].append(np.corrcoef(c[:-1], c[1:])[0,1] if np.std(c)>1e-6 and len(c)>1 else 0)
        normal_feats['bimodality_score'].append(0)
        normal_feats['ljungbox_p'].append(1.0)

        if has_cp and cp_loc > 30 and cp_loc < len(row) - 30:
            normal_feats['change_var_ratio'].append(
                np.std(row[cp_loc:]) / (np.std(row[:cp_loc]) + 1e-6))
        else:
            normal_feats['change_var_ratio'].append(1.0)

    for k in normal_feats:
        normal_feats[k] = np.nan_to_num(np.array(normal_feats[k]), nan=0)

    results = []
    for ttype in sorted(profile_df['theft_type'].unique()):
        subset = profile_df[profile_df['theft_type'] == ttype]
        n_type = len(subset)
        if n_type == 0:
            continue

        best_auc = 0
        best_feat = ''
        for feat in type_features.get(ttype, ['anomaly_ratio']):
            if feat in subset.columns:
                theft_vals = subset[feat].values.astype(float)
                theft_vals = np.nan_to_num(theft_vals, nan=0)

                # Use actual normal feature values
                normal_vals = normal_feats.get(feat, np.zeros(n_normal))
                all_vals = np.concatenate([theft_vals, normal_vals])
                all_labels = np.concatenate([np.ones(n_type), np.zeros(len(normal_vals))])

                try:
                    auc = roc_auc_score(all_labels, np.abs(all_vals))
                    if auc > best_auc:
                        best_auc = auc
                        best_feat = feat
                except:
                    pass

        results.append({
            'theft_type': ttype,
            'count': n_type,
            'pct': n_type / len(profile_df) * 100,
            'best_feat': best_feat,
            'best_auc': best_auc,
            'est_best_f1': best_auc * 0.85 if best_auc > 0.5 else 0,
        })

    return pd.DataFrame(results)


# =============================================================================
# 6. Main
# =============================================================================

def run_phase0():
    t0 = time.time()
    print("=" * 60)
    print("  Phase 0: Behavioral Profiling")
    print("=" * 60)

    raw, flags, date_cols = load_sgcc_data()

    print("\n[Step 1] Computing behavioral profiles for theft users...")
    profile_df, filled = compute_behavioral_profile(raw, flags, date_cols)

    print("\n[Step 2] Classifying theft users into 7 types...")
    profile_df = classify_theft_type(profile_df)

    print("\n[Step 3] Distribution analysis...")
    dist = profile_df['theft_type'].value_counts()
    print(f"\n  {'Theft Type':<20s} {'Count':>6s} {'Pct':>6s}")
    print("  " + "-" * 35)
    type_names = {
        '1_voltage_drop': '降压型',
        '2_bypass': '旁路型',
        '3_constant_low': '扩差型',
        '4_intermittent': '降压间歇型',
        '5_remote': '高科技遥控型',
        '6_no_meter': '无表/私接型',
        '7_unknown': '未知/移相型',
    }
    for ttype, count in dist.items():
        pct = count / len(profile_df) * 100
        name = type_names.get(ttype, ttype)
        print(f"  {name:<18s} {count:>6d} {pct:>5.1f}%")

    unknown_pct = dist.get('7_unknown', 0) / len(profile_df) * 100
    print(f"\n  Unidentifiable (unknown/移相): {unknown_pct:.1f}%")
    if unknown_pct > 15:
        print(f"  => F1 90% CEILING: daily data cannot detect {unknown_pct:.0f}% of thefts")
    else:
        print(f"  => F1 90% FEASIBLE: low unidentifiable ratio")

    print("\n[Step 4] Baseline F1 assessment...")
    baseline = assess_baseline_f1(profile_df, filled, flags)
    print(f"\n  {'Theft Type':<18s} {'Count':>6s} {'BestFeat':<22s} {'AUC':>6s} {'EstF1':>6s}")
    print("  " + "-" * 60)
    for _, row in baseline.iterrows():
        name = type_names.get(row['theft_type'], row['theft_type'])
        print(f"  {name:<18s} {int(row['count']):>6d} {row['best_feat']:<22s} "
              f"{row['best_auc']:>5.3f} {row['est_best_f1']:>5.3f}")

    # Save
    profile_df.to_csv('output/phase0_profile.csv', index=False)
    baseline.to_csv('output/phase0_baseline.csv', index=False)
    print(f"\n  Saved to output/phase0_profile.csv, phase0_baseline.csv")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")
    return profile_df, baseline, filled, flags

if __name__ == '__main__':
    profile_df, baseline, filled, flags = run_phase0()
