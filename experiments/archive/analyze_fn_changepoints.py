import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, r'D:\Project\ThiefElectricity')

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score
from dl_data import load_raw_data

SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

print("="*60)
print("FN Change-Point Detection Analysis")
print("="*60)

print("Loading raw data and V225 OOF...")
X_raw, y = load_raw_data()
print(f"  X_raw: {X_raw.shape}, y: {y.shape}")

oof_path = os.path.join(OUTPUT_DIR, 'sgcc_final_oof.npz')
oof_v225 = np.load(oof_path)['oof_v225']

THR = 0.740
fn_mask = (y == 1) & (oof_v225 < THR)
tp_mask = (y == 1) & (oof_v225 >= THR)
fp_mask = (y == 0) & (oof_v225 >= THR)
tn_mask = (y == 0) & (oof_v225 < THR)

print(f"\n  V225 threshold = {THR}")
print(f"  FN (theft, missed): {fn_mask.sum()}")
print(f"  TP (theft, caught): {tp_mask.sum()}")
print(f"  FP (normal, false alarm): {fp_mask.sum()}")
print(f"  TN (normal, correct): {tn_mask.sum()}")
print(f"  FN rate among theft: {fn_mask.sum() / (y==1).sum() * 100:.1f}%")

fn_indices = np.where(fn_mask)[0]
np.random.seed(SEED)
if len(fn_indices) > 20:
    sample_idx = np.random.choice(fn_indices, 20, replace=False)
else:
    sample_idx = fn_indices

def compute_cusum(series, ref_mean=None):
    if ref_mean is None:
        ref_mean = np.nanmean(series)
    deviations = series - ref_mean
    cusum = np.cumsum(deviations)
    cusum_max = np.max(cusum)
    cusum_min = np.min(cusum)
    cusum_range = cusum_max - cusum_min
    cusum_peak_loc = np.argmax(np.abs(cusum))
    return cusum, cusum_max, cusum_min, cusum_range, cusum_peak_loc

def compute_regime_stats(series, split_point=None):
    n = len(series)
    if split_point is None:
        split_point = max(1, n * 3 // 4)
    
    pre = series[:split_point]
    post = series[split_point:]
    
    pre_mean = np.nanmean(pre)
    post_mean = np.nanmean(post)
    pre_std = np.nanstd(pre)
    post_std = np.nanstd(post)
    pre_miss = np.isnan(pre).mean()
    post_miss = np.isnan(post).mean()
    pre_zero = np.nansum(pre == 0) / max(np.sum(~np.isnan(pre)), 1)
    post_zero = np.nansum(post == 0) / max(np.sum(~np.isnan(post)), 1)
    
    ratio = post_mean / (pre_mean + 1e-6)
    ratio_miss = post_miss / (pre_miss + 1e-6) if pre_miss > 0 else (2.0 if post_miss > 0 else 1.0)
    ratio_zero = post_zero / (pre_zero + 1e-6) if pre_zero > 0 else (2.0 if post_zero > 0 else 1.0)
    
    return {
        'pre_mean': pre_mean, 'post_mean': post_mean,
        'pre_std': pre_std, 'post_std': post_std,
        'pre_miss': pre_miss, 'post_miss': post_miss,
        'pre_zero': pre_zero, 'post_zero': post_zero,
        'ratio_mean': ratio, 'ratio_miss': ratio_miss, 'ratio_zero': ratio_zero
    }

def max_consecutive_low(series, threshold=None, min_gap=5):
    if threshold is None:
        threshold = np.nanmedian(series)
    clean = np.nan_to_num(series, nan=threshold + 1)
    low = clean < threshold
    max_run = 0
    run = 0
    for v in low:
        if v:
            run += 1
            max_run = max(max_run, run)
        else:
            if run >= min_gap:
                pass
            run = 0
    return max_run if max_run >= min_gap else 0

def compute_drawdown(series):
    clean = np.nan_to_num(series, nan=0)
    cumulative = np.cumsum(np.maximum(clean, 0))
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_dd = np.max(drawdown)
    max_dd_pct = max_dd / (running_max[-1] + 1e-6)
    dd_end = np.argmax(drawdown)
    return max_dd, max_dd_pct, dd_end

print("\n" + "="*60)
print("Per-sample analysis on FN curves...")
print("="*60)

fn_stats = []
for idx in sample_idx:
    ts = X_raw[idx].copy()
    ts_clean = np.nan_to_num(ts, nan=0)
    miss_mask = np.isnan(ts)
    
    n = len(ts)
    split3q = max(1, n * 3 // 4)
    split2q = n // 2
    
    cusum, cmax, cmin, crange, cpeak = compute_cusum(ts_clean)
    
    stats3q = compute_regime_stats(ts, split3q)
    stats2q = compute_regime_stats(ts, split2q)
    
    rolling_mean_30 = np.array([np.nanmean(ts[max(0,i-14):min(n,i+15)]) for i in range(n)])
    roll_cusum, rcmax, rcmin, rcrange, rcpeak = compute_cusum(rolling_mean_30)
    
    max_low_run = max_consecutive_low(ts_clean)
    max_dd, max_dd_pct, dd_end = compute_drawdown(ts_clean)
    
    fn_stats.append({
        'idx': idx,
        'cusum_range': crange,
        'cusum_peak_frac': cpeak / n,
        'cusum_roll_range': rcrange,
        'ratio_mean_3q': stats3q['ratio_mean'],
        'ratio_mean_2q': stats2q['ratio_mean'],
        'ratio_miss_3q': stats3q['ratio_miss'],
        'ratio_zero_3q': stats3q['ratio_zero'],
        'max_low_run': max_low_run,
        'max_drawdown_pct': max_dd_pct,
        'pct_missing': miss_mask.mean(),
        'pct_zero': (ts_clean == 0).mean(),
    })

for s in fn_stats:
    print(f"  idx {s['idx']:6d}: mean_2q={s['ratio_mean_2q']:.2f} mean_3q={s['ratio_mean_3q']:.2f} "
          f"cusum_r={s['cusum_range']:.1f} miss_3q={s['ratio_miss_3q']:.2f} "
          f"low_run={s['max_low_run']} dd={s['max_drawdown_pct']:.3f}")

n_regime_change = sum(1 for s in fn_stats if s['ratio_mean_3q'] < 0.7)
print(f"\n  FN with post/pre ratio < 0.7 (likely regime change): {n_regime_change}/{len(fn_stats)}")

print("\n" + "="*60)
print("Population-level comparison (FN vs TP vs FP vs TN)")
print("="*60)

def compute_group_stats(indices, label):
    n_sample = min(500, len(indices))
    if n_sample < len(indices):
        sample = np.random.choice(indices, n_sample, replace=False)
    else:
        sample = indices
    
    ratios_3q = []
    ratios_2q = []
    ratios_miss = []
    ratios_zero = []
    cusum_ranges = []
    max_low_runs = []
    
    for idx in sample:
        ts = X_raw[idx].copy()
        ts_clean = np.nan_to_num(ts, nan=0)
        n = len(ts)
        
        s3q = compute_regime_stats(ts, max(1, n*3//4))
        s2q = compute_regime_stats(ts, n//2)
        cusum, cmax, cmin, crange, _ = compute_cusum(ts_clean)
        max_low = max_consecutive_low(ts_clean)
        
        ratios_3q.append(s3q['ratio_mean'])
        ratios_2q.append(s2q['ratio_mean'])
        ratios_miss.append(s3q['ratio_miss'])
        ratios_zero.append(s3q['ratio_zero'])
        cusum_ranges.append(crange)
        max_low_runs.append(max_low)
    
    ratios_3q = np.array(ratios_3q)
    ratios_2q = np.array(ratios_2q)
    ratios_miss = np.array(ratios_miss)
    ratios_zero = np.array(ratios_zero)
    cusum_ranges = np.array(cusum_ranges)
    max_low_runs = np.array(max_low_runs)
    
    regime_frac_3q = (ratios_3q < 0.7).mean()
    regime_frac_2q = (ratios_2q < 0.7).mean()
    
    print(f"\n  {label} (n={len(sample)}):")
    print(f"    mean_ratio_3q: {np.median(ratios_3q):.3f} [{np.percentile(ratios_3q,25):.3f}, {np.percentile(ratios_3q,75):.3f}]")
    print(f"    mean_ratio_2q: {np.median(ratios_2q):.3f} [{np.percentile(ratios_2q,25):.3f}, {np.percentile(ratios_2q,75):.3f}]")
    print(f"    miss_ratio_3q: {np.median(ratios_miss):.3f} [{np.percentile(ratios_miss,25):.3f}, {np.percentile(ratios_miss,75):.3f}]")
    print(f"    zero_ratio_3q: {np.median(ratios_zero):.3f} [{np.percentile(ratios_zero,25):.3f}, {np.percentile(ratios_zero,75):.3f}]")
    print(f"    cusum_range: {np.median(cusum_ranges):.0f} [{np.percentile(cusum_ranges,25):.0f}, {np.percentile(cusum_ranges,75):.0f}]")
    print(f"    max_low_run: {np.median(max_low_runs):.0f} [{np.percentile(max_low_runs,25):.0f}, {np.percentile(max_low_runs,75):.0f}]")
    print(f"    regime_frac_3q (ratio<0.7): {regime_frac_3q:.3f}")
    print(f"    regime_frac_2q (ratio<0.7): {regime_frac_2q:.3f}")
    
    return {
        'ratios_3q': ratios_3q, 'ratios_2q': ratios_2q,
        'ratios_miss': ratios_miss, 'ratios_zero': ratios_zero,
        'cusum_ranges': cusum_ranges, 'max_low_runs': max_low_runs,
        'regime_frac_3q': regime_frac_3q, 'regime_frac_2q': regime_frac_2q,
    }

np.random.seed(SEED)
stats_fn = compute_group_stats(fn_indices, 'FN')
stats_tp = compute_group_stats(np.where(tp_mask)[0], 'TP')
stats_fp = compute_group_stats(np.where(fp_mask)[0], 'FP')
stats_tn = compute_group_stats(np.where(tn_mask)[0], 'TN')

print("\n" + "="*60)
print("Visualizing 20 FN curves...")
print("="*60)

fig, axes = plt.subplots(5, 4, figsize=(24, 22))
axes = axes.flatten()

for i, s in enumerate(fn_stats[:20]):
    ax = axes[i]
    idx = s['idx']
    ts = X_raw[idx].copy()
    ts_clean = np.nan_to_num(ts, nan=0)
    miss_mask = np.isnan(ts)
    n = len(ts)
    
    split = max(1, n * 3 // 4)
    
    ax.plot(range(n), ts_clean, 'b-', alpha=0.5, linewidth=0.8, label='consumption')
    ax.plot(range(n), ts_clean, 'b-', alpha=0.8, linewidth=0.3)
    
    ax.fill_between(range(n), 0, ts_clean.max()*1.1, 
                    where=miss_mask, color='red', alpha=0.3, label='missing')
    
    trend_90 = np.array([np.nanmean(ts[max(0,j-44):min(n,j+45)]) for j in range(n)])
    ax.plot(range(n), trend_90, 'r-', linewidth=1.5, alpha=0.7, label='trend(90d)')
    
    ax.axvline(x=split, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    
    pre_mean = np.nanmean(ts[:split])
    post_mean = np.nanmean(ts[split:])
    ax.axhline(y=pre_mean, color='green', alpha=0.3)
    ax.axhline(y=post_mean, color='red', alpha=0.3)
    
    ratio = s['ratio_mean_3q']
    ax.set_title(f"FN idx={idx} | post/pre={ratio:.2f} | miss={s['pct_missing']:.1%} | zero={s['pct_zero']:.1%}",
                 fontsize=8)
    ax.set_xlabel('day', fontsize=7)
    ax.set_ylabel('kWh', fontsize=7)
    
    if i == 0:
        ax.legend(fontsize=6, loc='upper right')

plt.suptitle('20 Randomly Sampled FN (False Negative) Consumption Curves\n'
             'Blue=consumption, Red=missing, Green=pre/post split at 3/4, Trend line=90d moving avg',
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fn_curves_changepoint.png'), dpi=150)
plt.close()
print(f"  Saved fn_curves_changepoint.png")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

for ax_idx, (group_name, color, ratios, regime_frac) in enumerate([
    ('FN (theft missed)', 'red', stats_fn['ratios_3q'], stats_fn['regime_frac_3q']),
    ('TP (theft caught)', 'green', stats_tp['ratios_3q'], stats_tp['regime_frac_3q']),
]):
    ax_row = axes2[0, ax_idx]
    clean_ratios = ratios[~np.isnan(ratios) & ~np.isinf(ratios)]
    ax_row.hist(np.clip(clean_ratios, 0, 3), bins=40, alpha=0.6, color=color)
    ax_row.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
    ax_row.axvline(x=0.7, color='red', linestyle=':', alpha=0.5)
    med = np.median(clean_ratios) if len(clean_ratios) > 0 else 0
    ax_row.set_title(f'Post/Pre ratio: {group_name}\n'
                      f'(median={med:.3f}, regime<0.7={regime_frac:.1%})',
                      fontsize=11)
    ax_row.set_xlabel('Post / Pre mean ratio')
    ax_row.set_ylabel('Count')

for ax_idx, (group_name, color, ratios, regime_frac) in enumerate([
    ('FP (normal, false alarm)', 'orange', stats_fp['ratios_3q'], stats_fp['regime_frac_3q']),
    ('TN (normal, correct)', 'blue', stats_tn['ratios_3q'], stats_tn['regime_frac_3q']),
]):
    ax_row = axes2[1, ax_idx]
    clean_ratios = ratios[~np.isnan(ratios) & ~np.isinf(ratios)]
    ax_row.hist(np.clip(clean_ratios, 0, 3), bins=40, alpha=0.6, color=color)
    ax_row.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
    ax_row.axvline(x=0.7, color='red', linestyle=':', alpha=0.5)
    med = np.median(clean_ratios) if len(clean_ratios) > 0 else 0
    ax_row.set_title(f'Post/Pre ratio: {group_name}',
                      fontsize=11)
    ax_row.set_xlabel('Post / Pre mean ratio')
    ax_row.set_ylabel('Count')

plt.suptitle('Regime Change Analysis: Post/Pre Consumption Ratio by Group\n'
             'Red dotted line = 0.7 (potential regime change threshold)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, 'fn_regime_change_distribution.png'), dpi=150)
plt.close()
print(f"  Saved fn_regime_change_distribution.png")

print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"  FN regime change rate (3q ratio < 0.7): {stats_fn['regime_frac_3q']:.1%}")
print(f"  TP regime change rate (3q ratio < 0.7): {stats_tp['regime_frac_3q']:.1%}")
print(f"  FP regime change rate (3q ratio < 0.7): {stats_fp['regime_frac_3q']:.1%}")
print(f"  TN regime change rate (3q ratio < 0.7): {stats_tn['regime_frac_3q']:.1%}")

delta_fn_tp = stats_fn['regime_frac_3q'] - stats_tp['regime_frac_3q']
print(f"\n  FN - TP regime change delta: {delta_fn_tp:+.1%}")
if delta_fn_tp > 0.05:
    print(f"  ==> POSITIVE: FN have meaningfully more regime changes than TP. Change-point features WOULD help.")
elif delta_fn_tp > 0.02:
    print(f"  ==> MARGINAL: Small signal exists. Change-point features might help marginally.")
else:
    print(f"  ==> NEGATIVE: FN and TP have similar regime-change rates. Change-point features unlikely to help.")

np.savez_compressed(os.path.join(OUTPUT_DIR, 'fn_changepoint_stats.npz'),
                    fn_indices=fn_indices, tp_indices=np.where(tp_mask)[0],
                    fp_indices=np.where(fp_mask)[0], tn_indices=np.where(tn_mask)[0],
                    fn_sample_idx=sample_idx, fn_stats_list=np.array(fn_stats),
                    stats_fn_regime=stats_fn['regime_frac_3q'],
                    stats_tp_regime=stats_tp['regime_frac_3q'],
                    stats_fp_regime=stats_fp['regime_frac_3q'],
                    stats_tn_regime=stats_tn['regime_frac_3q'])

print(f"\nResults saved to {OUTPUT_DIR}/fn_changepoint_stats.npz")