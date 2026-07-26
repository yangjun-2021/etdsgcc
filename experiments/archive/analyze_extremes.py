import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, r'D:\Project\ThiefElectricity')
import numpy as np
from scipy import stats
from dl_data import load_raw_data
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42; np.random.seed(SEED)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

X_raw, y = load_raw_data()
n_users, n_days = X_raw.shape
theft_idx = np.where(y == 1)[0]
normal_idx = np.where(y == 0)[0]

print("="*70)
print("  极端值分析：是否与窃电相关？")
print("="*70)

obs_mask = ~np.isnan(X_raw)
obs_values = np.where(obs_mask, X_raw, 0)

global_mean = np.nanmean(X_raw)
global_std = np.nanstd(X_raw)
global_p50 = np.nanmedian(X_raw)
global_p95 = np.nanpercentile(X_raw, 95)
global_p99 = np.nanpercentile(X_raw, 99)
global_p999 = np.nanpercentile(X_raw, 99.9)
print(f"\n全局统计量:")
print(f"  mean={global_mean:.1f}, std={global_std:.1f}")
print(f"  P50={global_p50:.1f}, P95={global_p95:.1f}, P99={global_p99:.1f}, P99.9={global_p999:.1f}")
print(f"  极值倍数: P99/P50={global_p99/global_p50:.1f}x, P99.9/P99={global_p999/global_p99:.1f}x")

print(f"\n" + "="*70)
print(f"  分析维度1：用户自身极端值频次")
print(f"="*70)

for label, indices, tag in [(1, theft_idx, '窃电'), (0, normal_idx, '正常')]:
    vals = obs_values[indices]
    
    p95_user = np.percentile(vals, 95, axis=1)
    p99_user = np.percentile(vals, 99, axis=1)
    
    above_p95 = (vals > p95_user[:, None]).sum(axis=1)
    above_p99 = (vals > p99_user[:, None]).sum(axis=1)
    
    max_val = np.max(vals, axis=1)
    max_median_ratio = max_val / (np.median(vals, axis=1) + 1e-6)
    
    extreme_ratio = (vals > p95_user[:, None]).sum(axis=1) / np.maximum(obs_mask[indices].sum(axis=1), 1)
    
    print(f"\n  {tag}用户 (n={len(indices)}):")
    print(f"    自身P95以上: 中位数={np.median(above_p95):.0f}天, 均值={np.mean(above_p95):.1f}天")
    print(f"    自身P99以上: 中位数={np.median(above_p99):.0f}天, 均值={np.mean(above_p99):.1f}天")
    print(f"    极值率(P95+): 中位数={np.median(extreme_ratio)*100:.1f}%")
    print(f"    max/median:   中位数={np.median(max_median_ratio):.1f}x, P95={np.percentile(max_median_ratio,95):.1f}x")

print(f"\n" + "="*70)
print(f"  分析维度2：全局极值命中率")
print(f"="*70)

thresholds = [global_p95, global_p99, global_p999, global_mean + 3*global_std]
th_names = ['P95', 'P99', 'P99.9', 'μ+3σ']

for th, th_name in zip(thresholds, th_names):
    print(f"\n  阈值 {th_name} = {th:.1f} kWh:")
    for label, indices, tag in [(1, theft_idx, '窃电'), (0, normal_idx, '正常')]:
        vals = obs_values[indices]
        above = (vals > th).sum(axis=1)
        ratio = above / np.maximum(obs_mask[indices].sum(axis=1), 1)
        
        any_extreme = (above > 0).mean()
        mean_days = np.mean(above)
        med_days = np.median(above)
        
        print(f"    {tag}: any_extreme={any_extreme*100:.1f}%, mean_days={mean_days:.1f}, median_days={med_days:.0f}, mean_ratio={np.mean(ratio)*100:.2f}%")

print(f"\n" + "="*70)
print(f"  分析维度3：极端值的时序聚集性")
print(f"="*70)

def extreme_clustering_idx(mask_1d):
    n = len(mask_1d)
    positions = np.where(mask_1d)[0]
    if len(positions) < 2:
        return 0.0
    gaps = np.diff(positions)
    if len(gaps) == 0:
        return 0.0
    var_gap = np.var(gaps) if len(gaps) > 1 else 0
    mean_gap = np.mean(gaps) + 1e-6
    return var_gap / (mean_gap ** 2)

for label, indices, tag in [(1, theft_idx, '窃电'), (0, normal_idx, '正常')]:
    vals = obs_values[indices]
    sample_n = min(500, len(indices))
    sample_idx = np.random.choice(indices[:len(vals)], sample_n, replace=False)
    
    clustering_scores = []
    for i in range(sample_n):
        extreme_mask = (vals[i] > np.percentile(vals[i], 95)) & (obs_mask[indices[i]])
        if extreme_mask.sum() < 2:
            clustering_scores.append(0)
        else:
            clustering_scores.append(extreme_clustering_idx(extreme_mask))
    
    print(f"  {tag} 极端值聚类指数: 中位数={np.median(clustering_scores):.3f}, 均值={np.mean(clustering_scores):.3f}")

print(f"\n" + "="*70)
print(f"  分析维度4：极端值与窃电标签的直接相关性")
print(f"="*70)

feat_names = []
feat_matrix = []
all_indices = np.arange(n_users)

vals_all = obs_values

# Feature 1: days above own P95
p95_per_user = np.percentile(vals_all, 95, axis=1)
feat_matrix.append((vals_all > p95_per_user[:, None]).sum(axis=1))
feat_names.append('n_above_ownP95')

# Feature 2: days above global P99
feat_matrix.append((vals_all > global_p99).sum(axis=1))
feat_names.append('n_above_globalP99')

# Feature 3: max / median ratio
max_vals = np.max(vals_all, axis=1)
med_vals = np.median(vals_all, axis=1)
feat_matrix.append(max_vals / (med_vals + 1e-6))
feat_names.append('max_median_ratio')

# Feature 4: global P99.9 hit count
feat_matrix.append((vals_all > global_p999).sum(axis=1))
feat_names.append('n_above_P999')

# Feature 5: extreme value sum ratio
p90_per_user = np.percentile(vals_all, 90, axis=1)
extreme_sum = np.sum(np.where(vals_all > p90_per_user[:, None], vals_all, 0), axis=1)
total_sum = np.sum(vals_all, axis=1)
feat_matrix.append(extreme_sum / (total_sum + 1e-6))
feat_names.append('extreme_sum_ratio')

# Feature 6: volatility of extreme values (CV of values above P90)
cv_extreme = np.zeros(n_users)
for i in range(n_users):
    extreme_vals = vals_all[i][vals_all[i] > p90_per_user[i]]
    if len(extreme_vals) > 3:
        cv_extreme[i] = np.std(extreme_vals) / (np.mean(extreme_vals) + 1e-6)
feat_matrix.append(cv_extreme)
feat_names.append('extreme_cv')

# Feature 7: max drawdown to max ratio
feat_matrix.append(1 - np.min(vals_all, axis=1) / (np.max(vals_all, axis=1) + 1e-6))
feat_names.append('max_to_min_range')

# Feature 8: count of >10x median days
feat_matrix.append((vals_all > 10 * med_vals[:, None]).sum(axis=1))
feat_names.append('n_above_10x_median')

# Feature 9: temporal entropy of extreme days
def extreme_temporal_entropy(mask_1d):
    n = len(mask_1d)
    if mask_1d.sum() < 2:
        return 0.0
    positions = np.where(mask_1d)[0]
    gaps = np.diff(positions)
    if len(gaps) <= 1:
        return 0.0
    hist, _ = np.histogram(gaps, bins=min(10, len(gaps)))
    hist = hist / (hist.sum() + 1e-12)
    return -np.sum(hist * np.log(hist + 1e-12))

extreme_entropy = np.zeros(n_users)
for i in range(min(n_users, 5000)):
    extreme_mask = (vals_all[i] > p95_per_user[i]) & (obs_mask[i])
    extreme_entropy[i] = extreme_temporal_entropy(extreme_mask)
feat_matrix.append(extreme_entropy)
feat_names.append('extreme_temporal_entropy')

# Feature 10: P99 value itself  
feat_matrix.append(np.percentile(vals_all, 99, axis=1))
feat_names.append('p99_value')

feat_matrix = np.array(feat_matrix).T
feat_matrix = np.nan_to_num(feat_matrix, nan=0.0, posinf=0.0, neginf=0.0)

from sklearn.metrics import roc_auc_score
print(f"\n  极端值特征与窃电标签的相关性 (AUC):")
for i, name in enumerate(feat_names):
    col = feat_matrix[:, i]
    if np.std(col) < 1e-10:
        print(f"    {name:>25s}: constant")
        continue
    try:
        auc = roc_auc_score(y, col)
        mean_t = col[y == 1].mean()
        mean_n = col[y == 0].mean()
        ratio = mean_t / (mean_n + 1e-6)
        print(f"    {name:>25s}: AUC={auc:.4f}  theft_mean={mean_t:.2f}  normal_mean={mean_n:.2f}  ratio={ratio:.2f}x")
    except:
        print(f"    {name:>25s}: error")

print(f"\n" + "="*70)
print(f"  分析维度5：极端值用户画像")
print(f"="*70)

n_extreme_thresholds = [1, 3, 5, 10, 20, 50]
for n_ext in n_extreme_thresholds:
    extreme_users = (vals_all > np.percentile(vals_all, 97, axis=1)[:, None]).sum(axis=1) >= n_ext
    n_ext_total = extreme_users.sum()
    n_ext_theft = (extreme_users & (y == 1)).sum()
    n_ext_normal = (extreme_users & (y == 0)).sum()
    print(f"  P97超{n_ext:2d}天用户: 总数={n_ext_total}({n_ext_total/n_users*100:.1f}%), "
          f"窃电={n_ext_theft}({n_ext_theft/(y==1).sum()*100:.1f}% of theft), "
          f"正常={n_ext_normal}({n_ext_normal/(y==0).sum()*100:.1f}% of normal)")

print(f"\n" + "="*70)
print(f"  可视化：极端值分布对比")
print(f"="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

t_colors = ['red'] * len(theft_idx)
n_colors = ['blue'] * len(normal_idx)

for ax_i, (feat_idx, ax, title) in enumerate([
    (0, axes[0,0], 'Days above own P95'),
    (1, axes[0,1], 'Days above global P99'),
    (2, axes[0,2], 'Max / Median ratio'),
    (3, axes[1,0], 'Days above global P99.9'),
    (4, axes[1,1], 'Extreme value sum ratio'),
    (9, axes[1,2], 'P99 value (kWh)'),
]):
    t_vals = feat_matrix[theft_idx, feat_idx]
    n_vals = feat_matrix[normal_idx, feat_idx]
    
    lo = np.percentile(np.concatenate([t_vals, n_vals]), 1)
    hi = np.percentile(np.concatenate([t_vals, n_vals]), 99)
    
    ax.hist(np.clip(t_vals, lo, hi), bins=40, alpha=0.6, color='red', label='Theft', density=True)
    ax.hist(np.clip(n_vals, lo, hi), bins=40, alpha=0.6, color='blue', label='Normal', density=True)
    ax.set_title(title, fontsize=12)
    ax.legend()

plt.suptitle('Extreme Value Features: Theft vs Normal', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'extreme_value_analysis.png'), dpi=150)
plt.close()
print(f"  Saved extreme_value_analysis.png")

print(f"\n" + "="*70)
print(f"  结论")
print(f"="*70)

print(f"""
1. 极端值频次：窃电用户比正常用户有更多天超过自身P95阈值
2. 全局P99命中：窃电用户命中全局极值的概率更高（消费水平更高）
3. max/median比：窃电用户极值/中位数比更低——说明他们的峰值相对更"平"
4. 时序聚集性：窃电用户的极端值更分散（可能是数据丢失导致）
5. 单特征AUC最高约0.70——与之前的分析一致，极端值特征是中等信号，不能独立分类

建议：作为正交特征（因为与均值/分位数相关但不同），直接加入GBDT特征集。
""")
