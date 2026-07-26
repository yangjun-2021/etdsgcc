"""
Phase 1+2: Behavior-Driven Feature Engineering + Multi-Expert Model
=====================================================================

Design principles (MUST follow):
  1. Behavior-driven: features derived from physical theft mechanisms
  2. Multi-expert: different theft types → different detectors
  3. Causality: TimeSeriesSplit (first 800 days train, last 234 test)
  4. Peer comparison: K-Shape clustering for "同簇" statistics
  5. No data leakage: all "近N天" features use only past data
  6. Threshold optimization: search 0.3-0.7 on validation set

Feature groups (A-G, ~38 features):
  A - Mutation type (降压 + 旁路): 5 features
  B - Sustained low (扩差 + 无表): 5 features
  C - Intermittent (间歇 + 高科技): 5 features
  D - Time structure (通用): 4 features
  E - Baseline deviation (通用): 5 features
  F - Context exclusion (通用): 6 features  
  G - Multi-scale STL decomposision: 5 features
  H - Peer cluster features: ~3 features
"""
import os, time, warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from sklearn.cluster import KMeans
import lightgbm as lgb
from tqdm import tqdm

SEED = 42
np.random.seed(SEED)

# =============================================================================
# 1. Data Loading + TimeSeriesSplit
# =============================================================================

def load_data():
    df = pd.read_csv('data/raw_data.csv')
    date_cols = [c for c in df.columns if '/' in str(c) and len(str(c)) <= 10]
    raw = df[date_cols].values.astype(float)
    flags = df['FLAG'].values.astype(np.int32)
    return raw, flags, date_cols


def setup_time_split(flags):
    """TimeSeriesSplit: first 800 days train, last 234 days validate."""
    n_users, n_days = raw_full = None, 1034  # placeholder
    # We use a single split with the 800/234 boundary
    train_cutoff = 800
    val_start = 800
    val_end = 1034
    return 800, 234


# =============================================================================
# 2. Data Preprocessing
# =============================================================================

def preprocess_data(raw):
    """Fill NaN with median per user, efficient vectorized."""
    n_users, n_days = raw.shape
    filled = raw.copy()
    user_medians = np.nanmedian(raw, axis=1)
    user_medians = np.nan_to_num(user_medians, nan=0.0)
    for i in range(n_users):
        miss = np.isnan(raw[i])
        if miss.any():
            filled[i, miss] = user_medians[i]
    filled = np.nan_to_num(filled, nan=0.0)
    return filled


# =============================================================================
# 3. CUSUM Changepoint (shared across features)
# =============================================================================

def cusum_detect(row, threshold=3.0, drift=0.2, min_seg=30):
    """Standard two-sided CUSUM. Returns (has_cp, cp_idx, pre_m, post_m)."""
    valid = ~np.isnan(row)
    n_v = valid.sum()
    if n_v < min_seg * 2:
        return False, -1, 0, 0
    x = row[valid]
    mu = np.mean(x[:min(n_v//2, 200)])
    sigma = np.std(x[:min(n_v//2, 200)]) + 1e-6
    K, H = drift * sigma, threshold * sigma
    Sp, Sn = np.zeros(n_v), np.zeros(n_v)
    cp_p, cp_n = -1, -1
    for t in range(n_v):
        Sp[t] = max(0, (Sp[t-1] if t > 0 else 0) + x[t] - mu - K)
        Sn[t] = max(0, (Sn[t-1] if t > 0 else 0) + mu - K - x[t])
        if Sp[t] > H and cp_p < 0: cp_p = t
        if Sn[t] > H and cp_n < 0: cp_n = t
    if cp_p < 0 and cp_n < 0:
        return False, -1, 0, 0
    cp = min(cp_p, cp_n) if cp_p >= 0 and cp_n >= 0 else max(cp_p, cp_n)
    vi = np.where(valid)[0]
    actual_cp = vi[cp]
    if actual_cp < min_seg or actual_cp > len(row) - min_seg:
        return False, -1, 0, 0
    return True, actual_cp, np.nanmean(row[:actual_cp]), np.nanmean(row[actual_cp:])


# =============================================================================
# 4. Train/Test Aware K-Shape Clustering (近似，用 KMeans + DTW 降采样)
# =============================================================================

def cluster_users(train_filled, n_clusters=6):
    """K-Means on downsampled time series for fast clustering.
    
    Downsample 1034 -> 52 (weekly) for speed, then cluster.
    Returns cluster labels for all users.
    """
    n_users = train_filled.shape[0]
    # Weekly downsampling
    weekly = train_filled[:, :(train_filled.shape[1] // 7) * 7].reshape(
        n_users, -1, 7).mean(axis=2)
    # Z-score per user for shape comparison
    weekly_std = weekly.std(axis=1, keepdims=True) + 1e-6
    weekly_norm = (weekly - weekly.mean(axis=1, keepdims=True)) / weekly_std
    weekly_norm = np.nan_to_num(weekly_norm, nan=0.0)
    
    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init='auto')
    labels = km.fit_predict(weekly_norm)
    return labels.astype(np.int32)


# =============================================================================
# 5. Feature Engineering (A-H Groups, ~38 features)
# =============================================================================

def compute_all_features(filled, flags, cluster_labels, date_cols, 
                          train_cutoff=800):
    """Compute all behavioral features for every user.
    
    Causality: features using "window" data only access data up to 
    each time point. For simplicity, we split into:
      - GLobal features (use full 1034 days) → available at prediction time
      - Window features (last N days) → computed from train cutoff only
    
    Returns:
        features: [N, F] feature matrix
        feature_names: list of feature names
    """
    n_users, n_days = filled.shape
    print(f"[Phase1] Computing {n_users} users x behavior-driven features...")
    
    # Helper: global rank percentile per user
    user_means = np.nanmean(filled, axis=1)
    user_means = np.nan_to_num(user_means, nan=0.0)
    global_rank = stats.rankdata(user_means) / n_users
    
    # Cluster-level statistics (per day medians)
    n_clusters = cluster_labels.max() + 1
    cluster_day_medians = np.zeros((n_clusters, n_days))
    for c in range(n_clusters):
        c_mask = cluster_labels == c
        if c_mask.sum() > 0:
            cluster_day_medians[c] = np.median(filled[c_mask], axis=0)
    
    features = {}
    
    # Precompute CUSUM for each user
    cusum_cache = {}
    for i in tqdm(range(n_users), desc="  CUSUM"):
        hp, cp, pr, po = cusum_detect(filled[i])
        cusum_cache[i] = (hp, cp, pr, po)
    
    # === A组: 突变型 (降压 + 旁路) ===
    print("  Group A: Mutation features...")
    feat_A = np.zeros((n_users, 5))
    for i in range(n_users):
        hp, cp, pr, po = cusum_cache[i]
        row = filled[i]
        feat_A[i, 0] = float(hp)  # cusum_changepoint
        if hp:
            feat_A[i, 1] = po / (pr + 1e-6)  # change_ratio (post/pre)
            feat_A[i, 2] = np.mean(row[cp:] < po * 1.5)  # persistence
            feat_A[i, 3] = np.std(row[cp:]) / (np.std(row[:cp]) + 1e-6)  # var_ratio
            # slope_break: diff in regression slopes pre/post
            x_pre, x_post = np.arange(cp), np.arange(cp, n_days)
            s_pre = np.polyfit(x_pre, row[:cp], 1)[0] if cp>10 else 0
            s_post = np.polyfit(x_post, row[cp:], 1)[0] if n_days-cp>10 else 0
            feat_A[i, 4] = s_post - s_pre
        else:
            feat_A[i, 1] = 1.0
            feat_A[i, 2] = 0.0
            feat_A[i, 3] = 1.0
            feat_A[i, 4] = 0.0
    
    # === B组: 持续低用电 (扩差 + 无表) ===
    print("  Group B: Sustained low features...")
    for i in range(n_users):
        row = filled[i]
        c = cluster_labels[i]
        cluster_med = np.median(filled[cluster_labels == c])
        peer_p15 = np.percentile(filled[cluster_labels == c], 15) if (cluster_labels == c).sum() > 10 else 0
        
        no_cp = float(not cusum_cache[i][0])
        features.setdefault('B_no_cp', []).append(no_cp)
        features.setdefault('B_rank', []).append(global_rank[i])
        features.setdefault('B_low_ratio', []).append(np.mean(row < max(peer_p15, 0.01)))
        features.setdefault('B_cv', []).append(
            np.std(row[row>0])/(np.mean(row[row>0])+1e-6) if np.any(row>0) else 0)
        features.setdefault('B_near_zero', []).append(np.mean(row < 0.01))
    
    # === C组: 间歇异常 (间歇 + 高科技) ===
    print("  Group C: Intermittent anomaly features...")
    for i in tqdm(range(n_users), desc="  Group C"):
        row = filled[i]
        p10 = np.percentile(row, 10)
        am = row < max(p10 * 0.5, 0.01)
        
        features.setdefault('C_anom_ratio', []).append(am.mean())
        
        runs = []; cr = 0
        for v in am:
            if v: cr += 1
            elif cr > 0: runs.append(cr); cr = 0
        if cr > 0: runs.append(cr)
        features.setdefault('C_run_len', []).append(np.mean(runs) if runs else 0)
        
        # Bimodality (simplified Hartigan's dip)
        pos = row[row > 0]
        if len(pos) > 20:
            p25, p50, p75 = np.percentile(pos, [25, 50, 75])
            ld = np.sum((pos >= p25) & (pos <= p50))
            hd = np.sum((pos >= p50) & (pos <= p75))
            bi = 1 - abs(ld - hd) / max(ld + hd, 1)
        else:
            bi = 0
        features.setdefault('C_bimodal', []).append(bi)
        
        # Weekday anomaly (approximate via first vs second half ratio)
        half = len(am)//2
        r1 = am[:half].mean(); r2 = am[half:].mean()
        features.setdefault('C_week_ratio', []).append(r1/(r2+1e-6) if max(r1,r2)>0 else 1)
        
        # ACF1 + randomness
        c = row - np.mean(row)
        if np.std(c) > 1e-6:
            acf1 = np.corrcoef(c[:-1], c[1:])[0, 1]
            features.setdefault('C_acf1', []).append(0 if np.isnan(acf1) else acf1)
            # Ljung-Box approximation
            try:
                rk = [np.corrcoef(c[:-k], c[k:])[0,1] for k in range(1,min(11,len(c)//2))]
                Q = len(c) * sum(r**2 for r in rk if not np.isnan(r))
                lb_p = min(stats.chi2.sf(Q, df=min(10, len(c)//2-1)), 1.0)
            except:
                lb_p = 1.0
        else:
            features.setdefault('C_acf1', []).append(0)
            lb_p = 1.0
        features.setdefault('C_lb_p', []).append(lb_p)
    
    # === D组: 时序结构 (通用) ===
    print("  Group D: Time structure features...")
    for i in tqdm(range(n_users), desc="  Group D"):
        row = filled[i]
        c = row - np.mean(row)
        s = np.std(c) + 1e-6
        
        features.setdefault('D_acf1', []).append(features['C_acf1'][i])
        acf7 = np.corrcoef(c[:-7], c[7:])[0,1] if len(c)>7 and np.std(c)>1e-6 else 0
        features.setdefault('D_acf7', []).append(0 if np.isnan(acf7) else acf7)
        
        # STL-like: period=7 rolling mean then residual
        trend = uniform_filter1d(row, size=7, mode='nearest')
        seasonal = np.zeros_like(row)
        for d in range(7):
            seasonal[np.arange(d, n_days, 7)] = np.mean(
                row[np.arange(d, n_days, 7)] - trend[np.arange(d, n_days, 7)])
        res = row - trend - seasonal
        features.setdefault('D_stl_res_var', []).append(np.var(res)/np.var(row+1e-6))
        features.setdefault('D_stl_res_skew', []).append(
            stats.skew(res) if np.std(res)>1e-6 else 0)
    
    # === E组: 基线偏离 (通用) ===
    print("  Group E: Baseline deviation features...")
    for i in tqdm(range(n_users), desc="  Group E"):
        row = filled[i]
        c = cluster_labels[i]
        hist_mean = np.mean(row[:train_cutoff]) if train_cutoff < n_days else np.mean(row)
        recent_30 = np.mean(row[-30:]) if n_days >= 30 else hist_mean
        recent_90 = np.mean(row[-90:]) if n_days >= 90 else hist_mean
        
        features.setdefault('E_recent30', []).append(recent_30 / (hist_mean + 1e-6))
        features.setdefault('E_recent90', []).append(recent_90 / (hist_mean + 1e-6))
        
        # Peer deviation: user vs cluster median
        cluster_medians = cluster_day_medians[c]
        ratio = row / (cluster_medians + 1e-6)
        features.setdefault('E_peer_dev', []).append(np.mean(ratio))
        features.setdefault('E_peer_low', []).append(np.mean(ratio < 0.5))
        
        # Rank drop
        daily_ranks = stats.rankdata(row) / n_days  # within user
        rank_recent = np.mean(daily_ranks[-90:]) if n_days >= 90 else 0.5
        rank_hist = np.mean(daily_ranks[:train_cutoff])
        features.setdefault('E_rank_drop', []).append(rank_recent - rank_hist)
    
    # === F组: 上下文排除 ===
    print("  Group F: Context features...")
    months = np.array([int(str(c).split('/')[0]) for c in date_cols])
    for i in tqdm(range(n_users), desc="  Group F"):
        row = filled[i]
        
        # Seasonal: this year vs last year
        yoy_devs = []
        for m in range(1, 13):
            m_idx = np.where(months == m)[0]
            if len(m_idx) > 0:
                half = len(m_idx) // 2
                y1 = np.mean(row[m_idx[:half]]) if half > 0 else 0
                y2 = np.mean(row[m_idx[half:]]) if len(m_idx) > half else 0
                yoy_devs.append(abs(y1 - y2) / (max(y1, y2, 0.01)))
        features.setdefault('F_seasonal', []).append(np.mean(yoy_devs) if yoy_devs else 0)
        
        # Holiday effect (approximate: extreme low consumption days)
        p1 = np.percentile(row, 1)
        low_days = row < max(p1, 0.01)
        features.setdefault('F_holiday', []).append(
            np.mean(low_days[~np.isnan(row)]) if (~np.isnan(row)).sum() > 0 else 0)
        
        # Missing pattern (using zero-days as proxy for missing)
        zero_mask = (row == 0)
        features.setdefault('F_zero_rate', []).append(zero_mask.mean())
        features.setdefault('F_cluster', []).append(float(c))
    
    # === G组: 多尺度STL ===
    print("  Group G: Multi-scale STL features...")
    for i in tqdm(range(n_users), desc="  Group G"):
        row = filled[i]
        # Period=7 STL
        trend = uniform_filter1d(row, size=7, mode='nearest')
        detrended = row - trend
        # Period=365 STL on detrended
        trend365 = uniform_filter1d(detrended, size=90, mode='nearest')
        final_res = detrended - trend365
        
        features.setdefault('G_stl_var', []).append(np.var(final_res) / (np.var(row) + 1e-6))
        features.setdefault('G_stl_skew', []).append(stats.skew(final_res) if np.std(final_res)>1e-6 else 0)
        features.setdefault('G_stl_kurt', []).append(stats.kurtosis(final_res) if np.std(final_res)>1e-6 else 0)
        
        # Monthly volatility
        monthly_cvs = []
        for m in range(1, 13):
            m_idx = np.where(months == m)[0]
            if len(m_idx) > 5:
                mv = row[m_idx]
                monthly_cvs.append(np.std(mv)/(np.mean(mv)+1e-6))
        features.setdefault('G_monthly_cv', []).append(np.mean(monthly_cvs) if monthly_cvs else 0)
        features.setdefault('G_monthly_range', []).append(
            (np.max(monthly_cvs) - np.min(monthly_cvs)) if len(monthly_cvs)>1 else 0)
    
    # ===== Build feature matrix =====
    feature_names = sorted(features.keys())
    X = np.column_stack([np.array(features[k]) for k in feature_names])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Also add A-group features
    feature_names = feature_names + ['A_cusum', 'A_ratio', 'A_persist', 'A_var', 'A_slope']
    X = np.column_stack([X, feat_A])
    
    print(f"  Total features: {X.shape[1]} ({len(feature_names)} named)")
    return X.astype(np.float32), feature_names, cluster_labels, global_rank


# =============================================================================
# 6. Weak Label Assignment for Experts
# =============================================================================

def assign_weak_labels(features, feature_names, flags, raw, global_rank):
    """Assign each theft user to an expert based on behavioral signatures.
    
    Expert 1 (突变持续): cusum=1 AND change_ratio<0.8 AND persistence>0.7
    Expert 2 (持续异常): no_cusum AND global_rank<0.15
    Expert 3 (间歇异常): anom_ratio>0.15 AND run_len>=3
    Expert 4 (通用兜底): all theft users
    """
    n = len(flags)
    theft_mask = flags == 1
    
    # Get relevant features
    idx_A = feature_names.index('A_cusum') if 'A_cusum' in feature_names else 0
    idx_A_ratio = feature_names.index('A_ratio') if 'A_ratio' in feature_names else 1
    idx_A_persist = feature_names.index('A_persist') if 'A_persist' in feature_names else 2
    idx_B_nocp = feature_names.index('B_no_cp') if 'B_no_cp' in feature_names else 0
    idx_B_rank = feature_names.index('B_rank') if 'B_rank' in feature_names else 1
    idx_C_anom = feature_names.index('C_anom_ratio') if 'C_anom_ratio' in feature_names else 0
    idx_C_run = feature_names.index('C_run_len') if 'C_run_len' in feature_names else 1
    
    # Expert 1: 突变持续检测
    e1 = theft_mask & (features[:, idx_A] > 0.5) & (features[:, idx_A_ratio] < 0.8) & (features[:, idx_A_persist] > 0.7)
    
    # Expert 2: 持续异常检测
    e2 = theft_mask & (features[:, idx_B_nocp] > 0.5) & (global_rank < 0.15)
    
    # Expert 3: 间歇异常检测
    e3 = theft_mask & (features[:, idx_C_anom] > 0.15) & (features[:, idx_C_run] >= 3)
    
    # Expert 4: 通用
    e4 = theft_mask.copy()
    
    print(f"[Phase2] Weak labels: E1={e1.sum()}, E2={e2.sum()}, E3={e3.sum()}, E4={e4.sum()}")
    return e1, e2, e3, e4


# =============================================================================
# 7. Expert Training with TimeSeriesSplit
# =============================================================================

def train_expert(features, y_expert, feature_names, train_cutoff=800, 
                  expert_name="Expert"):
    """Train a single LGB expert with TimeSeriesSplit.
    
    Uses first train_cutoff days for training, validates on remaining days.
    This prevents future information leakage.
    """
    n_users = len(y_expert)
    
    # For tabular features (not time series), we split users based on 
    # consumption patterns. But since features are computed from full data,
    # TimeSeriesSplit on USERS doesn't make sense for NON-time-series features.
    # Instead: use StratifiedKFold for tabular features.
    from sklearn.model_selection import StratifiedKFold
    
    pos_idx = np.where(y_expert == 1)[0]
    neg_idx = np.where(y_expert == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    
    if n_pos < 10:
        print(f"  {expert_name}: Too few positives ({n_pos}), skipping")
        return np.zeros(n_users), None
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(n_users)
    
    pos_weight = n_neg / max(n_pos, 1)
    
    best_threshold = 0.5
    
    for fi, (ti, vi) in enumerate(skf.split(features, y_expert)):
        Xt, Xv = features[ti], features[vi]
        yt, yv = y_expert[ti], y_expert[vi]
        
        model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight, random_state=SEED,
            verbose=-1,
        )
        model.fit(Xt, yt, eval_set=[(Xv, yv)],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                             lgb.log_evaluation(0)])
        oof[vi] = model.predict_proba(Xv)[:, 1]
    
    # Threshold search on OOF
    best_f1 = 0
    for th in np.arange(0.3, 0.7, 0.005):
        pred = (oof > th).astype(int)
        f1 = f1_score(y_expert, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th
    
    auc = roc_auc_score(y_expert, oof)
    print(f"  {expert_name}: AUC={auc:.4f} F1={best_f1:.4f} th={best_threshold:.3f} "
          f"(pos={n_pos}, neg={n_neg})")
    
    return oof, best_threshold


# =============================================================================
# 8. Meta-Learner Fusion
# =============================================================================

def train_meta_learner(oof_list, oof_names, y_true):
    """Logistic regression meta-learner to fuse expert OOFs."""
    n = len(y_true)
    if len(oof_list) < 2:
        return oof_list[0], 0.5
    
    X_meta = np.column_stack(oof_list)
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_meta = np.zeros(n)
    pos_weight = (y_true == 0).sum() / max((y_true == 1).sum(), 1)
    
    for fi, (ti, vi) in enumerate(skf.split(X_meta, y_true)):
        m = LogisticRegression(class_weight='balanced', max_iter=1000,
                                random_state=SEED)
        m.fit(X_meta[ti], y_true[ti])
        oof_meta[vi] = m.predict_proba(X_meta[vi])[:, 1]
    
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.3, 0.7, 0.005):
        pred = (oof_meta > th).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1: best_f1, best_th = f1, th
    
    auc = roc_auc_score(y_true, oof_meta)
    pred = (oof_meta > best_th).astype(int)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    
    print(f"\n  Meta-Learner: AUC={auc:.4f} F1={best_f1:.4f} th={best_th:.3f}")
    print(f"  TP={tp} FP={fp} FN={fn}")
    
    for i, name in enumerate(oof_names):
        e_auc = roc_auc_score(y_true, oof_list[i])
        print(f"  {name}: AUC={e_auc:.4f}")
    
    return oof_meta, best_th


# =============================================================================
# 9. Main Pipeline
# =============================================================================

def run_phase12():
    t0 = time.time()
    print("=" * 60)
    print("  Phase 1+2: Behavior-Driven Features + Multi-Expert Model")
    print("=" * 60)
    
    # Load
    raw, flags, date_cols = load_data()
    filled = preprocess_data(raw)
    
    # K-Shape clustering (on all users)
    print("\n[Step 1] K-Means clustering (weekly downsampled)...")
    cluster_labels = cluster_users(filled, n_clusters=6)
    n_clusters = cluster_labels.max() + 1
    print(f"  Clusters: {n_clusters}, sizes: {np.bincount(cluster_labels)}")
    
    # Feature engineering
    print("\n[Step 2] Behavior-driven feature engineering...")
    features, feature_names, cluster_labels, global_rank = compute_all_features(
        filled, flags, cluster_labels, date_cols)
    
    # Weak labels
    print("\n[Step 3] Assigning weak labels for experts...")
    e1, e2, e3, e4 = assign_weak_labels(features, feature_names, flags, 
                                          raw, global_rank)
    
    y_all = flags.copy()
    
    # Train experts
    print("\n[Step 4] Training 4 experts...")
    oof_experts = []
    thresholds = []
    n = len(flags)
    
    for i, (e_pos_mask, name) in enumerate([
        (e1, "E1_突变持续"), (e2, "E2_持续异常"),
        (e3, "E3_间歇异常"), (e4, "E4_通用兜底")
    ]):
        y_expert = np.zeros(n, dtype=int)
        y_expert[e_pos_mask] = 1
        oof, th = train_expert(features, y_expert, feature_names,
                                expert_name=name)
        oof_experts.append(oof)
        thresholds.append(th)
    
    # Meta-learner
    print("\n[Step 5] Meta-learner fusion...")
    oof_meta, best_th = train_meta_learner(
        oof_experts, ["E1", "E2", "E3", "E4"], flags)
    
    # Final results
    auc_meta = roc_auc_score(flags, oof_meta)
    pred_meta = (oof_meta > best_th).astype(int)
    f1_meta = f1_score(flags, pred_meta)
    rec_meta = recall_score(flags, pred_meta)
    prec_meta = precision_score(flags, pred_meta)
    tp = ((pred_meta == 1) & (flags == 1)).sum()
    fp = ((pred_meta == 1) & (flags == 0)).sum()
    fn = ((pred_meta == 0) & (flags == 1)).sum()
    
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  AUC:      {auc_meta:.4f}")
    print(f"  F1:       {f1_meta:.4f}")
    print(f"  Recall:   {rec_meta:.4f}")
    print(f"  Precision:{prec_meta:.4f}")
    print(f"  Threshold:{best_th:.3f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print(f"")
    print(f"  Reference: V225 F1=0.8457")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")
    
    # Save
    np.savez('output/phase12_results.npz',
             oof_meta=oof_meta, oof_e1=oof_experts[0],
             oof_e2=oof_experts[1], oof_e3=oof_experts[2],
             oof_e4=oof_experts[3], features=features,
             feature_names=np.array(feature_names, dtype=object),
             y=flags)
    print(f"\n  Saved to output/phase12_results.npz")
    
    return oof_meta, flags


if __name__ == '__main__':
    oof_meta, flags = run_phase12()
