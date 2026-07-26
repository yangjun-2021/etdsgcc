"""
Phase 1+2: Fast Behavior-Driven Multi-Expert Model

Optimized for speed: all features vectorized, minimal per-user loops.
Only includes features that can be computed in < 5 min total.
"""
import os, time, warnings
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
import lightgbm as lgb

SEED = 42; np.random.seed(SEED)

# ======== DATA LOADING ========
def load_data():
    df = pd.read_csv('data/raw_data.csv')
    dc = [c for c in df.columns if '/' in str(c) and len(str(c)) <= 10]
    raw = df[dc].values.astype(float); flags = df['FLAG'].values.astype(np.int32)
    return raw, flags, dc

# ======== VECTORIZED FEATURE EXTRACTION ========
def extract_features(filled, raw, flags, date_cols):
    """Vectorized feature extraction. ~20 core behavioral features."""
    n_users, n_days = filled.shape
    print(f"[Phase1] Extracting {n_users} users, {n_days} days...")
    
    miss_mask = np.isnan(raw)
    valid = ~miss_mask
    
    # === Basic stats (vectorized) ===
    mean_cons = np.nanmean(raw, axis=1)
    std_cons = np.nanstd(raw, axis=1)
    cv_daily = std_cons / (np.nan_to_num(mean_cons, nan=0) + 1e-6)
    zero_rate = np.nan_to_num((raw == 0) & valid, nan=0).mean(axis=1)
    miss_rate = miss_mask.mean(axis=1)
    
    # === Rankings ===
    global_rank = stats.rankdata(np.nan_to_num(mean_cons, nan=0)) / n_users
    
    # === CUSUM (vectorized: run on batch with threshold) ===
    # Approximate CUSUM: max cumulative deviation from mean
    centered = filled - filled.mean(axis=1, keepdims=True)
    cumsum_dev = np.cumsum(centered, axis=1)
    # Normalize by rolling std
    roll_std = np.sqrt(uniform_filter1d(centered ** 2, size=60, axis=1) + 1e-6)
    cusum_score = np.max(np.abs(cumsum_dev), axis=1) / (np.mean(roll_std, axis=1) + 1e-6)
    
    # === Changepoint features ===
    # Simple: mean of second half / mean of first half
    half = n_days // 2
    first_mean = np.nanmean(raw[:, :half], axis=1)
    second_mean = np.nanmean(raw[:, half:], axis=1)
    change_ratio = second_mean / (first_mean + 1e-6)
    change_persistence = np.mean(raw[:, half:] < (second_mean * 1.5).reshape(-1,1), axis=1)
    
    # === Anomaly days ===
    p10 = np.percentile(filled, 10, axis=1).reshape(-1, 1)
    anomaly_mask = filled < np.maximum(p10 * 0.5, 0.01)
    anomaly_ratio = anomaly_mask.mean(axis=1)
    
    # === Run length of anomaly periods ===
    diffs = np.diff(anomaly_mask.astype(int), axis=1)
    # Simplified: std of anomaly indicator (high std = alternating pattern)
    anom_std = np.std(anomaly_mask.astype(float), axis=1)
    
    # === ACF features ===
    acf1 = np.zeros(n_users)
    for i in range(min(n_users, 5000)):  # Sample for speed
        c = filled[i] - filled[i].mean()
        if np.std(c) > 1e-6:
            a = np.corrcoef(c[:-1], c[1:])[0, 1]
            acf1[i] = 0 if np.isnan(a) else a
    # Use global mean for rest (faster)
    if n_users > 5000:
        acf1[5000:] = np.mean(acf1[:5000])
    
    # === STL-like residual ===
    # Period=7 rolling mean → detrended → residual
    trend7 = uniform_filter1d(filled, size=7, axis=1, mode='nearest')
    residual = filled - trend7
    res_std = np.std(residual, axis=1) / (np.std(filled, axis=1) + 1e-6)
    res_skew = np.zeros(n_users)
    for i in range(min(n_users, 5000)):
        if np.std(residual[i]) > 1e-6:
            res_skew[i] = stats.skew(residual[i])
    if n_users > 5000:
        res_skew[5000:] = np.mean(res_skew[:5000])
    
    # === Periodicity ===
    # Weekly autocorrelation (lag 7)
    acf7 = np.zeros(n_users)
    for i in range(min(n_users, 5000)):
        c = filled[i] - filled[i].mean()
        if len(c) > 7 and np.std(c) > 1e-6:
            a = np.corrcoef(c[:-7], c[7:])[0, 1]
            acf7[i] = 0 if np.isnan(a) else a
    if n_users > 5000:
        acf7[5000:] = np.mean(acf7[:5000])
    
    # === Near-zero ===
    near_zero = (filled < 0.01).mean(axis=1)
    
    # === Peer deviation (cluster-based) ===
    # Fast clustering
    weekly = filled[:, :(n_days//7)*7].reshape(n_users, -1, 7).mean(axis=2)
    weekly_norm = (weekly - weekly.mean(axis=1, keepdims=True)) / (weekly.std(axis=1, keepdims=True) + 1e-6)
    weekly_norm = np.nan_to_num(weekly_norm, nan=0)
    km = KMeans(n_clusters=6, random_state=SEED, n_init='auto')
    cluster_labels = km.fit_predict(weekly_norm)
    
    # Peer mean per user
    peer_mean = np.zeros(n_users)
    for c in range(6):
        mask = cluster_labels == c
        if mask.sum() > 0:
            peer_mean[mask] = np.median(mean_cons[mask])
    peer_dev = mean_cons / (peer_mean + 1e-6)
    
    # === Monthly volatility ===
    months = np.array([int(str(c).split('/')[0]) for c in date_cols])
    monthly_cv = np.zeros(n_users)
    monthly_range = np.zeros(n_users)
    for i in range(n_users):
        cvs = []
        for m in range(1, 13):
            mi = np.where(months == m)[0]
            if len(mi) > 5:
                mv = filled[i, mi]
                cvs.append(np.std(mv) / (np.mean(mv) + 1e-6))
        if cvs:
            monthly_cv[i] = np.mean(cvs)
            monthly_range[i] = np.max(cvs) - np.min(cvs)
    
    # === Assemble feature matrix ===
    X = np.column_stack([
        mean_cons, std_cons, cv_daily, zero_rate, miss_rate,
        global_rank, change_ratio, change_persistence,
        cusum_score, anomaly_ratio, anom_std,
        acf1, acf7, res_std, res_skew,
        near_zero, peer_dev,
        monthly_cv, monthly_range,
    ])
    feature_names = [
        'mean', 'std', 'cv', 'zero_rate', 'miss_rate',
        'global_rank', 'change_ratio', 'change_persistence',
        'cusum_score', 'anomaly_ratio', 'anom_std',
        'acf1', 'acf7', 'res_std', 'res_skew',
        'near_zero', 'peer_dev',
        'monthly_cv', 'monthly_range',
    ]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  {len(feature_names)} features extracted")
    
    return X.astype(np.float32), feature_names, cluster_labels, global_rank

# ======== WEAK LABEL ASSIGNMENT ========
def assign_weak_labels(X, feature_names, flags):
    idx = {n: i for i, n in enumerate(feature_names)}
    theft_mask = flags == 1
    n = len(flags)
    
    e1 = theft_mask & (X[:, idx['cusum_score']] > 0.3) & (X[:, idx['change_persistence']] > 0.7)
    e2 = theft_mask & (X[:, idx['near_zero']] > 0.9) | (X[:, idx['global_rank']] < 0.05)
    e3 = theft_mask & (X[:, idx['anomaly_ratio']] > 0.1) & (X[:, idx['anom_std']] > 0.15)
    e4 = theft_mask.copy()
    
    print(f"  E1(突变)={e1.sum()} E2(持续)={e2.sum()} E3(间歇)={e3.sum()} E4(通用)={e4.sum()}")
    return e1, e2, e3, e4

# ======== EXPERT TRAINING ========
def train_expert(features, y_pos_mask, expert_name):
    n = len(y_pos_mask)
    y_expert = y_pos_mask.astype(int)
    n_pos = y_expert.sum()
    n_neg = n - n_pos
    
    if n_pos < 10:
        print(f"  {expert_name}: <10 positives, skip")
        return np.zeros(n)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(n)
    pw = n_neg / n_pos
    
    for fi,(ti,vi) in enumerate(skf.split(features, y_expert)):
        m = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.03,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pw, random_state=SEED, verbose=-1)
        m.fit(features[ti], y_expert[ti], eval_set=[(features[vi], y_expert[vi])],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        oof[vi] = m.predict_proba(features[vi])[:,1]
    
    bf, bt = 0, 0.5
    for th in np.arange(0.3, 0.7, 0.005):
        f1 = f1_score(y_expert, (oof > th).astype(int), zero_division=0)
        if f1 > bf: bf, bt = f1, th
    auc = roc_auc_score(y_expert, oof)
    print(f"  {expert_name}: AUC={auc:.4f} F1={bf:.4f} th={bt:.3f} (pos={n_pos})")
    return oof

# ======== MAIN ========
def run():
    t0 = time.time()
    print("=" * 60)
    print("  Phase 1+2: Behavior-Driven Multi-Expert")
    print("=" * 60)
    
    raw, flags, dc = load_data()
    filled = np.nan_to_num(raw, nan=0)
    # Simple median fill for NaN
    user_med = np.nanmedian(raw, axis=1)
    user_med = np.nan_to_num(user_med, nan=0)
    for i in range(len(raw)):
        miss = np.isnan(raw[i])
        if miss.any(): filled[i, miss] = user_med[i]
    
    X, fnames, clusters, granks = extract_features(filled, raw, flags, dc)
    
    e1, e2, e3, e4 = assign_weak_labels(X, fnames, flags)
    
    print("\n[Phase2] Training experts...")
    oofs = [
        train_expert(X, e1, "E1_突变持续"),
        train_expert(X, e2, "E2_持续异常"),
        train_expert(X, e3, "E3_间歇异常"),
        train_expert(X, e4, "E4_通用兜底"),
    ]
    
    # Meta-learner
    X_meta = np.column_stack(oofs)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_meta = np.zeros(len(flags))
    for fi,(ti,vi) in enumerate(skf.split(X_meta, flags)):
        m = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED)
        m.fit(X_meta[ti], flags[ti])
        oof_meta[vi] = m.predict_proba(X_meta[vi])[:,1]
    
    bf, bt = 0, 0.5
    for th in np.arange(0.3, 0.7, 0.005):
        f1 = f1_score(flags, (oof_meta > th).astype(int), zero_division=0)
        if f1 > bf: bf, bt = f1, th
    
    auc = roc_auc_score(flags, oof_meta)
    pred = (oof_meta > bt).astype(int)
    tp = ((pred==1)&(flags==1)).sum()
    fp = ((pred==1)&(flags==0)).sum()
    fn = ((pred==0)&(flags==1)).sum()
    
    print(f"\n{'='*60}")
    print(f"  FINAL (Meta-Learner)")
    print(f"{'='*60}")
    print(f"  AUC={auc:.4f} F1={bf:.4f} Rec={recall_score(flags,pred):.4f} Prec={precision_score(flags,pred):.4f} th={bt:.3f}")
    print(f"  TP={tp} FP={fp} FN={fn}")
    print(f"  V225 ref: F1=0.8457 | Our best: F1=0.8544")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")
    
    np.savez('output/phase12_fast.npz', oof_meta=oof_meta, oofs=np.array(oofs), flags=flags)
    return oof_meta, flags

if __name__ == '__main__':
    run()
