"""
Behavioral Enhancement for Super-GBDT

Absorbs data engineering insights from Phase 0 (behavioral profiling):
  1. CUSUM changepoint features (has_cp, change_ratio, persistence)
  2. Peer cluster deviation (K-Means cluster + user vs cluster median)
  3. Behavioral weak labels (E1/E2/E3 type indicators from Phase 0 rules)
  4. Multi-expert architecture: 4 experts × 307 features + different sample weights

Target: Improve Super-GBDT from AUC=0.9870/F1=0.853 to beyond.
"""
import os, time, glob
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings; warnings.filterwarnings('ignore')

import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from utils import seed_everything, best_f1_score

SEED = 42; seed_everything(SEED)
np.random.seed(SEED)

# ======== 1. CUSUM CHANGEPOINT (batch) ========
def compute_cusum_batch(filled, min_seg=30):
    """Batch CUSUM detection for all users.
    Returns: has_cp, cp, pre_mean, post_mean, change_ratio, persistence
    """
    n_users, n_days = filled.shape
    print(f"  CUSUM on {n_users} users...")
    
    has_cp = np.zeros(n_users, dtype=int)
    change_ratio = np.ones(n_users)
    persistence = np.zeros(n_users)
    cp_loc = np.full(n_users, -1, dtype=int)
    
    for i in range(n_users):
        row = filled[i]
        valid = row > 0
        n_v = valid.sum()
        if n_v < min_seg * 2: continue
        
        x = row[valid]
        mu = np.mean(x[:min(n_v//2, 200)])
        sigma = np.std(x[:min(n_v//2, 200)]) + 1e-6
        K, H = 0.2 * sigma, 3.0 * sigma
        
        Sp = np.zeros(n_v); Sn = np.zeros(n_v)
        cp_p = cp_n = -1
        for t in range(n_v):
            Sp[t] = max(0, (Sp[t-1] if t > 0 else 0) + x[t] - mu - K)
            Sn[t] = max(0, (Sn[t-1] if t > 0 else 0) + mu - K - x[t])
            if Sp[t] > H and cp_p < 0: cp_p = t
            if Sn[t] > H and cp_n < 0: cp_n = t
        
        if cp_p < 0 and cp_n < 0: continue
        cp = min(cp_p, cp_n) if cp_p >= 0 and cp_n >= 0 else max(cp_p, cp_n)
        vi = np.where(valid)[0]; actual_cp = vi[cp]
        if actual_cp < min_seg or actual_cp > n_days - min_seg: continue
        
        has_cp[i] = 1; cp_loc[i] = actual_cp
        pre = np.mean(row[:actual_cp])
        post = np.mean(row[actual_cp:])
        change_ratio[i] = post / (pre + 1e-6)
        persistence[i] = np.mean(row[actual_cp:] < post * 1.5 if post > 0 else row[actual_cp:] < 0.01)
    
    return has_cp, cp_loc, change_ratio, persistence

# ======== 2. CLUSTER + PEER DEVIATION ========
def compute_cluster_features(filled, n_clusters=6):
    """K-Means on weekly downsampled data. Returns cluster_id + peer_dev."""
    n_users, n_days = filled.shape
    weekly = filled[:, :(n_days//7)*7].reshape(n_users, -1, 7).mean(axis=2)
    wp = (weekly - weekly.mean(axis=1, keepdims=True)) / (weekly.std(axis=1, keepdims=True) + 1e-6)
    wp = np.nan_to_num(wp, nan=0)
    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init='auto')
    cluster_id = km.fit_predict(wp).astype(np.float32)
    
    user_mean = filled.mean(axis=1)
    peer_mean = np.zeros(n_users)
    for c in range(n_clusters):
        mask = cluster_id == c
        if mask.sum() > 0: peer_mean[mask] = np.median(user_mean[mask])
    peer_dev = user_mean / (peer_mean + 1e-6)
    
    return cluster_id, peer_dev


# ======== 3. BEHAVIORAL WEAK LABELS ========
def compute_weak_labels(filled, change_ratio, persistence, has_cp, 
                         peer_dev, n_users, n_days):
    """Behavioral type indicators from Phase 0 rules.
    
    E1 (突变): has_cp + ratio<0.8 + persist>0.7
    E2 (持续): near_zero>0.9 OR peer_dev<0.05
    E3 (间歇): anomaly ratio > 0.15 + intermittent pattern
    """
    near_zero = (filled < 0.01).mean(axis=1)
    p10 = np.percentile(filled, 10, axis=1)
    anom_ratio = (filled < np.maximum(p10.reshape(-1,1) * 0.5, 0.01)).mean(axis=1)
    
    e1 = ((has_cp == 1) & (change_ratio < 0.8) & (persistence > 0.7)).astype(float)
    e2 = ((near_zero > 0.9) | (peer_dev < 0.05)).astype(float)
    
    anom_mask = (anom_ratio > 0.15).astype(int)
    runs = []; cr = 0
    # Approximate run length via anomaly std
    anom_std = np.std((filled < np.maximum(p10.reshape(-1,1) * 0.5, 0.01)).astype(float), axis=1)
    e3 = ((anom_ratio > 0.15) & (anom_std > 0.15)).astype(float)
    
    return e1, e2, e3, near_zero, anom_ratio


# ======== 4. GBDT TRAINER (shared) ========
def train_gbdt(X, y, skf, model_label="GBDT"):
    """Standard GBDT ensemble training."""
    n = len(y); oof = np.zeros(n)
    for fi,(ti,vi) in enumerate(skf.split(X, y)):
        pw = (y[ti]==0).sum() / max((y[ti]==1).sum(), 1)
        m1 = lgb.LGBMClassifier(n_estimators=1000,max_depth=7,learning_rate=0.05,
            num_leaves=63,subsample=0.8,colsample_bytree=0.8,scale_pos_weight=pw,
            random_state=SEED,verbose=-1)
        m1.fit(X[ti],y[ti],eval_set=[(X[vi],y[vi])],
               callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(0)])
        m2 = xgb.XGBClassifier(n_estimators=1000,max_depth=7,learning_rate=0.05,
            subsample=0.8,colsample_bytree=0.8,scale_pos_weight=pw,
            tree_method='hist',random_state=SEED,verbosity=0)
        m2.fit(X[ti],y[ti],eval_set=[(X[vi],y[vi])],verbose=False)
        m3 = CatBoostClassifier(iterations=1000,depth=7,learning_rate=0.05,
            auto_class_weights='Balanced',random_seed=SEED,verbose=0)
        m3.fit(X[ti],y[ti],eval_set=(X[vi],y[vi]),early_stopping_rounds=50,verbose=0)
        oof[vi] = 0.4*m1.predict_proba(X[vi])[:,1] + 0.3*m2.predict_proba(X[vi])[:,1] + 0.3*m3.predict_proba(X[vi])[:,1]
        print(f'  {model_label} Fold{fi+1}: AUC={roc_auc_score(y[vi],oof[vi]):.4f}')
    return oof


def train_lgb_simple(X, y, skf, sample_weight=None, label=""):
    """Fast LGB-only training (for experts)."""
    n = len(y); oof = np.zeros(n)
    for fi,(ti,vi) in enumerate(skf.split(X, y)):
        pw = (y[ti]==0).sum() / max((y[ti]==1).sum(), 1)
        sw = sample_weight[ti] if sample_weight is not None else None
        m = lgb.LGBMClassifier(n_estimators=1000,max_depth=7,learning_rate=0.05,
            num_leaves=63,subsample=0.8,colsample_bytree=0.8,
            scale_pos_weight=pw,random_state=SEED+fi,verbose=-1)
        if sw is not None:
            m.fit(X[ti], y[ti], sample_weight=sw,
                  eval_set=[(X[vi], y[vi])],
                  callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(0)])
        else:
            m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])],
                  callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(0)])
        oof[vi] = m.predict_proba(X[vi])[:,1]
    return oof


# ======== MAIN ========
def run():
    t0 = time.time()
    print("="*60)
    print("  Behavioral Enhancement for Super-GBDT")
    print("="*60)
    
    # Load raw data for behavioral features
    print("\n[1] Loading raw data...")
    import pandas as pd
    df = pd.read_csv('data/raw_data.csv')
    dc = [c for c in df.columns if '/' in str(c) and len(str(c)) <= 10]
    raw = df[dc].values.astype(float); y = df['FLAG'].values.astype(np.int32)
    filled = np.nan_to_num(raw, nan=0)
    user_med = np.nan_to_num(np.nanmedian(raw, axis=1), nan=0)
    for i in range(len(raw)):
        miss = np.isnan(raw[i])
        if miss.any(): filled[i, miss] = user_med[i]
    
    # Load existing 307 features AND OOF stack (Super-GBDT config)
    print("[2] Loading 307-dim + OOF stack features...")
    d = np.load('output/sgcc_preprocessed.npz')
    stat = np.nan_to_num(d['stat_features'], nan=0.0, posinf=0.0, neginf=0.0)
    im = d['impute_mask']; mr = im.mean(axis=1).reshape(-1, 1)
    
    # Load V71 OOFs (key Super-GBDT ingredients)
    OD = r'D:\Project\ThiefElectricity\output'
    v71 = np.load(sorted(glob.glob(f'{OD}/v71_oofs_*.npz'), reverse=True)[0], allow_pickle=True)
    v71_oofs = np.column_stack([v71['lgb'], v71['xgb'], v71['cat'], v71['tcn'], v71['innov']])
    
    # External OOFs
    ext_oofs = np.column_stack([
        np.load(sorted(glob.glob(f'{OD}/v213_results_*.npz'), reverse=True)[0], allow_pickle=True)['oof_v213'],
        np.load(sorted(glob.glob(f'{OD}/v219_results_*.npz'), reverse=True)[0], allow_pickle=True)['oof_final'],
        np.load(sorted(glob.glob(f'{OD}/v225_results_*.npz'), reverse=True)[0], allow_pickle=True)['oof_final'],
        np.load(sorted(glob.glob(f'{OD}/v216_results_*.npz'), reverse=True)[0], allow_pickle=True)['oof_final'],
    ])
    
    # Our OOFs
    our_oofs = np.load('output/tcn_kd_results.npz')
    our_oofs_s = np.column_stack([our_oofs['oof_tcn_kd'], our_oofs['oof_stacker']])
    
    # Super feature matrix (same as Super-GBDT)
    X_base = np.column_stack([stat, mr, v71_oofs, ext_oofs, our_oofs_s])
    X_base = np.nan_to_num(X_base, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    print(f"  Super-GBDT base: {X_base.shape} (307+1+5+4+2={stat.shape[1]+1+5+4+2})")
    
    # Behavioral enhancements
    print("[3] Computing behavioral enhancements...")
    has_cp, cp_loc, change_ratio, persistence = compute_cusum_batch(filled)
    cluster_id, peer_dev = compute_cluster_features(filled)
    e1, e2, e3, near_zero, anom_ratio = compute_weak_labels(
        filled, change_ratio, persistence, has_cp, peer_dev, len(y), filled.shape[1])
    
    behavioral = np.column_stack([
        has_cp, change_ratio, persistence, near_zero, anom_ratio,
        peer_dev, cluster_id, e1, e2, e3
    ]).astype(np.float32)
    behavioral = np.nan_to_num(behavioral, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Behavioral features: {behavioral.shape}")
    
    # Enhanced Super-GBDT
    print("\n[4] Training Enhanced Super-GBDT (307+10=317 dims)...")
    X_enhanced = np.column_stack([X_base, behavioral]).astype(np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_base = train_gbdt(X_enhanced, y, skf, "GBDT+Behav")
    
    auc_b = roc_auc_score(y, oof_base)
    f1_b, th_b, rec_b, prec_b = best_f1_score(y, oof_base)
    tp_b = ((oof_base > th_b) & (y == 1)).sum()
    fp_b = ((oof_base > th_b) & (y == 0)).sum()
    fn_b = ((oof_base <= th_b) & (y == 1)).sum()
    print(f"\n  Enhanced GBDT: AUC={auc_b:.4f} F1={f1_b:.4f} Rec={rec_b:.4f} Prec={prec_b:.4f}")
    print(f"  TP={tp_b} FP={fp_b} FN={fn_b}")
    
    # Multi-Expert with 307 features + behavioral labels as sample_weight
    print("\n[5] Training Multi-Expert (307 features, behavioral weights)...")
    
    # E1: weight mutation-type users higher
    w1 = np.ones(len(y)); w1[y == 1] *= 1.0 + e1[y == 1] * 3.0
    oof_e1 = train_lgb_simple(X_base, y, skf, sample_weight=w1, label="E1_突变加权")
    
    # E2: weight sustained-low users higher  
    w2 = np.ones(len(y)); w2[y == 1] *= 1.0 + e2[y == 1] * 3.0
    oof_e2 = train_lgb_simple(X_base, y, skf, sample_weight=w2, label="E2_持续加权")
    
    # E3: weight intermittent users higher
    w3 = np.ones(len(y)); w3[y == 1] *= 1.0 + e3[y == 1] * 3.0
    oof_e3 = train_lgb_simple(X_base, y, skf, sample_weight=w3, label="E3_间歇加权")
    
    # E4: uniform (same as base)
    oof_e4 = train_lgb_simple(X_base, y, skf, label="E4_均匀")
    
    # Meta-learner fusion
    print("\n[6] Meta-Learner fusion...")
    X_meta = np.column_stack([oof_e1, oof_e2, oof_e3, oof_e4, e1, e2, e3])
    oof_meta = np.zeros(len(y))
    for fi,(ti,vi) in enumerate(skf.split(X_meta, y)):
        m = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED)
        m.fit(X_meta[ti], y[ti])
        oof_meta[vi] = m.predict_proba(X_meta[vi])[:,1]
    
    auc_m = roc_auc_score(y, oof_meta)
    f1_m, th_m, rec_m, prec_m = best_f1_score(y, oof_meta)
    tp_m = ((oof_meta > th_m) & (y == 1)).sum()
    fp_m = ((oof_meta > th_m) & (y == 0)).sum()
    fn_m = ((oof_meta <= th_m) & (y == 1)).sum()
    
    # ======== FINAL COMPARISON ========
    sep = "=" * 60
    print(f"\n{sep}")
    print("  FINAL COMPARISON")
    print(sep)
    print(f"  {'Model':<25s}  {'AUC':>7s}  {'F1':>7s}  {'Rec':>7s}  {'Prec':>7s}  {'TP':>6s}  {'FP':>6s}  {'FN':>6s}")
    print("  " + "-" * 76)
    
    results = [
        ("Original Super-GBDT", 0.9870, 0.8527, 0.8503, 0.8551, 3074, 521, 541),
        ("Enhanced GBDT+Behav", auc_b, f1_b, rec_b, prec_b, int(tp_b), int(fp_b), int(fn_b)),
        ("Multi-Expert Meta", auc_m, f1_m, rec_m, prec_m, int(tp_m), int(fp_m), int(fn_m)),
    ]
    
    for name, auc, f1, rec, prec, tp, fp, fn in results:
        print(f"  {name:<25s}  {auc:.4f}  {f1:.4f}  {rec:.4f}  {prec:.4f}  {tp:6d}  {fp:6d}  {fn:6d}")
    
    # Delta
    d_auc = auc_b - 0.9870
    d_f1 = f1_b - 0.8527
    d_auc_m = auc_m - 0.9870
    d_f1_m = f1_m - 0.8527
    print(f"\n  Enhanced vs Original: dAUC={d_auc:+.4f} dF1={d_f1:+.4f}")
    print(f"  Multi-Expert vs Original: dAUC={d_auc_m:+.4f} dF1={d_f1_m:+.4f}")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")
    
    np.savez('output/behavior_enhanced.npz',
             oof_base=oof_base, oof_meta=oof_meta,
             oof_e1=oof_e1, oof_e2=oof_e2, oof_e3=oof_e3, oof_e4=oof_e4,
             e1=e1, e2=e2, e3=e3, y=y)
    return oof_base, oof_meta

if __name__ == '__main__':
    run()
