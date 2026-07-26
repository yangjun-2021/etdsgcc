"""
Self-Contained Pipeline: Pure Raw Data → F1 ≥ 0.86
=====================================================
Zero external dependencies. Only input: data/raw_data.csv

Pipeline design:
  Layer 1 (5 base models): different feature subsets → 5 diverse OOFs
  Layer 2 (3 meta models): use Layer 1 OOFs as features → 3 deeper OOFs
  Layer 3 (stacking): LR meta-learner on all 8 OOFs
"""
import os, time, random, warnings
import numpy as np, pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')
np.random.seed(42); random.seed(42)

from config import DATA_DIR
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb, xgboost as xgb

OUT = 'output'
os.makedirs(OUT, exist_ok=True)
t_start = time.time()
S = '-' * 60

# ═══════════════════════════════════════════════════════════════════
# 1. LOAD RAW DATA & BUILD FEATURE POOL
# ═══════════════════════════════════════════════════════════════════
print(f'{S}')
print('  [1/5] Building feature pool from raw data')
print(f'{S}')

df = pd.read_csv(os.path.join(DATA_DIR, 'raw_data.csv'))
dc = [c for c in df.columns if '/' in str(c) and len(str(c)) <= 10]
raw = df[dc].values.astype(float)
y = df['FLAG'].values.astype(int)
n = len(y)
nd = raw.shape[1]
half = nd // 2
filled = np.nan_to_num(raw, nan=0)
print(f'  Raw: {raw.shape}  Theft={y.sum()}/{n} ({y.mean()*100:.1f}%)  Missing={np.isnan(raw).mean()*100:.1f}%')

# ─── Feature groups ───
def build_all_features():
    """Build comprehensive feature pool from raw data only."""
    all_groups = {}

    # G1: Per-user basic statistics (on log1p-transformed, NaN-filled raw)
    filled_clean = np.nan_to_num(raw, nan=0)
    log_data = np.log1p(np.maximum(filled_clean, 0))
    impute_mask = np.isnan(raw)

    stats = np.zeros((n, 0), dtype=np.float32)
    for fn_name, fn in [
        ('mean', lambda x: np.nanmean(x, axis=1)),
        ('std', lambda x: np.nanstd(x, axis=1)),
        ('median', lambda x: np.nanmedian(x, axis=1)),
        ('min', lambda x: np.nanmin(x, axis=1)),
        ('max', lambda x: np.nanmax(x, axis=1)),
        ('skew', lambda x: np.nan_to_num(np.apply_along_axis(
            lambda r: np.nan if np.all(np.isnan(r)) else (
                np.nanmean((r - np.nanmean(r))**3) / (np.nanstd(r)**3 + 1e-6)
            ), 1, x), nan=0)),
    ]:
        arr = fn(log_data).reshape(-1, 1)
        stats = np.column_stack([stats, np.nan_to_num(arr, nan=0)]) if stats.shape[1] > 0 else np.nan_to_num(arr, nan=0).reshape(-1, 1)

    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        arr = np.nanpercentile(log_data, q, axis=1).reshape(-1, 1)
        stats = np.column_stack([stats, np.nan_to_num(arr, nan=0)])
    all_groups['Stats'] = stats.astype(np.float32)

    # G2: Missing pattern features
    miss = impute_mask.astype(float)
    miss_feats = [miss.mean(axis=1)]
    for ss, se in [(0, nd//4), (nd//4, nd//2), (nd//2, 3*nd//4), (3*nd//4, nd),
                    (0, half), (half, nd)]:
        miss_feats.append(miss[:, ss:se].mean(axis=1))
    mr = np.zeros(n)
    for i in range(n):
        runs, cr = [], 0
        for m in impute_mask[i]:
            if m: cr += 1
            else:
                if cr > 0: runs.append(cr); cr = 0
        if cr > 0: runs.append(cr)
        mr[i] = max(runs) if runs else 0
    miss_feats.append(mr)
    miss_feats.append(impute_mask.sum(axis=1) / nd)
    all_groups['Missing'] = np.nan_to_num(np.column_stack([f.reshape(-1, 1) for f in miss_feats]), nan=0).astype(np.float32)

    # G3: PAA multi-resolution
    paa_list = []
    for n_seg in [25, 50, 100]:
        seg = nd / n_seg
        for i in range(n_seg):
            s = int(round(i * seg))
            e = int(round((i + 1) * seg))
            if e > s:
                val = np.nanmean(raw[:, s:e], axis=1)
            elif s < nd:
                val = raw[:, s]
            else:
                val = np.zeros(n)
            paa_list.append(val)
    all_groups['PAA'] = np.nan_to_num(np.column_stack([p.reshape(-1, 1) for p in paa_list]), nan=0).astype(np.float32)

    # G4: Temporal profiles (monthly + DOW + trend + seasonal)
    temp_list = []
    # Monthly
    mi_arr = np.digitize(np.array([j % 365 for j in range(nd)]), np.linspace(0, 365, 13)) - 1
    for m in range(12):
        mm = (mi_arr == m)
        temp_list.append(np.nanmean(filled[:, mm], axis=1) if mm.sum() > 0 else np.zeros(n))
    # Day-of-week
    dow_arr = np.array([j % 7 for j in range(nd)])
    for d in range(7):
        dd = (dow_arr == d)
        temp_list.append(np.nanmean(filled[:, dd], axis=1) if dd.sum() > 0 else np.zeros(n))
    # Short-term trends
    for w in [7, 14, 30, 60, 90]:
        if nd >= w:
            fw = filled[:, :w].mean(axis=1)
            lw = filled[:, -w:].mean(axis=1)
            temp_list.append(np.where(np.abs(fw) > 1e-6, (lw - fw) / (np.abs(fw) + 1e-6), 0))
            temp_list.append(np.nanmean(np.diff(filled[:, -w:], axis=1), axis=1))
    # Seasonal
    for period in [7, 30, 90]:
        if nd >= period * 3:
            for offset in range(period):
                idx = np.arange(offset, nd, period)
                if len(idx) > 0:
                    temp_list.append(np.nanmean(filled[:, idx], axis=1) - np.nanmean(filled, axis=1))
    # Global quantiles
    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        temp_list.append(np.nanpercentile(filled, q, axis=1))
    all_groups['Temporal'] = np.nan_to_num(np.column_stack([t.reshape(-1, 1) for t in temp_list]), nan=0).astype(np.float32)

    # G5: Zero/consumption ratio features
    zero_feats = []
    zero_feats.append((filled < 0.01).mean(axis=1))
    zero_feats.append((log_data < 0.1).mean(axis=1))
    all_groups['ZeroRatio'] = np.nan_to_num(np.column_stack([z.reshape(-1, 1) for z in zero_feats]), nan=0).astype(np.float32)

    # Build 4 feature sets with different group combinations (diversity!)
    all_keys = list(all_groups.keys())
    X_all = np.column_stack([all_groups[k] for k in all_keys])
    X_all = np.nan_to_num(X_all, nan=0, posinf=0, neginf=0).astype(np.float32)
    
    X_no_stats = np.column_stack([all_groups[k] for k in all_keys if k != 'Stats'])
    X_no_stats = np.nan_to_num(X_no_stats, nan=0, posinf=0, neginf=0).astype(np.float32)
    
    X_no_paa = np.column_stack([all_groups[k] for k in all_keys if k != 'PAA'])
    X_no_paa = np.nan_to_num(X_no_paa, nan=0, posinf=0, neginf=0).astype(np.float32)

    X_no_temp = np.column_stack([all_groups[k] for k in all_keys if k != 'Temporal'])
    X_no_temp = np.nan_to_num(X_no_temp, nan=0, posinf=0, neginf=0).astype(np.float32)

    print(f'  Feature groups: {[(k, v.shape[1]) for k, v in sorted(all_groups.items())]}')
    print(f'  X_all: {X_all.shape[1]}d  X_noStats: {X_no_stats.shape[1]}d  X_noPAA: {X_no_paa.shape[1]}d  X_noTemp: {X_no_temp.shape[1]}d')

    return X_all, X_no_stats, X_no_paa, X_no_temp, y, all_groups


X_all, X_no_stats, X_no_paa, X_no_temp, y, groups = build_all_features()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ═══════════════════════════════════════════════════════════════════
# 2. LAYER 1: 5 diverse base models on different feature subsets
# ═══════════════════════════════════════════════════════════════════
print(f'\n{S}')
print('  [2/5] Layer 1: 5 diverse base models')
print(f'{S}')

def train_cv_oof(X, y, model_name, factory_fn, skf):
    oof = np.zeros(n)
    for fi, (ti, vi) in enumerate(skf.split(X, y)):
        pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
        m = factory_fn(pw, fi + 42)
        if model_name.startswith('LGB'):
            m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        else:
            m.fit(X[ti], y[ti])
        oof[vi] = m.predict_proba(X[vi])[:, 1]
    oof = np.nan_to_num(oof, nan=0.5)
    return oof

layer1 = {}

# M1: LightGBM on ALL features
layer1['LGB_All'] = train_cv_oof(X_all, y, 'LGB', lambda pw, s: lgb.LGBMClassifier(
    n_estimators=1000, max_depth=7, learning_rate=0.03, num_leaves=63,
    min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=pw, random_state=s, verbose=-1), skf)

# M2: XGBoost on ALL features
layer1['XGB_All'] = train_cv_oof(X_all, y, 'XGB', lambda pw, s: xgb.XGBClassifier(
    n_estimators=800, max_depth=6, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.7, scale_pos_weight=pw, tree_method='hist', verbosity=0,
    random_state=s), skf)

# M3: LightGBM on features WITHOUT temporal profiles (different perspective)
layer1['LGB_noTemp'] = train_cv_oof(X_no_temp, y, 'LGB', lambda pw, s: lgb.LGBMClassifier(
    n_estimators=800, max_depth=7, learning_rate=0.03, num_leaves=63,
    min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=pw, random_state=s, verbose=-1), skf)

# M4: LightGBM on features WITHOUT PAA (different perspective)
layer1['LGB_noPAA'] = train_cv_oof(X_no_paa, y, 'LGB', lambda pw, s: lgb.LGBMClassifier(
    n_estimators=800, max_depth=7, learning_rate=0.03, num_leaves=63,
    min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=pw, random_state=s, verbose=-1), skf)

# M5: RandomForest on ALL (different decision boundaries)
layer1['RF_All'] = train_cv_oof(X_all, y, 'RF', lambda pw, s: RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_leaf=20, max_features=0.3,
    class_weight='balanced', n_jobs=-1, random_state=s), skf)

for name, oof in sorted(layer1.items()):
    bf = max(f1_score(y, (oof > th).astype(int), zero_division=0)
             for th in np.arange(0.05, 0.95, 0.01))
    auc = roc_auc_score(y, oof)
    print(f'  {name:18s}: F1={bf:.4f} AUC={auc:.4f}')


# ═══════════════════════════════════════════════════════════════════
# 3. LAYER 2: 3 meta models using Layer 1 OOFs as features
# ═══════════════════════════════════════════════════════════════════
print(f'\n{S}')
print('  [3/5] Layer 2: 3 meta models (OOF features)')
print(f'{S}')

# Build OOF feature matrix from Layer 1
layer1_names = list(layer1.keys())
P1 = np.column_stack([layer1[nm] for nm in layer1_names])

# Augment with all features — this creates "OOF-enhanced" training data
X_meta2 = np.column_stack([X_all, P1]).astype(np.float32)
X_meta2 = np.nan_to_num(X_meta2, nan=0, posinf=0, neginf=0)

layer2 = {}

# M6: LightGBM on features + Layer1 OOFs
layer2['LGB_L2'] = train_cv_oof(X_meta2, y, 'LGB', lambda pw, s: lgb.LGBMClassifier(
    n_estimators=800, max_depth=6, learning_rate=0.03, num_leaves=63,
    min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=pw, random_state=s, verbose=-1), skf)

# M7: XGBoost on features + Layer1 OOFs
layer2['XGB_L2'] = train_cv_oof(X_meta2, y, 'XGB', lambda pw, s: xgb.XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.7, scale_pos_weight=pw, tree_method='hist', verbosity=0,
    random_state=s), skf)

# M8: ExtraTrees on features + Layer1 OOFs
layer2['ET_L2'] = train_cv_oof(X_meta2, y, 'ET', lambda pw, s: ExtraTreesClassifier(
    n_estimators=300, max_depth=12, min_samples_leaf=20, max_features=0.3,
    class_weight='balanced', n_jobs=-1, random_state=s), skf)

for name, oof in sorted(layer2.items()):
    bf = max(f1_score(y, (oof > th).astype(int), zero_division=0)
             for th in np.arange(0.05, 0.95, 0.01))
    auc = roc_auc_score(y, oof)
    corr_w_l1 = np.mean([np.corrcoef(oof, layer1[n])[0, 1] for n in layer1_names[:2]])
    print(f'  {name:18s}: F1={bf:.4f} AUC={auc:.4f}  corr_L1_avg={corr_w_l1:.3f}')


# ═══════════════════════════════════════════════════════════════════
# 4. LAYER 3: Stacking all 8 OOFs
# ═══════════════════════════════════════════════════════════════════
print(f'\n{S}')
print('  [4/5] Layer 3: Meta-learner stacking')
print(f'{S}')

all_oofs = {**layer1, **layer2}
all_names = list(all_oofs.keys())
P_all = np.column_stack([all_oofs[nm] for nm in all_names])
print(f'  Total OOF pool: {len(all_names)} ({len(layer1)} L1 + {len(layer2)} L2)')

# Correlation analysis
corrs = np.corrcoef(P_all.T)
avg_corrs = [(np.sum(np.abs(corrs[i])) - 1) / (len(all_names) - 1) for i in range(len(all_names))]
for i, nm in enumerate(all_names):
    print(f'  {nm:18s}: avg_corr={avg_corrs[i]:.3f}')

# Train multiple meta-learners, pick best
skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_oof, best_f1, best_name = None, 0, ''

from sklearn.ensemble import HistGradientBoostingClassifier

for meta_name, factory in [
    ('XGB-d3', lambda pw: xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
        scale_pos_weight=pw, tree_method='hist', verbosity=0, random_state=42)),
    ('LR-C1.0', lambda _: LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=42)),
    ('LR-C0.5', lambda _: LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000, random_state=42)),
    ('HistGB', lambda _: HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.05, random_state=42)),
]:
    oof = np.zeros(n)
    for fi, (ti, vi) in enumerate(skf_meta.split(P_all, y)):
        pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
        m = factory(pw)
        m.fit(P_all[ti], y[ti])
        oof[vi] = m.predict_proba(P_all[vi])[:, 1]
    bf = max(f1_score(y, (oof > th).astype(int), zero_division=0)
             for th in np.arange(0.05, 0.95, 0.001))
    auc = roc_auc_score(y, oof)
    print(f'  {meta_name:10s}: F1={bf:.4f} AUC={auc:.4f}')
    if bf > best_f1:
        best_f1, best_oof, best_name = bf, oof, meta_name

# Top-3 ensemble
sorted_meta = [('XGB-d3',None),('LR-C1.0',None),('LR-C0.5',None)]  # placeholder
all_meta = {}
for meta_name in ['XGB-d3','LR-C1.0','LR-C0.5']:
    oof = np.zeros(n)
    for fi, (ti, vi) in enumerate(skf_meta.split(P_all, y)):
        pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
        if meta_name == 'XGB-d3':
            m = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                   scale_pos_weight=pw, tree_method='hist', verbosity=0, random_state=42)
        else:
            c = 1.0 if '1.0' in meta_name else 0.5
            m = LogisticRegression(C=c, class_weight='balanced', max_iter=2000, random_state=42)
        m.fit(P_all[ti], y[ti])
        oof[vi] = m.predict_proba(P_all[vi])[:, 1]
    all_meta[meta_name] = oof

# Average top 3
ensemble_oof = (all_meta['XGB-d3'] + all_meta['LR-C1.0'] + all_meta['LR-C0.5']) / 3
bf_ens = max(f1_score(y, (ensemble_oof > th).astype(int), zero_division=0)
             for th in np.arange(0.05, 0.95, 0.001))
print(f'  Top3-Ens : F1={bf_ens:.4f} AUC={roc_auc_score(y, ensemble_oof):.4f}')
if bf_ens > best_f1:
    best_f1, best_oof, best_name = bf_ens, ensemble_oof, 'Top3-Ensemble'

# Final threshold
bt = 0.5
bf0 = 0
for th in np.arange(0.05, 0.95, 0.001):
    pred = (best_oof > th).astype(int)
    if pred.sum() == 0: continue
    f = f1_score(y, pred, zero_division=0)
    if f > bf0: bf0, bt = f, th
pred_final = (best_oof > bt).astype(int)
tp = ((pred_final == 1) & (y == 1)).sum()
fp = ((pred_final == 1) & (y == 0)).sum()
fn = ((pred_final == 0) & (y == 1)).sum()


# ═══════════════════════════════════════════════════════════════════
# 5. REPORT
# ═══════════════════════════════════════════════════════════════════
print(f'\n{S}')
print('  [5/5] FINAL: Self-Contained (Pure Raw Data)')
print(f'{S}')
print(f'  Input:          data/raw_data.csv only')
print(f'  Features:       {X_all.shape[1]} dims (5 groups: '
      f'{", ".join(f"{k}({v.shape[1]})" for k,v in sorted(groups.items()))})')
print(f'  Layer 1 models: {len(layer1)} (LGB_All, XGB_All, LGB_noTemp, LGB_noPAA, RF_All)')
print(f'  Layer 2 models: {len(layer2)} (LGB_L2, XGB_L2, ET_L2 on features+OOFs)')
print(f'  Total OOFs:     {len(all_oofs)}')
print(f'  Best meta:      {best_name}')
print(f'')
print(f'  F1=      {bf0:.4f}')
print(f'  AUC=     {roc_auc_score(y, best_oof):.4f}')
print(f'  Rec=     {tp/(tp+fn):.4f}')
print(f'  Prec=    {tp/(tp+fp) if tp+fp>0 else 0:.4f}')
print(f'  th=      {bt:.3f}')
print(f'  TP={tp}  FP={fp}  FN={fn}')
print(f'')
print(f'  COMPARISON:')
print(f'  V225 (external):          F1=0.8457 (10 external OOF stack)')
print(f'  Mega Boost Enhanced:      F1=0.8616 (27 OOFs, 11 external)')
print(f'  Self-Contained Pipeline:  F1={bf0:.4f} (0 external, pure raw)')
print(f'  vs V225:                  {bf0 - 0.8457:+.4f}')
print(f'')
print(f'  Time: {(time.time() - t_start) / 60:.1f} min')

np.savez_compressed(os.path.join(OUT, 'self_contained_final.npz'),
                     oof_final=best_oof, y=y, f1=bf0, names=np.array(all_names))
print(f'  Saved to output/self_contained_final.npz')
