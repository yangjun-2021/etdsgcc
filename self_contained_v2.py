"""
Self-Contained Pipeline v2: preprocess_sgcc + extra features + multi-stage OOF chain
=====================================================================================
Input: data/raw_data.csv only
Step 1: Run preprocess_sgcc → 353 stat features
Step 2: Build extra features (PAA + temporal + missing patterns) → +349 = ~702 dims
Step 3: Train 8 base models on different feature subsets → Layer 1 OOFs
Step 4: Train 4 models using Layer 1 OOFs as features → Layer 2 OOFs
Step 5: Stack all 12 OOFs with multi-meta → final prediction
"""
import os, time, random, warnings, sys
import numpy as np, pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')
np.random.seed(42); random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
import lightgbm as lgb, xgboost as xgb
from config import OUTPUT_DIR

OUT = OUTPUT_DIR
os.makedirs(OUT, exist_ok=True)
t_start = time.time()
S = '=' * 60


# ═══════════════════════════════════════════════════════════════════
# 1. PREPROCESS (uses only raw_data.csv → outputs sgcc_preprocessed.npz)
# ═══════════════════════════════════════════════════════════════════
print(f'{S}')
print('  Step 1: Preprocessing (src.data.preprocess_sgcc)')
print(f'{S}')
from src.data.preprocess_sgcc import preprocess_sgcc
X_seq, stat_features, y, impute_mask = preprocess_sgcc(
    use_advanced_features=True, remove_corr=True, corr_threshold=0.99, ae_epochs=200
)
n = len(y)
nd = X_seq.shape[2]
half = nd // 2
print(f'  stat_features: {stat_features.shape[1]} dims')
print(f'  X_seq: {X_seq.shape}')

# ═══════════════════════════════════════════════════════════════════
# 2. BUILD EXTRA FEATURES
# ═══════════════════════════════════════════════════════════════════
print(f'\n{S}')
print('  Step 2: Building extra features (PAA + temporal + patterns)')
print(f'{S}')

raw_df = pd.read_csv('data/raw_data.csv')
dc = [c for c in raw_df.columns if '/' in str(c) and len(str(c)) <= 10]
raw = raw_df[dc].values.astype(float)
del raw_df
filled = np.nan_to_num(raw, nan=0)
residuals = np.load(os.path.join(OUT, 'sgcc_preprocessed.npz'))['residuals']

extra_groups = {}

# PAA (25 + 50 + 100 = 175)
paa_list = []
for n_seg in [25, 50, 100]:
    seg = nd / n_seg
    for i in range(n_seg):
        s = int(round(i * seg))
        e = int(round((i + 1) * seg))
        if e > s: val = np.nanmean(raw[:, s:e], axis=1)
        elif s < nd: val = raw[:, s]
        else: val = np.zeros(n)
        paa_list.append(val)
extra_groups['PAA'] = np.nan_to_num(np.column_stack([p.reshape(-1, 1) for p in paa_list]), nan=0).astype(np.float32)

# Temporal profiles
temp_list = []
mi_arr = np.digitize(np.array([j % 365 for j in range(nd)]), np.linspace(0, 365, 13)) - 1
for m in range(12):
    mm = (mi_arr == m)
    temp_list.append(np.nanmean(filled[:, mm], axis=1) if mm.sum() > 0 else np.zeros(n))
dow_arr = np.array([j % 7 for j in range(nd)])
for d in range(7):
    dd = (dow_arr == d)
    temp_list.append(np.nanmean(filled[:, dd], axis=1) if dd.sum() > 0 else np.zeros(n))
for w in [7, 14, 30, 60, 90]:
    if nd >= w:
        fw = filled[:, :w].mean(axis=1); lw = filled[:, -w:].mean(axis=1)
        temp_list.append(np.where(np.abs(fw) > 1e-6, (lw - fw) / (np.abs(fw) + 1e-6), 0))
        temp_list.append(np.nanmean(np.diff(filled[:, -w:], axis=1), axis=1))
for period in [7, 30, 90]:
    if nd >= period * 3:
        for offset in range(period):
            idx = np.arange(offset, nd, period)
            if len(idx) > 0:
                temp_list.append(np.nanmean(filled[:, idx], axis=1) - np.nanmean(filled, axis=1))
for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    temp_list.append(np.nanpercentile(filled, q, axis=1))
extra_groups['Temporal'] = np.nan_to_num(np.column_stack([t.reshape(-1, 1) for t in temp_list]), nan=0).astype(np.float32)

# Missing patterns
miss = impute_mask.astype(float)
miss_feats = [miss.mean(axis=1)]
for ss, se in [(0, nd // 4), (nd // 4, nd // 2), (nd // 2, 3 * nd // 4), (3 * nd // 4, nd),
               (0, half), (half, nd)]:
    miss_feats.append(miss[:, ss:se].mean(axis=1))
mr_arr = np.zeros(n)
for i in range(n):
    runs, cr = [], 0
    for m in impute_mask[i]:
        if m: cr += 1
        else:
            if cr > 0: runs.append(cr); cr = 0
    if cr > 0: runs.append(cr)
    mr_arr[i] = max(runs) if runs else 0
miss_feats.append(mr_arr)
miss_feats.append(impute_mask.sum(axis=1) / nd)
extra_groups['Missing'] = np.nan_to_num(np.column_stack([f.reshape(-1, 1) for f in miss_feats]), nan=0).astype(np.float32)

# Residual aggregation
res_feats = [np.nanmean(residuals, axis=1), np.nanstd(residuals, axis=1),
             np.nanmean(np.abs(residuals), axis=1)]
r1 = np.nanmean(residuals[:, :half], axis=1); r2 = np.nanmean(residuals[:, half:], axis=1)
res_feats.append(np.where(np.abs(r1) > 1e-6, (r2 - r1) / (np.abs(r1) + 1e-6), 0))
extra_groups['Residual'] = np.nan_to_num(np.column_stack([r.reshape(-1, 1) for r in res_feats]), nan=0).astype(np.float32)

for k, v in sorted(extra_groups.items()):
    print(f'  {k:12s}: {v.shape[1]:4d} dims')

# Build combined feature sets
all_keys = list(extra_groups.keys())
X_extra = np.column_stack([extra_groups[k] for k in all_keys])
X_base = np.nan_to_num(stat_features, nan=0, posinf=0, neginf=0)
X_full = np.column_stack([X_base, X_extra])
X_full = np.nan_to_num(X_full, nan=0, posinf=0, neginf=0).astype(np.float32)

# Feature subsets for diversity
X_no_PAA = np.nan_to_num(np.column_stack([X_base] + [extra_groups[k] for k in all_keys if k != 'PAA']), nan=0).astype(np.float32)
X_no_Temp = np.nan_to_num(np.column_stack([X_base] + [extra_groups[k] for k in all_keys if k != 'Temporal']), nan=0).astype(np.float32)
X_no_Res = np.nan_to_num(np.column_stack([X_base] + [extra_groups[k] for k in all_keys if k != 'Residual']), nan=0).astype(np.float32)

print(f'\n  Feature sets: X_full={X_full.shape[1]}d  X_noPAA={X_no_PAA.shape[1]}d')
print(f'  X_noTemp={X_no_Temp.shape[1]}d  X_noRes={X_no_Res.shape[1]}d  X_base={X_base.shape[1]}d')


# ═══════════════════════════════════════════════════════════════════
# 3. TRAINING HELPERS
# ═══════════════════════════════════════════════════════════════════
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def train_oof(X, name, factory_fn):
    oof = np.zeros(n)
    for fi, (ti, vi) in enumerate(skf.split(X, y)):
        pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
        m = factory_fn(pw, 42 + fi)
        if 'LGB' in name or 'lgb' in name.lower():
            try: m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            except: m.fit(X[ti], y[ti])
        else: m.fit(X[ti], y[ti])
        oof[vi] = m.predict_proba(X[vi])[:, 1]
    return np.nan_to_num(oof, nan=0.5)

def factory_lgb(pw, s): return lgb.LGBMClassifier(
    n_estimators=1000, max_depth=7, learning_rate=0.03, num_leaves=63,
    min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=pw, random_state=s, verbose=-1)
def factory_xgb(pw, s): return xgb.XGBClassifier(
    n_estimators=800, max_depth=6, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.7, scale_pos_weight=pw, tree_method='hist', verbosity=0, random_state=s)
def factory_rf(pw, s): return RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_leaf=20, max_features=0.3,
    class_weight='balanced', n_jobs=-1, random_state=s)


# ═══════════════════════════════════════════════════════════════════
# 4. LAYER 1: 8 diverse base models
# ═══════════════════════════════════════════════════════════════════
print(f'\n{S}')
print('  Step 3: Layer 1 — 8 diverse base models')
print(f'{S}')

specs = [
    ('LGB_Full',   X_full,   factory_lgb),
    ('XGB_Full',   X_full,   factory_xgb),
    ('LGB_noPAA',  X_no_PAA, factory_lgb),
    ('LGB_noTemp', X_no_Temp,factory_lgb),
    ('LGB_noRes',  X_no_Res, factory_lgb),
    ('RF_Full',    X_full,   factory_rf),
    ('LGB_Base',   X_base,   factory_lgb),
    ('XGB_Base',   X_base,   factory_xgb),
]

layer1 = {}
for name, X, fac in specs:
    oof = train_oof(X, name, fac)
    layer1[name] = oof
    bf = max(f1_score(y, (oof > th).astype(int), zero_division=0)
             for th in np.arange(0.05, 0.95, 0.01))
    auc = roc_auc_score(y, oof)
    print(f'  {name:18s} ({X.shape[1]:4d}d): F1={bf:.4f} AUC={auc:.4f}')


# ═══════════════════════════════════════════════════════════════════
# 5. LAYER 2: 4 models using Layer 1 OOFs as features
# ═══════════════════════════════════════════════════════════════════
print(f'\n{S}')
print('  Step 4: Layer 2 — OOF-enhanced models')
print(f'{S}')

l1_names = list(layer1.keys())
P1 = np.nan_to_num(np.column_stack([layer1[nm] for nm in l1_names]), nan=0)
X_meta = np.column_stack([X_full, P1[:, :5]]).astype(np.float32)  # top 5 OOFs for speed
X_meta = np.nan_to_num(X_meta, nan=0, posinf=0, neginf=0)

layer2 = {}
l2_specs = [
    ('LGB_L2', factory_lgb),
    ('XGB_L2', factory_xgb),
    ('RF_L2',  factory_rf),
    ('ET_L2',  lambda pw, s: ExtraTreesClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20, max_features=0.3, class_weight='balanced', n_jobs=-1, random_state=s)),
]

for name, fac in l2_specs:
    oof = train_oof(X_meta, name, fac)
    layer2[name] = oof
    bf = max(f1_score(y, (oof > th).astype(int), zero_division=0)
             for th in np.arange(0.05, 0.95, 0.01))
    auc = roc_auc_score(y, oof)
    print(f'  {name:18s}: F1={bf:.4f} AUC={auc:.4f}')


# ═══════════════════════════════════════════════════════════════════
# 6. STACKING: Meta-learner on all 12 OOFs
# ═══════════════════════════════════════════════════════════════════
print(f'\n{S}')
print('  Step 5: Meta-learner stacking')
print(f'{S}')

all_oofs = {**layer1, **layer2}
all_names = list(all_oofs.keys())
P_all = np.column_stack([all_oofs[nm] for nm in all_names])
print(f'  Total OOF pool: {len(all_names)} ({len(layer1)} L1 + {len(layer2)} L2)')

# Correlation check
corrs = np.corrcoef(P_all.T)
print(f'  Correlation range: {corrs.min():.3f} ~ {corrs.max():.3f}')
for i, nm in enumerate(all_names):
    ac = (np.sum(np.abs(corrs[i])) - 1) / (len(all_names) - 1)
    print(f'    {nm:18s}: avg_corr={ac:.3f}')

# Train multiple meta-learners
skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_oof, best_f1, best_name = None, 0, ''
meta_oofs = {}

for meta_name, factory in [
    ('XGB-d3', lambda pw: xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
        scale_pos_weight=pw, tree_method='hist', verbosity=0, random_state=42)),
    ('LR-C1.0', lambda _: LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=42)),
    ('LR-C0.5', lambda _: LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000, random_state=42)),
    ('HistGB', lambda _: HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.05, random_state=42)),
    ('LGB', lambda pw: lgb.LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
        scale_pos_weight=pw, verbose=-1, random_state=42)),
]:
    oof = np.zeros(n)
    for fi, (ti, vi) in enumerate(skf_meta.split(P_all, y)):
        pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
        m = factory(pw)
        if meta_name == 'LGB':
            m.fit(P_all[ti], y[ti], eval_set=[(P_all[vi], y[vi])],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        else:
            m.fit(P_all[ti], y[ti])
        oof[vi] = m.predict_proba(P_all[vi])[:, 1]
    bf = max(f1_score(y, (oof > th).astype(int), zero_division=0)
             for th in np.arange(0.05, 0.95, 0.001))
    auc = roc_auc_score(y, oof)
    print(f'  {meta_name:10s}: F1={bf:.4f} AUC={auc:.4f}')
    meta_oofs[meta_name] = oof
    if bf > best_f1:
        best_f1, best_oof, best_name = bf, oof, meta_name

# Top-3 Ensemble
sorted_keys = sorted(meta_oofs, key=lambda k: max(
    f1_score(y, (meta_oofs[k] > th).astype(int), zero_division=0)
    for th in np.arange(0.05, 0.95, 0.001)), reverse=True)
top3_oofs = [meta_oofs[k] for k in sorted_keys[:3]]
ens_oof = sum(top3_oofs) / 3
bf_ens = max(f1_score(y, (ens_oof > th).astype(int), zero_division=0)
             for th in np.arange(0.05, 0.95, 0.001))
print(f'  Top3-Ens ({sorted_keys[0][:7]}+{sorted_keys[1][:7]}+{sorted_keys[2][:7]}): '
      f'F1={bf_ens:.4f} AUC={roc_auc_score(y, ens_oof):.4f}')
if bf_ens > best_f1:
    best_f1, best_oof, best_name = bf_ens, ens_oof, 'Top3-Ensemble'


# ═══════════════════════════════════════════════════════════════════
# 7. FINAL REPORT
# ═══════════════════════════════════════════════════════════════════
bt = 0.5; bf0 = 0
for th in np.arange(0.05, 0.95, 0.001):
    pred = (best_oof > th).astype(int)
    if pred.sum() == 0: continue
    f = f1_score(y, pred, zero_division=0)
    if f > bf0: bf0, bt = f, th
pred_final = (best_oof > bt).astype(int)
tp = ((pred_final == 1) & (y == 1)).sum()
fp = ((pred_final == 1) & (y == 0)).sum()
fn = ((pred_final == 0) & (y == 1)).sum()

from sklearn.metrics import recall_score, precision_score

print(f'\n{S}')
print('  FINAL: Self-Contained Pipeline (Pure raw_data.csv)')
print(f'{S}')
print(f'  Input:      data/raw_data.csv only (zero external files)')
print(f'  Features:   {X_full.shape[1]} dims ({X_base.shape[1]} stat + {X_extra.shape[1]} extra)')
print(f'  Layer 1:    {len(layer1)} models on diff feature subsets')
print(f'  Layer 2:    {len(layer2)} models with OOF features')
print(f'  Total OOFs: {len(all_names)}')
print(f'  Best meta:  {best_name}')
print(f'')
print(f'  F1=      {bf0:.4f}')
print(f'  AUC=     {roc_auc_score(y, best_oof):.4f}')
rec = tp/(tp+fn) if (tp+fn)>0 else 0
prec = tp/(tp+fp) if (tp+fp)>0 else 0
print(f'  Rec=     {rec:.4f}')
print(f'  Prec=    {prec:.4f}')
print(f'  th=      {bt:.3f}')
print(f'  TP={tp}  FP={fp}  FN={fn}')
print(f'')
print(f'  COMPARISON:')
print(f'  V225 (10 external OOFs):       F1=0.8457')
print(f'  Pipeline + 11 external OOFs:   F1=0.8621')
print(f'  Self-Contained (0 external):   F1={bf0:.4f}')
print(f'  ∆ vs external version:         {bf0-0.8621:+.4f}')
print(f'  ∆ vs V225:                     {bf0-0.8457:+.4f}')
print(f'  Time: {(time.time()-t_start)/60:.1f} min')

np.savez_compressed(os.path.join(OUT, 'self_contained_v2.npz'),
                     oof_final=best_oof, y=y, f1=bf0, names=np.array(all_names))
print(f'  Saved to output/self_contained_v2.npz')
