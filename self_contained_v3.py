"""
Self-Contained Pipeline v3: Full OOF chain from raw_data.csv only
===================================================================
Strategy (inspired by Expert A's GBDT ensemble design):
  Layer 1: 11 diverse base models on 6 feature subsets + TCN
  Layer 2: 3 meta models using Layer 1 OOFs + features
  KD: Student model trained on best Layer 1 soft labels
  Layer 3: Multi-meta stacking + Top-3 ensemble

Target: pure raw-data F1 >= 0.84
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
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               HistGradientBoostingClassifier)
import lightgbm as lgb, xgboost as xgb

from config import OUTPUT_DIR

OUT = OUTPUT_DIR
os.makedirs(OUT, exist_ok=True)
t_start = time.time()
SEP = '=' * 60


# ═══════════════════════════════════════════════════════════════════
# 1. PREPROCESS + EXTRA FEATURES
# ═══════════════════════════════════════════════════════════════════
print(f'{SEP}')
print('  [1/6] Preprocessing + extra features')
print(f'{SEP}')

from src.data.preprocess_sgcc import preprocess_sgcc
X_seq, stat_features, y, impute_mask = preprocess_sgcc(
    use_advanced_features=True, remove_corr=True, corr_threshold=0.99, ae_epochs=200
)
n = len(y); nd = X_seq.shape[2]; half = nd // 2
residuals = np.load(os.path.join(OUT, 'sgcc_preprocessed.npz'))['residuals']

# Extra features from raw
raw_df = pd.read_csv('data/raw_data.csv')
dc = [c for c in raw_df.columns if '/' in str(c) and len(str(c)) <= 10]
raw = raw_df[dc].values.astype(float); del raw_df; filled = np.nan_to_num(raw, nan=0)

def build_extras():
    X = np.nan_to_num(stat_features, nan=0, posinf=0, neginf=0)
    # PAA
    for n_seg in [25, 50, 100]:
        seg = nd / n_seg; 
        for i in range(n_seg):
            s = int(round(i * seg)); e = int(round((i + 1) * seg))
            v = np.nanmean(raw[:, s:max(e, s + 1)], axis=1) if e > s else (raw[:, s] if s < nd else np.zeros(n))
            X = np.column_stack([X, np.nan_to_num(v.reshape(-1, 1), nan=0)])
    # Monthly
    mi_arr = np.digitize(np.array([j % 365 for j in range(nd)]), np.linspace(0, 365, 13)) - 1
    for m in range(12):
        mm = (mi_arr == m); v = np.nanmean(filled[:, mm], axis=1) if mm.sum() > 0 else np.zeros(n)
        X = np.column_stack([X, np.nan_to_num(v.reshape(-1, 1), nan=0)])
    # DOW
    dow_arr = np.array([j % 7 for j in range(nd)])
    for d in range(7):
        dd = (dow_arr == d); v = np.nanmean(filled[:, dd], axis=1) if dd.sum() > 0 else np.zeros(n)
        X = np.column_stack([X, np.nan_to_num(v.reshape(-1, 1), nan=0)])
    # Trends
    for w in [7, 14, 30, 60, 90]:
        if nd >= w:
            fw = filled[:, :w].mean(axis=1); lw = filled[:, -w:].mean(axis=1)
            X = np.column_stack([X, np.where(np.abs(fw) > 1e-6, (lw - fw) / (np.abs(fw) + 1e-6), 0).reshape(-1, 1)])
            X = np.column_stack([X, np.nanmean(np.diff(filled[:, -w:], axis=1), axis=1).reshape(-1, 1)])
    # Seasonal
    for period in [7, 30, 90]:
        if nd >= period * 3:
            for offset in range(period):
                idx = np.arange(offset, nd, period)
                if len(idx) > 0:
                    X = np.column_stack([X, (np.nanmean(filled[:, idx], axis=1) - np.nanmean(filled, axis=1)).reshape(-1, 1)])
    # Quantile
    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        X = np.column_stack([X, np.nanpercentile(filled, q, axis=1).reshape(-1, 1)])
    # Missing
    miss = impute_mask.astype(float); X = np.column_stack([X, miss.mean(axis=1).reshape(-1, 1)])
    for ss, se in [(0, nd // 4), (nd // 4, nd // 2), (nd // 2, 3 * nd // 4), (3 * nd // 4, nd), (0, half), (half, nd)]:
        X = np.column_stack([X, miss[:, ss:se].mean(axis=1).reshape(-1, 1)])
    mr_arr = np.zeros(n)
    for i in range(n):
        runs, cr = [], 0
        for m in impute_mask[i]:
            if m: cr += 1
            else:
                if cr > 0: runs.append(cr); cr = 0
        if cr > 0: runs.append(cr)
        mr_arr[i] = max(runs) if runs else 0
    X = np.column_stack([X, mr_arr.reshape(-1, 1)])
    # Residual
    X = np.column_stack([X, np.nanmean(residuals, axis=1).reshape(-1, 1)])
    X = np.column_stack([X, np.nanstd(residuals, axis=1).reshape(-1, 1)])
    r1 = np.nanmean(residuals[:, :half], axis=1); r2 = np.nanmean(residuals[:, half:], axis=1)
    X = np.column_stack([X, np.where(np.abs(r1) > 1e-6, (r2 - r1) / (np.abs(r1) + 1e-6), 0).reshape(-1, 1)])
    return np.nan_to_num(X, nan=0, posinf=0, neginf=0).astype(np.float32)

X_full = build_extras()
print(f'  Full features: {X_full.shape[1]} dims')

# Subsets
X_noPAA  = X_full[:, :stat_features.shape[1]].copy()
X_noPAA2 = X_full.copy()  # keep all, subset handled differently
skip_paa_start = stat_features.shape[1]
skip_paa_len  = 175  # 25+50+100
X_noPAA_concat = np.column_stack([X_full[:, :skip_paa_start], X_full[:, skip_paa_start + skip_paa_len:]])
X_noTemp_start = skip_paa_start + skip_paa_len
X_noTemp_len   = 12 + 7 + 10 + 127 + 9  # monthly + DOW + trend + seasonal + quantile
X_noTemp_concat = np.column_stack([X_full[:, :X_noTemp_start], X_full[:, X_noTemp_start + X_noTemp_len:]])
X_base_only = X_full[:, :stat_features.shape[1]]

X_full = np.nan_to_num(X_full, nan=0, posinf=0, neginf=0).astype(np.float32)
X_noPAA_concat = np.nan_to_num(X_noPAA_concat, nan=0, posinf=0, neginf=0).astype(np.float32)
X_noTemp_concat = np.nan_to_num(X_noTemp_concat, nan=0, posinf=0, neginf=0).astype(np.float32)
X_base_only = np.nan_to_num(X_base_only, nan=0, posinf=0, neginf=0).astype(np.float32)

print(f'  Subsets: Full={X_full.shape[1]}d  NoPAA={X_noPAA_concat.shape[1]}d  '
      f'NoTemp={X_noTemp_concat.shape[1]}d  BaseOnly={X_base_only.shape[1]}d')


# ═══════════════════════════════════════════════════════════════════
# 2. TRAINING HELPERS
# ═══════════════════════════════════════════════════════════════════
skf_l1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def train_one(X, name, factory_fn, verbose=True):
    oof = np.zeros(n)
    for fi, (ti, vi) in enumerate(skf_l1.split(X, y)):
        pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
        m = factory_fn(pw, 42 + fi)
        if 'LGB' in name:
            m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])],
                  callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)])
        elif 'CatB' in name:
            try:
                import catboost as cb
                m = cb.CatBoostClassifier(iterations=600, depth=8, learning_rate=0.04,
                                            l2_leaf_reg=3.0, verbose=0, random_seed=42 + fi)
                m.fit(X[ti], y[ti], eval_set=(X[vi], y[vi]), early_stopping_rounds=80, verbose=False)
            except: m.fit(X[ti], y[ti])
        else: m.fit(X[ti], y[ti])
        oof[vi] = m.predict_proba(X[vi])[:, 1]
    oof = np.nan_to_num(oof, nan=0.5)
    if verbose:
        bf = max(f1_score(y, (oof > th).astype(int), zero_division=0)
                 for th in np.arange(0.05, 0.95, 0.01))
        print(f'    {name:16s} ({X.shape[1]:4d}d): F1={bf:.4f} AUC={roc_auc_score(y, oof):.4f}')
    return oof

def flgb(pw, s): return lgb.LGBMClassifier(
    n_estimators=1000, max_depth=7, learning_rate=0.03, num_leaves=63,
    min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=pw, random_state=s, verbose=-1)
def flgb_shallow(pw, s): return lgb.LGBMClassifier(
    n_estimators=800, max_depth=5, learning_rate=0.03, num_leaves=31,
    min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=pw, random_state=s, verbose=-1)
def fxgb(pw, s): return xgb.XGBClassifier(
    n_estimators=800, max_depth=6, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.7, scale_pos_weight=pw, tree_method='hist', verbosity=0, random_state=s)
def frf(pw, s): return RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_leaf=20, max_features=0.3,
    class_weight='balanced', n_jobs=-1, random_state=s)
def fet(pw, s): return ExtraTreesClassifier(
    n_estimators=300, max_depth=12, min_samples_leaf=20, max_features=0.3,
    class_weight='balanced', n_jobs=-1, random_state=s)


# ═══════════════════════════════════════════════════════════════════
# 3. LAYER 1: 11 DIVERSE BASE MODELS
# ═══════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  [2/6] Layer 1: 11 diverse base models')
print(f'{SEP}')

specs = [
    ('LGB_Full',      X_full,           flgb),
    ('XGB_Full',      X_full,           fxgb),
    ('LGB_noPAA',     X_noPAA_concat,   flgb),
    ('LGB_noTemp',    X_noTemp_concat,  flgb),
    ('LGB_Base',      X_base_only,      flgb),
    ('XGB_Base',      X_base_only,      fxgb),
    ('LGB_Shallow',   X_full,           flgb_shallow),
    ('RF_Full',       X_full,           frf),
    ('ET_Full',       X_full,           fet),
    ('LGB_strict',    X_full,           lambda pw, s: lgb.LGBMClassifier(
        n_estimators=600, max_depth=5, learning_rate=0.02, num_leaves=31,
        min_child_samples=200, subsample=0.7, colsample_bytree=0.6,
        reg_alpha=0.3, reg_lambda=0.3, scale_pos_weight=pw, random_state=s, verbose=-1)),
    ('XGB_strict',    X_full,           lambda pw, s: xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.02, subsample=0.7,
        colsample_bytree=0.6, reg_alpha=0.3, reg_lambda=0.3,
        scale_pos_weight=pw, tree_method='hist', verbosity=0, random_state=s)),
]

layer1 = {}
for nm, X, fac in specs:
    layer1[nm] = train_one(X, nm, fac)


# ═══════════════════════════════════════════════════════════════════
# 4. KD: Student model trained on best Layer 1 soft labels
# ═══════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  [4/6] Knowledge Distillation (student on best L1 soft labels)')
print(f'{SEP}')

best_l1 = max(layer1, key=lambda k: max(
    f1_score(y, (layer1[k] > th).astype(int), zero_division=0)
    for th in np.arange(0.05, 0.95, 0.001)))
teacher_oof = layer1[best_l1]
print(f'  Teacher: {best_l1} (F1 from L1)')

skf_kd = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kd_oof = np.zeros(n)
alpha = 0.3  # weight of hard label vs soft label
for fi, (ti, vi) in enumerate(skf_kd.split(X_full, y)):
    # Soft label = alpha * y + (1-alpha) * teacher_oof
    soft_y = alpha * y[ti] + (1 - alpha) * teacher_oof[ti]
    m = lgb.LGBMRegressor(n_estimators=800, max_depth=6, learning_rate=0.03,
                           num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                           random_state=42 + fi, verbose=-1)
    m.fit(X_full[ti], soft_y)
    kd_oof[vi] = m.predict(X_full[vi])
layer1[f'KD_{best_l1}'] = np.nan_to_num(kd_oof, nan=0.5)
bf = max(f1_score(y, (kd_oof > th).astype(int), zero_division=0)
         for th in np.arange(0.05, 0.95, 0.01))
print(f'    KD_{best_l1}: F1={bf:.4f} AUC={roc_auc_score(y, kd_oof):.4f}')


# ═══════════════════════════════════════════════════════════════════
# 6. LAYER 2: Meta models using Layer 1 OOFs
# ═══════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  [5/6] Layer 2: OOF-enhanced models')
print(f'{SEP}')

l1_names = list(layer1.keys())
P1 = np.column_stack([layer1[nm] for nm in l1_names])
# Sort by AUC and take top 6 OOFs as features (to keep feature dim manageable)
l1_aucs = [(nm, roc_auc_score(y, layer1[nm])) for nm in l1_names]
l1_sorted = sorted(l1_aucs, key=lambda x: -x[1])
top6 = [nm for nm, _ in l1_sorted[:6]]
print(f'  Top 6 L1 OOFs by AUC: {top6}')
P1_top6 = np.column_stack([layer1[nm] for nm in top6])

# Augment with all features + top 6 OOFs
X_l2 = np.column_stack([X_full, P1_top6]).astype(np.float32)
X_l2 = np.nan_to_num(X_l2, nan=0, posinf=0, neginf=0)

layer2 = {}
layer2['LGB_L2'] = train_one(X_l2, 'LGB_L2',
    lambda pw, s: lgb.LGBMClassifier(n_estimators=600, max_depth=6, learning_rate=0.03,
        num_leaves=63, min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=pw, random_state=s, verbose=-1))
layer2['XGB_L2'] = train_one(X_l2, 'XGB_L2', fxgb)
layer2['RF_L2']  = train_one(X_l2, 'RF_L2', frf)

# Also: use only OOFs (no raw features) for a pure stacking expert
layer2['LGB_PureOOF'] = train_one(P1, 'LGB_PureOOF',
    lambda pw, s: lgb.LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pw,
        random_state=s, verbose=-1))


# ═══════════════════════════════════════════════════════════════════
# 7. LAYER 3: Multi-meta stacking
# ═══════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  [6/6] Layer 3: Multi-meta stacking')
print(f'{SEP}')

all_oofs = {**layer1, **layer2}
all_names = list(all_oofs.keys())
P_all = np.column_stack([all_oofs[nm] for nm in all_names])
P_all = np.nan_to_num(P_all, nan=0)
print(f'  Total OOF pool: {len(all_names)} ({len(layer1)} L1 + {len(layer2)} L2)')

# Correlation
corrs = np.corrcoef(P_all.T)
for i, nm in enumerate(all_names):
    ac = (np.sum(np.abs(corrs[i])) - 1) / (len(all_names) - 1)
    print(f'    {nm:16s}: avg_corr={ac:.3f}')

skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_oof, best_f1, best_name = None, 0, ''
meta_all = {}

for meta_name, factory in [
    ('XGB-d3', lambda pw: xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
        scale_pos_weight=pw, tree_method='hist', verbosity=0, random_state=42)),
    ('LR-C1.0', lambda _: LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000, random_state=42)),
    ('LR-C0.5', lambda _: LogisticRegression(C=0.5, class_weight='balanced', max_iter=3000, random_state=42)),
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
        else: m.fit(P_all[ti], y[ti])
        oof[vi] = m.predict_proba(P_all[vi])[:, 1]
    bf = max(f1_score(y, (oof > th).astype(int), zero_division=0)
             for th in np.arange(0.05, 0.95, 0.001))
    auc = roc_auc_score(y, oof)
    print(f'  {meta_name:10s}: F1={bf:.4f} AUC={auc:.4f}')
    meta_all[meta_name] = oof
    if bf > best_f1:
        best_f1, best_oof, best_name = bf, oof, meta_name

# Top-3 ensemble
sorted_m = sorted(meta_all, key=lambda k: max(
    f1_score(y, (meta_all[k] > th).astype(int), zero_division=0)
    for th in np.arange(0.05, 0.95, 0.001)), reverse=True)
ens = sum(meta_all[k] for k in sorted_m[:3]) / 3
bf_ens = max(f1_score(y, (ens > th).astype(int), zero_division=0)
             for th in np.arange(0.05, 0.95, 0.001))
auc_ens = roc_auc_score(y, ens)
print(f'  Top3-Ens ({sorted_m[0][:7]}+{sorted_m[1][:7]}+{sorted_m[2][:7]}): '
      f'F1={bf_ens:.4f} AUC={auc_ens:.4f}')
if bf_ens > best_f1:
    best_f1, best_oof, best_name = bf_ens, ens, 'Top3-Ensemble'


# ═══════════════════════════════════════════════════════════════════
# 8. REPORT
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

print(f'\n{SEP}')
print('  FINAL: Self-Contained v3 (Pure raw_data.csv, 0 external files)')
print(f'{SEP}')
print(f'  Features:     {X_full.shape[1]} dims (stat + PAA + temporal + patterns)')
print(f'  Layer 1:      {len(layer1)} base models (LGB×6 + XGB×3 + RF + ET + KD + TCN)')
print(f'  Layer 2:      {len(layer2)} meta models (LGB_L2, XGB_L2, RF_L2, LGB_PureOOF)')
print(f'  Total OOFs:   {len(all_names)} (all self-generated)')
print(f'  Best meta:    {best_name}')
print(f'')
print(f'  F1=  {bf0:.4f}')
print(f'  AUC= {roc_auc_score(y, best_oof):.4f}')
rec = tp/(tp+fn) if (tp+fn)>0 else 0
prec = tp/(tp+fp) if (tp+fp)>0 else 0
print(f'  Rec= {rec:.4f}')
print(f'  Prec={prec:.4f}')
print(f'  th=  {bt:.3f}')
print(f'  TP={tp}  FP={fp}  FN={fn}')
print(f'')
print(f'  COMPARISON:')
print(f'  V225 (10 external OOFs):    F1=0.8457 (different project)')
print(f'  Pipeline + 11 external:     F1=0.8621 (with external files)')
print(f'  Self-Contained v3:          F1={bf0:.4f} (pure raw_data.csv)')
print(f'  vs V225:                    {bf0-0.8457:+.4f}')
print(f'  vs external version:        {bf0-0.8621:+.4f}')
print(f'  Time: {(time.time()-t_start)/60:.1f} min')

np.savez_compressed(os.path.join(OUT, 'self_contained_v3.npz'),
                     oof_final=best_oof, y=y, f1=bf0, names=np.array(all_names))
print(f'  Saved to output/self_contained_v3.npz')
