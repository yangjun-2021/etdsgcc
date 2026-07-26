"""
Heterogeneous Feature×Algorithm Grid Search
=============================================
For each algorithm (LGBM, XGBoost, CatBoost, RandomForest),
find the best feature subset among 4 pre-defined heterogeneous sets.
Then ensemble all best-config experts + existing strong OOFs.
"""
import numpy as np, os, time, glob, warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb, xgboost as xgb, catboost as cb
from sklearn.ensemble import RandomForestClassifier
from utils import seed_everything
seed_everything(42)

OUT = 'output'
OD = r'D:\Project\ThiefElectricity\output'
t0 = time.time()

sepline = '=' * 60

# ─── 1. Build all feature groups ──────────────────────────────────
print(sepline)
print('  Building feature groups')
print(sepline)

base = np.load(os.path.join(OUT, 'sgcc_preprocessed.npz'))
y = base['flags'].astype(int)
n = len(y)
stat = np.nan_to_num(base['stat_features'], nan=0, posinf=0, neginf=0)
novel = np.nan_to_num(np.load(os.path.join(OUT, 'novel_features.npz'))['features'], nan=0)
deng = np.nan_to_num(np.load(os.path.join(OUT, 'dengine_features.npz'))['X'], nan=0)
residuals = base['residuals']
impute_mask = base['impute_mask']
nd = residuals.shape[1]
half = nd // 2

import pandas as pd
raw_df = pd.read_csv('data/raw_data.csv')
dc = [c for c in raw_df.columns if '/' in str(c) and len(str(c)) <= 10]
raw = raw_df[dc].values.astype(float)
del raw_df
filled = np.nan_to_num(raw, nan=0)

groups = {}
groups['stat'] = stat
groups['novel'] = novel
groups['deng'] = deng

PAA = []
for n_seg in [25, 50, 100]:
    seg = raw.shape[1] / n_seg
    for i in range(n_seg):
        s = int(round(i * seg))
        e = int(round((i + 1) * seg))
        PAA.append(np.nanmean(raw[:, s:max(e, s + 1)], axis=1))
groups['paa'] = np.nan_to_num(np.column_stack([p.reshape(-1, 1) for p in PAA]), nan=0)

res_list = [
    np.nanmean(residuals, axis=1),
    np.nanstd(residuals, axis=1),
    np.nanmean(np.abs(residuals), axis=1),
    np.nanmax(np.abs(residuals), axis=1),
    (np.nanmean(residuals[:, half:], axis=1) - np.nanmean(residuals[:, :half], axis=1))
    / (np.maximum(np.abs(np.nanmean(residuals[:, :half], axis=1)), 1e-6)),
]
for w in [30, 60, 90, 180, 365]:
    if nd >= w:
        res_list.append(np.nanmean(np.abs(residuals[:, -w:]), axis=1))
groups['res'] = np.nan_to_num(np.column_stack([r.reshape(-1, 1) for r in res_list]), nan=0)

mask_list = [impute_mask.astype(float).mean(axis=1)]
for ss, se in [(0, nd // 4), (nd // 4, nd // 2),
               (nd // 2, 3 * nd // 4), (3 * nd // 4, nd),
               (0, half), (half, nd)]:
    mask_list.append(impute_mask[:, ss:se].astype(float).mean(axis=1))
mr = np.zeros(n)
for i in range(n):
    runs, cr = [], 0
    for m in impute_mask[i]:
        if m:
            cr += 1
        else:
            if cr > 0:
                runs.append(cr)
            cr = 0
    if cr > 0:
        runs.append(cr)
    mr[i] = max(runs) if runs else 0
mask_list.append(mr)
mask_list.append((~impute_mask).sum(axis=1) / nd)
groups['mask'] = np.nan_to_num(np.column_stack([m.reshape(-1, 1) for m in mask_list]), nan=0)

month_of = np.array([j % 365 for j in range(nd)])
mi = np.digitize(month_of, np.linspace(0, 365, 13)) - 1
groups['month'] = np.nan_to_num(
    np.column_stack([np.nanmean(filled[:, (mi == m)], axis=1).reshape(-1, 1) for m in range(12)]),
    nan=0,
)

dow = np.array([j % 7 for j in range(nd)])
groups['dow'] = np.nan_to_num(
    np.column_stack([np.nanmean(filled[:, (dow == d)], axis=1).reshape(-1, 1) for d in range(7)]),
    nan=0,
)

trend_feats = []
for w in [7, 14, 30, 60, 90]:
    if nd >= w:
        first = filled[:, :w].mean(axis=1)
        last = filled[:, -w:].mean(axis=1)
        trend_feats.append((last - first) / (np.maximum(np.abs(first), 1e-6)))
        trend_feats.append(np.nanmean(np.diff(filled[:, -w:], axis=1), axis=1))
groups['trend'] = np.nan_to_num(np.column_stack([t.reshape(-1, 1) for t in trend_feats]), nan=0)

seas_list = []
for period in [7, 30, 90]:
    if nd >= period * 3:
        for offset in range(period):
            idx = np.arange(offset, nd, period)
            if len(idx) > 0:
                seas_list.append(np.nanmean(filled[:, idx], axis=1) - np.nanmean(filled, axis=1))
groups['seasonal'] = np.nan_to_num(np.column_stack([s.reshape(-1, 1) for s in seas_list]), nan=0)

qlist = []
for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    qlist.append(np.nanpercentile(filled, q, axis=1))
groups['quantile'] = np.nan_to_num(np.column_stack([q.reshape(-1, 1) for q in qlist]), nan=0)

all_keys = list(groups.keys())
for k, v in groups.items():
    print(f'  {k:10s}: {v.shape[1]:4d} dims')

# ─── 2. Heterogeneous feature sets ─────────────────────────────────
feature_sets = {
    'Statistical': ['stat', 'novel', 'deng', 'quantile'],     # 495 dims
    'Temporal':    ['paa', 'trend', 'seasonal', 'dow'],        # 319 dims
    'Anomaly':     ['res', 'mask', 'month'],                    #  31 dims
    'ALL':         all_keys,                                    # 850 dims
}

# ─── 3. Algorithms ────────────────────────────────────────────────
algorithms = {
    'LGBM': lambda: lgb.LGBMClassifier(n_estimators=500, max_depth=7,
        learning_rate=0.05, num_leaves=63, min_child_samples=100,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
        reg_lambda=0.1, verbose=-1),
    'XGBoost': lambda: xgb.XGBClassifier(n_estimators=500, max_depth=6,
        learning_rate=0.04, subsample=0.8, colsample_bytree=0.7,
        tree_method='hist', verbosity=0),
    'CatBoost': lambda: cb.CatBoostClassifier(iterations=500, depth=8,
        learning_rate=0.06, l2_leaf_reg=3.0, verbose=0),
    'RandomForest': lambda: RandomForestClassifier(n_estimators=200,
        max_depth=12, min_samples_leaf=20, max_features=0.3,
        class_weight='balanced', n_jobs=-1),
}

# ─── 4. Grid search ───────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

print(f'\n{sepline}')
print('  Algorithm x Feature Set Grid Search')
print(sepline)

for alg_name, factory in algorithms.items():
    for fs_name, fgs in feature_sets.items():
        X = np.column_stack([groups[g] for g in fgs])
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0).astype(np.float32)

        oof = np.zeros(n)
        for fi, (ti, vi) in enumerate(skf.split(X, y)):
            clf = factory()
            pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
            if alg_name == 'LGBM':
                clf.set_params(scale_pos_weight=pw, random_state=42 + fi)
                clf.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])],
                        callbacks=[lgb.early_stopping(80, verbose=False),
                                   lgb.log_evaluation(0)])
            elif alg_name == 'XGBoost':
                clf.set_params(scale_pos_weight=pw, random_state=42 + fi)
                clf.fit(X[ti], y[ti])
            elif alg_name == 'CatBoost':
                clf.set_params(random_seed=42 + fi)
                clf.fit(X[ti], y[ti], eval_set=(X[vi], y[vi]),
                        early_stopping_rounds=80, verbose=False)
            elif alg_name == 'RandomForest':
                clf.set_params(random_state=42 + fi)
                clf.fit(X[ti], y[ti])
            oof[vi] = clf.predict_proba(X[vi])[:, 1]

        oof = np.nan_to_num(oof, nan=0.5)
        bf = max(f1_score(y, (oof > th).astype(int), zero_division=0)
                 for th in np.arange(0.05, 0.95, 0.01))
        auc = roc_auc_score(y, oof)
        results.append((alg_name, fs_name, X.shape[1], bf, auc, oof))
        elapsed = (time.time() - t0) / 60
        flag = '***' if bf > 0.70 else ''
        print(f'  {alg_name:12s} x {fs_name:12s} ({X.shape[1]:4d}d): '
              f'F1={bf:.4f} AUC={auc:.4f}  [{elapsed:.0f}min] {flag}')

# ─── 5. Best per algorithm ────────────────────────────────────────
print(f'\n{sepline}')
print('  BEST PER ALGORITHM')
print(sepline)
best_per_alg = {}
for alg_name in algorithms:
    alg_results = [r for r in results if r[0] == alg_name]
    best_r = max(alg_results, key=lambda x: x[3])
    best_per_alg[alg_name] = best_r
    print(f'  {alg_name:12s}: {best_r[1]:12s} ({best_r[2]:3d}d) '
          f'F1={best_r[3]:.4f} AUC={best_r[4]:.4f}')

# ─── 6. Build expert OOFs from best configs ───────────────────────
expert_oofs = {}
for alg_name, (alg, fs, dims, bf, auc, oof) in best_per_alg.items():
    expert_oofs[alg_name] = oof

# ─── 7. Load existing strong OOFs ─────────────────────────────────
existing = {}

our = np.load(os.path.join(OUT, 'tcn_kd_results.npz'))
for k, key in [('TCN+KD', 'oof_tcn_kd'), ('TCN-stacker', 'oof_stacker'),
                ('TCN-blend', 'oof_blend'), ('TCN-hill', 'oof_hill')]:
    if key in our.files:
        existing[k] = our[key]

existing['Super-GBDT'] = np.load(os.path.join(OUT, 'super_gbdt.npz'))['oof_super']
existing['SmartBlend'] = np.load(os.path.join(OUT, 'smart_blend.npz'))['oof_final']

multi = np.load(os.path.join(OUT, 'multi_oof_results.npz'))
for k in ['oof_lgb', 'oof_xgb', 'oof_cb']:
    if k in multi.files:
        existing['multi_' + k.replace('oof_', '')] = multi[k]

for tag, prefix, key in [('V213', 'v213_results_', 'oof_v213'),
                          ('V219', 'v219_results_', 'oof_final'),
                          ('V225', 'v225_results_', 'oof_final'),
                          ('V216', 'v216_results_', 'oof_final')]:
    fs = sorted(glob.glob(os.path.join(OD, f'{prefix}*.npz')), reverse=True)
    if fs:
        d = np.load(fs[0], allow_pickle=True)
        if key in d.files:
            existing[tag] = d[key]

for prefix, keys_, fmt in [('v71_oofs_', ['lgb', 'xgb', 'cat', 'tcn', 'innov'], 'V71_{}'),
                            ('v229_results_', ['oof_iso', 'oof_platt'], 'V229_{}')]:
    fs = sorted(glob.glob(os.path.join(OD, f'{prefix}*.npz')), reverse=True)
    if fs:
        d = np.load(fs[0], allow_pickle=True)
        for k in keys_:
            if k in d.files:
                existing[fmt.format(k)] = d[k]

# ─── 8. Correlation analysis ──────────────────────────────────────
all_oofs = {**expert_oofs, **existing}
P_all = np.column_stack([all_oofs[k] for k in all_oofs])
nn_all = list(all_oofs.keys())

corrs = np.corrcoef(P_all.T)
print(f'\n{sepline}')
print('  Expert OOF Correlation Matrix')
print(sepline)
print(f'  {"":14s} ', end='')
for nm in nn_all:
    print(f'{nm[:10]:>10s} ', end='')
print()
for i, nm_i in enumerate(nn_all):
    print(f'  {nm_i:14s} ', end='')
    for j in range(len(nn_all)):
        c = corrs[i, j]
        if c > 0.95:
             marker = 'H'
        elif c > 0.90:
             marker = 'h'
        else:
             marker = ' '
        print(f'{c:9.3f}{marker}', end=' ')
    print()

# ─── 9. Meta-learner stacking ─────────────────────────────────────
print(f'\n{sepline}')
print('  Meta-Learner Stacking')
print(sepline)

skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_configs = [
    ('XGB-4', lambda pw: xgb.XGBClassifier(n_estimators=300, max_depth=4,
        learning_rate=0.05, scale_pos_weight=pw, tree_method='hist',
        verbosity=0, random_state=42)),
    ('XGB-3', lambda pw: xgb.XGBClassifier(n_estimators=300, max_depth=3,
        learning_rate=0.05, scale_pos_weight=pw, tree_method='hist',
        verbosity=0, random_state=42)),
    ('LGB', lambda pw: lgb.LGBMClassifier(n_estimators=300, max_depth=3,
        learning_rate=0.05, scale_pos_weight=pw, verbose=-1, random_state=42)),
]

best_f1 = 0
best_oof = None
best_info = ''

for meta_name, factory in best_configs:
    oof = np.zeros(n)
    for fi, (ti, vi) in enumerate(skf2.split(P_all, y)):
        pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
        m = factory(pw)
        if 'LGB' in meta_name:
            m.fit(P_all[ti], y[ti], eval_set=[(P_all[vi], y[vi])],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(0)])
        else:
            m.fit(P_all[ti], y[ti])
        oof[vi] = m.predict_proba(P_all[vi])[:, 1]

    bf = max(f1_score(y, (oof > th).astype(int), zero_division=0)
             for th in np.arange(0.05, 0.95, 0.001))
    auc = roc_auc_score(y, oof)
    print(f'  {meta_name:10s}: F1={bf:.4f} AUC={auc:.4f}')
    if bf > best_f1:
        best_f1, best_oof, best_info = bf, oof, f'{meta_name} F1={bf:.4f}'

# ─── 10. Final report ─────────────────────────────────────────────
from sklearn.metrics import recall_score, precision_score

bt = 0.5
bf0 = 0
for th in np.arange(0.05, 0.95, 0.001):
    pred = (best_oof > th).astype(int)
    if pred.sum() == 0:
        continue
    f = f1_score(y, pred, zero_division=0)
    if f > bf0:
        bf0, bt = f, th
pred_final = (best_oof > bt).astype(int)
tp = ((pred_final == 1) & (y == 1)).sum()
fp = ((pred_final == 1) & (y == 0)).sum()
fn = ((pred_final == 0) & (y == 1)).sum()

print(f'\n{sepline}')
print('  FINAL: Heterogeneous Feature x Algorithm Ensemble')
print(sepline)
print(f'  New experts:  {len(expert_oofs)} (each on best feature set)')
print(f'  Existing OOFs: {len(existing)}')
print(f'  Total:         {len(all_oofs)}')
print(f'  Best meta:     {best_info}')
print(f'  F1=           {bf0:.4f}')
print(f'  AUC=          {roc_auc_score(y, best_oof):.4f}')
print(f'  Rec=          {tp/(tp+fn):.4f}')
print(f'  Prec=         {tp/(tp+fp) if tp+fp>0 else 0:.4f}')
print(f'  th=           {bt:.3f}')
print(f'  TP={tp}  FP={fp}  FN={fn}')
print()
print(f'  COMPARISON:')
print(f'  V225:                 F1=0.8457')
print(f'  Super-GBDT:           F1=0.8527')
print(f'  Mega Boost Enhanced:  F1=0.8616')
print(f'  Heterogeneous:        F1={bf0:.4f}')
print(f'  vs Enhanced:          {bf0 - 0.8616:+.4f}')
print(f'  vs V225:              {bf0 - 0.8457:+.4f}')

np.savez_compressed(
    os.path.join(OUT, 'heterogeneous_ensemble.npz'),
    oof_final=best_oof, y=y, f1=bf0, auc=roc_auc_score(y, best_oof),
    threshold=bt, tp=tp, fp=fp, fn=fn,
    names=np.array(nn_all),
)
print(f'\n  Saved to output/heterogeneous_ensemble.npz')
print(f'  Total time: {(time.time() - t0) / 60:.1f} min')
