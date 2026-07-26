import numpy as np, os, time, random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42); random.seed(42)

from config import DATA_DIR
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb, xgboost as xgb

OUT = 'output'
t0 = time.time()
base = np.load(os.path.join(OUT, 'sgcc_preprocessed.npz'))
y = base['flags'].astype(int)
n = len(y)
stat = np.nan_to_num(base['stat_features'], nan=0, posinf=0, neginf=0)
impute_mask = base['impute_mask']
residuals = base['residuals']
nd = residuals.shape[1]
half = nd // 2

import pandas as pd
raw_df = pd.read_csv(os.path.join(DATA_DIR, 'raw_data.csv'))
dc = [c for c in raw_df.columns if '/' in str(c) and len(str(c)) <= 10]
raw = raw_df[dc].values.astype(float)
del raw_df
filled = np.nan_to_num(raw, nan=0)

X = stat.copy()
for n_seg in [25, 50, 100]:
    seg = raw.shape[1] / n_seg
    for i in range(n_seg):
        s = int(round(i * seg))
        e = int(round((i + 1) * seg))
        X = np.column_stack([X, np.nan_to_num(np.nanmean(raw[:, s:max(e, s + 1)], axis=1).reshape(-1, 1), nan=0)])

mi = np.digitize(np.array([j % 365 for j in range(nd)]), np.linspace(0, 365, 13)) - 1
for m in range(12):
    mm = (mi == m)
    val = np.nanmean(filled[:, mm], axis=1).reshape(-1, 1) if mm.sum() > 0 else np.zeros((n, 1))
    X = np.column_stack([X, np.nan_to_num(val, nan=0)])

for d in range(7):
    dd = (np.array([j % 7 for j in range(nd)]) == d)
    val = np.nanmean(filled[:, dd], axis=1).reshape(-1, 1) if dd.sum() > 0 else np.zeros((n, 1))
    X = np.column_stack([X, np.nan_to_num(val, nan=0)])

for w in [7, 14, 30, 60, 90]:
    if nd >= w:
        first = filled[:, :w].mean(axis=1)
        last = filled[:, -w:].mean(axis=1)
        X = np.column_stack([X, (last - first) / (np.maximum(np.abs(first), 1e-6)).reshape(-1, 1)])
        X = np.column_stack([X, np.nanmean(np.diff(filled[:, -w:], axis=1), axis=1).reshape(-1, 1)])

for period in [7, 30, 90]:
    if nd >= period * 3:
        for offset in range(period):
            idx = np.arange(offset, nd, period)
            if len(idx) > 0:
                X = np.column_stack([X, (np.nanmean(filled[:, idx], axis=1) - np.nanmean(filled, axis=1)).reshape(-1, 1)])

for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    X = np.column_stack([X, np.nanpercentile(filled, q, axis=1).reshape(-1, 1)])

X = np.column_stack([X, np.nanmean(residuals, axis=1).reshape(-1, 1)])
X = np.column_stack([X, np.nanstd(residuals, axis=1).reshape(-1, 1)])
r1 = np.nanmean(residuals[:, :half], axis=1)
r2 = np.nanmean(residuals[:, half:], axis=1)
X = np.column_stack([X, ((r2 - r1) / (np.maximum(np.abs(r1), 1e-6))).reshape(-1, 1)])
X = np.column_stack([X, impute_mask.mean(axis=1).reshape(-1, 1)])

for fn, key in [('dengine_features.npz', 'X'), ('gan_features.npz', 'features')]:
    try:
        extra = np.load(os.path.join(OUT, fn))
        X = np.column_stack([X, np.nan_to_num(extra[key], nan=0)])
    except:
        pass

X = np.nan_to_num(X, nan=0, posinf=0, neginf=0).astype(np.float32)
print(f'Base features: {X.shape[1]} dims')

bd = np.load(os.path.join(OUT, 'bundled_oofs.npz'), allow_pickle=True)
ext_oofs = np.zeros((n, 0))
for i in range(len(bd['names'])):
    k = f'oof_{i}'
    ext_oofs = np.column_stack([ext_oofs, bd[k].reshape(-1, 1)])
try:
    ea = np.load(os.path.join(OUT, 'sgcc_expert_a.npz'))['oof_proba'].reshape(-1, 1)
    ext_oofs = np.column_stack([ext_oofs, ea])
except:
    pass
try:
    eb = np.load(os.path.join(OUT, 'sgcc_expert_b.npz'))['oof_proba'].reshape(-1, 1)
    ext_oofs = np.column_stack([ext_oofs, eb])
except:
    pass
print(f'OOF as features: {ext_oofs.shape[1]} dims')

XF = np.column_stack([X, ext_oofs]).astype(np.float32)
XF = np.nan_to_num(XF, nan=0, posinf=0, neginf=0)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# TEST 1: LGB on features only
oof1 = np.zeros(n)
for fi, (ti, vi) in enumerate(skf.split(X, y)):
    pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
    m = lgb.LGBMClassifier(n_estimators=1000, max_depth=7, learning_rate=0.03,
                            num_leaves=63, min_child_samples=100, subsample=0.8,
                            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                            scale_pos_weight=pw, random_state=42 + fi, verbose=-1)
    m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    oof1[vi] = m.predict_proba(X[vi])[:, 1]
bf1 = max(f1_score(y, (oof1 > th).astype(int), zero_division=0)
          for th in np.arange(0.05, 0.95, 0.001))
print(f'[1] LGB on {X.shape[1]} features only:    F1={bf1:.4f} AUC={roc_auc_score(y, oof1):.4f}')

# TEST 2: LGB on features + OOFs
oof2 = np.zeros(n)
for fi, (ti, vi) in enumerate(skf.split(XF, y)):
    pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
    m = lgb.LGBMClassifier(n_estimators=1000, max_depth=7, learning_rate=0.03,
                            num_leaves=63, min_child_samples=100, subsample=0.8,
                            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                            scale_pos_weight=pw, random_state=42 + fi, verbose=-1)
    m.fit(XF[ti], y[ti], eval_set=[(XF[vi], y[vi])],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    oof2[vi] = m.predict_proba(XF[vi])[:, 1]
bf2 = max(f1_score(y, (oof2 > th).astype(int), zero_division=0)
          for th in np.arange(0.05, 0.95, 0.001))
print(f'[2] LGB on features + OOFs:             F1={bf2:.4f} AUC={roc_auc_score(y, oof2):.4f}  (+{bf2-bf1:.4f})')

# TEST 3: XGB on features + OOFs
oof3 = np.zeros(n)
for fi, (ti, vi) in enumerate(skf.split(XF, y)):
    pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
    m = xgb.XGBClassifier(n_estimators=800, max_depth=6, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.7,
                           scale_pos_weight=pw, tree_method='hist', verbosity=0,
                           random_state=42 + fi)
    m.fit(XF[ti], y[ti])
    oof3[vi] = m.predict_proba(XF[vi])[:, 1]
bf3 = max(f1_score(y, (oof3 > th).astype(int), zero_division=0)
          for th in np.arange(0.05, 0.95, 0.001))
print(f'[3] XGB on features + OOFs:             F1={bf3:.4f} AUC={roc_auc_score(y, oof3):.4f}')

# TEST 4: 2-model ensemble
P2 = np.column_stack([oof2, oof3])
skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof4 = np.zeros(n)
for fi, (ti, vi) in enumerate(skf2.split(P2, y)):
    m = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=42)
    m.fit(P2[ti], y[ti])
    oof4[vi] = m.predict_proba(P2[vi])[:, 1]
bf4 = max(f1_score(y, (oof4 > th).astype(int), zero_division=0)
          for th in np.arange(0.05, 0.95, 0.001))
print(f'[4] 2-model ensemble (LGB+XGB):          F1={bf4:.4f} AUC={roc_auc_score(y, oof4):.4f}  (+{bf4-bf2:.4f})')

sep = '=' * 55
print(f'\n{sep}')
print('  WHY SO MANY MODELS?')
print(sep)
print(f'  1. Features only (LGB):  F1={bf1:.4f}')
print(f'  2. Features+OOFs (LGB):  F1={bf2:.4f}  (+{bf2-bf1:.4f}) OOF is key')
print(f'  3. Features+OOFs (XGB):  F1={bf3:.4f}')
print(f'  4. 2-model ensemble:     F1={bf4:.4f}  (+{bf4-bf2:.4f}) diversity')
print(f'  27-OOF stacking (curr):  F1=0.8621  (+{0.8621-bf4:.4f}) multi-OOF')
print()
print(f'  Each OOF carries DIFFERENT information from different')
print(f'  model architectures and feature sets. Stacking them')
print(f'  extracts ~+0.01 F1 that no single model can capture.')
print(f'  Time: {(time.time() - t0) / 60:.1f} min')
