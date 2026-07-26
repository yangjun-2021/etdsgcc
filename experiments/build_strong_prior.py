"""Build a strong GBDT prior for Informer using available OOFs."""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import lightgbm as lgb
import numpy as np

seed_everything(SEED)

y = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags']
stat = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['stat_features']
miss = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['impute_mask'].mean(axis=1).reshape(-1, 1)

# Collect strong OOFs
oofs = {}

bd = np.load(os.path.join(OUTPUT_DIR, 'bundled_oofs.npz'), allow_pickle=True)
for i, name in enumerate(bd['names']):
    key = f'oof_{i}'
    if key in bd.files:
        oofs[name] = np.nan_to_num(bd[key], nan=0.5)

tcn = np.load(os.path.join(OUTPUT_DIR, 'tcn_kd_results.npz'))
for k in ['oof_tcn_kd', 'oof_stacker']:
    if k in tcn.files:
        oofs[k] = np.nan_to_num(tcn[k], nan=0.5)

for f, k in [('smart_blend.npz', 'oof_final'), ('super_gbdt.npz', 'oof_super')]:
    try:
        d = np.load(os.path.join(OUTPUT_DIR, f))
        oofs[k] = np.nan_to_num(d[k], nan=0.5)
    except Exception:
        pass

# Select OOFs with AUC > 0.98
strong_oofs = []
for name, oof in oofs.items():
    if roc_auc_score(y, oof) > 0.98:
        strong_oofs.append(oof)
        print(f'Using {name}: AUC={roc_auc_score(y, oof):.4f}')

X_oofs = np.column_stack(strong_oofs)
X = np.column_stack([
    np.nan_to_num(stat, nan=0, posinf=0, neginf=0),
    miss,
    X_oofs,
]).astype(np.float32)
print(f'Feature matrix: {X.shape}')

# Train 5-fold LGB prior
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
prior = np.zeros(len(y))
for fi, (ti, vi) in enumerate(skf.split(X, y)):
    pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
    m = lgb.LGBMClassifier(
        n_estimators=1000, max_depth=7, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pw, random_state=SEED, verbose=-1)
    m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    prior[vi] = m.predict_proba(X[vi])[:, 1]

best_f1, best_th = 0, 0.5
for th in np.arange(0.05, 0.95, 0.005):
    pred = (prior > th).astype(int)
    if pred.sum() == 0:
        continue
    f = f1_score(y, pred, zero_division=0)
    if f > best_f1:
        best_f1, best_th = f, th
pred = (prior > best_th).astype(int)
print(f'\nStrong prior: F1={f1_score(y, pred):.4f}, Rec={recall_score(y, pred):.4f}, '
      f'Prec={precision_score(y, pred, zero_division=0):.4f}, AUC={roc_auc_score(y, prior):.4f}, th={best_th:.3f}')

np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'strong_gbdt_prior.npz'),
    prior=prior,
    flags=y,
)
print(f'Saved to {os.path.join(OUTPUT_DIR, "strong_gbdt_prior.npz")}')
