"""Add false-negative-targeted features and train a GBDT ensemble (vectorized)."""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

seed_everything(SEED)

pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
y = pre['flags'].astype(int)
stat = np.nan_to_num(pre['stat_features'], nan=0.0, posinf=0.0, neginf=0.0)
mask = pre['impute_mask'].astype(bool)
X_seq = pre['X_seq']

val = X_seq[:, 0, :].astype(np.float64)
n, t = val.shape
print(f'n={n}, t={t}, base stat dims={stat.shape[1]}')

obs = val.copy()
obs[~mask] = np.nan

new_feats = []

# 1. Long zero-streak features (vectorized)
print('Computing zero-streak features...')
zero_obs = (obs == 0).astype(int)
# Pad to detect transitions
padded = np.concatenate([np.zeros((n, 1)), zero_obs, np.zeros((n, 1))], axis=1)
diff = np.diff(padded, axis=1)
# start positions: diff == 1, end positions: diff == -1
starts = [np.where(diff[i] == 1)[0] for i in range(n)]
ends = [np.where(diff[i] == -1)[0] for i in range(n)]
max_zero_run = np.array([np.max(e - s) if len(s) > 0 else 0 for s, e in zip(starts, ends)])
n_zero_runs = np.array([len(s) for s in starts])
new_feats.extend([max_zero_run, n_zero_runs])

# 2. Low-consumption ratio among observed data
print('Computing low-consumption ratio...')
low_thresh = 0.5
low_ratio = np.nansum(obs < low_thresh, axis=1) / np.maximum(mask.sum(axis=1), 1)
new_feats.append(low_ratio)

# 3. Flatline detection: proportion of low-variance windows (batched vectorized)
print('Computing flatline features...')
def rolling_var_batched(x, m, w):
    """x: [N, T], m: [N, T] bool mask. Return [N, T-w+1] window variance ignoring NaN."""
    n_local = x.shape[0]
    tw = t - w + 1
    out = np.full((n_local, tw), np.nan, dtype=np.float64)
    batch_size = 2000
    for start in range(0, n_local, batch_size):
        end = min(start + batch_size, n_local)
        # [B, tw, w]
        windows = np.lib.stride_tricks.sliding_window_view(x[start:end], window_shape=(w,), axis=1)
        masks = np.lib.stride_tricks.sliding_window_view(m[start:end], window_shape=(w,), axis=1)
        counts = masks.sum(axis=2)
        sums = np.nansum(np.where(masks, windows, 0), axis=2)
        sumsq = np.nansum(np.where(masks, windows ** 2, 0), axis=2)
        valid = counts > 1
        mean = sums / np.maximum(counts, 1)
        var = sumsq / np.maximum(counts, 1) - mean ** 2
        var[~valid] = np.nan
        out[start:end] = var
    return out

for w in [7, 30]:
    if t >= w:
        rv = rolling_var_batched(obs, mask, w)
        flat_ratio_w = np.nansum(rv < 0.01, axis=1) / max(rv.shape[1], 1)
        new_feats.append(flat_ratio_w)
        print(f'  w={w}: done')

# 4. Head-vs-tail drop
print('Computing head-tail drop features...')
for w in [30, 90]:
    if t >= 2 * w:
        head = np.nanmean(obs[:, :w], axis=1)
        tail = np.nanmean(obs[:, -w:], axis=1)
        drop = np.where(np.abs(head) > 1e-6, (head - tail) / (np.abs(head) + 1e-6), 0)
        new_feats.append(drop)

# 5. Consumption trend
print('Computing trend features...')
time = np.arange(t)
trend = np.zeros(n)
for i in range(n):
    valid = mask[i]
    if valid.sum() >= 2:
        coef = np.polyfit(time[valid], val[i, valid], 1)
        trend[i] = coef[0]
new_feats.append(trend)

# 6. Missing interaction: missing ratio in low-consumption regions
print('Computing missing-in-low features...')
miss_in_low = np.zeros(n)
for i in range(n):
    low_mask = val[i] < low_thresh
    if low_mask.sum() > 0:
        miss_in_low[i] = (~mask[i] & low_mask).sum() / low_mask.sum()
new_feats.append(miss_in_low)

X_new = np.column_stack([f.reshape(-1, 1) for f in new_feats])
X_new = np.nan_to_num(X_new, nan=0, posinf=0, neginf=0).astype(np.float32)
X_full = np.column_stack([stat, X_new])
print(f'Full feature matrix: {X_full.shape}')

# Train GBDT ensemble
print('Training GBDT ensemble...')
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_lgb = np.zeros(n)
oof_xgb = np.zeros(n)

for fi, (ti, vi) in enumerate(skf.split(X_full, y)):
    print(f'Fold {fi+1}/5')
    pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)

    m = lgb.LGBMClassifier(
        n_estimators=1000, max_depth=7, learning_rate=0.03, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=0.05,
        min_child_samples=50, scale_pos_weight=pw, random_state=SEED + fi, verbose=-1)
    m.fit(X_full[ti], y[ti], eval_set=[(X_full[vi], y[vi])],
          callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    oof_lgb[vi] = m.predict_proba(X_full[vi])[:, 1]

    m = xgb.XGBClassifier(
        n_estimators=1000, max_depth=6, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=0.05, min_child_weight=5,
        scale_pos_weight=pw, tree_method='hist', verbosity=0, random_state=SEED + fi)
    m.fit(X_full[ti], y[ti])
    oof_xgb[vi] = m.predict_proba(X_full[vi])[:, 1]

# Quick blend sweep
print('Blend sweep:')
for w_lgb in np.arange(0.3, 0.8, 0.1):
    ens = w_lgb * oof_lgb + (1 - w_lgb) * oof_xgb
    best_f1 = 0
    for th in np.arange(0.05, 0.95, 0.005):
        f = f1_score(y, (ens > th).astype(int), zero_division=0)
        if f > best_f1: best_f1 = f
    print(f'  LGB weight {w_lgb:.1f}: best F1={best_f1:.4f}')

ens = 0.5 * oof_lgb + 0.5 * oof_xgb
best_f1, best_th = 0, 0.5
for th in np.arange(0.05, 0.95, 0.001):
    pred = (ens > th).astype(int)
    if pred.sum() == 0: continue
    f = f1_score(y, pred, zero_division=0)
    if f > best_f1: best_f1, best_th = f, th
pred = (ens > best_th).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
print(f'\nBest: F1={best_f1:.4f}, th={best_th:.3f}, Rec={recall_score(y,pred):.4f}, '
      f'Prec={precision_score(y,pred):.4f}, AUC={roc_auc_score(y,ens):.4f}, TP={tp}, FP={fp}, FN={fn}')

np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'fn_targeted_gbdt_oof.npz'),
    oof_fn_targeted_gbdt=ens, flags=y,
)
print(f'Saved to output/fn_targeted_gbdt_oof.npz')
