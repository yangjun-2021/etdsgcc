"""Build recall-oriented strong GBDT priors with different scale_pos_weights."""
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
try:
    bd = np.load(os.path.join(OUTPUT_DIR, 'bundled_oofs.npz'), allow_pickle=True)
    for i, name in enumerate(bd['names']):
        key = f'oof_{i}'
        if key in bd.files:
            oofs[name] = np.nan_to_num(bd[key], nan=0.5)
except Exception as e:
    print(f'bundled_oofs.npz not loaded: {e}')

try:
    tcn = np.load(os.path.join(OUTPUT_DIR, 'tcn_kd_results.npz'))
    for k in ['oof_tcn_kd', 'oof_stacker']:
        if k in tcn.files:
            oofs[k] = np.nan_to_num(tcn[k], nan=0.5)
except Exception:
    pass

for f, k in [('smart_blend.npz', 'oof_final'), ('super_gbdt.npz', 'oof_super')]:
    try:
        d = np.load(os.path.join(OUTPUT_DIR, f))
        oofs[k] = np.nan_to_num(d[k], nan=0.5)
    except Exception:
        pass

# Also include current strong internal OOFs
for f, k in [
    ('amst_3ch_recall10_oof.npz', 'oof_amst_3ch_recall10'),
    ('informer_oof.npz', 'oof_informer'),
    ('patch_transformer_raw_3ch_oof.npz', 'oof_patch_transformer_raw_3ch'),
    ('supcon_raw_3ch_oof.npz', 'oof_supcon_raw_3ch'),
    ('strong_gbdt_prior_oof.npz', 'oof_strong_gbdt_prior'),
]:
    try:
        d = np.load(os.path.join(OUTPUT_DIR, f))
        if k in d.files:
            oofs[f'{f}_{k}'] = np.nan_to_num(d[k], nan=0.5)
    except Exception:
        pass

strong_oofs = []
for name, oof in oofs.items():
    try:
        if roc_auc_score(y, oof) > 0.98:
            strong_oofs.append(oof)
            print(f'Using {name}: AUC={roc_auc_score(y, oof):.4f}')
    except Exception:
        pass

if not strong_oofs:
    print('No strong OOFs found, using stat+miss only')

X_oofs = np.column_stack(strong_oofs) if strong_oofs else np.zeros((len(y), 0))
X = np.column_stack([
    np.nan_to_num(stat, nan=0, posinf=0, neginf=0),
    miss,
    X_oofs,
]).astype(np.float32)
print(f'Feature matrix: {X.shape}')

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for sw_mult in [1.0, 2.0, 3.0, 5.0]:
    prior = np.zeros(len(y))
    for fi, (ti, vi) in enumerate(skf.split(X, y)):
        pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
        m = lgb.LGBMClassifier(
            n_estimators=1000, max_depth=7, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pw * sw_mult, random_state=SEED, verbose=-1)
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
    print(f'\nscale_pos_weight={sw_mult:.1f}x: F1={f1_score(y, pred):.4f}, '
          f'Rec={recall_score(y, pred):.4f}, Prec={precision_score(y, pred, zero_division=0):.4f}, '
          f'AUC={roc_auc_score(y, prior):.4f}, th={best_th:.3f}')

    # Save best recall-oriented prior
    if sw_mult == 3.0:
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, 'strong_gbdt_prior_recall3x.npz'),
            prior=prior, flags=y,
        )
        print('Saved strong_gbdt_prior_recall3x.npz')
