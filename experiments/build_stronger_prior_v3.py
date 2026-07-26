"""Build stronger GBDT prior v3 by including all recent strong OOF signals."""
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

# Collect all available OOFs (external + internal)
oofs = {}

# External bundled
try:
    bd = np.load(os.path.join(OUTPUT_DIR, 'bundled_oofs.npz'), allow_pickle=True)
    for i, name in enumerate(bd['names']):
        key = f'oof_{i}'
        if key in bd.files:
            oofs[f'external_{name}'] = np.nan_to_num(bd[key], nan=0.5)
except Exception as e:
    print(f'No bundled_oofs: {e}')

# TCN-KD
try:
    tcn = np.load(os.path.join(OUTPUT_DIR, 'tcn_kd_results.npz'))
    for k in ['oof_tcn_kd', 'oof_stacker', 'oof_blend', 'oof_hill']:
        if k in tcn.files:
            oofs[k] = np.nan_to_num(tcn[k], nan=0.5)
except Exception as e:
    print(f'No tcn_kd: {e}')

# Other saved OOFs / blends
for f, k in [
    ('smart_blend.npz', 'oof_final'),
    ('super_gbdt.npz', 'oof_super'),
    ('mega_hillclimb.npz', 'oof_final'),
    ('autoresearch_best.npz', 'oof_final'),
    ('mega_boost_enhanced.npz', 'oof_final'),
    ('feature_rich_meta_oof.npz', 'oof_feature_rich_meta'),
    ('meta_raw_only_oof.npz', 'oof_meta_raw_only'),
    ('nn_meta_oof.npz', 'oof_nn_meta'),
    ('informer_oof.npz', 'oof_informer'),
    ('amst_3ch_recall10_oof.npz', 'oof_amst_3ch_recall10'),
    ('amst_3ch_strong_prior_oof.npz', 'oof_amst_3ch_strong_prior'),
    ('amst_3ch_synthetic_fast_oof.npz', 'oof_amst_3ch_synthetic_fast'),
    ('amst_3ch_synthetic_mixed_fast_oof.npz', 'oof_amst_3ch_synthetic_mixed_fast'),
    ('amst_3ch_synthetic_mixed_ls_fast_oof.npz', 'oof_amst_3ch_synthetic_mixed_ls_fast'),
    ('amst_3ch_synthetic_x3_sp_fast_oof.npz', 'oof_amst_3ch_synthetic_x3_sp_fast'),
    ('patch_transformer_raw_3ch_recall_oof.npz', 'oof_patch_transformer_raw_3ch_recall'),
    ('patch_transformer_raw_3ch_synthetic_sp_oof.npz', 'oof_patch_transformer_raw_3ch_synthetic_sp'),
    ('informer_3ch_synthetic_sp_oof.npz', 'oof_informer_3ch_synthetic_sp'),
    ('supcon_raw_3ch_oof.npz', 'oof_supcon_raw_3ch'),
    ('strong_gbdt_prior_oof.npz', 'oof_strong_gbdt_prior'),
]:
    try:
        d = np.load(os.path.join(OUTPUT_DIR, f))
        if k in d.files and len(d[k]) == len(y):
            oofs[f'{f.replace(".npz","")}_{k}'] = np.nan_to_num(d[k], nan=0.5)
    except Exception:
        pass

# Filter to OOFs with AUC > 0.965
strong_oofs = []
for name, oof in oofs.items():
    try:
        auc = roc_auc_score(y, oof)
        if auc > 0.965:
            strong_oofs.append(oof)
            print(f'Using {name}: AUC={auc:.4f}')
    except Exception:
        pass

if len(strong_oofs) == 0:
    raise ValueError('No strong OOFs found')

X_oofs = np.column_stack(strong_oofs)
X = np.column_stack([
    np.nan_to_num(stat, nan=0, posinf=0, neginf=0),
    miss,
    X_oofs,
]).astype(np.float32)
print(f'Feature matrix: {X.shape}')

# Train 5-fold LGB prior with stronger config and recall orientation
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
prior = np.zeros(len(y))
for fi, (ti, vi) in enumerate(skf.split(X, y)):
    pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
    m = lgb.LGBMClassifier(
        n_estimators=3000, max_depth=8, learning_rate=0.02,
        num_leaves=127, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        scale_pos_weight=pw * 1.5,
        random_state=SEED, verbose=-1)
    m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
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
print(f'\nStronger prior v3: F1={f1_score(y, pred):.4f}, Rec={recall_score(y, pred):.4f}, '
      f'Prec={precision_score(y, pred, zero_division=0):.4f}, AUC={roc_auc_score(y, prior):.4f}, th={best_th:.3f}')

np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'stronger_gbdt_prior_v3.npz'),
    prior=prior,
    flags=y,
)
print(f'Saved to {os.path.join(OUTPUT_DIR, "stronger_gbdt_prior_v3.npz")}')
