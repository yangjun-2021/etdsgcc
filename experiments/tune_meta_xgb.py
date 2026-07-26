"""Tune XGB/LGB meta-learners on the current OOF matrix."""
import os
import sys
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything
from src.training.meta_learner import _load_internal_oofs, _load_external_oofs

seed_everything(SEED)

flags = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags']
impute_mask = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['impute_mask']
a = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_a.npz'))['oof_proba']
b = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_b.npz'))['oof_proba']

all_oofs = {}
for name, oof in _load_internal_oofs(flags).items():
    all_oofs[name] = oof
for name, oof in _load_external_oofs(flags).items():
    all_oofs[name] = oof
all_oofs['Expert-A(GBDT)'] = a
all_oofs['Expert-B(TCN)'] = b

names = sorted(all_oofs.keys())
P_tmp = np.column_stack([all_oofs[nm] for nm in names])
P_tmp = np.nan_to_num(P_tmp, nan=0.5, posinf=1.0, neginf=0.0)
corrs = np.corrcoef(P_tmp.T)
kept = []
for i, nm in enumerate(names):
    drop = False
    for j in kept:
        if abs(corrs[i, names.index(j)]) > 0.999:
            drop = True; break
    if not drop:
        kept.append(nm)
P = np.column_stack([all_oofs[nm] for nm in kept])
P = np.nan_to_num(P, nan=0.5, posinf=1.0, neginf=0.0)
P = np.column_stack([P, impute_mask.mean(axis=1).reshape(-1, 1)])
print(f'OOF matrix: {P.shape}')

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

def eval_meta(factory, name):
    oof = np.zeros(len(flags))
    for fi, (ti, vi) in enumerate(skf.split(P, flags)):
        pw = (flags[ti]==0).sum() / max((flags[ti]==1).sum(), 1)
        m = factory(pw)
        if 'LGB' in name:
            m.fit(P[ti], flags[ti], eval_set=[(P[vi], flags[vi])],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        else:
            m.fit(P[ti], flags[ti])
        oof[vi] = m.predict_proba(P[vi])[:, 1]
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (oof > th).astype(int)
        if pred.sum() == 0: continue
        f = f1_score(flags, pred, zero_division=0)
        if f > best_f1: best_f1, best_th = f, th
    pred = (oof > best_th).astype(int)
    print(f'{name:30s}: F1={best_f1:.4f} Rec={recall_score(flags,pred):.4f} '
          f'Prec={precision_score(flags,pred,zero_division=0):.4f} AUC={roc_auc_score(flags,oof):.4f} th={best_th:.3f}')
    return best_f1

configs = [
    ('XGB-d4-500', lambda pw: xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.03, scale_pos_weight=pw, tree_method='hist', verbosity=0, random_state=SEED)),
    ('XGB-d5-500', lambda pw: xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.03, scale_pos_weight=pw, tree_method='hist', verbosity=0, random_state=SEED)),
    ('XGB-d6-500', lambda pw: xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, scale_pos_weight=pw, tree_method='hist', verbosity=0, random_state=SEED)),
    ('XGB-d4-1000', lambda pw: xgb.XGBClassifier(n_estimators=1000, max_depth=4, learning_rate=0.01, scale_pos_weight=pw, tree_method='hist', verbosity=0, random_state=SEED)),
    ('XGB-d3-1000', lambda pw: xgb.XGBClassifier(n_estimators=1000, max_depth=3, learning_rate=0.01, scale_pos_weight=pw, tree_method='hist', verbosity=0, random_state=SEED)),
    ('LGB-800', lambda pw: lgb.LGBMClassifier(n_estimators=800, max_depth=5, num_leaves=31, learning_rate=0.03, scale_pos_weight=pw, verbose=-1, random_state=SEED)),
    ('LGB-1500', lambda pw: lgb.LGBMClassifier(n_estimators=1500, max_depth=4, num_leaves=31, learning_rate=0.02, scale_pos_weight=pw, verbose=-1, random_state=SEED)),
]
for name, factory in configs:
    eval_meta(factory, name)
