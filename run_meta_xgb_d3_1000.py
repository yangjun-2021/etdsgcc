"""Train a strong XGB meta-learner (d3, 1000 est) on the current OOF pool."""
import os
import sys
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything
from src.training.meta_learner import _load_internal_oofs, _load_external_oofs
from src.evaluation.evaluate import evaluate_dataset
import pickle

seed_everything(SEED)

flags = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags']
mask = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['impute_mask']
a = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_a.npz'))['oof_proba']
b = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_b.npz'))['oof_proba']

all_oofs = {}
for n, o in _load_internal_oofs(flags).items():
    all_oofs[n] = o
for n, o in _load_external_oofs(flags).items():
    all_oofs[n] = o
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
P = np.column_stack([P, mask.mean(axis=1).reshape(-1, 1)])
print(f'OOF matrix: {P.shape}')

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(flags))
for fi, (ti, vi) in enumerate(skf.split(P, flags)):
    pw = (flags[ti] == 0).sum() / max((flags[ti] == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=1000, max_depth=3, learning_rate=0.01,
        scale_pos_weight=pw, tree_method='hist',
        verbosity=0, random_state=SEED)
    model.fit(P[ti], flags[ti])
    oof[vi] = model.predict_proba(P[vi])[:, 1]

best_f1, best_th = 0, 0.5
for th in np.arange(0.05, 0.95, 0.005):
    pred = (oof > th).astype(int)
    if pred.sum() == 0: continue
    f = f1_score(flags, pred, zero_division=0)
    if f > best_f1: best_f1, best_th = f, th
pred = (oof > best_th).astype(int)
print(f'XGB-d3-1000 meta: F1={best_f1:.4f}, Rec={recall_score(flags, pred):.4f}, '
      f'Prec={precision_score(flags, pred, zero_division=0):.4f}, AUC={roc_auc_score(flags, oof):.4f}, th={best_th:.3f}')

np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'sgcc_mega_meta_xgb_d3_1000.npz'),
    oof_final=oof, labels=flags, f1=best_f1, auc=roc_auc_score(flags, oof),
    threshold=best_th,
)
results = {
    'oof_proba_meta': oof,
    'oof_proba_a': a,
    'oof_proba_b': b,
    'best_f1': best_f1,
    'best_th': best_th,
    'best_recall': recall_score(flags, pred, zero_division=0),
    'best_precision': precision_score(flags, pred, zero_division=0),
    'flags': flags,
}
with open(os.path.join(OUTPUT_DIR, 'sgcc_meta_results_xgb_d3_1000.pkl'), 'wb') as f:
    pickle.dump(results, f)

evaluate_dataset('sgcc', results, OUTPUT_DIR)
