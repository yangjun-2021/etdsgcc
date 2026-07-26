"""Train an XGB model to predict hillclimb's false negatives, and save OOF.

The target is binary: 1 if hillclimb (at its best threshold) is a false negative,
0 otherwise.  The OOF probabilities are produced by 5-fold CV and can be fed
into the meta-learner as an additional error-correction signal.
"""
import os
import sys
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SEED, N_FOLDS

flags = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags']
stat = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['stat_features']
hill = np.load(os.path.join(OUTPUT_DIR, 'hillclimb_best_oof.npz'))['oof_hillclimb']

best_f1, best_th = 0, 0.5
for th in np.arange(0.05, 0.95, 0.005):
    pred = (hill > th).astype(int)
    if pred.sum() == 0:
        continue
    f = f1_score(flags, pred, zero_division=0)
    if f > best_f1:
        best_f1, best_th = f, th
hill_pred = (hill > best_th).astype(int)
fn_target = ((hill_pred == 0) & (flags == 1)).astype(int)
print(f'Hillclimb th={best_th:.3f} F1={best_f1:.4f}, FN target rate={fn_target.mean()*100:.2f}%')

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(flags))
for fi, (ti, vi) in enumerate(skf.split(stat, fn_target)):
    pw = (fn_target[ti] == 0).sum() / max((fn_target[ti] == 1).sum(), 1)
    m = xgb.XGBClassifier(
        n_estimators=800, max_depth=6, learning_rate=0.03,
        scale_pos_weight=pw, tree_method='hist', verbosity=0,
        random_state=SEED + fi)
    m.fit(stat[ti], fn_target[ti])
    oof[vi] = m.predict_proba(stat[vi])[:, 1]
    fold_auc = roc_auc_score(fn_target[vi], oof[vi])
    print(f'  Fold {fi+1}: AUC={fold_auc:.4f}')

overall_auc = roc_auc_score(fn_target, oof)
print(f'Overall FN predictor AUC={overall_auc:.4f}')

np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'hillclimb_fn_predictor_oof.npz'),
    oof_hillclimb_fn_predictor=oof,
    flags=flags,
)
print(f'Saved to {os.path.join(OUTPUT_DIR, "hillclimb_fn_predictor_oof.npz")}')
