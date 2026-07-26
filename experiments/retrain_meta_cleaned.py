"""Retrain meta-learner on cleaned labels using updated Expert A OOF."""
import os, sys
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything
from src.training.meta_learner import _load_internal_oofs, _load_external_oofs

seed_everything(SEED)


def best_f1_score(y_true, y_prob):
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (y_prob > th).astype(int)
        if pred.sum() == 0: continue
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_th = f, th
    return best_f1, best_th


def main():
    # Load cleaned labels
    cl = np.load(os.path.join(OUTPUT_DIR, 'cleaned_labels_v1.npz'))
    y_clean = cl['y_clean'].astype(int)
    y_orig = cl['y_orig'].astype(int)

    # Load OOFs
    oofs = {}
    oofs.update(_load_internal_oofs(y_clean))
    oofs.update(_load_external_oofs(y_clean))

    # Use cleaned Expert A if available
    try:
        a = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_a_cleaned.npz'))
        oofs['Expert-A-cleaned'] = a['oof_proba']
        print('Loaded cleaned Expert A OOF')
    except Exception as e:
        print('Cleaned Expert A not found:', e)

    # Also keep original Expert A/B
    try:
        oofs['Expert-A(GBDT)'] = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_a.npz'))['oof_proba']
    except Exception:
        pass
    try:
        oofs['Expert-B(TCN)'] = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_b.npz'))['oof_proba']
    except Exception:
        pass

    oofs = {k: v for k, v in oofs.items() if len(v) == len(y_clean)}
    names = sorted(oofs.keys())
    print(f'Using {len(names)} OOFs')
    for n in names:
        bf = best_f1_score(y_clean, oofs[n])[0]
        print(f'  {n:30s}: cleaned-F1={bf:.4f}')

    P = np.column_stack([oofs[n] for n in names])
    P = np.nan_to_num(P, nan=0.5, posinf=1.0, neginf=0.0)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y_clean))

    for fi, (ti, vi) in enumerate(skf.split(P, y_clean)):
        pw = (y_clean[ti] == 0).sum() / max((y_clean[ti] == 1).sum(), 1)
        m = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                              scale_pos_weight=pw, tree_method='hist',
                              random_state=SEED+fi, verbosity=0)
        m.fit(P[ti], y_clean[ti])
        oof[vi] = m.predict_proba(P[vi])[:, 1]

    # Evaluate on cleaned labels
    bf, th = best_f1_score(y_clean, oof)
    pred = (oof > th).astype(int)
    print(f'\nMeta on cleaned labels: F1={bf:.4f}, Rec={recall_score(y_clean,pred):.4f}, '
          f'Prec={precision_score(y_clean,pred,zero_division=0):.4f}, AUC={roc_auc_score(y_clean,oof):.4f}, th={th:.3f}')

    # Evaluate on original labels
    bf, th = best_f1_score(y_orig, oof)
    pred = (oof > th).astype(int)
    print(f'Meta on original labels: F1={bf:.4f}, Rec={recall_score(y_orig,pred):.4f}, '
          f'Prec={precision_score(y_orig,pred,zero_division=0):.4f}, AUC={roc_auc_score(y_orig,oof):.4f}, th={th:.3f}')

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'meta_cleaned_oof.npz'),
        oof_meta_cleaned=oof,
        y_clean=y_clean,
        y_orig=y_orig,
    )


if __name__ == '__main__':
    main()
