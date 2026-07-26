"""Retrain Expert A on aggressive cleaned labels v2."""
import os, sys
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything

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
    base = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
    X = np.nan_to_num(base['stat_features'], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    cl = np.load(os.path.join(OUTPUT_DIR, 'cleaned_labels_v2.npz'))
    y_clean = cl['y_clean'].astype(int)
    y_orig = cl['y_orig'].astype(int)

    print(f'Training Expert A on cleaned v2 labels: pos={y_clean.sum()}, neg={len(y_clean)-y_clean.sum()}')
    print(f'Features: {X.shape}, NaN after fill: {np.isnan(X).sum()}')

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y_clean))
    n_trees = 400
    leaf_indices = np.zeros((len(y_clean), n_trees), dtype=np.int32)

    for fi, (ti, vi) in enumerate(skf.split(X, y_clean)):
        print(f'\n  Fold {fi+1}/{N_FOLDS}')
        print(f'    Train: {len(ti)}, Val: {len(vi)}')

        pw = (y_clean[ti] == 0).sum() / max((y_clean[ti] == 1).sum(), 1)

        # LightGBM
        m_lgb = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                                    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                                    scale_pos_weight=pw, random_state=SEED+fi, verbose=-1)
        m_lgb.fit(X[ti], y_clean[ti])
        p_lgb = m_lgb.predict_proba(X[vi])[:, 1]

        # XGBoost
        m_xgb = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   scale_pos_weight=pw, tree_method='hist',
                                   random_state=SEED+fi, verbosity=0)
        m_xgb.fit(X[ti], y_clean[ti])
        p_xgb = m_xgb.predict_proba(X[vi])[:, 1]

        # CatBoost
        m_cb = cb.CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05,
                                      l2_leaf_reg=3.0, verbose=0, random_seed=SEED+fi)
        m_cb.fit(X[ti], y_clean[ti])
        p_cb = m_cb.predict_proba(X[vi])[:, 1]

        # Blend weights by val F1
        best_f1, best_w = 0, (0.34, 0.33, 0.33)
        for wl in np.arange(0, 1.01, 0.1):
            for wx in np.arange(0, 1.0 - wl + 0.001, 0.1):
                wc = 1.0 - wl - wx
                ens = wl * p_lgb + wx * p_xgb + wc * p_cb
                f = best_f1_score(y_clean[vi], ens)[0]
                if f > best_f1:
                    best_f1, best_w = f, (wl, wx, wc)

        ens = best_w[0]*p_lgb + best_w[1]*p_xgb + best_w[2]*p_cb
        oof[vi] = ens
        print(f'    Fold ensemble weights: LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, Cat={best_w[2]:.2f}')
        print(f'    Fold {fi+1}: cleaned-F1={best_f1:.4f}')

        # Collect leaf indices from XGB
        leaf_indices[vi] = m_xgb.apply(X[vi])

    bf, th = best_f1_score(y_clean, oof)
    pred = (oof > th).astype(int)
    print(f'\n[Expert A SGCC v2] Overall: cleaned-F1={bf:.4f}, AUC={roc_auc_score(y_clean, oof):.4f}')
    print(f'  Best threshold: {th:.3f}')
    print(f'On cleaned v2 labels: F1={f1_score(y_clean,pred):.4f}, Rec={recall_score(y_clean,pred):.4f}, Prec={precision_score(y_clean,pred,zero_division=0):.4f}, AUC={roc_auc_score(y_clean,oof):.4f}, th={th:.3f}')

    bf, th = best_f1_score(y_orig, oof)
    pred = (oof > th).astype(int)
    print(f'On original labels: F1={f1_score(y_orig,pred):.4f}, Rec={recall_score(y_orig,pred):.4f}, Prec={precision_score(y_orig,pred,zero_division=0):.4f}, AUC={roc_auc_score(y_orig,oof):.4f}, th={th:.3f}')

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'sgcc_expert_a_cleaned_v2.npz'),
        oof_proba=oof,
        leaf_indices=leaf_indices,
        labels=y_clean,
    )


if __name__ == '__main__':
    main()
