"""V3 voter: untainted GBDT ensemble on handcrafted+extended features, ORIGINAL labels.

CPU-only voter to add a non-deep family to the v3 consensus pool.
Features: sgcc_extended_features.npz (548d) + base stat_features (110d) + impute-mask
fraction. Labels: y_orig. No OOF-stacking features (unlike build_stronger_prior_original),
so this voter is fully untainted.

Usage:
    conda run -n ml python experiments/v3_gbdt_orig.py
"""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from config import OUTPUT_DIR, SEED, N_FOLDS, SGCC_CONFIG
from src.utils.utils import seed_everything

seed_everything(SEED)


def main():
    t0 = time.time()
    cl = np.load(os.path.join(OUTPUT_DIR, 'cleaned_labels_v1.npz'))
    y = cl['y_orig'].astype(int)

    ext = np.load(os.path.join(OUTPUT_DIR, 'sgcc_extended_features.npz'))
    assert np.array_equal(ext['flags'].astype(int), y), 'extended features flags != y_orig'
    X_ext = np.nan_to_num(ext['features'], nan=0.0, posinf=0.0, neginf=0.0)

    pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
    stat = np.nan_to_num(pre['stat_features'], nan=0.0, posinf=0.0, neginf=0.0)
    mask_frac = pre['impute_mask'].astype(float).mean(axis=1).reshape(-1, 1)

    X = np.hstack([X_ext, stat, mask_frac]).astype(np.float32)
    print(f'Feature matrix: {X.shape}, positives: {y.sum()} ({y.mean()*100:.2f}%)')

    n = len(y)
    oofs = {k: np.zeros(n) for k in ('lgb', 'xgb', 'cb')}
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fi, (ti, vi) in enumerate(skf.split(X, y)):
        pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)

        cfg = SGCC_CONFIG['gbdt_params']['lgb'].copy()
        cfg.update(scale_pos_weight=pw, random_state=SEED + fi, verbose=-1)
        m = lgb.LGBMClassifier(**cfg)
        m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        oofs['lgb'][vi] = m.predict_proba(X[vi])[:, 1]

        cfg = SGCC_CONFIG['gbdt_params']['xgb'].copy()
        cfg.update(scale_pos_weight=pw, random_state=SEED + fi, verbosity=0, tree_method='hist')
        m = xgb.XGBClassifier(**cfg)
        m.fit(X[ti], y[ti])
        oofs['xgb'][vi] = m.predict_proba(X[vi])[:, 1]

        cfg = SGCC_CONFIG['gbdt_params']['catboost'].copy()
        cfg.update(random_seed=SEED + fi)
        m = CatBoostClassifier(**cfg)
        m.fit(X[ti], y[ti], eval_set=(X[vi], y[vi]), early_stopping_rounds=80, verbose=False)
        oofs['cb'][vi] = m.predict_proba(X[vi])[:, 1]
        print(f'Fold {fi+1}/{N_FOLDS} done, elapsed {(time.time()-t0)/60:.1f} min')

    best_f1, best_w = 0, (0.4, 0.3, 0.3)
    for wl in np.arange(0.0, 1.01, 0.1):
        for wx in np.arange(0.0, 1.0 - wl + 0.001, 0.1):
            wc = 1.0 - wl - wx
            ens = wl * oofs['lgb'] + wx * oofs['xgb'] + wc * oofs['cb']
            for th in np.arange(0.05, 0.95, 0.01):
                f = f1_score(y, (ens > th).astype(int), zero_division=0)
                if f > best_f1:
                    best_f1, best_w = f, (wl, wx, wc)
    ens = best_w[0] * oofs['lgb'] + best_w[1] * oofs['xgb'] + best_w[2] * oofs['cb']
    print(f'Blend weights: LGB={best_w[0]:.2f} XGB={best_w[1]:.2f} Cat={best_w[2]:.2f}')

    best_th = 0.5
    best_f1 = 0
    for th in np.arange(0.05, 0.95, 0.005):
        f = f1_score(y, (ens > th).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_th = f, th
    pred = (ens > best_th).astype(int)
    print(f'\nV3-voter GBDT (original labels, untainted): F1={best_f1:.4f}, '
          f'Rec={recall_score(y, pred):.4f}, Prec={precision_score(y, pred, zero_division=0):.4f}, '
          f'AUC={roc_auc_score(y, ens):.4f}, th={best_th:.3f}')

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'v3voter_gbdt_oof.npz'),
        oof_v3voter_gbdt=ens,
        y_orig=y,
    )
    for k, o in oofs.items():
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, f'v3voter_gbdt_{k}_oof.npz'),
            **{f'oof_v3voter_gbdt_{k}': o}, y_orig=y,
        )
    print(f'Saved v3voter_gbdt_oof.npz (+ per-model), total {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
