"""Clean labels using OOF consensus and retrain GBDT ensemble.

This is a self-training / confident-learning style label cleaning:
- Flip label=1 samples with very low consensus (< fp_th)
- Flip label=0 samples with very high consensus (> fn_th)
- Retrain GBDT on cleaned labels
- Report F1 on both cleaned and original labels
"""
import os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SEED, N_FOLDS, SGCC_CONFIG
from src.utils.utils import seed_everything

seed_everything(SEED)


def load_features():
    pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
    y = pre['flags'].astype(int)
    stat = np.nan_to_num(pre['stat_features'], nan=0.0, posinf=0.0, neginf=0.0)
    mask = pre['impute_mask']
    X = np.hstack([stat, mask.astype(float).mean(axis=1).reshape(-1, 1)]).astype(np.float32)
    return X, y


def get_consensus(y):
    oofs = {}
    for f, k in [
        ('sgcc_mega_meta.npz', 'oof_final'),
        ('autoresearch_best.npz', 'oof_final'),
        ('mega_boost_enhanced.npz', 'oof_final'),
        ('informer_oof.npz', 'oof_informer'),
        ('amst_3ch_recall10_oof.npz', 'oof_amst_3ch_recall10'),
        ('patch_transformer_raw_3ch_oof.npz', 'oof_patch_transformer_raw_3ch'),
        ('supcon_raw_3ch_oof.npz', 'oof_supcon_raw_3ch'),
        ('strong_gbdt_prior_oof.npz', 'oof_strong_gbdt_prior'),
    ]:
        try:
            oofs[f] = np.load(os.path.join(OUTPUT_DIR, f))[k]
        except Exception:
            pass
    P = np.column_stack(list(oofs.values()))
    consensus = (P > 0.5).mean(axis=1)
    return consensus


def clean_labels(y, consensus, fp_th=0.3, fn_th=0.7):
    y_clean = y.copy()
    flip_fp = (y == 1) & (consensus < fp_th)
    flip_fn = (y == 0) & (consensus > fn_th)
    y_clean[flip_fp] = 0
    y_clean[flip_fn] = 1
    return y_clean, flip_fp, flip_fn


def train_gbdt_ensemble(X, y, y_orig, skf):
    n = len(y)
    oof_lgb = np.zeros(n)
    oof_xgb = np.zeros(n)
    oof_cb = np.zeros(n)

    for fi, (ti, vi) in enumerate(skf.split(X, y)):
        pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)

        cfg = SGCC_CONFIG['gbdt_params']['lgb'].copy()
        cfg['scale_pos_weight'] = pw
        cfg['random_state'] = SEED + fi
        cfg['verbose'] = -1
        m = lgb.LGBMClassifier(**cfg)
        m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        oof_lgb[vi] = m.predict_proba(X[vi])[:, 1]

        cfg = SGCC_CONFIG['gbdt_params']['xgb'].copy()
        cfg['scale_pos_weight'] = pw
        cfg['random_state'] = SEED + fi
        cfg['verbosity'] = 0
        cfg['tree_method'] = 'hist'
        m = xgb.XGBClassifier(**cfg)
        m.fit(X[ti], y[ti])
        oof_xgb[vi] = m.predict_proba(X[vi])[:, 1]

        cfg = SGCC_CONFIG['gbdt_params']['catboost'].copy()
        cfg['random_seed'] = SEED + fi
        m = CatBoostClassifier(**cfg)
        m.fit(X[ti], y[ti], eval_set=(X[vi], y[vi]), early_stopping_rounds=80, verbose=False)
        oof_cb[vi] = m.predict_proba(X[vi])[:, 1]

    # Blend
    best_f1, best_w = 0, (0.4, 0.3, 0.3)
    for wl in np.arange(0.0, 1.01, 0.1):
        for wx in np.arange(0.0, 1.0 - wl + 0.001, 0.1):
            wc = 1.0 - wl - wx
            ens = wl * oof_lgb + wx * oof_xgb + wc * oof_cb
            f = max(f1_score(y, (ens > th).astype(int), zero_division=0) for th in np.arange(0.05, 0.95, 0.01))
            if f > best_f1:
                best_f1, best_w = f, (wl, wx, wc)
    ens = best_w[0]*oof_lgb + best_w[1]*oof_xgb + best_w[2]*oof_cb
    return ens, best_w


def evaluate(oof, y, label=''):
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (oof > th).astype(int)
        if pred.sum() == 0: continue
        f = f1_score(y, pred, zero_division=0)
        if f > best_f1: best_f1, best_th = f, th
    pred = (oof > best_th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(f'{label}: F1={best_f1:.4f}, Rec={recall_score(y,pred):.4f}, '
          f'Prec={precision_score(y,pred,zero_division=0):.4f}, AUC={roc_auc_score(y,oof):.4f}, '
          f'th={best_th:.3f}, TP={tp} FP={fp} FN={fn}')
    return best_f1


def main():
    X, y_orig = load_features()
    consensus = get_consensus(y_orig)

    print('Testing different cleaning thresholds...')
    for fp_th in [0.1, 0.2, 0.3]:
        for fn_th in [0.7, 0.8, 0.9]:
            y_clean, flip_fp, flip_fn = clean_labels(y_orig, consensus, fp_th, fn_th)
            print(f'\nfp_th={fp_th:.1f}, fn_th={fn_th:.1f}: flipped {flip_fp.sum()} positives->neg, {flip_fn.sum()} negatives->pos')

            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            oof, w = train_gbdt_ensemble(X, y_clean, y_orig, skf)
            print(f'  Blend weights: LGB={w[0]:.2f} XGB={w[1]:.2f} Cat={w[2]:.2f}')
            f_clean = evaluate(oof, y_clean, '  On cleaned labels')
            f_orig = evaluate(oof, y_orig, '  On original labels')

            if fp_th == 0.2 and fn_th == 0.8:
                # Save this version
                np.savez_compressed(
                    os.path.join(OUTPUT_DIR, f'cleaned_labels_fp{fp_th}_fn{fn_th}.npz'),
                    y_clean=y_clean, y_orig=y_orig, consensus=consensus,
                    flipped_fp=flip_fp, flipped_fn=flip_fn,
                )
                np.savez_compressed(
                    os.path.join(OUTPUT_DIR, 'gbdt_cleaned_labels_oof.npz'),
                    oof_gbdt_cleaned=oof, y_clean=y_clean, y_orig=y_orig,
                )


if __name__ == '__main__':
    main()
