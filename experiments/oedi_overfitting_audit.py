"""OEDI overfitting audit: three diagnostics for the perfect-score question.

D1. Label-shuffle sanity: one fold with permuted labels must collapse (rules out
    pipeline/index leakage).
D2. Feature importance: what the GBDT actually uses (expected: attack-shape
    statistics; suspicious: index-like features).
D3. Leave-one-theft-type-out (LOTTO): train on Normal + 5 attack types, test on
    the held-out type. Measures generalization to NOVEL attack templates — the
    key question behind "is 1.0 just template memorization".

Usage:
    conda run -n ml python experiments/oedi_overfitting_audit.py
"""
import os
import sys
import pickle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

from config import OUTPUT_DIR, SEED, OEDI_CONFIG


def best_f1(y, p):
    best, bth = 0.0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        f = f1_score(y, (p > th).astype(int), zero_division=0)
        if f > best:
            best, bth = f, th
    return best, bth


def lgb_params(pw):
    cfg = OEDI_CONFIG['gbdt_params']['lgb'].copy()
    cfg.update(scale_pos_weight=pw, random_state=SEED, verbose=-1)
    return cfg


def main():
    pre = np.load(os.path.join(OUTPUT_DIR, 'oedi_preprocessed.npz'))
    X, y, fa = pre['stat_features'].astype(np.float32), pre['y'].astype(int), pre['fold_assignments']
    meta = pickle.load(open(os.path.join(OUTPUT_DIR, 'oedi_meta.pkl'), 'rb'))
    theft_labels = np.array(meta['theft_type_labels'])
    stat_names = meta['stat_names']

    # ---------------- D1: label shuffle on fold 0 ----------------
    print('=' * 60)
    print('[D1] Label-shuffle sanity (fold 0)')
    rng = np.random.RandomState(SEED)
    tr = fa != 0
    va = fa == 0
    y_shuf = y.copy()
    y_shuf[tr] = rng.permutation(y[tr])
    pw = (y_shuf[tr] == 0).sum() / max((y_shuf[tr] == 1).sum(), 1)
    m = lgb.LGBMClassifier(**lgb_params(pw))
    m.fit(X[tr], y_shuf[tr])
    p = m.predict_proba(X[va])[:, 1]
    f1, th = best_f1(y[va], p)
    print(f'  shuffled-train F1 on val = {f1:.4f} (expect collapse toward ~0.3-0.5; '
          f'high values would indicate leakage)')

    # ---------------- D2: feature importance ----------------
    print('=' * 60)
    print('[D2] Feature importance (fold-0 model, gain)')
    tr = fa != 0
    pw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
    m = lgb.LGBMClassifier(**lgb_params(pw))
    m.fit(X[tr], y[tr])
    imp = m.booster_.feature_importance(importance_type='gain')
    order = np.argsort(imp)[::-1][:15]
    total = imp.sum() + 1e-12
    for i in order:
        print(f'  {stat_names[i]:28s} gain_share={imp[i]/total:.3f}')

    # ---------------- D3: leave-one-theft-type-out ----------------
    print('=' * 60)
    print('[D3] Leave-one-theft-type-out (train on Normal+5 types, test on held-out type)')
    normal = theft_labels == 'Normal'
    for tt in sorted(set(theft_labels) - {'Normal'}):
        held = theft_labels == tt
        tr_mask = ~held                     # Normal + other 5 types
        te_mask = held | normal             # held-out type + all Normal
        y_tr, y_te = y[tr_mask], (y[te_mask] == 1).astype(int)
        # keep only positives of other types + normals in train (y already binary)
        pw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        m = lgb.LGBMClassifier(**lgb_params(pw))
        m.fit(X[tr_mask], y_tr)
        p = m.predict_proba(X[te_mask])[:, 1]
        f1, th = best_f1(y_te, p)
        pred = (p > th).astype(int)
        pos = y_te == 1
        print(f'  hold {tt:7s}: F1={f1:.4f} th={th:.2f} '
              f'Rec(novel)={recall_score(y_te, pred, zero_division=0):.4f} '
              f'Prec={precision_score(y_te, pred, zero_division=0):.4f} '
              f'AUC={roc_auc_score(y_te, p):.4f}  (n_pos={pos.sum()}, n_norm={(~pos).sum()})')


if __name__ == '__main__':
    main()
