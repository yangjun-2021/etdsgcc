"""Quick check: can the new poor OOF signals help a simple meta-model?"""
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SEED, N_FOLDS


def best_f1(y, proba):
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (proba > th).astype(int)
        if pred.sum() == 0:
            continue
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    pred = (proba > best_th).astype(int)
    return best_f1, best_th, recall_score(y, pred, zero_division=0), precision_score(y, pred, zero_division=0)


def main():
    y = np.load(os.path.join(OUTPUT_DIR, 'hillclimb_best_oof.npz'))['flags']
    hill = np.load(os.path.join(OUTPUT_DIR, 'hillclimb_best_oof.npz'))['oof_hillclimb']
    v3 = np.load(os.path.join(OUTPUT_DIR, 'amst_3ch_synthetic_mixed_ls_v3_oof.npz'))['oof_amst_3ch_synthetic_mixed_ls_v3']

    signals = {'hillclimb': hill, 'v3': v3}
    for fname, key in [
        ('confident_weighted_lgb_oof.npz', 'oof_confident_weighted_lgb'),
        ('usage_stratified_lgb_oof.npz', 'oof_usage_stratified_lgb'),
    ]:
        p = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(p):
            signals[fname.replace('_oof.npz', '')] = np.load(p)[key]

    print('Individual signals:')
    for name, proba in signals.items():
        f1, th, rec, prec = best_f1(y, proba)
        auc = roc_auc_score(y, proba)
        print(f'  {name:25s}: F1={f1:.4f} Rec={rec:.4f} Prec={prec:.4f} AUC={auc:.4f} th={th:.3f}')

    # Simple LR meta on all signals
    P = np.column_stack(list(signals.values()))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for ti, vi in skf.split(P, y):
        m = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=SEED)
        m.fit(P[ti], y[ti])
        oof[vi] = m.predict_proba(P[vi])[:, 1]
    f1, th, rec, prec = best_f1(y, oof)
    auc = roc_auc_score(y, oof)
    print(f'\nLR meta on all above: F1={f1:.4f} Rec={rec:.4f} Prec={prec:.4f} AUC={auc:.4f} th={th:.3f}')

    # LR meta excluding poor signals
    good = np.column_stack([hill, v3])
    oof2 = np.zeros(len(y))
    for ti, vi in skf.split(good, y):
        m = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=SEED)
        m.fit(good[ti], y[ti])
        oof2[vi] = m.predict_proba(good[vi])[:, 1]
    f1, th, rec, prec = best_f1(y, oof2)
    auc = roc_auc_score(y, oof2)
    print(f'LR meta on hillclimb+v3: F1={f1:.4f} Rec={rec:.4f} Prec={prec:.4f} AUC={auc:.4f} th={th:.3f}')


if __name__ == '__main__':
    main()
