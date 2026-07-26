"""Quickly integrate Foundation OOF (if available) into a meta-learner."""
import os, sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything
from src.training.meta_learner import _load_internal_oofs, _load_external_oofs

seed_everything(SEED)


def load_all_oofs(y):
    oofs = {}
    oofs.update(_load_internal_oofs(y))
    oofs.update(_load_external_oofs(y))

    # Add pipeline experts
    try:
        oofs['Expert-A(GBDT)'] = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_a.npz'))['oof_proba']
    except Exception:
        pass
    try:
        oofs['Expert-B(TCN)'] = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_b.npz'))['oof_proba']
    except Exception:
        pass

    # Add Foundation if available
    try:
        d = np.load(os.path.join(OUTPUT_DIR, 'sgcc_foundation.npz'))
        oofs['Foundation-Hybrid'] = d['oof_proba']
        oofs['Foundation-NN'] = d['nn_oof_proba']
        print('Loaded Foundation OOFs')
    except Exception as e:
        print('Foundation OOF not found:', e)

    # Filter to valid length
    oofs = {k: v for k, v in oofs.items() if len(v) == len(y)}
    return oofs


def correlation_prune(oofs):
    names = sorted(oofs.keys())
    P = np.column_stack([oofs[n] for n in names])
    P = np.nan_to_num(P, nan=0.5, posinf=1.0, neginf=0.0)
    corrs = np.corrcoef(P.T)
    kept = []
    for i, nm in enumerate(names):
        drop = False
        for j in kept:
            if abs(corrs[i, names.index(j)]) > 0.999:
                drop = True
                break
        if not drop:
            kept.append(nm)
    return {k: oofs[k] for k in kept}


def train_meta(P, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    results = {}

    for name, factory in [
        ('LR-C1.0', lambda pw: LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=SEED)),
        ('LR-C0.5', lambda pw: LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000, random_state=SEED)),
        ('XGB-d3', lambda pw: xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, scale_pos_weight=pw,
                                                 tree_method='hist', verbosity=0, random_state=SEED)),
        ('XGB-d4', lambda pw: xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, scale_pos_weight=pw,
                                                 tree_method='hist', verbosity=0, random_state=SEED)),
        ('LGB', lambda pw: lgb.LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, scale_pos_weight=pw,
                                              verbose=-1, random_state=SEED)),
        ('HistGB', lambda _: HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.05, random_state=SEED)),
    ]:
        oof = np.zeros(len(y))
        for fi, (ti, vi) in enumerate(skf.split(P, y)):
            pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
            m = factory(pw)
            if name == 'LGB':
                m.fit(P[ti], y[ti], eval_set=[(P[vi], y[vi])],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            else:
                m.fit(P[ti], y[ti])
            oof[vi] = m.predict_proba(P[vi])[:, 1]

        best_f1, best_th = 0, 0.5
        for th in np.arange(0.05, 0.95, 0.005):
            pred = (oof > th).astype(int)
            if pred.sum() == 0: continue
            f = f1_score(y, pred, zero_division=0)
            if f > best_f1: best_f1, best_th = f, th
        pred = (oof > best_th).astype(int)
        rec = recall_score(y, pred, zero_division=0)
        prec = precision_score(y, pred, zero_division=0)
        auc = roc_auc_score(y, oof)
        print(f'  {name:10s}: F1={best_f1:.4f} Rec={rec:.4f} Prec={prec:.4f} AUC={auc:.4f} th={best_th:.3f}')
        results[name] = {'oof': oof, 'f1': best_f1, 'th': best_th}

    # Top3 ensemble
    top3 = sorted(results, key=lambda k: results[k]['f1'], reverse=True)[:3]
    ens = np.mean([results[k]['oof'] for k in top3], axis=0)
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (ens > th).astype(int)
        if pred.sum() == 0: continue
        f = f1_score(y, pred, zero_division=0)
        if f > best_f1: best_f1, best_th = f, th
    pred = (ens > best_th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(f'\nTop3-Ensemble ({"+".join(top3)}):')
    print(f'  F1={best_f1:.4f} Rec={recall_score(y,pred):.4f} Prec={precision_score(y,pred,zero_division=0):.4f} '
          f'AUC={roc_auc_score(y,ens):.4f} th={best_th:.3f}')
    print(f'  TP={tp} FP={fp} FN={fn}')

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'meta_with_foundation_oof.npz'),
        oof_final=ens,
        flags=y,
        f1=best_f1,
        top3=np.array(top3),
    )


def main():
    y = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags']
    oofs = load_all_oofs(y)
    print(f'Loaded {len(oofs)} OOFs')
    oofs = correlation_prune(oofs)
    print(f'After pruning: {len(oofs)}')

    names = sorted(oofs.keys())
    print('Components:')
    for n in names:
        oof = oofs[n]
        best = max(f1_score(y, (oof > th).astype(int), zero_division=0) for th in np.arange(0.05, 0.95, 0.01))
        print(f'  {n:30s}: best F1={best:.4f}')

    P = np.column_stack([oofs[n] for n in names])
    P = np.nan_to_num(P, nan=0.5, posinf=1.0, neginf=0.0)
    train_meta(P, y)


if __name__ == '__main__':
    main()
