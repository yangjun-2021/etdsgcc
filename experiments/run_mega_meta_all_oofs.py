"""Mega meta-ensemble using all available OOF probability signals.

Loads every *_oof.npz in output/, extracts continuous probability arrays,
trains a stacked ensemble (LR + LGB + XGB) with 5-fold CV, and reports
best F1/recall/precision as well as recall-prioritized results.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import numpy as np
import glob
import lightgbm as lgb
import xgboost as xgb

seed_everything(SEED)


def load_all_oofs(output_dir):
    """Load all continuous OOF signals from output/*_oof.npz."""
    files = sorted(glob.glob(os.path.join(output_dir, '*_oof.npz')))
    # Load labels from first file that has flags
    y = None
    signals = {}
    for f in files:
        try:
            d = np.load(f)
            if y is None:
                if 'flags' in d:
                    y = d['flags'].astype(int)
                elif 'y_orig' in d:
                    y = d['y_orig'].astype(int)
                elif 'y' in d:
                    y = d['y'].astype(int)
            # Find probability array
            for k in d.keys():
                if k.startswith('oof_') or k == 'best_meta':
                    arr = d[k]
                    if arr.ndim > 1:
                        arr = arr[:, 1] if arr.shape[1] == 2 else arr.ravel()
                    arr = np.nan_to_num(arr.astype(np.float32), nan=0.5, posinf=1.0, neginf=0.0)
                    name = f"{os.path.basename(f).replace('.npz', '')}_{k}"
                    signals[name] = arr
                    break
        except Exception as e:
            print(f'  skip {f}: {e}')
    return signals, y


def best_f1(y_true, p):
    best = (0, 0, 0, 0)
    for th in np.linspace(0.01, 0.99, 199):
        pred = (p >= th).astype(int)
        if pred.sum() == 0:
            continue
        f = f1_score(y_true, pred, zero_division=0)
        if f > best[0]:
            best = (f, recall_score(y_true, pred), precision_score(y_true, pred), th)
    return best


def train_meta_lr(P, y, skf):
    oof = np.zeros(len(y))
    for fi, (ti, vi) in enumerate(skf.split(P, y)):
        m = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=SEED+fi, C=1.0)
        m.fit(P[ti], y[ti])
        oof[vi] = m.predict_proba(P[vi])[:, 1]
    return oof


def train_meta_lgb(P, y, skf):
    oof = np.zeros(len(y))
    for fi, (ti, vi) in enumerate(skf.split(P, y)):
        dtrain = lgb.Dataset(P[ti], label=y[ti])
        dval = lgb.Dataset(P[vi], label=y[vi], reference=dtrain)
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 5,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbose': -1,
            'seed': SEED+fi,
        }
        m = lgb.train(
            params, dtrain, num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        oof[vi] = m.predict(P[vi], num_iteration=m.best_iteration)
    return oof


def train_meta_xgb(P, y, skf):
    oof = np.zeros(len(y))
    for fi, (ti, vi) in enumerate(skf.split(P, y)):
        m = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            tree_method='hist',
            eval_metric='auc',
            random_state=SEED+fi,
            verbosity=0,
        )
        m.fit(P[ti], y[ti])
        oof[vi] = m.predict_proba(P[vi])[:, 1]
    return oof


def main():
    print('Loading all OOF signals...')
    signals, y = load_all_oofs(OUTPUT_DIR)
    print(f'Loaded {len(signals)} signals, n={len(y)}, pos={y.sum()}')

    # Build feature matrix
    names = list(signals.keys())
    P = np.column_stack([signals[n] for n in names])
    print(f'Feature matrix shape: {P.shape}')

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    print('\nTraining LR meta...')
    oof_lr = train_meta_lr(P, y, skf)
    print('Training LGB meta...')
    oof_lgb = train_meta_lgb(P, y, skf)
    print('Training XGB meta...')
    oof_xgb = train_meta_xgb(P, y, skf)

    # Also simple average
    oof_avg = P.mean(axis=1)

    results = {
        'avg': oof_avg,
        'lr': oof_lr,
        'lgb': oof_lgb,
        'xgb': oof_xgb,
    }

    print('\n=== Mega meta-ensemble results ===')
    for name, oof in results.items():
        f1, rec, prec, th = best_f1(y, oof)
        auc = roc_auc_score(y, oof)
        print(f'{name:6s}: F1={f1:.4f} Rec={rec:.4f} Prec={prec:.4f} AUC={auc:.4f} th={th:.3f}')

    # Blend all metas
    best_overall = (0, None, None)
    for wl in np.arange(0, 1.01, 0.1):
        for wx in np.arange(0, 1.01 - wl + 0.001, 0.1):
            wlgb = 1.0 - wl - wx
            blend = wl * oof_lr + wx * oof_xgb + wlgb * oof_lgb
            f1, rec, prec, th = best_f1(y, blend)
            if f1 > best_overall[0]:
                best_overall = (f1, (wl, wx, wlgb), blend.copy())

    f1, rec, prec, th = best_f1(y, best_overall[2])
    auc = roc_auc_score(y, best_overall[2])
    print(f'\nBest blend LR={best_overall[1][0]:.1f} XGB={best_overall[1][1]:.1f} LGB={best_overall[1][2]:.1f}: '
          f'F1={f1:.4f} Rec={rec:.4f} Prec={prec:.4f} AUC={auc:.4f} th={th:.3f}')

    # Save best
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'mega_meta_all_oofs_oof.npz'),
        flags=y,
        oof_mega_meta_all_oofs=best_overall[2],
        names=np.array(names),
        weights=np.array(best_overall[1]),
    )
    print(f'Saved to {os.path.join(OUTPUT_DIR, "mega_meta_all_oofs_oof.npz")}')


if __name__ == '__main__':
    main()
