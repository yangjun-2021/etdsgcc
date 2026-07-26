"""Train meta-learner only on raw-label strong OOFs, evaluated on original labels."""
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


def evaluate(y_true, y_prob, label):
    bf, th = best_f1_score(y_true, y_prob)
    pred = (y_prob > th).astype(int)
    rec = recall_score(y_true, pred, zero_division=0)
    prec = precision_score(y_true, pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    print(f'{label}: F1={bf:.4f}, Rec={rec:.4f}, Prec={prec:.4f}, AUC={auc:.4f}, th={th:.3f} | TP={tp} FP={fp} FN={fn}')
    return bf, th


def train_meta(P, y, name, factory):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for fi, (ti, vi) in enumerate(skf.split(P, y)):
        pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
        m = factory(pw)
        if name == 'LGB':
            m.fit(P[ti], y[ti], eval_set=[(P[vi], y[vi])],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        elif name == 'CatB':
            import catboost as cb
            m.fit(P[ti], y[ti], eval_set=(P[vi], y[vi]), early_stopping_rounds=50, verbose=False)
        else:
            m.fit(P[ti], y[ti])
        oof[vi] = m.predict_proba(P[vi])[:, 1]
    return oof


def main():
    y_orig = np.load(os.path.join(OUTPUT_DIR, 'cleaned_labels_v1.npz'))['y_orig'].astype(int)

    oofs = {}
    files = [
        ('sgcc_expert_a.npz', 'oof_proba', 'Expert-A-raw'),
        ('sgcc_expert_b.npz', 'oof_proba', 'Expert-B-raw'),
        ('amst_3ch_recall10_oof.npz', 'oof_amst_3ch_recall10', 'AMST-raw-recall10'),
        ('amst_3ch_strong_prior_oof.npz', 'oof_amst_3ch_strong_prior', 'AMST-raw-strong'),
        ('informer_3ch_strong_prior_oof.npz', 'oof_informer_3ch_strong_prior', 'Informer-raw-3ch'),
        ('patch_transformer_raw_3ch_recall_oof.npz', 'oof_patch_transformer_raw_3ch_recall', 'PatchT-raw-recall'),
        ('supcon_raw_3ch_oof.npz', 'oof_supcon_raw_3ch', 'SupCon-raw'),
        ('mega_boost_enhanced.npz', 'oof_final', 'MegaBoost-raw'),
        ('mega_hillclimb.npz', 'oof_final', 'MegaHill-raw'),
    ]
    for fn, key, label in files:
        try:
            d = np.load(os.path.join(OUTPUT_DIR, fn))
            if key in d.files and len(d[key]) == len(y_orig):
                oofs[label] = d[key]
                print(f'Loaded {label}')
        except Exception as e:
            print(f'Not loaded {label}: {e}')

    print(f'\nUsing {len(oofs)} OOFs:')
    for name, oof in oofs.items():
        bf = best_f1_score(y_orig, oof)[0]
        print(f'  {name:30s}: orig-F1={bf:.4f}')

    names = list(oofs.keys())
    P = np.column_stack([oofs[n] for n in names])
    P = np.nan_to_num(P, nan=0.5, posinf=1.0, neginf=0.0)

    print('\nTraining meta-learners on original labels...')
    meta_results = {}

    factories = [
        ('XGB-d4', lambda pw: xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                                scale_pos_weight=pw, tree_method='hist',
                                                random_state=SEED, verbosity=0)),
        ('XGB-d3', lambda pw: xgb.XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05,
                                                scale_pos_weight=pw, tree_method='hist',
                                                random_state=SEED, verbosity=0)),
        ('LR-C1.0', lambda _: LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=SEED)),
        ('LR-C0.5', lambda _: LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000, random_state=SEED)),
        ('LGB', lambda pw: lgb.LGBMClassifier(n_estimators=400, max_depth=3, learning_rate=0.05,
                                              scale_pos_weight=pw, verbose=-1, random_state=SEED)),
        ('HistGB', lambda _: HistGradientBoostingClassifier(max_iter=200, max_depth=3,
                                                             learning_rate=0.05, random_state=SEED)),
    ]
    try:
        import catboost as cb
        factories.append(('CatB', lambda _: cb.CatBoostClassifier(iterations=400, depth=3, learning_rate=0.05,
                                                                   auto_class_weights='Balanced', verbose=0,
                                                                   random_seed=SEED)))
    except ImportError:
        pass

    for name, factory in factories:
        oof = train_meta(P, y_orig, name, factory)
        f1, th = best_f1_score(y_orig, oof)
        print(f'  {name:10s}: orig-F1={f1:.4f}')
        meta_results[name] = oof

    # Top-3 average
    top3 = sorted(meta_results, key=lambda k: best_f1_score(y_orig, meta_results[k])[0], reverse=True)[:3]
    ens = np.mean([meta_results[n] for n in top3], axis=0)
    meta_results['Top3Avg'] = ens
    print(f'\nTop3 meta: {top3}')

    print('\n=== Evaluation on original labels ===')
    for name, oof in meta_results.items():
        evaluate(y_orig, oof, name)

    best_name = max(meta_results, key=lambda k: best_f1_score(y_orig, meta_results[k])[0])
    best_oof = meta_results[best_name]
    print(f'\nBest meta on original labels: {best_name}')

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'meta_raw_only_oof.npz'),
        oof_meta_raw_only=best_oof,
        y_orig=y_orig,
        names=np.array(names),
        best_meta=best_name,
    )


if __name__ == '__main__':
    main()
