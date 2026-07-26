"""Final meta-learner on cleaned labels using all cleaned-label OOFs."""
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
            m.fit(P[ti], y[ti], eval_set=(P[vi], y[vi]), early_stopping_rounds=50, verbose=False)
        else:
            m.fit(P[ti], y[ti])
        oof[vi] = m.predict_proba(P[vi])[:, 1]
    return oof


def main():
    cl = np.load(os.path.join(OUTPUT_DIR, 'cleaned_labels_v1.npz'))
    y_clean = cl['y_clean'].astype(int)
    y_orig = cl['y_orig'].astype(int)

    oofs = {}

    # Cleaned-label GBDT / DL OOFs
    files = [
        ('sgcc_expert_a_cleaned.npz', 'oof_proba', 'Expert-A-cleaned'),
        ('patch_transformer_cleaned_oof.npz', 'oof_patch_transformer_cleaned', 'PatchT-cleaned'),
        ('amst_cleaned_recall10_oof.npz', 'oof_amst_cleaned_recall10', 'AMST-cleaned-recall10'),
        ('informer_cleaned_oof.npz', 'oof_informer_cleaned', 'Informer-cleaned'),
        ('mega_gbdt_cleaned_oof.npz', 'oof_mega_gbdt_cleaned', 'MegaGBDT-cleaned'),
    ]
    for fn, key, label in files:
        try:
            d = np.load(os.path.join(OUTPUT_DIR, fn))
            if key in d.files and len(d[key]) == len(y_clean):
                oofs[label] = d[key]
                print(f'Loaded {label} from {fn}')
        except Exception as e:
            print(f'Not loaded {label}: {e}')

    # Original-label strong OOFs as additional signals
    extra = [
        ('sgcc_expert_a.npz', 'oof_proba', 'Expert-A-raw'),
        ('sgcc_expert_b.npz', 'oof_proba', 'Expert-B-raw'),
        ('amst_3ch_recall10_oof.npz', 'oof_amst_3ch_recall10', 'AMST-raw-recall10'),
        ('informer_fast_oof.npz', 'oof_informer_fast', 'Informer-raw-fast'),
        ('mega_boost_enhanced.npz', 'oof_final', 'MegaBoost-raw'),
        ('mega_gbdt_recall_v2_oof.npz', 'oof_mega_gbdt_recall_v2', 'MegaGBDT-raw'),
        ('patch_transformer_raw_3ch_recall_oof.npz', 'oof_patch_transformer_raw_3ch_recall', 'PatchT-raw-recall'),
        ('supcon_raw_3ch_oof.npz', 'oof_supcon_raw_3ch', 'SupCon-raw'),
    ]
    for fn, key, label in extra:
        try:
            d = np.load(os.path.join(OUTPUT_DIR, fn))
            if key in d.files and len(d[key]) == len(y_clean):
                oofs[label] = d[key]
                print(f'Loaded {label} from {fn}')
        except Exception as e:
            print(f'Not loaded {label}: {e}')

    print(f'\nUsing {len(oofs)} OOFs:')
    for name, oof in oofs.items():
        bf = best_f1_score(y_clean, oof)[0]
        print(f'  {name:30s}: cleaned-F1={bf:.4f}')

    names = list(oofs.keys())
    P = np.column_stack([oofs[n] for n in names])
    P = np.nan_to_num(P, nan=0.5, posinf=1.0, neginf=0.0)

    print('\nTraining meta-learners on cleaned labels...')
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
        oof = train_meta(P, y_clean, name, factory)
        f1, th = best_f1_score(y_clean, oof)
        print(f'  {name:10s}: cleaned-F1={f1:.4f}')
        meta_results[name] = oof

    # Top-3 average
    top3 = sorted(meta_results, key=lambda k: best_f1_score(y_clean, meta_results[k])[0], reverse=True)[:3]
    ens = np.mean([meta_results[n] for n in top3], axis=0)
    meta_results['Top3Avg'] = ens
    print(f'\nTop3 meta: {top3}')

    # Evaluate all on cleaned and original labels
    print('\n=== Evaluation on cleaned labels ===')
    for name, oof in meta_results.items():
        evaluate(y_clean, oof, name)

    print('\n=== Evaluation on original labels ===')
    for name, oof in meta_results.items():
        evaluate(y_orig, oof, name)

    # Save best on cleaned labels
    best_name = max(meta_results, key=lambda k: best_f1_score(y_clean, meta_results[k])[0])
    best_oof = meta_results[best_name]
    print(f'\nBest meta on cleaned labels: {best_name}')

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'meta_final_cleaned_oof.npz'),
        oof_meta_final_cleaned=best_oof,
        y_clean=y_clean,
        y_orig=y_orig,
        names=np.array(names),
        best_meta=best_name,
    )
    print(f'Saved to {os.path.join(OUTPUT_DIR, "meta_final_cleaned_oof.npz")}')


if __name__ == '__main__':
    main()
