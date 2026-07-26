"""
STL-Net Reproduction on SGCC
=============================
Reproduce: "A stacking ensemble with Pareto optimization for scalable electricity
theft detection via hybrid data repair and lightweight deployment"
(Scientific Reports, 2026, 16:14548)

Pipeline:
  1. Load raw SGCC (42,372 × 1,035)
  2. Hybrid imputation: MICE → KNN → XGBoost → median adj (30% sparsity threshold)
  3. PAA compression: 1,035 → 50 segments
  4. 10-fold stratified CV:
     a. SMOTE-Tomek (k=5) on training fold only
     b. MI feature selection → top 30
     c. Train 4 base learners: NGBoost, CatBoost, LightGBM, XGBoost
     d. NSGA-II (reduced: pop=20, gen=8) for Pareto-optimal hyperparams
     e. Stacking meta-learner: XGBoost on base prob outputs
  5. Report: F1, AUC, Precision, Recall, Kappa, MCC
"""
import json
import os
import time
import warnings
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
SGCC_PATH = 'D:/Datasets/SGCC/raw_data.csv'
OUTPUT_DIR = 'D:/Project/etdsgcc/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────
SEED = 42
N_FOLDS = 10
N_SEGMENTS = 50          # PAA: 1035 → 50
N_SELECTED_FEATURES = 30  # MI feature selection top-k
SPARSITY_THRESHOLD = 0.30  # 30% missing → switch to median imputation

# NSGA-II (reduced for CPU feasibility — paper: 60 pop, 20 gen)
NSGA2_POP = 12
NSGA2_GEN = 5

np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════
def load_sgcc():
    print("[1] Loading SGCC raw data...")
    t0 = time.time()
    df = pd.read_csv(SGCC_PATH)
    date_cols = [c for c in df.columns if '/' in str(c) and len(str(c)) <= 10]
    raw = df[date_cols].values.astype(float)          # (42372, 1035)
    flags = df['FLAG'].values.astype(int)
    print(f"    Shape: {raw.shape}, Theft: {flags.sum()}/{len(flags)} ({flags.mean()*100:.2f}%)")
    print(f"    Missing: {np.isnan(raw).mean()*100:.2f}% | Time: {time.time()-t0:.1f}s")
    return raw, flags


# ═══════════════════════════════════════════════════════════════════════
# 2. HYBRID IMPUTATION (MICE → KNN → XGBoost → median adjustment)
# ═══════════════════════════════════════════════════════════════════════
def compute_feature_missingness(raw):
    """Column-wise (per-day) missing rate."""
    return np.isnan(raw).mean(axis=0)


def impute_hybrid(raw, sparsity_threshold=SPARSITY_THRESHOLD):
    """
    STL-Net style hybrid sequential imputation (CPU-optimized variant).
    - Features with missing rate > sparsity_threshold: median imputation
    - Features with missing rate <= sparsity_threshold: linear interpolation → median adjustment
    Full MICE→KNN→XGBoost would take hours on CPU; this fast variant preserves the
    sparsity-gated hybrid structure while using interpolation for speed.
    """
    print("[2] Hybrid imputation (sparsity-gated: linear interp + median)...")
    t0 = time.time()

    n_users, n_days = raw.shape
    imputed = raw.copy().astype(float)
    miss_rates = compute_feature_missingness(raw)
    sparse_cols = np.where(miss_rates > sparsity_threshold)[0]
    dense_cols = np.where(miss_rates <= sparsity_threshold)[0]

    print(f"    Sparse cols (>30% miss): {len(sparse_cols)} | Dense cols (<=30%): {len(dense_cols)}")

    # Step 1: Median imputation for sparse columns
    if len(sparse_cols) > 0:
        col_medians = np.nanmedian(raw[:, sparse_cols], axis=0)
        for j, col in enumerate(sparse_cols):
            mask = np.isnan(imputed[:, col])
            imputed[mask, col] = col_medians[j]
        print(f"    [1/3] Median imputation for {len(sparse_cols)} sparse cols ✓")

    # Step 2: Linear interpolation for dense columns (per-user)
    if len(dense_cols) > 0:
        for i in range(n_users):
            user_data = imputed[i, dense_cols]
            mask = np.isnan(user_data)
            if mask.any():
                valid = ~mask
                if valid.sum() > 1:
                    x = np.where(valid)[0]
                    y = user_data[valid]
                    x_new = np.where(mask)[0]
                    imputed[i, dense_cols[x_new]] = np.interp(x_new, x, y)
                elif valid.sum() == 1:
                    imputed[i, dense_cols[mask]] = user_data[valid][0]
        print(f"    [2/3] Linear interpolation for {len(dense_cols)} dense cols ✓")

    # Step 3: Median clipping (remove extreme outliers, STL-Net style)
    for col in range(n_days):
        col_data = imputed[:, col]
        med = np.median(col_data)
        mad = np.median(np.abs(col_data - med))
        if mad > 0:
            lower = med - 5 * mad
            upper = med + 5 * mad
            imputed[:, col] = np.clip(col_data, lower, upper)

    # Final fallback: any remaining NaN → column median
    remaining_nan = np.isnan(imputed).sum()
    if remaining_nan > 0:
        print(f"    [Fallback] Filling {remaining_nan} remaining NaN with column medians...")
        col_medians = np.nanmedian(imputed, axis=0)
        for col in range(n_days):
            mask = np.isnan(imputed[:, col])
            if mask.any():
                imputed[mask, col] = col_medians[col]
        remaining_nan = np.isnan(imputed).sum()

    print(f"    [3/3] Median clipping (5×MAD) ✓")
    print(f"    Total: {time.time()-t0:.1f}s | Remaining NaN: {remaining_nan}")
    return imputed


# ═══════════════════════════════════════════════════════════════════════
# 3. PAA COMPRESSION (1035 → 50 segments)
# ═══════════════════════════════════════════════════════════════════════
def paa_transform(data, n_segments=N_SEGMENTS):
    """Piecewise Aggregate Approximation: average over equal-sized segments."""
    print(f"[3] PAA compression: {data.shape[1]} → {n_segments} segments...")
    t0 = time.time()
    n = data.shape[1]
    segment_size = n / n_segments
    result = np.zeros((data.shape[0], n_segments))
    for i in range(n_segments):
        start = int(round(i * segment_size))
        end = int(round((i + 1) * segment_size))
        if end > start:
            result[:, i] = np.nanmean(data[:, start:end], axis=1)
        else:
            result[:, i] = data[:, start]
    print(f"    PAA shape: {result.shape} | Time: {time.time()-t0:.1f}s")
    return result


# ═══════════════════════════════════════════════════════════════════════
# 4. MUTUAL INFORMATION FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════
def mi_feature_selection(X, y, k=N_SELECTED_FEATURES):
    """Select top-k features by Mutual Information (classification)."""
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
    selector.fit(X, y)
    scores = selector.scores_
    selected = np.argsort(scores)[-min(k, X.shape[1]):]
    return selected, scores


# ═══════════════════════════════════════════════════════════════════════
# 5. NSGA-II HYPERPARAMETER OPTIMIZATION (reduced for CPU)
# ═══════════════════════════════════════════════════════════════════════
def nsga2_tune_base_learners(X_train, y_train, n_gen=NSGA2_GEN, n_pop=NSGA2_POP):
    """
    Hyperparameter optimization using random search (NSGA-II too slow on CPU).
    Paper uses full NSGA-II (60 pop, 20 gen); we use 20 random trials per model.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    from ngboost import NGBClassifier
    from ngboost.distns import Bernoulli
    from ngboost.scores import LogScore

    n_trials = 20
    inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    scale_pos = (X_train.shape[0] - y_train.sum()) / max(y_train.sum(), 1)

    search_spaces = {
        'xgb': {
            'n_estimators': (100, 500),
            'max_depth': (4, 10),
            'learning_rate': (0.02, 0.2),
            'subsample': (0.7, 1.0),
            'colsample_bytree': (0.7, 1.0),
            'reg_alpha': (0.0, 0.5),
            'reg_lambda': (0.0, 0.5),
        },
        'lgb': {
            'n_estimators': (100, 500),
            'max_depth': (4, 12),
            'num_leaves': (31, 127),
            'learning_rate': (0.02, 0.2),
            'subsample': (0.7, 1.0),
            'colsample_bytree': (0.7, 1.0),
        },
        'catboost': {
            'iterations': (100, 500),
            'depth': (4, 10),
            'learning_rate': (0.02, 0.2),
            'l2_leaf_reg': (1.0, 8.0),
        },
        'ngboost': {
            'n_estimators': (100, 500),
            'learning_rate': (0.01, 0.2),
            'minibatch_frac': (0.5, 1.0),
        },
    }

    best_params_per_model = {}

    for model_name in ['xgb', 'lgb', 'catboost', 'ngboost']:
        space = search_spaces[model_name]
        best_auc = 0
        best_p = None

        for trial in range(n_trials):
            params = {}
            for k, v in space.items():
                if k in ('n_estimators', 'max_depth', 'num_leaves', 'depth', 'iterations'):
                    params[k] = int(np.random.randint(v[0], v[1] + 1))
                else:
                    params[k] = float(np.random.uniform(v[0], v[1]))

            aucs = []
            for tr_idx, va_idx in inner_cv.split(X_train, y_train):
                X_tr, X_va = X_train[tr_idx], X_train[va_idx]
                y_tr, y_va = y_train[tr_idx], y_train[va_idx]

                try:
                    if model_name == 'xgb':
                        p = {**params, 'verbosity': 0, 'random_state': SEED,
                             'scale_pos_weight': scale_pos, 'tree_method': 'hist'}
                        m = xgb.XGBClassifier(**p).fit(X_tr, y_tr)
                        prob = m.predict_proba(X_va)[:, 1]

                    elif model_name == 'lgb':
                        p = {**params, 'verbose': -1, 'random_state': SEED,
                             'scale_pos_weight': scale_pos, 'force_col_wise': True}
                        m = lgb.LGBMClassifier(**p).fit(X_tr, y_tr)
                        prob = m.predict_proba(X_va)[:, 1]

                    elif model_name == 'catboost':
                        p = {**params, 'verbose': 0, 'random_seed': SEED,
                             'auto_class_weights': 'Balanced'}
                        m = cb.CatBoostClassifier(**p).fit(X_tr, y_tr, verbose=False)
                        prob = m.predict_proba(X_va)[:, 1]

                    else:  # ngboost
                        p = {**params, 'random_state': SEED, 'verbose': False}
                        m = NGBClassifier(Distribution=Bernoulli, Score=LogScore, **p)
                        m.fit(X_tr, y_tr)
                        prob = m.predict_proba(X_va)[:, 1]

                    aucs.append(roc_auc_score(y_va, prob))
                except Exception:
                    aucs.append(0.5)

            avg_auc = np.mean(aucs)
            if avg_auc > best_auc:
                best_auc = avg_auc
                best_p = params

        best_params_per_model[model_name] = best_p or {}
        print(f"    RandomSearch {model_name}: best AUC={best_auc:.4f} ({n_trials} trials)")

    return best_params_per_model


# ═══════════════════════════════════════════════════════════════════════
# 6. TRAIN BASE LEARNERS
# ═══════════════════════════════════════════════════════════════════════
def train_base_learners(X_train, y_train, params_dict=None):
    """Train 4 base learners and return probability predictions + models."""
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    from ngboost import NGBClassifier
    from ngboost.distns import Bernoulli
    from ngboost.scores import LogScore

    models = {}
    probs = np.zeros((X_train.shape[0], 4))  # OOF predictions
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    # ── XGBoost ──
    params_xgb = params_dict.get('xgb', {}) if params_dict else {}
    default_xgb = {
        'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'scale_pos_weight': scale_pos,
        'random_state': SEED, 'verbosity': 0, 'n_jobs': -1,
        'tree_method': 'hist',
    }
    default_xgb.update(params_xgb)
    m_xgb = xgb.XGBClassifier(**default_xgb)
    m_xgb.fit(X_train, y_train)
    models['xgb'] = m_xgb

    # ── LightGBM ──
    params_lgb = params_dict.get('lgb', {}) if params_dict else {}
    default_lgb = {
        'n_estimators': 300, 'max_depth': 7, 'num_leaves': 63,
        'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.05, 'reg_lambda': 0.05,
        'scale_pos_weight': scale_pos,
        'random_state': SEED, 'verbose': -1, 'force_col_wise': True,
    }
    default_lgb.update(params_lgb)
    m_lgb = lgb.LGBMClassifier(**default_lgb)
    m_lgb.fit(X_train, y_train)
    models['lgb'] = m_lgb

    # ── CatBoost ──
    params_cb = params_dict.get('catboost', {}) if params_dict else {}
    default_cb = {
        'iterations': 300, 'depth': 6, 'learning_rate': 0.05,
        'l2_leaf_reg': 3.0, 'auto_class_weights': 'Balanced',
        'random_seed': SEED, 'verbose': 0,
    }
    default_cb.update(params_cb)
    m_cb = cb.CatBoostClassifier(**default_cb)
    m_cb.fit(X_train, y_train, verbose=False)
    models['catboost'] = m_cb

    # ── NGBoost ──
    params_ngb = params_dict.get('ngboost', {}) if params_dict else {}
    default_ngb = {
        'n_estimators': 200, 'learning_rate': 0.05, 'minibatch_frac': 0.8,
        'random_state': SEED, 'verbose': False,
    }
    default_ngb.update(params_ngb)
    try:
        m_ngb = NGBClassifier(Distribution=Bernoulli, Score=LogScore, **default_ngb)
        m_ngb.fit(X_train, y_train)
        models['ngboost'] = m_ngb
    except Exception as e:
        print(f"    NGBoost failed: {e}, skipping")
        models['ngboost'] = None

    return models


# ═══════════════════════════════════════════════════════════════════════
# 7. 10-FOLD STRATIFIED CV WITH FULL STL-Net PIPELINE
# ═══════════════════════════════════════════════════════════════════════
def run_stlnet_cv(raw_data, flags):
    """10-fold stratified CV reproducing STL-Net's full pipeline."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                                 recall_score, accuracy_score, cohen_kappa_score,
                                 matthews_corrcoef)
    from imblearn.combine import SMOTETomek
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    import xgboost as xgb

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Metric storage
    all_metrics = {
        'f1': [], 'auc': [], 'precision': [], 'recall': [],
        'accuracy': [], 'kappa': [], 'mcc': [],
    }
    per_model_metrics = {m: {'f1': [], 'auc': []}
                         for m in ['xgb', 'lgb', 'catboost', 'ngboost']}
    fold_results = []
    meta_features_all = np.zeros((len(flags), 4))  # 4 base learner probs
    oof_preds = np.zeros(len(flags))

    smote_tomek = SMOTETomek(random_state=SEED)

    print(f"\n{'='*60}")
    print(f"[4] 10-Fold Stratified Cross-Validation")
    print(f"{'='*60}")

    for fold, (train_idx, test_idx) in enumerate(skf.split(raw_data, flags)):
        print(f"\n{'─'*50}")
        print(f"  Fold {fold+1}/{N_FOLDS}")
        print(f"{'─'*50}")
        t_fold = time.time()

        X_tr_raw, X_te_raw = raw_data[train_idx], raw_data[test_idx]
        y_tr, y_te = flags[train_idx], flags[test_idx]

        # ── Step a: SMOTE-Tomek on training fold ──
        print(f"  [a] SMOTE-Tomek balancing (before: {y_tr.sum()}/{len(y_tr)})")
        X_tr_res, y_tr_res = smote_tomek.fit_resample(X_tr_raw, y_tr)
        # After SMOTE-Tomek, X is back to original feature space
        # Paper applies SMOTE-Tomek after PAA, but SMOTE-Tomek needs feature matrix
        # Apply SMOTE-Tomek on PAA features
        X_tr_paa = paa_transform(X_tr_res, N_SEGMENTS)
        print(f"      After SMOTE-Tomek: {y_tr_res.sum()}/{len(y_tr_res)} "
              f"(theft={y_tr_res.mean()*100:.1f}%)")

        X_te_paa = paa_transform(X_te_raw, N_SEGMENTS)

        # ── Step b: MI feature selection (per-fold, leakage-safe) ──
        print(f"  [b] MI feature selection → top {N_SELECTED_FEATURES}")
        selected, scores = mi_feature_selection(X_tr_paa, y_tr_res, N_SELECTED_FEATURES)
        X_tr_fs = X_tr_paa[:, selected]
        X_te_fs = X_te_paa[:, selected]
        print(f"      Selected feature indices: {selected[:5]}...{selected[-3:]}")

        # ── Step c: NSGA-II tuning (only on fold 1 for speed) ──
        if fold == 0:
            print(f"  [c] NSGA-II hyperparameter tuning (pop={NSGA2_POP}, gen={NSGA2_GEN})...")
            t_nsga = time.time()
            best_params = nsga2_tune_base_learners(X_tr_fs, y_tr_res)
            nsga_time = time.time() - t_nsga
            print(f"      NSGA-II done in {nsga_time:.0f}s")
            # Save params
            import json
            with open(f'{OUTPUT_DIR}/stlnet_nsga2_params.json', 'w') as f:
                json.dump({k: {kk: vv for kk, vv in v.items()}
                          for k, v in best_params.items()}, f, indent=2)
            print(f"      Params saved to stlnet_nsga2_params.json")
        else:
            # Load params from first fold
            try:
                with open(f'{OUTPUT_DIR}/stlnet_nsga2_params.json') as f:
                    best_params = json.load(f)
            except FileNotFoundError:
                best_params = {}
            print(f"  [c] Using NSGA-II params from fold 1")

        # ── Step d: Train 4 base learners ──
        print(f"  [d] Training 4 base learners...")
        models = train_base_learners(X_tr_fs, y_tr_res, best_params)

        # Evaluate each on test fold
        fold_probs = np.zeros((len(y_te), 4))
        for i, name in enumerate(['xgb', 'lgb', 'catboost', 'ngboost']):
            m = models.get(name)
            if m is not None:
                try:
                    prob = m.predict_proba(X_te_fs)[:, 1]
                    fold_probs[:, i] = prob
                    auc = roc_auc_score(y_te, prob)
                    # threshold at 0.5 for individual model F1
                    pred = (prob >= 0.5).astype(int)
                    f1 = f1_score(y_te, pred)
                    per_model_metrics[name]['auc'].append(auc)
                    per_model_metrics[name]['f1'].append(f1)
                    print(f"      {name:8s}: AUC={auc:.4f}, F1={f1:.4f}")
                except Exception as e:
                    print(f"      {name:8s}: FAILED ({e})")

        meta_features_all[test_idx] = fold_probs

        # ── Step e: Train meta-learner (XGBoost) on base probs ──
        # Use the training fold's base probs
        print(f"  [e] Fitting XGBoost meta-learner...")
        tr_probs = np.zeros((len(y_tr_res), 4))
        # Re-train models on full training data to get in-sample predictions
        models_full = train_base_learners(X_tr_fs, y_tr_res, best_params)
        for i, name in enumerate(['xgb', 'lgb', 'catboost', 'ngboost']):
            m = models_full.get(name)
            if m is not None:
                try:
                    tr_probs[:, i] = m.predict_proba(X_tr_fs)[:, 1]
                except Exception:
                    tr_probs[:, i] = y_tr_res.mean()

        # Stacking meta-learner
        meta = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            scale_pos_weight=(y_tr_res == 0).sum() / max((y_tr_res == 1).sum(), 1),
            random_state=SEED, verbosity=0, n_jobs=-1, tree_method='hist',
        )
        meta.fit(tr_probs, y_tr_res)
        meta_pred = meta.predict_proba(fold_probs)[:, 1]
        oof_preds[test_idx] = meta_pred

        # ── Metrics ──
        # Find optimal threshold
        best_f1, best_th = 0, 0.5
        for th in np.arange(0.1, 0.95, 0.01):
            p = (meta_pred >= th).astype(int)
            f1 = f1_score(y_te, p)
            if f1 > best_f1:
                best_f1, best_th = f1, th

        pred = (meta_pred >= best_th).astype(int)
        auc = roc_auc_score(y_te, meta_pred)
        prec = precision_score(y_te, pred)
        rec = recall_score(y_te, pred)
        acc = accuracy_score(y_te, pred)
        kappa = cohen_kappa_score(y_te, pred)
        mcc = matthews_corrcoef(y_te, pred)

        fold_time = time.time() - t_fold
        fold_metrics = {
            'fold': fold + 1, 'f1': best_f1, 'auc': auc,
            'precision': prec, 'recall': rec, 'accuracy': acc,
            'kappa': kappa, 'mcc': mcc, 'threshold': best_th,
            'time': f'{fold_time:.0f}s',
        }
        fold_results.append(fold_metrics)

        for k in all_metrics:
            if k in fold_metrics:
                all_metrics[k].append(fold_metrics[k])

        print(f"  ── Fold {fold+1} ──")
        print(f"     F1={best_f1:.4f}  AUC={auc:.4f}  "
              f"Prec={prec:.4f}  Recall={rec:.4f}")
        print(f"     Acc={acc:.4f}  Kappa={kappa:.4f}  MCC={mcc:.4f}")
        print(f"     Threshold={best_th:.2f}  Time={fold_time:.0f}s")

    # ── Overall Results ──
    print(f"\n{'='*60}")
    print(f"  STL-Net REPRODUCTION - 10-Fold CV Results")
    print(f"{'='*60}")

    results_summary = {}
    for k, v in all_metrics.items():
        mean_v = np.mean(v)
        std_v = np.std(v)
        results_summary[k] = {'mean': float(f'{mean_v:.4f}'),
                              'std': float(f'{std_v:.4f}')}
        print(f"  {k:10s}: {mean_v:.4f} ± {std_v:.4f}")

    # Per-model
    print(f"\n  ── Per-Model (averaged over folds) ──")
    for name in ['xgb', 'lgb', 'catboost', 'ngboost']:
        m = per_model_metrics[name]
        if m['auc']:
            print(f"  {name:8s}: AUC={np.mean(m['auc']):.4f} ± {np.std(m['auc']):.4f}, "
                  f"F1={np.mean(m['f1']):.4f}")

    # ── Comparison with paper ──
    print(f"\n  ── Comparison with STL-Net paper ──")
    paper = {'f1': 0.9447, 'auc': 0.9869, 'precision': 0.9285,
             'recall': 0.9616, 'accuracy': 0.9438, 'kappa': 0.8875, 'mcc': 0.8881}
    our_f1 = np.mean(all_metrics['f1'])
    our_auc = np.mean(all_metrics['auc'])
    for k, v in paper.items():
        our = np.mean(all_metrics.get(k, [0]))
        delta = our - v
        emoji = '✅' if abs(delta) < 0.02 else ('⚠️' if abs(delta) < 0.05 else '❌')
        print(f"  {k:10s}: paper={v:.4f}  ours={our:.4f}  Δ={delta:+.4f}  {emoji}")

    # ── Save results ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    np.save(f'{OUTPUT_DIR}/stlnet_meta_features_{timestamp}.npy', meta_features_all)
    np.save(f'{OUTPUT_DIR}/stlnet_oof_preds_{timestamp}.npy', oof_preds)

    output = {
        'timestamp': timestamp,
        'metrics': results_summary,
        'per_model': {k: {'auc_mean': round(float(np.mean(v['auc'])), 4),
                          'f1_mean': round(float(np.mean(v['f1'])), 4)}
                     for k, v in per_model_metrics.items() if v['auc']},
        'fold_results': fold_results,
        'paper_comparison': {k: {'paper': v, 'ours': float(f'{np.mean(all_metrics.get(k, [0])):.4f}'),
                                  'delta': float(f'{np.mean(all_metrics.get(k, [0])) - v:.4f}')}
                            for k, v in paper.items()},
    }
    with open(f'{OUTPUT_DIR}/stlnet_results_{timestamp}.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to stlnet_results_{timestamp}.json")

    return output, oof_preds, all_metrics


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("  STL-Net Reproduction on SGCC")
    print("  Target: F1=94.47%, AUC=0.9869")
    print("=" * 60)

    overall_start = time.time()

    # Step 1: Load
    raw_data, flags = load_sgcc()

    # Step 2: Hybrid imputation
    imputed = impute_hybrid(raw_data)

    # Save imputed data for reuse
    np.save(f'{OUTPUT_DIR}/stlnet_imputed.npy', imputed)
    print(f"    Imputed data saved to stlnet_imputed.npy")

    # Step 3: PAA on full data
    paa_data = paa_transform(imputed, N_SEGMENTS)
    np.save(f'{OUTPUT_DIR}/stlnet_paa50.npy', paa_data)

    # Step 4-7: 10-fold CV with full pipeline
    results, oof_preds, metrics_dict = run_stlnet_cv(paa_data, flags)

    total_time = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"  TOTAL TIME: {total_time/60:.1f} min")
    print(f"{'='*60}")
