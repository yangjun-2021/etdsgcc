"""
MegaMetaLearner: Multi-OOF stacking with external OOF integration.

Key changes vs old meta_learner:
  1. Loads ALL available OOFs (external + internal), not just 2 experts
  2. Meta-features = OOF predictions only (not raw features), prevents noise
  3. Trains 4+ meta-learners (LR, XGB, LGB, HistGB, Avg) and picks best
  4. Uses consistent 5-fold CV
   5. Loads bundled external OOFs from local bundled files

Expected pure-auto F1: 0.86+
"""
import os, pickle, time, warnings
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from scipy.optimize import nnls
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

warnings.filterwarnings('ignore')

from config import SGCC_CONFIG, OEDI_CONFIG, SEED, N_FOLDS, OUTPUT_DIR, DATA_DIR




def _best_f1_score(y_true, y_prob):
    """Pure function to compute best F1 score by threshold search."""
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (y_prob > th).astype(int)
        if pred.sum() == 0:
            continue
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_th = f, th
    pred = (y_prob > best_th).astype(int)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    return best_f1, best_th, tp, fp, fn


def _load_external_oofs(y):
    """Load bundled external OOFs from local bundled npz/csv files.
    
    Filters to TRUE_OOF_KEYS only — excludes stat_feat_*, res_*, missing_* etc.
    """
    TRUE_OOF_KEYS = {
        'ExpertA_OOF', 'ExpertB_OOF',
        'V71_cat', 'V71_innov', 'V71_lgb', 'V71_tcn', 'V71_xgb',
        'V213', 'V216', 'V219', 'V225',
        'V229_oof_iso', 'V229_oof_platt',
        'AnomalyAE_OOF', 'AnomalyIF_OOF',
        'NeighborTheftRatio', 'NeighborDistance',
    }
    existing = {}

    # Primary: bundled CSV (human-readable, cross-tool compatible)
    bundled_csv = os.path.join(OUTPUT_DIR, 'bundled_oofs.csv')
    if os.path.exists(bundled_csv):
        try:
            import pandas as pd
            df = pd.read_csv(bundled_csv)
            for col in df.columns:
                if col in TRUE_OOF_KEYS and len(df[col].values) == len(y):
                    existing[col] = df[col].values.astype(np.float64)
        except Exception:
            pass

    # Additional clean baseline OOFs
    clean_csv = os.path.join(OUTPUT_DIR, 'clean_baseline_oofs.csv')
    if os.path.exists(clean_csv):
        try:
            import pandas as pd
            df = pd.read_csv(clean_csv)
            for col in df.columns:
                if col == 'FLAG':
                    continue
                if len(df[col].values) == len(y):
                    existing[f'Clean-{col}'] = df[col].values.astype(np.float64)
        except Exception:
            pass

    if existing:
        return existing

    # Alternative: bundled NPZ (binary, faster)
    bundled_npz = os.path.join(OUTPUT_DIR, 'bundled_oofs.npz')
    if os.path.exists(bundled_npz):
        try:
            bd = np.load(bundled_npz, allow_pickle=True)
            names = bd['names']
            for i, name in enumerate(names):
                name_str = str(name)
                if name_str in TRUE_OOF_KEYS:
                    key = f'oof_{i}'
                    if key in bd.files and len(bd[key]) == len(y):
                        existing[name_str] = bd[key]
            if existing:
                return existing
        except Exception:
            pass

    # No matching external OOFs (e.g. OEDI where bundled SGCC OOFs have a
    # different length): proceed with an empty external pool instead of failing.
    return {}


def _load_internal_oofs(y):
    """Load our own saved OOFs from output directory."""
    oofs = {}

    try:
        our = np.load(os.path.join(OUTPUT_DIR, 'tcn_kd_results.npz'))
        for k in ['oof_tcn_kd', 'oof_stacker', 'oof_blend', 'oof_hill']:
            if k in our.files and len(our[k]) == len(y):
                oofs[k.replace('oof_', '').replace('_', '-')] = our[k]
    except Exception:
        pass

    try:
        super_gbdt = np.load(os.path.join(OUTPUT_DIR, 'super_gbdt.npz'))
        if len(super_gbdt['oof_super']) == len(y):
            oofs['Super-GBDT'] = super_gbdt['oof_super']
    except Exception:
        pass

    try:
        smart = np.load(os.path.join(OUTPUT_DIR, 'smart_blend.npz'))
        if len(smart['oof_final']) == len(y):
            oofs['SmartBlend'] = smart['oof_final']
    except Exception:
        pass

    try:
        multi = np.load(os.path.join(OUTPUT_DIR, 'multi_oof_results.npz'))
        for k in ['oof_lgb', 'oof_xgb', 'oof_cb']:
            if k in multi.files and len(multi[k]) == len(y):
                oofs[f'multi_{k.replace("oof_", "")}'] = multi[k]
    except Exception:
        pass

    try:
        tcn_enh = np.load(os.path.join(OUTPUT_DIR, 'tcn_enhanced.npz'))
        if len(tcn_enh['oof_tcn']) == len(y):
            oofs['TCN-Enhanced'] = tcn_enh['oof_tcn']
    except Exception:
        pass

    # Additional strong internal ensembles discovered in output/
    extra_files = {
        'autoresearch_best.npz': ['oof_final'],
        'mega_boost.npz': ['oof_final'],
        'mega_boost_enhanced.npz': ['oof_final'],
        'mega_boost_final.npz': ['oof_final'],
        'heterogeneous_ensemble.npz': ['oof_final'],
        'behavior_enhanced.npz': ['oof_meta', 'oof_e1', 'oof_e2', 'oof_e3', 'oof_e4'],
        'final_blend.npz': ['oof_final'],
        'final_fusion.npz': ['oof_tcn'],
        'ultimate.npz': ['oof_final'],
        'mega_hillclimb.npz': ['oof_final'],
        'informer_strong_prior_oof.npz': ['oof_informer_strong_prior'],
        'informer_large_strong_prior_oof.npz': ['oof_informer_large_strong_prior'],
        'strong_gbdt_prior_oof.npz': ['oof_strong_gbdt_prior'],
        'stronger_gbdt_prior_v2.npz': ['prior'],
        'amst_strong_prior_oof.npz': ['oof_amst_strong_prior'],
        'amst_3ch_strong_prior_oof.npz': ['oof_amst_3ch_strong_prior'],
        'amst_3ch_large_strong_prior_oof.npz': ['oof_amst_3ch_large_strong_prior'],
        'amst_3ch_medium_strong_prior_oof.npz': ['oof_amst_3ch_medium_strong_prior'],
        'amst_3ch_medium_tsa_amp_oof.npz': ['oof_amst_3ch_medium_tsa_amp'],
        'informer_3ch_strong_prior_oof.npz': ['oof_informer_3ch_strong_prior'],
        'amst_3ch_supcon_oof.npz': ['oof_amst_3ch_supcon'],
        'amst_3ch_recall10_oof.npz': ['oof_amst_3ch_recall10'],
        'amst_3ch_synthetic_oof.npz': ['oof_amst_3ch_synthetic'],
        'amst_3ch_synthetic_fast_oof.npz': ['oof_amst_3ch_synthetic_fast'],
        'amst_3ch_preprocessed_synthetic_oof.npz': ['oof_amst_3ch_preprocessed_synthetic'],
        'amst_3ch_synthetic_x3_oof.npz': ['oof_amst_3ch_synthetic_x3'],
        'amst_3ch_synthetic_x3_sp_oof.npz': ['oof_amst_3ch_synthetic_x3_sp'],
        'amst_3ch_synthetic_x3_sp_fast_oof.npz': ['oof_amst_3ch_synthetic_x3_sp_fast'],
        'amst_3ch_synthetic_mixed_fast_oof.npz': ['oof_amst_3ch_synthetic_mixed_fast'],
        'amst_3ch_synthetic_mixed_ls_fast_oof.npz': ['oof_amst_3ch_synthetic_mixed_ls_fast'],
        'amst_3ch_large_synthetic_mixed_ls_oof.npz': ['oof_amst_3ch_large_synthetic_mixed_ls'],
        'amst_3ch_synthetic_mixed_ls_v3_oof.npz': ['oof_amst_3ch_synthetic_mixed_ls_v3'],
        'amst_3ch_synthetic_mixed_ls_v3_gce_oof.npz': ['oof_amst_3ch_synthetic_mixed_ls_v3_gce'],
        'amst_3ch_synthetic_subtle_v3_oof.npz': ['oof_amst_3ch_synthetic_subtle_v3'],
        'hillclimb_fn_predictor_oof.npz': ['oof_hillclimb_fn_predictor'],
        'amst_3ch_synthetic_targeted_oof.npz': ['oof_amst_3ch_synthetic_targeted'],
        'amst_3ch_tsa_mixup_oof.npz': ['oof_amst_3ch_tsa_mixup'],
        'informer_3ch_synthetic_oof.npz': ['oof_informer_3ch_synthetic'],
        'informer_3ch_synthetic_sp_oof.npz': ['oof_informer_3ch_synthetic_sp'],
        'patch_transformer_raw_3ch_synthetic_oof.npz': ['oof_patch_transformer_raw_3ch_synthetic'],
        'patch_transformer_raw_3ch_synthetic_sp_oof.npz': ['oof_patch_transformer_raw_3ch_synthetic_sp'],
        'hard_fn_gbdt_oof.npz': ['oof_hard_fn_gbdt'],
        'patch_transformer_robust_oof.npz': ['oof_patch_transformer_robust'],
        'residual_cnn_oof.npz': ['oof_residual_cnn'],
        'amst_3ch_raw_oof.npz': ['oof_amst_3ch_raw'],
        'patch_transformer_raw_3ch_oof.npz': ['oof_patch_transformer_raw_3ch'],
        'supcon_raw_3ch_oof.npz': ['oof_supcon_raw_3ch'],
        'patch_transformer_raw_3ch_recall_oof.npz': ['oof_patch_transformer_raw_3ch_recall'],
        'meta_fn_predictor_oof.npz': ['oof_meta_fn_predictor'],
        'meta_error_predictor_oof.npz': ['oof_meta_error_predictor'],
        'patch_transformer_oof.npz': ['oof_patch_transformer'],
        'informer_fast_oof.npz': ['oof_informer_fast'],
    }
    for fname, keys in extra_files.items():
        try:
            d = np.load(os.path.join(OUTPUT_DIR, fname))
            for key in keys:
                if key in d.files and len(d[key]) == len(y):
                    label = f"{fname.replace('.npz', '').replace('_', '-')}-{key.replace('oof_', '').replace('_', '-')}"
                    oofs[label] = d[key]
        except Exception:
            pass

    return oofs


def _train_new_experts(X_features, y, skf, n_experts=4):
    """Train 4 diverse experts on feature pool (no OOF features — keeps diversity high)."""
    n = len(y)
    new_oofs = {}

    # Expert 1: LightGBM
    oof = np.zeros(n)
    for fi, (ti, vi) in enumerate(skf.split(X_features, y)):
        pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
        m = lgb.LGBMClassifier(n_estimators=800, max_depth=7, learning_rate=0.05,
                                num_leaves=63, min_child_samples=100, subsample=0.8,
                                colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                                scale_pos_weight=pw, random_state=SEED + fi, verbose=-1)
        m.fit(X_features[ti], y[ti], eval_set=[(X_features[vi], y[vi])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[vi] = m.predict_proba(X_features[vi])[:, 1]
    new_oofs['LGB-new'] = np.nan_to_num(oof, nan=0.5)

    # Expert 2: XGBoost
    oof = np.zeros(n)
    for fi, (ti, vi) in enumerate(skf.split(X_features, y)):
        pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
        m = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.04,
                               subsample=0.8, colsample_bytree=0.7,
                               scale_pos_weight=pw, tree_method='hist', verbosity=0,
                               random_state=SEED + fi)
        m.fit(X_features[ti], y[ti])
        oof[vi] = m.predict_proba(X_features[vi])[:, 1]
    new_oofs['XGB-new'] = np.nan_to_num(oof, nan=0.5)

    # Expert 3: CatBoost
    try:
        import catboost as cb
        oof = np.zeros(n)
        for fi, (ti, vi) in enumerate(skf.split(X_features, y)):
            m = cb.CatBoostClassifier(iterations=500, depth=8, learning_rate=0.06,
                                       l2_leaf_reg=3.0, verbose=0, random_seed=SEED + fi)
            m.fit(X_features[ti], y[ti], eval_set=(X_features[vi], y[vi]),
                  early_stopping_rounds=80, verbose=False)
            oof[vi] = m.predict_proba(X_features[vi])[:, 1]
        new_oofs['CatB-new'] = np.nan_to_num(oof, nan=0.5)
    except ImportError:
        pass

    # Expert 4: RandomForest
    from sklearn.ensemble import RandomForestClassifier
    oof = np.zeros(n)
    for fi, (ti, vi) in enumerate(skf.split(X_features, y)):
        m = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=20,
                                    max_features=0.3, class_weight='balanced',
                                    n_jobs=-1, random_state=SEED + fi)
        m.fit(X_features[ti], y[ti])
        oof[vi] = m.predict_proba(X_features[vi])[:, 1]
    new_oofs['RF-new'] = np.nan_to_num(oof, nan=0.5)

    return new_oofs


def _build_extended_features(stat_features, impute_mask=None):
    """Extend feature pool with PAA, seasonal, monthly, DOW, trend, quantile from raw data."""
    import pandas as pd

    # Load raw data
    data_path = os.path.join(DATA_DIR, 'raw_data.csv')
    raw_df = pd.read_csv(data_path)
    dc = [c for c in raw_df.columns if '/' in str(c) and len(str(c)) <= 10]
    raw = raw_df[dc].values.astype(float)
    del raw_df
    filled = np.nan_to_num(raw, nan=0)
    n, nd = raw.shape
    half = nd // 2

    X = np.nan_to_num(stat_features, nan=0, posinf=0, neginf=0)

    # ─── PAA multi-resolution ───
    for n_seg in [25, 50, 100]:
        seg = nd / n_seg
        paa = np.zeros((n, n_seg), dtype=np.float32)
        for i in range(n_seg):
            s = int(round(i * seg))
            e = int(round((i + 1) * seg))
            if e > s:
                paa[:, i] = np.nanmean(raw[:, s:e], axis=1)
            elif s < nd:
                paa[:, i] = raw[:, s]
        X = np.column_stack([X, np.nan_to_num(paa, nan=0)])

    # ─── Monthly profiles ───
    month_idx = np.digitize(np.array([j % 365 for j in range(nd)]),
                            np.linspace(0, 365, 13)) - 1
    monthly = np.zeros((n, 12), dtype=np.float32)
    for m in range(12):
        mask = (month_idx == m)
        if mask.sum() > 0:
            monthly[:, m] = np.nanmean(filled[:, mask], axis=1)
    X = np.column_stack([X, np.nan_to_num(monthly, nan=0)])

    # ─── Day-of-week profiles ───
    dow_idx = np.array([j % 7 for j in range(nd)])
    dow = np.zeros((n, 7), dtype=np.float32)
    for d in range(7):
        mask = (dow_idx == d)
        if mask.sum() > 0:
            dow[:, d] = np.nanmean(filled[:, mask], axis=1)
    X = np.column_stack([X, np.nan_to_num(dow, nan=0)])

    # ─── Short-term trends ───
    trends = []
    for w in [7, 14, 30, 60, 90]:
        if nd >= w:
            first = filled[:, :w].mean(axis=1)
            last = filled[:, -w:].mean(axis=1)
            trends.append((last - first) / (np.maximum(np.abs(first), 1e-6)))
            trends.append(np.nanmean(np.diff(filled[:, -w:], axis=1), axis=1))
    X = np.column_stack([X, np.nan_to_num(np.column_stack([t.reshape(-1, 1) for t in trends]), nan=0)])

    # ─── Seasonal decomposition ───
    seas = []
    for period in [7, 30, 90]:
        if nd >= period * 3:
            for offset in range(period):
                idx = np.arange(offset, nd, period)
                if len(idx) > 0:
                    seas.append(np.nanmean(filled[:, idx], axis=1) - np.nanmean(filled, axis=1))
    X = np.column_stack([X, np.nan_to_num(np.column_stack([s.reshape(-1, 1) for s in seas]), nan=0)])

    # ─── Global quantiles ───
    quants = []
    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        quants.append(np.nanpercentile(filled, q, axis=1))
    X = np.column_stack([X, np.nan_to_num(np.column_stack([q.reshape(-1, 1) for q in quants]), nan=0)])

    # ─── Residual aggregation ───
    try:
        base = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
        residuals = base['residuals']
        res_feats = [np.nanmean(residuals, axis=1), np.nanstd(residuals, axis=1),
                     np.nanmean(np.abs(residuals), axis=1), np.nanmax(np.abs(residuals), axis=1)]
        r1 = np.nanmean(residuals[:, :half], axis=1)
        r2 = np.nanmean(residuals[:, half:], axis=1)
        res_feats.append((r2 - r1) / (np.maximum(np.abs(r1), 1e-6)))
        for w in [30, 60, 90, 180]:
            if residuals.shape[1] >= w:
                res_feats.append(np.nanmean(np.abs(residuals[:, -w:]), axis=1))
        X = np.column_stack([X, np.nan_to_num(np.column_stack([r.reshape(-1, 1) for r in res_feats]), nan=0)])
    except Exception:
        pass

    # ─── Missing ratio ───
    if impute_mask is not None:
        X = np.column_stack([X, impute_mask.mean(axis=1).reshape(-1, 1)])

    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    X = np.clip(X, -1e4, 1e4).astype(np.float32)
    return X


class MegaMetaLearner:
    """Multi-OOF stacking meta-learner with external OOF integration."""

    def __init__(self, dataset='sgcc'):
        self.dataset = dataset
        self.config = SGCC_CONFIG if dataset == 'sgcc' else OEDI_CONFIG
        self.dataset_name = self.config['name']

    def train(self, stat_features, labels, impute_mask=None,
              oof_proba_a=None, oof_proba_b=None, oof_proba_c=None,
              fold_assignments=None, X_seq=None, skip_new_experts=False):
        print("=" * 70)
        print(f"  MegaMetaLearner ({self.dataset_name.upper()})")
        print("=" * 70)
        t0 = time.time()

        n = len(labels)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        # ─── 1. Build enhanced feature pool / train new experts ───
        if skip_new_experts:
            print("\n[1] Fast path: skipping new experts and extended feature pool.")
            new_experts = {}
        else:
            print("\n[1] Building extended feature pool...")
            X_pool = _build_extended_features(stat_features, impute_mask)

            # Try loading additional feature sources
            for fn, key in [('dengine_features.npz', 'X'),
                             ('gan_features.npz', 'features')]:
                try:
                    extra = np.load(os.path.join(OUTPUT_DIR, fn))
                    extra_X = np.nan_to_num(extra[key], nan=0, posinf=0, neginf=0)
                    X_pool = np.column_stack([X_pool, extra_X])
                except Exception:
                    pass

            X_pool = X_pool.astype(np.float32)
            print(f"  Feature pool: {X_pool.shape[1]} dims")

            print("\n[2] Training new experts...")
            new_experts = _train_new_experts(X_pool, labels, skf)

            for name, oof in sorted(new_experts.items()):
                f1, _, _, _, _ = _best_f1_score(labels, oof)
                auc = roc_auc_score(labels, oof)
                print(f"  {name:12s}: F1={f1:.4f} AUC={auc:.4f}")

        # ─── 3. Load existing OOFs ───
        print("\n[3] Loading existing OOFs...")
        all_oofs = {}

        # Internal OOFs
        internal = _load_internal_oofs(labels)
        for name, oof in sorted(internal.items()):
            all_oofs[name] = oof
            f1, _, _, _, _ = _best_f1_score(labels, oof)
            print(f"  [internal] {name:15s}: F1={f1:.4f}")

        # External OOFs
        external = _load_external_oofs(labels)
        for name, oof in sorted(external.items()):
            all_oofs[name] = oof
            f1, _, _, _, _ = _best_f1_score(labels, oof)
            print(f"  [external] {name:15s}: F1={f1:.4f}")

        # Add pipeline experts
        if oof_proba_a is not None:
            all_oofs['Expert-A(GBDT)'] = oof_proba_a
        if oof_proba_b is not None:
            all_oofs['Expert-B(TCN)'] = oof_proba_b
        if oof_proba_c is not None:
            all_oofs['Expert-C(Informer)'] = oof_proba_c

        # Add new experts
        for name, oof in new_experts.items():
            all_oofs[name] = oof

        # ─── Quality filter ───
        # Keep all OOFs by default: the meta-learners (regularized LR/XGB)
        # are better at down-weighting noise than a hard filter.  Print
        # diagnostics only.
        print(f"\n  Quality filter (disabled — printing diagnostics only)...")
        for name, oof in all_oofs.items():
            try:
                auc = roc_auc_score(labels, oof)
                f1, _, _, _, _ = _best_f1_score(labels, oof)
                print(f"  {name:18s}: AUC={auc:.4f} F1={f1:.4f}")
            except Exception:
                print(f"  {name:18s}: metric computation failed")

        # ─── Correlation pruning ───
        print(f"\n  Correlation pruning (drop if max_corr > 0.999 with kept)...")
        names = sorted(all_oofs.keys())
        P_tmp = np.column_stack([all_oofs[nm] for nm in names])
        P_tmp = np.nan_to_num(P_tmp, nan=0.5, posinf=1.0, neginf=0.0)
        corrs = np.corrcoef(P_tmp.T)
        kept_names = []
        for i, nm in enumerate(names):
            drop = False
            for j in kept_names:
                idx_j = names.index(j)
                if abs(corrs[i, idx_j]) > 0.999:
                    drop = True
                    break
            if not drop:
                kept_names.append(nm)
            else:
                print(f"  [drop] {nm:18s}: highly correlated with kept OOF")
        all_oofs = {nm: all_oofs[nm] for nm in kept_names}

        print(f"\n  Total OOF pool after filtering: {len(all_oofs)}")

        # ─── 4. Build OOF matrix ───
        names = list(all_oofs.keys())
        P = np.column_stack([all_oofs[nm] for nm in names])
        P = np.nan_to_num(P, nan=0.5, posinf=1.0, neginf=0.0)

        # Add missing-ratio as a meta-feature (helps calibrate threshold by data completeness)
        if impute_mask is not None:
            miss_ratio_feat = impute_mask.mean(axis=1).reshape(-1, 1)
        else:
            miss_ratio_feat = np.zeros((n, 1))
        P = np.column_stack([P, miss_ratio_feat])
        names = names + ['miss_ratio']

        # Correlation check
        corrs = np.corrcoef(P.T)
        for i, nm in enumerate(names):
            row_corrs = [corrs[i, j] for j in range(len(names)) if j != i]
            print(f"  {nm:18s}: mean_corr={np.mean(row_corrs):.3f}  max_corr={np.max(row_corrs):.3f}")

        # ─── 5. Train multiple meta-learners ───
        print(f"\n[4] Training meta-learners...")
        meta_results = {}

        for meta_name, factory in [
            ('XGB-d4', lambda pw: xgb.XGBClassifier(n_estimators=300, max_depth=4,
                learning_rate=0.05, scale_pos_weight=pw, tree_method='hist',
                verbosity=0, random_state=SEED)),
            ('XGB-d3', lambda pw: xgb.XGBClassifier(n_estimators=300, max_depth=3,
                learning_rate=0.05, scale_pos_weight=pw, tree_method='hist',
                verbosity=0, random_state=SEED)),
            ('LR-C1.0', lambda _: LogisticRegression(C=1.0, class_weight='balanced',
                max_iter=2000, random_state=SEED)),
            ('LR-C0.5', lambda _: LogisticRegression(C=0.5, class_weight='balanced',
                max_iter=2000, random_state=SEED)),
            ('LGB', lambda pw: lgb.LGBMClassifier(n_estimators=300, max_depth=3,
                learning_rate=0.05, scale_pos_weight=pw, verbose=-1,
                random_state=SEED)),
            ('HistGB', lambda _: HistGradientBoostingClassifier(max_iter=200,
                max_depth=3, learning_rate=0.05, random_state=SEED)),
            ('CatB', lambda _: __import__('catboost').CatBoostClassifier(
                iterations=500, depth=3, learning_rate=0.05,
                auto_class_weights='Balanced', verbose=0, random_seed=SEED)),
        ]:
            oof = np.zeros(n)
            for fi, (ti, vi) in enumerate(skf.split(P, labels)):
                pw = (labels[ti] == 0).sum() / max((labels[ti] == 1).sum(), 1)
                m = factory(pw)
                if meta_name == 'LGB':
                    m.fit(P[ti], labels[ti], eval_set=[(P[vi], labels[vi])],
                          callbacks=[lgb.early_stopping(50, verbose=False),
                                     lgb.log_evaluation(0)])
                elif meta_name == 'CatB':
                    m.fit(P[ti], labels[ti], eval_set=(P[vi], labels[vi]),
                          early_stopping_rounds=50, verbose=False)
                else:
                    m.fit(P[ti], labels[ti])
                oof[vi] = m.predict_proba(P[vi])[:, 1]

            f1, th, tp, fp, fn = _best_f1_score(labels, oof)
            auc = roc_auc_score(labels, oof)
            meta_results[meta_name] = {
                'f1': f1, 'auc': auc, 'th': th,
                'tp': tp, 'fp': fp, 'fn': fn,
                'oof': oof,
            }
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            print(f"  {meta_name:10s}: F1={f1:.4f} AUC={auc:.4f}  "
                  f"Rec={rec:.4f} Prec={prec:.4f}  th={th:.3f}")

        # Simple average
        avg = P.mean(axis=1)
        f1_avg, th_avg, tp, fp, fn = _best_f1_score(labels, avg)
        meta_results['SimpleAvg'] = {
            'f1': f1_avg, 'auc': roc_auc_score(labels, avg), 'th': th_avg,
            'tp': tp, 'fp': fp, 'fn': fn, 'oof': avg,
        }
        print(f"  SimpleAvg : F1={f1_avg:.4f} AUC={meta_results['SimpleAvg']['auc']:.4f}")

        # ─── 6. Meta ensemble: average of top 3 meta-learners ───
        print(f"\n[5] Meta ensemble (average of top 3)...")
        sorted_meta = sorted(meta_results, key=lambda k: meta_results[k]['f1'], reverse=True)
        top3 = sorted_meta[:3]

        ensemble_oof = np.zeros(n)
        for name in top3:
            ensemble_oof += meta_results[name]['oof']
        ensemble_oof /= 3

        f1_ens, th_ens, tp_ens, fp_ens, fn_ens = _best_f1_score(labels, ensemble_oof)
        auc_ens = roc_auc_score(labels, ensemble_oof)
        meta_results['Top3-Ensemble'] = {
            'f1': f1_ens, 'auc': auc_ens, 'th': th_ens,
            'tp': tp_ens, 'fp': fp_ens, 'fn': fn_ens, 'oof': ensemble_oof,
        }
        rec = tp_ens / (tp_ens + fn_ens) if (tp_ens + fn_ens) > 0 else 0
        prec = tp_ens / (tp_ens + fp_ens) if (tp_ens + fp_ens) > 0 else 0
        print(f"  Top3-Ensemble ({'+'.join(top3)}): F1={f1_ens:.4f} AUC={auc_ens:.4f}  "
              f"Rec={rec:.4f} Prec={prec:.4f}")

        # ─── 7. Pick best ───
        best_name = max(meta_results, key=lambda k: meta_results[k]['f1'])
        best = meta_results[best_name]
        elapsed = (time.time() - t0) / 60

        print(f"\n{'=' * 70}")
        print(f"  FINAL: MegaMetaLearner")
        print(f"{'=' * 70}")
        print(f"  Best meta: {best_name}")
        print(f"  F1=        {best['f1']:.4f}")
        print(f"  AUC=       {best['auc']:.4f}")
        print(f"  Rec=       {best['tp']/(best['tp']+best['fn']):.4f}")
        print(f"  Prec=      {best['tp']/(best['tp']+best['fp']):.4f}"
              if (best['tp'] + best['fp']) > 0 else "  Prec=      0.0000")
        print(f"  th=        {best['th']:.3f}")
        print(f"  TP={best['tp']}  FP={best['fp']}  FN={best['fn']}")
        print(f"  OOFs:      {len(all_oofs)}")
        print(f"  Time:      {elapsed:.1f} min")

        # Comparison
        print(f"\n  COMPARISON:")
        print(f"  V225 (external):     F1=0.8457")
        print(f"  Super-GBDT:          F1=0.8527")
        print(f"  Mega Boost Enhanced: F1=0.8616")
        print(f"  MegaMetaLearner:    F1={best['f1']:.4f}")

        # ─── 7. Save ───
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, f'{self.dataset_name}_mega_meta.npz'),
            oof_final=best['oof'], labels=labels,
            f1=best['f1'], auc=best['auc'], threshold=best['th'],
            tp=best['tp'], fp=best['fp'], fn=best['fn'],
            names=np.array(names),
        )

        results = {
            'oof_proba_meta': best['oof'],
            'oof_proba_a': oof_proba_a,
            'oof_proba_b': oof_proba_b,
            'best_f1': best['f1'],
            'best_f1_unconstrained': best['f1'],
            'best_th': best['th'],
            'best_th_unconstrained': best['th'],
            'best_recall': best['tp'] / (best['tp'] + best['fn']),
            'best_precision': best['tp'] / (best['tp'] + best['fp']) if (best['tp'] + best['fp']) > 0 else 0,
            'flags': labels,
        }

        with open(os.path.join(OUTPUT_DIR, f'{self.dataset_name}_meta_results.pkl'), 'wb') as f:
            pickle.dump(results, f)

        return results


# Backward-compatible wrappers
class MetaLearner(MegaMetaLearner):
    """Backward-compatible alias."""
    pass


def two_stage_stacking_sgcc(stat_features, flags, impute_mask, oof_proba_a, oof_proba_b):
    learner = MetaLearner(dataset='sgcc')
    return learner.train(stat_features, flags, impute_mask=impute_mask,
                         oof_proba_a=oof_proba_a, oof_proba_b=oof_proba_b)


def two_stage_stacking_oedi(stat_features, y, fold_assignments, oof_proba_a, oof_proba_b):
    learner = MetaLearner(dataset='oedi')
    return learner.train(stat_features, y, fold_assignments=fold_assignments,
                         oof_proba_a=oof_proba_a, oof_proba_b=oof_proba_b)


if __name__ == '__main__':
    print("Run through run_pipeline.py")
