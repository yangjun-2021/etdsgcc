"""
Improved Meta-Learner v2 for ETD-SGCC.

Targets F1 > 0.90 through:
  1. Greedy ensemble selection (forward selection) on OOF predictions
  2. Non-negative least squares (NNLS) weight optimization
  3. Bayesian Detection Rate (BDR) threshold optimization
  4. FPR-constrained threshold search
  5. Diverse meta-learners on selected OOF subset

The v2 learner is designed to be a drop-in replacement for MegaMetaLearner.
"""
import os
import pickle
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from scipy.optimize import nnls
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

warnings.filterwarnings('ignore')

from config import SGCC_CONFIG, OEDI_CONFIG, SEED, N_FOLDS, OUTPUT_DIR

# Autoresearch hyperparameters (tune these between runs)
TOP_K = 50
CORR_THRESHOLD = 0.9999
MAX_ENSEMBLE_SIZE = 15
N_CANDIDATES = 10


def _best_f1_score(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray = None) -> Tuple[float, float, int, int, int]:
    """Pure function to compute best F1 score by threshold search."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.005)
    best_f1, best_th = 0, 0.5
    for th in thresholds:
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


def _best_bdr_score(y_true: np.ndarray, y_prob: np.ndarray, prior: float = None,
                    thresholds: np.ndarray = None) -> Tuple[float, float]:
    """Threshold search maximizing Bayesian Detection Rate.

    BDR = (TPR * P(thief)) / (TPR * P(thief) + FPR * P(normal))
    This is the posterior probability that a flagged user is truly a thief.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.005)
    if prior is None:
        prior = y_true.mean()
    best_bdr, best_th = 0, 0.5
    for th in thresholds:
        pred = (y_prob > th).astype(int)
        if pred.sum() == 0:
            continue
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        tpr = tp / max((y_true == 1).sum(), 1)
        fpr = fp / max((y_true == 0).sum(), 1)
        numerator = tpr * prior
        denominator = numerator + fpr * (1 - prior)
        if denominator <= 0:
            continue
        bdr = numerator / denominator
        if bdr > best_bdr:
            best_bdr, best_th = bdr, th
    return best_bdr, best_th


def _best_recall_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, max_fpr: float = 0.01,
                        thresholds: np.ndarray = None) -> Tuple[float, float, float]:
    """Find threshold with maximum recall subject to FPR <= max_fpr."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.005)
    n_neg = max((y_true == 0).sum(), 1)
    best_rec, best_f1, best_th = 0, 0, 0.5
    for th in thresholds:
        pred = (y_prob > th).astype(int)
        if pred.sum() == 0:
            continue
        fp = ((pred == 1) & (y_true == 0)).sum()
        fpr = fp / n_neg
        if fpr > max_fpr:
            continue
        tp = ((pred == 1) & (y_true == 1)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        rec = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        f = 2 * tp / max(2 * tp + fp + fn, 1)
        if rec > best_rec or (rec == best_rec and f > best_f1):
            best_rec, best_f1, best_th = rec, f, th
    return best_rec, best_th, best_f1


def _metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, th: float) -> Dict[str, float]:
    pred = (y_prob > th).astype(int)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f = 2 * tp / max(2 * tp + fp + fn, 1)
    return {'f1': f, 'recall': rec, 'precision': prec, 'tp': tp, 'fp': fp, 'fn': fn}


def _load_external_oofs(y: np.ndarray) -> Dict[str, np.ndarray]:
    """Load bundled external OOFs."""
    TRUE_OOF_KEYS = {
        'ExpertA_OOF', 'ExpertB_OOF',
        'V71_cat', 'V71_innov', 'V71_lgb', 'V71_tcn', 'V71_xgb',
        'V213', 'V216', 'V219', 'V225',
        'V229_oof_iso', 'V229_oof_platt',
        'AnomalyAE_OOF', 'AnomalyIF_OOF',
        'NeighborTheftRatio', 'NeighborDistance',
    }
    existing = {}

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

    return {}


def _discover_all_npz_oofs(y: np.ndarray) -> Dict[str, np.ndarray]:
    """Auto-discover all full-length OOF-like arrays in OUTPUT_DIR npz files."""
    oofs = {}
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if not fname.endswith('.npz'):
            continue
        path = os.path.join(OUTPUT_DIR, fname)
        try:
            d = np.load(path)
            for key in d.files:
                # Skip obvious non-OOF keys
                if key in {'flags', 'y', 'labels', 'names', 'n_oofs', 'oofs', 'fold', 'idx'}:
                    continue
                # Accept keys that look like OOF predictions
                if not ('oof' in key.lower() or key in {'prior', 'v225_oof'}):
                    continue
                arr = d[key]
                if arr.shape != y.shape:
                    continue
                if arr.dtype.kind not in ('f', 'i'):
                    continue
                # Heuristic: predictions should be in [0, 1] roughly
                if arr.max() > 10 or arr.min() < -10:
                    continue
                label = f"{fname.replace('.npz', '').replace('_', '-')}-{key.replace('oof_', '').replace('_', '-')}"
                # Avoid duplicates
                if label in oofs:
                    continue
                oofs[label] = arr.astype(np.float64)
        except Exception:
            continue
    return oofs


def _load_internal_oofs(y: np.ndarray) -> Dict[str, np.ndarray]:
    """Load internal OOFs from output directory."""
    oofs = {}

    # Auto-discover all npz OOFs first
    discovered = _discover_all_npz_oofs(y)
    oofs.update(discovered)

    # Legacy internal results (fallback / explicit keys)
    for fname, keys in [
        ('tcn_kd_results.npz', ['oof_tcn_kd', 'oof_stacker', 'oof_blend', 'oof_hill']),
        ('super_gbdt.npz', ['oof_super']),
        ('smart_blend.npz', ['oof_final']),
        ('multi_oof_results.npz', ['oof_lgb', 'oof_xgb', 'oof_cb']),
        ('tcn_enhanced.npz', ['oof_tcn']),
    ]:
        try:
            d = np.load(os.path.join(OUTPUT_DIR, fname))
            for key in keys:
                if key in d.files and len(d[key]) == len(y):
                    label = f"{fname.replace('.npz', '').replace('_', '-')}-{key.replace('oof_', '').replace('_', '-')}"
                    oofs[label] = d[key]
        except Exception:
            pass

    # Large list of extra files
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
        # Additional strong blends discovered in output/
        'final_blend_best_oof.npz': ['oof_final_blend_best'],
        'gated_rescue_refined_oof.npz': ['oof_gated_rescue_refined'],
        'gated_rescue_blend_oof.npz': ['oof_gated_rescue_blend'],
        'mega_meta_all_oofs_oof.npz': ['oof_mega_meta_all_oofs'],
        'meta_raw_only_oof.npz': ['oof_meta_raw_only'],
        'feature_rich_meta_oof.npz': ['oof_feature_rich_meta'],
        'nn_meta_oof.npz': ['oof_nn_meta'],
        'meta_final_cleaned_oof.npz': ['oof_meta_final_cleaned'],
        'meta_cleaned_oof.npz': ['oof_meta_cleaned'],
        'informer_oof.npz': ['oof_informer'],
        'hillclimb_best_oof.npz': ['oof_hillclimb'],
        'sgcc_expert_c_multiscale.npz': ['oof_proba'],
        'sgcc_expert_d_unsupervised.npz': ['oof_isf', 'oof_lof', 'oof_knn', 'oof_combined'],
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


def _correlation_prune(oofs: Dict[str, np.ndarray], max_corr: float = 0.999) -> Dict[str, np.ndarray]:
    """Drop OOFs that are nearly perfectly correlated with a kept OOF."""
    names = sorted(oofs.keys())
    if len(names) <= 1:
        return oofs
    P = np.column_stack([oofs[nm] for nm in names])
    P = np.nan_to_num(P, nan=0.5, posinf=1.0, neginf=0.0)
    corrs = np.corrcoef(P.T)
    kept = []
    for i, nm in enumerate(names):
        drop = False
        for j in kept:
            idx_j = names.index(j)
            if abs(corrs[i, idx_j]) > max_corr:
                drop = True
                break
        if not drop:
            kept.append(nm)
    return {nm: oofs[nm] for nm in kept}


def _greedy_ensemble_selection(y: np.ndarray, oofs: Dict[str, np.ndarray],
                                max_size: int = 15, n_candidates: int = 5,
                                verbose: bool = True) -> Tuple[List[str], np.ndarray]:
    """Greedy forward selection of OOFs to maximize F1 of their average.

    At each step, try adding each remaining OOF and keep the one that gives
    the largest F1 improvement. Start from the single best OOF.
    """
    names = list(oofs.keys())
    if not names:
        return [], np.zeros(len(y))

    best_single = None
    best_single_f1 = -1
    for nm in names:
        f1, _, _, _, _ = _best_f1_score(y, oofs[nm])
        if f1 > best_single_f1:
            best_single_f1 = f1
            best_single = nm

    selected = [best_single]
    remaining = [nm for nm in names if nm != best_single]
    current_blend = oofs[best_single].copy()
    current_f1 = best_single_f1

    if verbose:
        print(f"  Ensemble seed: {best_single} F1={current_f1:.4f}")

    while remaining and len(selected) < max_size:
        best_candidate = None
        best_f1 = current_f1
        best_blend = None

        # Evaluate top candidates by individual F1 to limit search
        candidates = sorted(remaining, key=lambda nm: _best_f1_score(y, oofs[nm])[0], reverse=True)[:n_candidates]
        for cand in candidates:
            trial_blend = (current_blend * len(selected) + oofs[cand]) / (len(selected) + 1)
            f1, _, _, _, _ = _best_f1_score(y, trial_blend)
            if f1 > best_f1:
                best_f1 = f1
                best_candidate = cand
                best_blend = trial_blend

        if best_candidate is None:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        current_blend = best_blend
        current_f1 = best_f1
        if verbose:
            print(f"  Added {best_candidate}: size={len(selected)} F1={current_f1:.4f}")

    return selected, current_blend


def _nnls_weighted_blend(y: np.ndarray, oofs: Dict[str, np.ndarray],
                         selected: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Learn non-negative weights for selected OOFs using NNLS on calibrated targets."""
    if selected is None:
        selected = list(oofs.keys())
    P = np.column_stack([oofs[nm] for nm in selected])
    P = np.nan_to_num(P, nan=0.5, posinf=1.0, neginf=0.0)
    # Target: try to fit a soft target that encourages high recall
    # Use OOF probabilities themselves as features and binary labels as target.
    # NNLS finds w >= 0 s.t. P @ w ≈ y.
    weights, _ = nnls(P, y.astype(float))
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(selected)) / len(selected)
    blend = P.dot(weights)
    return blend, weights


def _optimize_weighted_blend(y: np.ndarray, oofs: Dict[str, np.ndarray],
                              selected: List[str] = None,
                              n_grid: int = 11) -> Tuple[np.ndarray, np.ndarray]:
    """Grid-search non-negative weights on selected OOFs to maximize F1.

    Uses a coarse-to-fine search: first find best single weight, then refine.
    """
    if selected is None:
        selected = list(oofs.keys())
    k = len(selected)
    P = np.column_stack([oofs[nm] for nm in selected])
    P = np.nan_to_num(P, nan=0.5, posinf=1.0, neginf=0.0)

    # Start with equal weights
    best_w = np.ones(k) / k
    best_f1, _, _, _, _ = _best_f1_score(y, P.dot(best_w))

    # Greedy coordinate-wise grid search (multiple passes)
    for _ in range(3):
        improved = False
        for i in range(k):
            for alpha in np.linspace(0, 1, n_grid):
                w = best_w.copy()
                w[i] = alpha
                if w.sum() > 0:
                    w = w / w.sum()
                f1, _, _, _, _ = _best_f1_score(y, P.dot(w))
                if f1 > best_f1:
                    best_f1 = f1
                    best_w = w
                    improved = True
        if not improved:
            break

    return P.dot(best_w), best_w


class ImprovedMetaLearner:
    """Improved meta-learner with ensemble selection and advanced thresholding."""

    def __init__(self, dataset='sgcc'):
        self.dataset = dataset
        self.config = SGCC_CONFIG if dataset == 'sgcc' else OEDI_CONFIG
        self.dataset_name = self.config['name']

    def train(self, stat_features, labels, impute_mask=None,
              oof_proba_a=None, oof_proba_b=None, oof_proba_c=None,
              fold_assignments=None, X_seq=None, skip_new_experts=True,
              pool_include=None):
        print("=" * 70)
        print(f"  ImprovedMetaLearner v2 ({self.dataset_name.upper()})")
        print("=" * 70)
        t0 = time.time()

        n = len(labels)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        # Load all available OOFs
        print("\n[1] Loading OOF pool...")
        all_oofs = {}

        internal = _load_internal_oofs(labels)
        for name, oof in sorted(internal.items()):
            all_oofs[name] = oof

        external = _load_external_oofs(labels)
        for name, oof in sorted(external.items()):
            all_oofs[name] = oof

        if oof_proba_a is not None:
            all_oofs['Expert-A(GBDT)'] = oof_proba_a
        if oof_proba_b is not None:
            all_oofs['Expert-B(TCN)'] = oof_proba_b
        if oof_proba_c is not None:
            all_oofs['Expert-C(Informer)'] = oof_proba_c

        # Optional pool restriction (e.g. only untainted v3-family OOFs)
        if pool_include:
            pats = pool_include if isinstance(pool_include, (list, tuple)) else [pool_include]
            all_oofs = {k: v for k, v in all_oofs.items()
                        if any(p in k for p in pats)}
            print(f"  Pool filtered by {list(pats)}: {len(all_oofs)}")

        print(f"  Raw OOF pool: {len(all_oofs)}")

        # Quality diagnostics
        print("\n[2] OOF quality diagnostics (top 15 by F1):")
        oof_f1s = []
        for name, oof in all_oofs.items():
            try:
                f1, th, tp, fp, fn = _best_f1_score(labels, oof)
                auc = roc_auc_score(labels, oof)
                oof_f1s.append((name, f1, auc, th))
            except Exception:
                pass
        oof_f1s.sort(key=lambda x: x[1], reverse=True)
        for name, f1, auc, th in oof_f1s[:15]:
            print(f"  {name:45s} F1={f1:.4f} AUC={auc:.4f} th={th:.3f}")

        # If a single OOF already exceeds 0.90, note it and still ensemble for robustness
        if oof_f1s and oof_f1s[0][1] >= 0.90:
            print(f"\n  *** Single OOF {oof_f1s[0][0]} already reaches F1={oof_f1s[0][1]:.4f} ***")

        # Keep top-K diverse OOFs for ensemble selection to control computation
        top_k = min(TOP_K, len(oof_f1s))
        top_names = [name for name, _, _, _ in oof_f1s[:top_k]]
        top_oofs = {name: all_oofs[name] for name in top_names}

        # Light correlation pruning among top OOFs to reduce redundancy
        top_oofs = _correlation_prune(top_oofs, max_corr=CORR_THRESHOLD)
        print(f"\n  Top-{top_k} OOF pool after light pruning: {len(top_oofs)}")

        # Ensemble selection
        print("\n[3] Greedy ensemble selection...")
        selected, greedy_blend = _greedy_ensemble_selection(
            labels, top_oofs, max_size=MAX_ENSEMBLE_SIZE, n_candidates=N_CANDIDATES, verbose=True)
        f1_greedy, th_greedy, tp_g, fp_g, fn_g = _best_f1_score(labels, greedy_blend)
        print(f"  Greedy blend: F1={f1_greedy:.4f} th={th_greedy:.3f} "
              f"TP={tp_g} FP={fp_g} FN={fn_g}")

        # NNLS weighted blend on selected OOFs
        print("\n[4] NNLS weighted blend on selected OOFs...")
        nnls_blend, nnls_w = _nnls_weighted_blend(labels, top_oofs, selected)
        f1_nnls, th_nnls, tp_n, fp_n, fn_n = _best_f1_score(labels, nnls_blend)
        print(f"  NNLS blend: F1={f1_nnls:.4f} th={th_nnls:.3f} "
              f"TP={tp_n} FP={fp_n} FN={fn_n}")
        for nm, w in zip(selected, nnls_w):
            if w > 0.001:
                print(f"    {nm}: {w:.4f}")

        # Grid-search weighted blend on selected OOFs
        print("\n[5] Grid-search weighted blend on selected OOFs...")
        grid_blend, grid_w = _optimize_weighted_blend(labels, top_oofs, selected, n_grid=11)
        f1_grid, th_grid, tp_gr, fp_gr, fn_gr = _best_f1_score(labels, grid_blend)
        print(f"  Grid blend: F1={f1_grid:.4f} th={th_grid:.3f} "
              f"TP={tp_gr} FP={fp_gr} FN={fn_gr}")

        # Meta-learners on selected OOFs (with CV)
        print("\n[6] Training meta-learners on selected OOFs...")
        P = np.column_stack([top_oofs[nm] for nm in selected])
        P = np.nan_to_num(P, nan=0.5, posinf=1.0, neginf=0.0)
        if impute_mask is not None:
            miss_ratio_feat = impute_mask.mean(axis=1).reshape(-1, 1)
        else:
            miss_ratio_feat = np.zeros((n, 1))
        P_ext = np.column_stack([P, miss_ratio_feat])

        meta_results = {}
        for meta_name, factory in [
            ('XGB-d3', lambda pw: xgb.XGBClassifier(n_estimators=500, max_depth=3,
                learning_rate=0.05, scale_pos_weight=pw, tree_method='hist',
                verbosity=0, random_state=SEED)),
            ('XGB-d4', lambda pw: xgb.XGBClassifier(n_estimators=500, max_depth=4,
                learning_rate=0.05, scale_pos_weight=pw, tree_method='hist',
                verbosity=0, random_state=SEED)),
            ('LR-C0.3', lambda _: LogisticRegression(C=0.3, class_weight='balanced',
                max_iter=2000, random_state=SEED)),
            ('LR-C1.0', lambda _: LogisticRegression(C=1.0, class_weight='balanced',
                max_iter=2000, random_state=SEED)),
            ('LGB-d3', lambda pw: lgb.LGBMClassifier(n_estimators=500, max_depth=3,
                learning_rate=0.05, scale_pos_weight=pw, verbose=-1,
                random_state=SEED)),
            ('HistGB', lambda _: HistGradientBoostingClassifier(max_iter=300,
                max_depth=3, learning_rate=0.05, random_state=SEED)),
        ]:
            oof = np.zeros(n)
            for fi, (ti, vi) in enumerate(skf.split(P_ext, labels)):
                pw = (labels[ti] == 0).sum() / max((labels[ti] == 1).sum(), 1)
                m = factory(pw)
                if meta_name == 'LGB-d3':
                    m.fit(P_ext[ti], labels[ti], eval_set=[(P_ext[vi], labels[vi])],
                          callbacks=[lgb.early_stopping(50, verbose=False),
                                     lgb.log_evaluation(0)])
                else:
                    m.fit(P_ext[ti], labels[ti])
                oof[vi] = m.predict_proba(P_ext[vi])[:, 1]

            f1, th, tp, fp, fn = _best_f1_score(labels, oof)
            auc = roc_auc_score(labels, oof)
            meta_results[meta_name] = {'f1': f1, 'auc': auc, 'th': th,
                                        'tp': tp, 'fp': fp, 'fn': fn, 'oof': oof}
            print(f"  {meta_name:10s}: F1={f1:.4f} AUC={auc:.4f} th={th:.3f}")

        # Collect all candidate OOFs
        candidates = {
            'GreedyBlend': greedy_blend,
            'NNLS-Blend': nnls_blend,
            'GridBlend': grid_blend,
        }
        for name, res in meta_results.items():
            candidates[name] = res['oof']

        # Best F1 candidate
        best_name = max(candidates, key=lambda k: _best_f1_score(labels, candidates[k])[0])
        best_f1, best_th, best_tp, best_fp, best_fn = _best_f1_score(labels, candidates[best_name])
        best_oof = candidates[best_name]
        best_auc = roc_auc_score(labels, best_oof)

        # BDR threshold
        prior = labels.mean()
        bdr, th_bdr = _best_bdr_score(labels, best_oof, prior=prior)
        metrics_bdr = _metrics_at_threshold(labels, best_oof, th_bdr)

        # FPR-constrained thresholds
        fpr_targets = [0.005, 0.01, 0.015, 0.02]
        fpr_results = {}
        for max_fpr in fpr_targets:
            rec, th_fpr, f_at_fpr = _best_recall_at_fpr(labels, best_oof, max_fpr=max_fpr)
            fpr_results[max_fpr] = {'recall': rec, 'th': th_fpr, 'f1': f_at_fpr}

        elapsed = (time.time() - t0) / 60
        print(f"\n{'=' * 70}")
        print(f"  FINAL: ImprovedMetaLearner v2")
        print(f"{'=' * 70}")
        print(f"  Best candidate: {best_name}")
        print(f"  F1=        {best_f1:.4f}")
        print(f"  AUC=       {best_auc:.4f}")
        print(f"  Rec=       {best_tp/(best_tp+best_fn):.4f}")
        print(f"  Prec=      {best_tp/(best_tp+best_fp):.4f}")
        print(f"  th=        {best_th:.3f}")
        print(f"  TP={best_tp}  FP={best_fp}  FN={best_fn}")
        print(f"\n  BDR threshold: th={th_bdr:.3f}, BDR={bdr:.4f}, "
              f"F1={metrics_bdr['f1']:.4f}, Rec={metrics_bdr['recall']:.4f}, "
              f"Prec={metrics_bdr['precision']:.4f}")
        print(f"\n  FPR-constrained thresholds:")
        for max_fpr, res in fpr_results.items():
            print(f"    FPR<={max_fpr*100:.1f}%: th={res['th']:.3f}, "
                  f"Rec={res['recall']:.4f}, F1={res['f1']:.4f}")
        print(f"\n  Time:      {elapsed:.1f} min")

        # Save results
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, f'{self.dataset_name}_meta_v2.npz'),
            oof_final=best_oof, labels=labels,
            f1=best_f1, auc=best_auc, threshold=best_th,
            tp=best_tp, fp=best_fp, fn=best_fn,
            selected=np.array(selected),
            bdr_threshold=th_bdr, bdr_value=bdr,
        )

        results = {
            'oof_proba_meta': best_oof,
            'oof_proba_a': oof_proba_a,
            'oof_proba_b': oof_proba_b,
            'best_f1': best_f1,
            'best_f1_unconstrained': best_f1,
            'best_th': best_th,
            'best_th_unconstrained': best_th,
            'best_recall': best_tp / (best_tp + best_fn),
            'best_precision': best_tp / (best_tp + best_fp) if (best_tp + best_fp) > 0 else 0,
            'flags': labels,
        }

        with open(os.path.join(OUTPUT_DIR, f'{self.dataset_name}_meta_v2_results.pkl'), 'wb') as f:
            pickle.dump(results, f)

        return results


# Backward-compatible alias
class MetaLearnerV2(ImprovedMetaLearner):
    pass


if __name__ == '__main__':
    print("Run through run_pipeline.py or run_meta_v2.py")
