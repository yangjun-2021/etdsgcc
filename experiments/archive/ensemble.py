"""
Multi-model ensemble + selective classification for F1/Recall breakthrough.

Strategy:
  1. GBDT (Expert A): 232 statistical features → OOF
  2. TCN+Leaf+Prior (Expert B): multi-channel time series → OOF
  3. Transformer+Prior (Expert C): time-frequency → OOF
  4. XGBoost meta-learner: stack all 3 OOFs + features → final OOF
  5. Selective classification: high-confidence → auto, low-confidence → review

The key insight from V221 oracle analysis:
  - F1=0.90 requires correcting 50% of mistakes (234 samples)
  - Selective classification with 3-5% review achieves system F1=0.91
  - This is the ONLY viable path to F1=0.90+ given the information-theoretic limit
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

from config import SEED, N_FOLDS, OUTPUT_DIR
from utils import seed_everything, best_f1_score, best_f1_recall_constrained


def meta_stacking(oof_a, oof_b, oof_c, stat_features, impute_mask, flags):
    """XGBoost meta-learner stacking.

    Args:
        oof_a: [N] GBDT OOF
        oof_b: [N] TCN OOF
        oof_c: [N] Transformer OOF
        stat_features: [N, F] statistical features
        impute_mask: [N, T] missing mask
        flags: [N] labels

    Returns:
        oof_meta: [N] meta-learner OOF
    """
    print("=" * 60)
    print("Meta-Learner: Multi-Model Stacking")
    print("=" * 60)

    n = len(flags)
    miss_ratio = impute_mask.mean(axis=1).reshape(-1, 1)

    meta_features = np.column_stack([
        stat_features,
        miss_ratio,
        oof_a.reshape(-1, 1),
        oof_b.reshape(-1, 1),
        oof_c.reshape(-1, 1),
        np.abs(oof_a - oof_b).reshape(-1, 1),
        np.abs(oof_a - oof_c).reshape(-1, 1),
        np.abs(oof_b - oof_c).reshape(-1, 1),
        ((oof_a + oof_b + oof_c) / 3).reshape(-1, 1),
    ])
    print(f"  Meta features: {meta_features.shape}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_meta = np.zeros(n)

    pos_weight = (flags == 0).sum() / max((flags == 1).sum(), 1)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(meta_features, flags)):
        X_train, X_val = meta_features[train_idx], meta_features[val_idx]
        y_train, y_val = flags[train_idx], flags[val_idx]

        model = xgb.XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            scale_pos_weight=pos_weight,
            tree_method='hist', random_state=SEED, verbosity=0,
        )
        model.fit(X_train, y_train)
        oof_meta[val_idx] = model.predict_proba(X_val)[:, 1]

        f1, th, rec, prec = best_f1_score(y_val, oof_meta[val_idx])
        print(f"  Fold {fold_idx+1}: F1={f1:.4f} AUC={roc_auc_score(y_val, oof_meta[val_idx]):.4f}")

    overall_f1, best_th, overall_rec, overall_prec = best_f1_score(flags, oof_meta)
    overall_auc = roc_auc_score(flags, oof_meta)
    print(f"\n[Meta] F1={overall_f1:.4f} AUC={overall_auc:.4f} "
          f"Rec={overall_rec:.4f} Prec={overall_prec:.4f} th={best_th:.3f}")

    return oof_meta


def selective_classification(oof_meta, flags, auto_threshold=0.9):
    """Selective classification with conformal-style gating.

    High-confidence predictions (prob > auto_threshold or prob < 1-auto_threshold)
    are auto-classified. Uncertain predictions are sent to human review.

    Args:
        oof_meta: [N] meta-learner probabilities
        flags: [N] true labels
        auto_threshold: confidence threshold for auto-classification

    Returns:
        dict with auto metrics, review metrics, system metrics
    """
    print(f"\n{'=' * 60}")
    print(f"Selective Classification (auto_threshold={auto_threshold})")
    print(f"{'=' * 60}")

    n = len(flags)
    high_conf = (oof_meta >= auto_threshold) | (oof_meta <= 1 - auto_threshold)
    review_mask = ~high_conf

    n_auto = high_conf.sum()
    n_review = review_mask.sum()
    review_rate = n_review / n

    auto_pred = np.where(oof_meta >= auto_threshold, 1,
                         np.where(oof_meta <= 1 - auto_threshold, 0, -1))

    auto_mask = high_conf
    auto_tp = ((auto_pred == 1) & (flags == 1) & auto_mask).sum()
    auto_fp = ((auto_pred == 1) & (flags == 0) & auto_mask).sum()
    auto_fn = ((auto_pred == 0) & (flags == 1) & auto_mask).sum()
    auto_tn = ((auto_pred == 0) & (flags == 0) & auto_mask).sum()

    auto_f1 = 2 * auto_tp / (2 * auto_tp + auto_fp + auto_fn) if (2 * auto_tp + auto_fp + auto_fn) > 0 else 0
    auto_recall = auto_tp / (auto_tp + auto_fn) if (auto_tp + auto_fn) > 0 else 0
    auto_precision = auto_tp / (auto_tp + auto_fp) if (auto_tp + auto_fp) > 0 else 0

    fn_in_review = ((flags == 1) & review_mask).sum()
    fp_in_review = ((auto_pred == 1) & (flags == 0) & review_mask).sum()

    review_errors = fn_in_review + fp_in_review
    total_errors = ((flags == 1) & (oof_meta < 0.5)).sum() + ((flags == 0) & (oof_meta >= 0.5)).sum()
    error_capture = review_errors / max(total_errors, 1)

    print(f"\n  Auto-classified: {n_auto} ({n_auto/n*100:.1f}%)")
    print(f"  Review: {n_review} ({review_rate*100:.1f}%)")
    print(f"  Auto F1: {auto_f1:.4f}")
    print(f"  Auto Recall: {auto_recall:.4f}")
    print(f"  Auto Precision: {auto_precision:.4f}")
    print(f"  Errors in review: {review_errors} / {total_errors} total ({error_capture*100:.1f}%)")

    for review_correction_rate in [0.5, 0.7, 0.9, 1.0]:
        corrected_fn = int(fn_in_review * review_correction_rate)
        corrected_fp = int(fp_in_review * review_correction_rate)

        sys_tp = auto_tp + corrected_fn
        sys_fp = auto_fp - corrected_fp + (fp_in_review - corrected_fp)
        sys_fn = auto_fn + (fn_in_review - corrected_fn)
        sys_tn = auto_tn + corrected_fp + (fp_in_review - corrected_fp) * 0

        sys_f1 = 2 * sys_tp / (2 * sys_tp + sys_fp + sys_fn) if (2 * sys_tp + sys_fp + sys_fn) > 0 else 0
        sys_recall = sys_tp / (sys_tp + sys_fn) if (sys_tp + sys_fn) > 0 else 0
        sys_precision = sys_tp / (sys_tp + sys_fp) if (sys_tp + sys_fp) > 0 else 0

        print(f"\n  With {review_correction_rate*100:.0f}% review correction:")
        print(f"    System F1: {sys_f1:.4f}")
        print(f"    System Recall: {sys_recall:.4f}")
        print(f"    System Precision: {sys_precision:.4f}")
        print(f"    TP={sys_tp} FP={sys_fp} FN={sys_fn} TN={sys_tn}")

        if review_correction_rate == 0.9:
            return {
                'auto_f1': auto_f1, 'auto_recall': auto_recall,
                'auto_precision': auto_precision,
                'review_rate': review_rate,
                'error_capture': error_capture,
                'system_f1': sys_f1, 'system_recall': sys_recall,
                'system_precision': sys_precision,
                'sys_tp': int(sys_tp), 'sys_fp': int(sys_fp),
                'sys_fn': int(sys_fn), 'sys_tn': int(sys_tn),
            }

    return None


def run_ensemble_pipeline(oof_a, oof_b, oof_c, stat_features, impute_mask, flags):
    """Full ensemble pipeline: meta-stacking + selective classification.

    Args:
        oof_a: GBDT OOF
        oof_b: TCN OOF
        oof_c: Transformer OOF
        stat_features: statistical features
        impute_mask: missing mask
        flags: labels

    Returns:
        results dict
    """
    seed_everything(SEED)

    print("\n" + "=" * 60)
    print("  INDIVIDUAL MODEL PERFORMANCE")
    print("=" * 60)
    for name, oof in [("GBDT (Expert A)", oof_a),
                       ("TCN (Expert B)", oof_b),
                       ("Transformer (Expert C)", oof_c)]:
        f1, th, rec, prec = best_f1_score(flags, oof)
        auc = roc_auc_score(flags, oof)
        print(f"  {name:<25s}: F1={f1:.4f} AUC={auc:.4f} Rec={rec:.4f} Prec={prec:.4f}")

    oof_meta = meta_stacking(oof_a, oof_b, oof_c, stat_features, impute_mask, flags)

    f1, th, rec, prec = best_f1_score(flags, oof_meta)
    f1_rc, th_rc, rec_rc, prec_rc = best_f1_recall_constrained(flags, oof_meta, min_recall=0.90)
    print(f"\n[Meta Unconstrained]    F1={f1:.4f} Rec={rec:.4f} Prec={prec:.4f} th={th:.3f}")
    if rec_rc >= 0.90:
        print(f"[Meta Recall>=0.90]     F1={f1_rc:.4f} Rec={rec_rc:.4f} Prec={prec_rc:.4f} th={th_rc:.3f}")
    else:
        print(f"[Meta Recall>=0.90]     Not achievable (max recall at best F1 th: {rec:.4f})")

    for threshold in [0.85, 0.90, 0.95]:
        print(f"\n{'---' * 15}")
        result = selective_classification(oof_meta, flags, auto_threshold=threshold)

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'ensemble_results.npz'),
        oof_a=oof_a, oof_b=oof_b, oof_c=oof_c,
        oof_meta=oof_meta, flags=flags,
    )
    print(f"\nResults saved to {OUTPUT_DIR}/ensemble_results.npz")

    return oof_meta


if __name__ == '__main__':
    import sys

    data_path = os.path.join(OUTPUT_DIR, 'ensemble_data.npz')
    if not os.path.exists(data_path):
        print("Please run the full pipeline first to generate OOF predictions.")
        print("Usage: python pipeline.py sgcc --ensemble")
        sys.exit(1)

    data = np.load(data_path)
    run_ensemble_pipeline(
        data['oof_a'], data['oof_b'], data['oof_c'],
        data['stat_features'], data['impute_mask'], data['flags']
    )
