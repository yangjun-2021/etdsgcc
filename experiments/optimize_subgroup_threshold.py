"""Subgroup threshold optimization for hillclimb OOF (fast vectorized).

Goal: see if per-usage-quintile thresholds can push F1/recall beyond the
single global threshold on SGCC original labels.
"""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR
from sklearn.metrics import f1_score, recall_score, precision_score


def main():
    d = np.load(os.path.join(OUTPUT_DIR, 'hillclimb_best_oof.npz'))
    proba = d['oof_hillclimb'].astype(np.float64)
    y = d['flags'].astype(int)

    usage = np.load(os.path.join(OUTPUT_DIR, 'usage_features.npz'))
    log_max = usage['log_max_usage']

    thresholds = np.arange(0.05, 0.95, 0.005)
    best_f1, best_th, best_rec, best_prec = 0, 0.5, 0, 0
    for th in thresholds:
        pred = (proba > th).astype(int)
        if pred.sum() == 0:
            continue
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_rec = recall_score(y, pred, zero_division=0)
            best_prec = precision_score(y, pred, zero_division=0)
    print(f"Baseline global: F1={best_f1:.4f} Rec={best_rec:.4f} Prec={best_prec:.4f} th={best_th:.3f}")

    n_groups = 5
    quintiles = np.percentile(log_max, np.linspace(0, 100, n_groups + 1))
    group_ids = np.digitize(log_max, quintiles[1:-1], right=True)
    print("\nGroup sizes / theft rates:")
    for g in range(n_groups):
        mask = group_ids == g
        print(f"  G{g}: n={mask.sum()} theft_rate={y[mask].mean()*100:.2f}%")

    # Candidate thresholds per group
    cand_ths = np.arange(0.05, 0.90, 0.05)
    M = len(cand_ths)

    # Precompute TP/FP/FN per group per threshold
    tp = np.zeros((n_groups, M), dtype=int)
    fp = np.zeros((n_groups, M), dtype=int)
    fn = np.zeros((n_groups, M), dtype=int)
    for g in range(n_groups):
        mask = group_ids == g
        y_g = y[mask]
        p_g = proba[mask]
        for j, th in enumerate(cand_ths):
            pred = (p_g > th).astype(int)
            tp[g, j] = ((pred == 1) & (y_g == 1)).sum()
            fp[g, j] = ((pred == 1) & (y_g == 0)).sum()
            fn[g, j] = ((pred == 0) & (y_g == 1)).sum()

    def score_from_tpfpfn(TP, FP, FN):
        rec = TP / (TP + FN + 1e-12)
        prec = TP / (TP + FP + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        return f1, rec, prec

    # Broadcast search
    print(f"\nBroadcast grid search: {M}^{n_groups} combos")
    grids = np.meshgrid(*[np.arange(M) for _ in range(n_groups)], indexing='ij')
    idx = np.stack([g.ravel() for g in grids], axis=1)
    row_idx = np.arange(n_groups)[:, None]
    TP = tp[row_idx, idx.T].sum(axis=0)
    FP = fp[row_idx, idx.T].sum(axis=0)
    FN = fn[row_idx, idx.T].sum(axis=0)
    f1s, recs, precs = score_from_tpfpfn(TP, FP, FN)

    best_idx = f1s.argmax()
    best_combo = idx[best_idx]
    best_overall = {
        'f1': f1s[best_idx],
        'rec': recs[best_idx],
        'prec': precs[best_idx],
        'ths': cand_ths[best_combo],
    }
    print(f"Best coarse per-group thresholds: {best_overall['ths']}")
    print(f"  F1={best_overall['f1']:.4f} Rec={best_overall['rec']:.4f} Prec={best_overall['prec']:.4f}")

    # Recall-constrained search
    mask_rc = recs >= 0.90
    if mask_rc.any():
        rc_idx = f1s[mask_rc].argmax()
        combo_rc = idx[mask_rc][rc_idx]
        best_rc = {
            'f1': f1s[mask_rc][rc_idx],
            'rec': recs[mask_rc][rc_idx],
            'prec': precs[mask_rc][rc_idx],
            'ths': cand_ths[combo_rc],
        }
        print(f"Best recall>=0.90: thresholds={best_rc['ths']}")
        print(f"  F1={best_rc['f1']:.4f} Rec={best_rc['rec']:.4f} Prec={best_rc['prec']:.4f}")
    else:
        best_rc = None
        print("No coarse combo achieved recall>=0.90")

    # Coordinate-descent refinement using precomputed tables
    print("\nCoordinate-descent refinement (step 0.01)...")
    fine_ths = best_overall['ths'].copy()
    fine_cand = np.arange(0.05, 0.95, 0.01)
    # Recompute tp/fp/fn on fine grid per group
    fine_tp = np.zeros((n_groups, len(fine_cand)), dtype=int)
    fine_fp = np.zeros((n_groups, len(fine_cand)), dtype=int)
    fine_fn = np.zeros((n_groups, len(fine_cand)), dtype=int)
    for g in range(n_groups):
        mask = group_ids == g
        y_g = y[mask]
        p_g = proba[mask]
        for j, th in enumerate(fine_cand):
            pred = (p_g > th).astype(int)
            fine_tp[g, j] = ((pred == 1) & (y_g == 1)).sum()
            fine_fp[g, j] = ((pred == 1) & (y_g == 0)).sum()
            fine_fn[g, j] = ((pred == 0) & (y_g == 1)).sum()

    def total_score(th_idx_per_group):
        TP = fine_tp[np.arange(n_groups), th_idx_per_group].sum()
        FP = fine_fp[np.arange(n_groups), th_idx_per_group].sum()
        FN = fine_fn[np.arange(n_groups), th_idx_per_group].sum()
        return score_from_tpfpfn(TP, FP, FN)

    current_idx = np.array([np.abs(fine_cand - t).argmin() for t in fine_ths])
    current_f1, _, _ = total_score(current_idx)
    improved = True
    while improved:
        improved = False
        for g in range(n_groups):
            best_local_f1 = current_f1
            best_local_idx = current_idx[g]
            for j in range(len(fine_cand)):
                trial = current_idx.copy()
                trial[g] = j
                f1, _, _ = total_score(trial)
                if f1 > best_local_f1:
                    best_local_f1 = f1
                    best_local_idx = j
            if best_local_idx != current_idx[g]:
                current_idx[g] = best_local_idx
                current_f1 = best_local_f1
                improved = True

    fine_f1, fine_rec, fine_prec = total_score(current_idx)
    fine_ths = fine_cand[current_idx]
    print(f"Best refined per-group thresholds: {np.round(fine_ths, 3)}")
    print(f"  F1={fine_f1:.4f} Rec={fine_rec:.4f} Prec={fine_prec:.4f}")

    # Per-group metrics
    pred_ref = (proba > fine_ths[group_ids]).astype(int)
    print("\nPer-group metrics (refined thresholds):")
    for g in range(n_groups):
        mask = group_ids == g
        pred_g = pred_ref[mask]
        y_g = y[mask]
        if pred_g.sum() == 0:
            f1 = rec = prec = 0.0
        else:
            f1 = f1_score(y_g, pred_g, zero_division=0)
            rec = recall_score(y_g, pred_g, zero_division=0)
            prec = precision_score(y_g, pred_g, zero_division=0)
        print(f"  G{g} th={fine_ths[g]:.3f}: F1={f1:.4f} Rec={rec:.4f} Prec={prec:.4f}")

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'subgroup_threshold_result.npz'),
        baseline_f1=best_f1, baseline_rec=best_rec, baseline_prec=best_prec, baseline_th=best_th,
        refined_f1=fine_f1, refined_rec=fine_rec, refined_prec=fine_prec,
        refined_thresholds=fine_ths,
        group_ids=group_ids,
        pred=pred_ref,
        proba=proba,
        flags=y,
    )
    print(f"\nSaved to {os.path.join(OUTPUT_DIR, 'subgroup_threshold_result.npz')}")


if __name__ == '__main__':
    main()
