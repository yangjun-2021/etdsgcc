import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

SEED = 42; np.random.seed(SEED)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

print("="*60)
print("Cost-Aware Conformal Prediction for SGCC ETD")
print("="*60)

print("\nLoading V225 OOF...")
data = np.load(os.path.join(OUTPUT_DIR, 'sgcc_final_oof.npz'))
probs = data['oof_v225']
y = data['y']
print(f"  n={len(y)}, theft_rate={y.mean()*100:.1f}%")

n = len(y)
n_positive = int(y.sum())
n_negative = int((1-y).sum())

c_fp = 500
c_fn = 24000

print(f"\n  Cost model:")
print(f"    FP cost (false alarm, on-site inspection): {c_fp} CNY")
print(f"    FN cost (missed theft, annual loss): {c_fn} CNY (8000/year * 3 years)")

review_costs_per_sample = np.array([200, 500, 1000, 2000])
print(f"    Review costs (CNY/sample): {review_costs_per_sample}")

perm = np.random.permutation(n)
n_cal = n // 2
cal_idx = perm[:n_cal]
test_idx = perm[n_cal:]

probs_cal = probs[cal_idx]
y_cal = y[cal_idx]
probs_test = probs[test_idx]
y_test = y[test_idx]

print(f"\n  Calibration: {n_cal}, Test: {n - n_cal}")

def nonconformity_score(prob, label, cost_ratio=1.0):
    if label == 1:
        return 1.0 - prob
    else:
        return prob * cost_ratio

def conformal_predict(probs, scores_cal, alpha):
    n_cal = len(scores_cal)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    tau = np.quantile(scores_cal, min(q_level, 1.0))
    
    predictions = np.zeros(len(probs))
    rejected = np.zeros(len(probs), dtype=bool)
    confidence = np.zeros(len(probs))
    
    for i in range(len(probs)):
        score_0 = probs[i]
        score_1 = 1.0 - probs[i]
        
        if score_0 <= tau and score_1 > tau:
            predictions[i] = 0
            rejected[i] = False
        elif score_1 <= tau and score_0 > tau:
            predictions[i] = 1
            rejected[i] = False
        else:
            rejected[i] = True
            predictions[i] = probs[i] > 0.5
    
    return predictions, rejected, tau

print("\n" + "="*60)
print("Standard Conformal Prediction (marginal)")
print("="*60)

scores_cal_marginal = np.array([
    nonconformity_score(probs_cal[i], y_cal[i], cost_ratio=1.0)
    for i in range(len(probs_cal))
])

alphas = [0.01, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]
results_standard = []

for alpha in alphas:
    preds, rej, tau = conformal_predict(probs_test, scores_cal_marginal, alpha)
    
    auto_mask = ~rej
    n_review = rej.sum()
    review_rate = n_review / len(probs_test)
    
    tp = ((preds == 1) & (y_test == 1) & auto_mask).sum()
    fp = ((preds == 1) & (y_test == 0) & auto_mask).sum()
    fn = ((preds == 0) & (y_test == 1) & auto_mask).sum()
    tn = ((preds == 0) & (y_test == 0) & auto_mask).sum()
    n_auto = auto_mask.sum()
    
    auto_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    auto_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    auto_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    fn_review = ((y_test == 1) & rej).sum()
    fp_review = ((y_test == 0) & (preds == 1) & rej).sum()
    tn_review = ((y_test == 0) & (preds == 0) & rej).sum()
    tp_review = ((y_test == 1) & (preds == 1) & rej).sum()
    
    n_review_errors = fn_review + fp_review
    error_capture = fn_review / max((y_test == 1).sum(), 1)
    
    empirical_coverage = ((preds == y_test) & auto_mask).sum() / max(n_auto, 1)
    
    auto_tp = tp
    auto_fp = fp
    auto_fn = fn
    auto_tn = tn
    
    print(f"  alpha={alpha:5.3f} | auto={n_auto:5d} ({n_auto/len(y_test)*100:4.1f}%) "
          f"| review={n_review:5d} | err_cap={error_capture:.3f} "
          f"| F1_auto={auto_f1:.4f} | cov_emp={empirical_coverage:.3f}")
    
    results_standard.append({
        'alpha': alpha, 'review_rate': review_rate, 'auto_f1': auto_f1,
        'auto_recall': auto_recall, 'auto_precision': auto_precision,
        'error_capture': error_capture, 'empirical_coverage': empirical_coverage,
        'tau': tau, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    })

print("\n" + "="*60)
print("Cost-Aware Conformal Prediction (asymmetric)")
print("="*60)

cost_ratios = [1.0, 2.0, 5.0, 10.0, 20.0, 47.94]
results_cost = []

for cr in cost_ratios:
    scores_cal_cost = np.array([
        nonconformity_score(probs_cal[i], y_cal[i], cost_ratio=cr)
        for i in range(len(probs_cal))
    ])
    
    row_results = []
    for alpha in alphas:
        preds, rej, tau = conformal_predict(probs_test, scores_cal_cost, alpha)
        
        auto_mask = ~rej
        review_rate = rej.sum() / len(probs_test)
        
        tp_auto = ((preds == y_test) & (y_test == 1) & auto_mask).sum()
        fp_auto = ((y_test == 0) & (preds == 1) & auto_mask).sum()
        fn_auto = ((y_test == 1) & (preds == 0) & auto_mask).sum()
        tn_auto = ((y_test == 0) & (preds == 0) & auto_mask).sum()
        n_auto = auto_mask.sum()
        
        auto_f1 = 2 * tp_auto / (2 * tp_auto + fp_auto + fn_auto) if (2 * tp_auto + fp_auto + fn_auto) > 0 else 0
        auto_recall = tp_auto / (tp_auto + fn_auto) if (tp_auto + fn_auto) > 0 else 0
        auto_precision = tp_auto / (tp_auto + fp_auto) if (tp_auto + fp_auto) > 0 else 0
        
        fn_review_total = ((y_test == 1) & rej).sum()
        n_review = rej.sum()
        
        auto_cost_fp = fp_auto * c_fp
        auto_cost_fn = fn_auto * c_fn
        review_cost = n_review * 200
        total_cost = auto_cost_fp + auto_cost_fn + review_cost
        
        baseline_cost_fp = ((probs_test > 0.74) & (y_test == 0)).sum() * c_fp
        baseline_cost_fn = ((probs_test <= 0.74) & (y_test == 1)).sum() * c_fn
        baseline_total = baseline_cost_fp + baseline_cost_fn
        
        cost_saving = baseline_total - total_cost
        fn_cost_reduction = (baseline_cost_fn - auto_cost_fn) / max(baseline_cost_fn, 1)
        
        row_results.append({
            'alpha': alpha, 'review_rate': review_rate, 'auto_f1': auto_f1,
            'auto_recall': auto_recall, 'auto_precision': auto_precision,
            'fn_review': fn_review_total, 'total_cost': total_cost,
            'cost_saving': cost_saving, 'baseline_cost': baseline_total,
            'fn_cost_reduction': fn_cost_reduction,
            'tau': tau,
        })
    
    results_cost.append({'cost_ratio': cr, 'rows': row_results})
    
    best_row = max(row_results, key=lambda r: r['cost_saving'])
    print(f"\n  cost_ratio={cr:5.1f} (FN penalty / FP penalty):")
    print(f"    Best alpha={best_row['alpha']:.3f}: review_rate={best_row['review_rate']:.1%}, "
          f"auto_F1={best_row['auto_f1']:.4f}")
    print(f"    Cost: baseline={best_row['baseline_cost']/1e6:.2f}M -> "
          f"conformal={best_row['total_cost']/1e6:.2f}M, saving={best_row['cost_saving']/1e6:.2f}M")
    print(f"    FN cost reduction: {best_row['fn_cost_reduction']:.1%}")

print("\n" + "="*60)
print("Visualization: Cost-Risk Trade-off Curves")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0, 0]
for r in results_standard:
    ax.plot(r['review_rate'], r['auto_f1'], 'o-', alpha=0.5, markersize=4,
            label=f"alpha={r['alpha']:.2f}" if r['alpha'] in [0.01, 0.05, 0.10, 0.20] else "")
ax.axhline(y=0.8457, color='red', linestyle='--', alpha=0.5, label='V225 full coverage')
ax.set_xlabel('Review Rate')
ax.set_ylabel('Auto-classification F1')
ax.set_title('Standard Conformal: Review Rate vs Auto F1')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for r in results_standard:
    ax.plot(r['alpha'], r['empirical_coverage'], 'o-', markersize=5)
ax.plot([0, 0.3], [1, 0.7], 'k--', alpha=0.3, label='nominal 1-alpha')
ax.set_xlabel('Significance level alpha')
ax.set_ylabel('Empirical Coverage')
ax.set_title('Coverage Guarantee: Empirical vs Nominal')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
for rc in results_cost:
    cr = rc['cost_ratio']
    rows = rc['rows']
    x = [r['review_rate'] for r in rows]
    y = [r['auto_f1'] for r in rows]
    ax.plot(x, y, 'o-', alpha=0.6, markersize=3, label=f'CR={cr:.0f}')
ax.axhline(y=0.8457, color='red', linestyle='--', alpha=0.5, label='V225 baseline')
ax.set_xlabel('Review Rate')
ax.set_ylabel('Auto-classification F1')
ax.set_title('Cost-Aware Conformal: Review Rate vs Auto F1')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

ax = axes[2, 1] if len(axes.shape) > 1 and axes.shape == (2, 3) else axes[1, 1]
for rc in results_cost:
    cr = rc['cost_ratio']
    rows = rc['rows']
    best_by_alpha = {}
    for r in rows:
        a = r['alpha']
        if a not in best_by_alpha or r['cost_saving'] > best_by_alpha[a]['cost_saving']:
            best_by_alpha[a] = r
    x = [r['review_rate'] for r in rows]
    y = [r['cost_saving'] / 1e6 for r in rows]
    ax.plot(x, y, 's-', alpha=0.5, markersize=3, label=f'CR={cr:.0f}')
ax.set_xlabel('Review Rate')
ax.set_ylabel('Cost Saving (Million CNY)')
ax.set_title('Cost Savings vs Review Rate')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

ax = axes[1, 1]
for rc in results_cost:
    cr = rc['cost_ratio']
    rows = rc['rows']
    x = [r['review_rate'] for r in rows]
    y = [r['fn_cost_reduction'] for r in rows]
    ax.plot(x, y, 's-', alpha=0.5, markersize=3, label=f'CR={cr:.0f}')
ax.set_xlabel('Review Rate')
ax.set_ylabel('FN Cost Reduction')
ax.set_title('FN Cost Reduction vs Review Rate')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'cost_conformal_curves.png'), dpi=150)
plt.close()
print("  Saved cost_conformal_curves.png")

print("\n" + "="*60)
print("Cost-Effectiveness Analysis")
print("="*60)

print(f"\n  {'Review':>8s}  {'Auto F1':>8s}  {'Auto Rec':>8s}  {'Auto Prec':>8s}  {'Auto%':>6s}  {'Cost(M)':>8s}  {'Saving(M)':>8s}  {'FN_Red%':>8s}")
print(f"  {'-'*75}")

for rc in results_cost:
    if rc['cost_ratio'] in [1.0, 10.0, 47.94]:
        cr = rc['cost_ratio']
        print(f"\n  Cost Ratio = {cr:.1f}:")
        for r in rc['rows'][::2]:
            print(f"  {r['review_rate']:7.1%}  {r['auto_f1']:8.4f}  {r['auto_recall']:8.4f}  "
                  f"{r['auto_precision']:8.4f}  {1-r['review_rate']:5.1%}  "
                  f"{r['total_cost']/1e6:8.2f}  {r['cost_saving']/1e6:8.2f}  {r['fn_cost_reduction']:7.1%}")

print(f"\n  V225 Baseline (full coverage):")
tp_b = ((probs_test > 0.74) & (y_test == 1)).sum()
fp_b = ((probs_test > 0.74) & (y_test == 0)).sum()
fn_b = ((probs_test <= 0.74) & (y_test == 1)).sum()
tn_b = ((probs_test <= 0.74) & (y_test == 0)).sum()
f1_b = 2*tp_b/(2*tp_b+fp_b+fn_b)
cost_b = fp_b*c_fp + fn_b*c_fn
print(f"  F1={f1_b:.4f}, TP={tp_b}, FP={fp_b}, FN={fn_b}, TN={tn_b}")
print(f"  Cost: FP={fp_b*c_fp/1e6:.2f}M + FN={fn_b*c_fn/1e6:.2f}M = {cost_b/1e6:.2f}M")

np.savez_compressed(os.path.join(OUTPUT_DIR, 'cost_conformal_results.npz'),
                    results_standard=np.array(results_standard),
                    results_cost=np.array(results_cost))
print(f"\nResults saved to cost_conformal_results.npz")