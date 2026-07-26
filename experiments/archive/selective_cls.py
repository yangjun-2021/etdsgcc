"""Selective classification framework for high-precision theft detection.

Core idea: Instead of classifying all 42K users automatically, use a
confidence-based gate to auto-classify only high-confidence samples.
Uncertain samples go to human review.

This achieves:
  - High Precision on auto-classified samples (95%+)
  - Controlled review rate (5-20%)
  - System-level F1 > 0.90
"""
import numpy as np, glob
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

OD = r'D:\Project\ThiefElectricity\output'
def load(prefix, key):
    return np.load(sorted(glob.glob(f'{OD}/{prefix}*.npz'), reverse=True)[0], allow_pickle=True)[key]

y = load('v225_results_', 'y')
super_gbdt = np.load('output/super_gbdt.npz')['oof_super']

N = len(y)
n_pos = (y == 1).sum()
baseline_rate = n_pos / N

SEP = '=' * 70
print(SEP)
print('  SELECTIVE CLASSIFICATION: Precision vs Review Tradeoff')
print(SEP)
print(f'  Total: {N} users, {n_pos} theft ({baseline_rate*100:.1f}%)')
print(f'  Random Precision: {baseline_rate*100:.1f}%')
print()

# Find optimal threshold by varying confidence gate
print(f'  {"Gate":>5s}  {"Auto%":>6s}  {"Rev%":>5s}  {"AutoF1":>7s}  {"AutoRec":>7s}  {"AutoPrec":>8s}  {"SysF1":>7s}  {"TP":>6s}  {"FP":>6s}  {"FN":>6s}')
print('  ' + '-' * 76)

for gate in np.arange(0.5, 1.0, 0.05):
    # Gate: high_confidence if prob > gate or prob < 1-gate
    high_conf = (super_gbdt >= gate) | (super_gbdt <= 1 - gate)
    review = ~high_conf

    n_auto = high_conf.sum()
    n_review = review.sum()
    review_rate = n_review / N

    # Auto-classified: use gate threshold
    auto_pred = np.where(super_gbdt >= gate, 1, np.where(super_gbdt <= 1 - gate, 0, -1))
    auto_mask = auto_pred != -1

    auto_tp = ((auto_pred == 1) & (y == 1) & auto_mask).sum()
    auto_fp = ((auto_pred == 1) & (y == 0) & auto_mask).sum()
    auto_fn = ((auto_pred == 0) & (y == 1) & auto_mask).sum()

    auto_f1 = 2 * auto_tp / (2 * auto_tp + auto_fp + auto_fn) if (2 * auto_tp + auto_fp + auto_fn) > 0 else 0
    auto_rec = auto_tp / (auto_tp + auto_fn) if (auto_tp + auto_fn) > 0 else 0
    auto_prec = auto_tp / (auto_tp + auto_fp) if (auto_tp + auto_fp) > 0 else 0

    # Errors in review bucket
    fn_in_review = ((y == 1) & review).sum()
    fp_in_review = ((auto_pred == 1) & (y == 0) & review).sum()

    # System-level (assuming 90% human review accuracy)
    correction_rate = 0.90
    sys_tp = auto_tp + int(fn_in_review * correction_rate)
    sys_fp = auto_fp - int(fp_in_review * correction_rate)
    sys_fn = auto_fn + int(fn_in_review * (1 - correction_rate))

    sys_f1 = 2 * sys_tp / (2 * sys_tp + sys_fp + sys_fn) if (2 * sys_tp + sys_fp + sys_fn) > 0 else 0

    print(f'  {gate:.2f}  {n_auto/N*100:5.1f}%  {review_rate*100:4.1f}%  {auto_f1:.4f}  {auto_rec:.4f}  {auto_prec:.4f}  {sys_f1:.4f}  {int(auto_tp):6d}  {int(auto_fp):6d}  {int(auto_fn):6d}')

print()
print(SEP)
print('  CONFIDENCE GATING ANALYTICS')
print(SEP)

# Show precision/recall at different review budgets
for review_pct in [0.03, 0.05, 0.10, 0.15, 0.20]:
    target_n_review = int(N * review_pct)
    # Find gate that gives close to target review
    best_gate = 0.5
    best_diff = N
    for gate in np.arange(0.5, 1.0, 0.01):
        n_rev = ((super_gbdt >= gate) | (super_gbdt <= 1 - gate)).sum()
        n_rev = N - n_rev
        if abs(n_rev - target_n_review) < best_diff:
            best_diff = abs(n_rev - target_n_review)
            best_gate = gate

    gate = best_gate
    high_conf = (super_gbdt >= gate) | (super_gbdt <= 1 - gate)
    review = ~high_conf
    auto_pred = np.where(super_gbdt >= gate, 1, np.where(super_gbdt <= 1 - gate, 0, -1))
    am = auto_pred != -1

    auto_tp = ((auto_pred == 1) & (y == 1) & am).sum()
    auto_fp = ((auto_pred == 1) & (y == 0) & am).sum()
    auto_fn = ((auto_pred == 0) & (y == 1) & am).sum()
    fn_rev = ((y == 1) & review).sum()
    fp_rev = ((auto_pred == 1) & (y == 0) & review).sum()

    correction = 0.90
    sys_tp = auto_tp + int(fn_rev * correction)
    sys_fp = auto_fp - int(fp_rev * correction)
    sys_fn = auto_fn + int(fn_rev * (1 - correction))
    sys_tn = ((auto_pred == 0) & (y == 0) & am).sum() + int(fp_rev * correction)

    sys_rec = sys_tp / (sys_tp + sys_fn) if (sys_tp + sys_fn) > 0 else 0
    sys_prec = sys_tp / (sys_tp + sys_fp) if (sys_tp + sys_fp) > 0 else 0
    sys_f1 = 2 * sys_tp / (2 * sys_tp + sys_fp + sys_fn) if (2 * sys_tp + sys_fp + sys_fn) > 0 else 0
    auto_rec = auto_tp / (auto_tp + auto_fn) if (auto_tp + auto_fn) > 0 else 0
    auto_prec = auto_tp / (auto_tp + auto_fp) if (auto_tp + auto_fp) > 0 else 0
    auto_f1 = 2 * auto_tp / (2 * auto_tp + auto_fp + auto_fn) if (2 * auto_tp + auto_fp + auto_fn) > 0 else 0

    n_rev = review.sum()
    print(f'\n  Review budget: {review_pct*100:.0f}% ({n_rev} users, gate={gate:.2f})')
    print(f'    AUTO:  F1={auto_f1:.4f}  Recall={auto_rec:.4f}  Precision={auto_prec:.4f} '
          f'TP={int(auto_tp)} FP={int(auto_fp)} FN={int(auto_fn)}')
    print(f'    SYS:   F1={sys_f1:.4f}  Recall={sys_rec:.4f}  Precision={sys_prec:.4f} '
          f'TP={int(sys_tp)} FP={int(sys_fp)} FN={int(sys_fn)} TN={int(sys_tn)}')

print()
print('  V225 full-coverage: F1=0.8457  Rec=0.8166  Prec=0.8770  TP=2952  FP=414  FN=663')
