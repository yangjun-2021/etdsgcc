import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (f1_score, recall_score, precision_score,
                             roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve,
                             confusion_matrix)

from config import OUTPUT_DIR
from src.utils.utils import best_f1_score, best_f1_recall_constrained


def evaluate_dataset(name, results, output_dir):
    print(f"\n{'=' * 60}")
    print(f"  Evaluation Report: {name.upper()}")
    print(f"{'=' * 60}")

    if name == 'sgcc':
        y_true = results['flags']
    else:
        y_true = results['y']

    proba_meta = results['oof_proba_meta']
    proba_a = results.get('oof_proba_a', proba_meta)
    proba_b = results.get('oof_proba_b', None)
    primary_name = results.get('primary_model_name', 'Meta-Learner (Stacking)')

    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    print(f"\n  Dataset: {name.upper()}")
    print(f"  Total samples: {len(y_true)}")
    print(f"  Positive: {n_pos} ({n_pos/len(y_true)*100:.2f}%)")
    print(f"  Negative: {n_neg} ({n_neg/len(y_true)*100:.2f}%)")

    print(f"\n  --- Primary Model ---")
    evaluate_proba(y_true, proba_a, "  ")

    if proba_b is not None:
        print(f"\n  --- Expert B (TCN) ---")
        evaluate_proba(y_true, proba_b, "  ")

    print(f"\n  --- {primary_name} ---")
    evaluate_proba(y_true, proba_meta, "  ")

    print(f"\n  --- Threshold Analysis (Meta) ---")
    threshold_analysis(y_true, proba_meta)

    print(f"\n  --- Component Comparison ---")
    compare_components(y_true, proba_a, proba_b, proba_meta)

    plot_pr_curve(y_true, proba_meta, output_dir, name)
    plot_roc_curve(y_true, proba_meta, output_dir, name)

    if name == 'sgcc':
        best_th = results.get('best_th_constrained',
                              results.get('best_th_unconstrained', 0.5))
    else:
        best_th = results.get('best_th', 0.5)

    plot_confusion(y_true, proba_meta, best_th, output_dir, name)

    final_pred = (proba_meta > best_th).astype(int)
    cm = confusion_matrix(y_true, final_pred)
    print(f"\n  === Final Results (th={best_th:.3f}) ===")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]:6d}  FP={cm[0,1]:6d}")
    print(f"    FN={cm[1,0]:6d}  TP={cm[1,1]:6d}")
    print(f"  F1={f1_score(y_true, final_pred):.4f}")
    print(f"  Recall={recall_score(y_true, final_pred):.4f}")
    print(f"  Precision={precision_score(y_true, final_pred):.4f}")
    print(f"  AUC={roc_auc_score(y_true, proba_meta):.4f}")
    print(f"  AP={average_precision_score(y_true, proba_meta):.4f}")

    ablation_results = run_ablation(name, y_true, proba_a, proba_b, proba_meta)
    return ablation_results


def evaluate_proba(y_true, proba, indent=""):
    f1, th, rec, prec = best_f1_score(y_true, proba)
    pred = (proba > th).astype(int)
    print(f"{indent}AUC:       {roc_auc_score(y_true, proba):.4f}")
    print(f"{indent}AP:        {average_precision_score(y_true, proba):.4f}")
    print(f"{indent}Best th:   {th:.3f}")
    print(f"{indent}F1:        {f1:.4f}")
    print(f"{indent}Recall:    {rec:.4f}")
    print(f"{indent}Precision: {prec:.4f}")


def threshold_analysis(y_true, proba):
    print(f"  {'th':>5s}  {'F1':>6s}  {'Recall':>7s}  {'Prec':>6s}  {'TP':>5s}  {'FP':>5s}  {'FN':>5s}")
    print(f"  {'-'*50}")
    for th in np.arange(0.2, 0.7, 0.05):
        pred = (proba > th).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        f1 = f1_score(y_true, pred) if pred.sum() > 0 else 0
        recall = recall_score(y_true, pred)
        precision = precision_score(y_true, pred) if pred.sum() > 0 else 0
        print(f"  {th:.2f}  {f1:.4f}  {recall:.4f}  {precision:.4f}  {tp:5d}  {fp:5d}  {fn:5d}")


def compare_components(y_true, proba_a, proba_b, proba_meta):
    components = {'GBDT (Expert A)': proba_a}
    if proba_b is not None:
        components['TCN+Leaf (Expert B)'] = proba_b
    components['Meta Stacking'] = proba_meta
    if proba_b is not None:
        components['Simple Avg A+B'] = (proba_a + proba_b) / 2

    print(f"  {'Component':<25s}  {'AUC':>6s}  {'AP':>6s}  {'F1':>6s}  {'Recall':>7s}  {'Prec':>6s}  {'th':>5s}")
    print(f"  {'-'*75}")
    for cname, cproba in sorted(components.items()):
        f1, th, rec, prec = best_f1_score(y_true, cproba)
        pred = (cproba > th).astype(int)
        print(f"  {cname:<25s}  {roc_auc_score(y_true, cproba):.4f}  "
              f"{average_precision_score(y_true, cproba):.4f}  "
              f"{f1:.4f}  {rec:.4f}  {prec:.4f}  {th:.3f}")


def run_ablation(name, y_true, proba_a, proba_b, proba_meta):
    print(f"\n  --- Ablation Study ---")
    print(f"  {'Configuration':<40s}  {'F1':>6s}  {'Recall':>7s}  {'Prec':>6s}  {'AUC':>6s}")
    print(f"  {'-'*70}")

    results = {}

    def eval_config(config_name, proba):
        if proba is None:
            return
        f1, th, rec, prec = best_f1_score(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        print(f"  {config_name:<40s}  {f1:.4f}  {rec:.4f}  {prec:.4f}  {auc:.4f}")
        results[config_name] = {'f1': f1, 'recall': rec, 'precision': prec, 'auc': auc, 'th': th}

    eval_config("GBDT only (Expert A)", proba_a)
    if proba_b is not None:
        eval_config("TCN+Leaf only (Expert B)", proba_b)
        eval_config("Simple Average (A+B)", (proba_a + proba_b) / 2)
    eval_config("Meta Stacking (Full)", proba_meta)

    recall_constrained = 0.905 if name == 'sgcc' else 0.90
    f1_rc, th_rc, rec_rc, prec_rc = best_f1_recall_constrained(y_true, proba_meta, min_recall=recall_constrained)
    if rec_rc >= recall_constrained:
        print(f"  {'Meta (Recall>=' + str(recall_constrained) + ')':<40s}  {f1_rc:.4f}  {rec_rc:.4f}  {prec_rc:.4f}")
        results['Meta_RecallConstrained'] = {'f1': f1_rc, 'recall': rec_rc, 'precision': prec_rc, 'th': th_rc}
    else:
        print(f"  {'Meta (Recall>=' + str(recall_constrained) + ')':<40s}  impossible")

    return results


def plot_pr_curve(y_true, proba, output_dir, name):
    precision, recall, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)

    f1, th, best_recall, best_prec = best_f1_score(y_true, proba)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(recall, precision, 'b-', linewidth=2, label=f'AP={ap:.4f}')
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title(f'Precision-Recall Curve ({name.upper()})', fontsize=16)
    ax.plot(best_recall, best_prec, 'r*', markersize=15,
            label=f'Best F1={f1:.4f} (th={th:.3f})')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{name}_pr_curve.png'), dpi=150)
    plt.close()
    print(f"  PR curve saved to {output_dir}/{name}_pr_curve.png")


def plot_roc_curve(y_true, proba, output_dir, name):
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC={auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title(f'ROC Curve ({name.upper()})', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{name}_roc_curve.png'), dpi=150)
    plt.close()
    print(f"  ROC curve saved to {output_dir}/{name}_roc_curve.png")


def plot_confusion(y_true, proba, threshold, output_dir, name):
    pred = (proba > threshold).astype(int)
    cm = confusion_matrix(y_true, pred)
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    im1 = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title(f'Confusion Matrix ({name.upper()})\n(th={threshold:.3f})')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Normal', 'Theft'])
    ax1.set_yticklabels(['Normal', 'Theft'])
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')

    im2 = ax2.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.set_title(f'Normalized Confusion Matrix ({name.upper()})\n(th={threshold:.3f})')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Normal', 'Theft'])
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f'{cm_normalized[i, j]:.3f}', ha='center', va='center',
                     color='white' if cm_normalized[i, j] > 0.5 else 'black')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{name}_confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {output_dir}/{name}_confusion_matrix.png")


def evaluate_sgcc():
    with open(os.path.join(OUTPUT_DIR, 'sgcc_meta_results.pkl'), 'rb') as f:
        results = pickle.load(f)
    return evaluate_dataset('sgcc', results, OUTPUT_DIR)


def evaluate_oedi():
    with open(os.path.join(OUTPUT_DIR, 'oedi_meta_results.pkl'), 'rb') as f:
        results = pickle.load(f)
    return evaluate_dataset('oedi', results, OUTPUT_DIR)


if __name__ == '__main__':
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'sgcc'
    if dataset == 'sgcc':
        evaluate_sgcc()
    elif dataset == 'oedi':
        evaluate_oedi()
    else:
        print(f"Unknown dataset: {dataset}")
