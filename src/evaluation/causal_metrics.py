"""Causal evaluation metrics for electricity theft detection.

- policy_value_at_k: precision @ top-k selection (policy evaluation view)
- pehe_proxy: proxy PEHE using observed labels
- ate_gap: predicted lift between treatment groups
"""
import numpy as np


def policy_value_at_k(y_true, scores, k_ratios=(0.01, 0.05, 0.1, 0.2)):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    total_pos = max(int(y_true.sum()), 1)
    n = len(y_true)
    result = {}
    for r in k_ratios:
        k = max(int(round(r * n)), 1)
        top_pos = int(y_sorted[:k].sum())
        precision = top_pos / k
        recall = top_pos / total_pos
        lift = precision / max(y_true.mean(), 1e-8)
        result[f'top{int(r * 100)}%'] = {
            'k': k,
            'precision': precision,
            'recall': recall,
            'lift': lift,
        }
    return result


def pehe_proxy(y_true, cf_normal, cf_theft):
    """Proxy PEHE using observed outcomes as ground truth for the observed arm."""
    y_true = np.asarray(y_true).astype(float)
    cf_normal = np.asarray(cf_normal)
    cf_theft = np.asarray(cf_theft)
    y_hat_obs = np.where(y_true > 0.5, cf_theft, cf_normal)
    factual_err = np.mean((y_true - y_hat_obs) ** 2)
    ite = cf_theft - cf_normal
    return {
        'factual_mse': float(factual_err),
        'ite_mean': float(ite.mean()),
        'ite_std': float(ite.std()),
    }


def ate_gap(scores, treatment):
    treatment = np.asarray(treatment).astype(int)
    scores = np.asarray(scores)
    if treatment.sum() == 0 or (1 - treatment).sum() == 0:
        return 0.0
    return float(scores[treatment == 1].mean() - scores[treatment == 0].mean())


def print_policy_report(name, y_true, scores):
    pv = policy_value_at_k(y_true, scores)
    print(f"  [{name} policy value]")
    for key, vals in pv.items():
        print(f"    {key:>6s} (k={vals['k']}): precision={vals['precision']:.4f} "
              f"recall={vals['recall']:.4f} lift={vals['lift']:.2f}x")
    return pv
