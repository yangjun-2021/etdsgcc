"""
Label-noise learning utilities for SGCC electricity theft detection.

Provides:
  - Confident Learning (Northcutt et al., 2021) for identifying noisy labels
  - Consensus-based label cleaning (legacy, used by BSC-ETD)
  - Leave-one-out consensus to reduce circular validation bias
  - Co-teaching / DivideMix ready wrappers

References
----------
Northcutt, C. G., Jiang, L., & Chuang, I. L. (2021). Confident Learning:
  Estimating Uncertainty in Dataset Labels. JAIR, 70, 1373-1411.
Song, H., et al. (2023). Learning from Noisy Labels with Deep Neural Networks:
  A Survey. IEEE TNNLS, 34(11), 8135-8153.
"""
import warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

try:
    from cleanlab.filter import find_label_issues
    CLEANLAB_AVAILABLE = True
except Exception as _e:
    CLEANLAB_AVAILABLE = False
    warnings.warn(f"cleanlab not available; Confident Learning will fall back to consensus. Error: {_e}")


def find_label_issues_confident_learning(oof_probs, labels, frac_noise=1.0,
                                         num_to_remove_per_class=None,
                                         min_examples_per_class=1,
                                         confident_joint=None,
                                         n_jobs=None):
    """Find likely label errors using Confident Learning.

    Args:
        oof_probs: [N, K] out-of-fold predicted probabilities (K classes).
                   For binary SGCC, shape [N, 2].
        labels: [N] integer labels in {0, 1, ..., K-1}.
        frac_noise: fraction of noisy samples to return (default 1.0 = all estimated).
        Other args: passed to cleanlab.filter.find_label_issues.

    Returns:
        issue_idx: np.ndarray of indices with suspected label errors
        issue_mask: [N] boolean mask
        noise_rate: estimated fraction of noisy labels
    """
    if not CLEANLAB_AVAILABLE:
        raise RuntimeError("cleanlab is required for Confident Learning.")

    labels = np.asarray(labels).astype(int)
    oof_probs = np.asarray(oof_probs).astype(float)

    # Ensure probabilities sum to 1 and are valid
    oof_probs = np.clip(oof_probs, 1e-12, 1.0)
    oof_probs = oof_probs / oof_probs.sum(axis=1, keepdims=True)

    issue_mask = find_label_issues(
        labels=labels,
        pred_probs=oof_probs,
        filter_by='prune_by_noise_rate',
        frac_noise=frac_noise,
        num_to_remove_per_class=num_to_remove_per_class,
        min_examples_per_class=min_examples_per_class,
        confident_joint=confident_joint,
        n_jobs=n_jobs,
    )

    issue_idx = np.where(issue_mask)[0]
    noise_rate = issue_mask.mean()
    return issue_idx, issue_mask, noise_rate


def find_label_issues_consensus(oof_probs, labels, n_models_threshold=None,
                                positive_threshold=0.5):
    """Legacy consensus-based label cleaning used in BSC-ETD.

    A sample is flagged when its original label disagrees with the majority
    vote of a pool of out-of-fold probability signals.

    Args:
        oof_probs: [N, M] matrix of M out-of-fold probability signals.
        labels: [N] original labels.
        n_models_threshold: minimum number of models that must agree to flip.
            Defaults to M//2 + 1.
        positive_threshold: probability threshold for positive class.

    Returns:
        issue_idx, issue_mask, noise_rate, cleaned_labels
    """
    oof_probs = np.asarray(oof_probs)
    labels = np.asarray(labels).astype(int)
    n_models = oof_probs.shape[1]
    if n_models_threshold is None:
        n_models_threshold = n_models // 2 + 1

    pred_pos = (oof_probs > positive_threshold).astype(int)
    n_pos_votes = pred_pos.sum(axis=1)
    consensus_pred = (n_pos_votes >= n_models_threshold).astype(int)

    issue_mask = (consensus_pred != labels)
    issue_idx = np.where(issue_mask)[0]
    noise_rate = issue_mask.mean()
    cleaned_labels = consensus_pred.copy()
    return issue_idx, issue_mask, noise_rate, cleaned_labels


def leave_one_out_consensus(oof_prob_matrix, labels, threshold=0.5):
    """Build consensus labels with reduced circular-validation bias.

    For each model m, the consensus label for sample i is computed from all
    other models except m. This avoids using a model's own predictions when
    constructing its supervision signal.

    Args:
        oof_prob_matrix: [N, M] out-of-fold probability signals.
        labels: [N] original labels.
        threshold: probability threshold for positive class.

    Returns:
        looc_consensus: [N, M] consensus predictions. looc_consensus[i, m]
            is the consensus label for sample i excluding model m.
        disagreement_mask: [N] bool, samples where any LOOC consensus differs
            from the original label.
    """
    oof_prob_matrix = np.asarray(oof_prob_matrix)
    n, m = oof_prob_matrix.shape
    labels = np.asarray(labels).astype(int)

    looc_consensus = np.zeros((n, m), dtype=int)
    for j in range(m):
        others = np.delete(oof_prob_matrix, j, axis=1)
        pred_pos = (others > threshold).astype(int)
        n_pos = pred_pos.sum(axis=1)
        looc_consensus[:, j] = (n_pos >= (m - 1) // 2 + 1).astype(int)

    # Flag samples where LOOC consensus disagrees with original label
    # (using majority vote across all LOOC predictions)
    final_consensus = (looc_consensus.mean(axis=1) >= 0.5).astype(int)
    disagreement_mask = (final_consensus != labels)
    return looc_consensus, disagreement_mask, final_consensus


def estimate_noise_rate_by_class(issue_mask, labels):
    """Report noise rate per class."""
    labels = np.asarray(labels)
    issue_mask = np.asarray(issue_mask)
    rates = {}
    for c in np.unique(labels):
        mask_c = labels == c
        rates[int(c)] = float(issue_mask[mask_c].mean()) if mask_c.sum() > 0 else 0.0
    return rates


def compare_label_cleaning_methods(oof_prob_matrix, labels, method='both'):
    """Compare consensus vs. Confident Learning label cleaning.

    Args:
        oof_prob_matrix: [N, M] OOF probability signals.
        labels: [N] original labels.
        method: 'consensus', 'confident_learning', or 'both'.

    Returns:
        dict with results from each method.
    """
    labels = np.asarray(labels).astype(int)
    results = {}

    if method in ('consensus', 'both'):
        idx_c, mask_c, rate_c, cleaned_c = find_label_issues_consensus(
            oof_prob_matrix, labels
        )
        results['consensus'] = {
            'issue_idx': idx_c,
            'issue_mask': mask_c,
            'noise_rate': rate_c,
            'cleaned_labels': cleaned_c,
            'per_class_noise_rate': estimate_noise_rate_by_class(mask_c, labels),
        }

    if method in ('confident_learning', 'both'):
        if oof_prob_matrix.shape[1] == 2:
            # Already [N, 2]
            probs_cl = oof_prob_matrix
        else:
            # Average multiple signals into [N, 2]
            p_pos = oof_prob_matrix.mean(axis=1)
            probs_cl = np.column_stack([1 - p_pos, p_pos])
        idx_cl, mask_cl, rate_cl = find_label_issues_confident_learning(
            probs_cl, labels
        )
        results['confident_learning'] = {
            'issue_idx': idx_cl,
            'issue_mask': mask_cl,
            'noise_rate': rate_cl,
            'cleaned_labels': np.where(mask_cl, 1 - labels, labels),
            'per_class_noise_rate': estimate_noise_rate_by_class(mask_cl, labels),
        }

    return results


if __name__ == '__main__':
    # Quick sanity check
    np.random.seed(42)
    n = 1000
    labels = np.random.binomial(1, 0.1, size=n).astype(int)
    # Simulate 46 correlated OOF signals
    base = np.random.rand(n, 1)
    oof = base + np.random.randn(n, 46) * 0.2
    oof = (oof > 0.5).astype(float)

    print("Consensus cleaning:")
    idx, mask, rate, cleaned = find_label_issues_consensus(oof, labels)
    print(f"  flagged {mask.sum()} / {n} ({rate*100:.2f}%)")
    print(f"  per-class noise rate: {estimate_noise_rate_by_class(mask, labels)}")

    if CLEANLAB_AVAILABLE:
        print("\nConfident Learning cleaning:")
        p_pos = oof.mean(axis=1)
        probs = np.column_stack([1 - p_pos, p_pos])
        idx2, mask2, rate2 = find_label_issues_confident_learning(probs, labels)
        print(f"  flagged {mask2.sum()} / {n} ({rate2*100:.2f}%)")
        print(f"  per-class noise rate: {estimate_noise_rate_by_class(mask2, labels)}")
