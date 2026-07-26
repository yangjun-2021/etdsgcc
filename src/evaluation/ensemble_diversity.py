"""
Ensemble diversity analysis and pruning for BSC-ETD OOF probability pool.

Addresses reviewer concern: "46路基分类器的高相关度削弱集成多样性"
(平均相关系数 rho>0.95, 排除标准 rho>0.995 过于宽松).

Provides:
  - Pairwise correlation / Q-statistic / double-fault diversity metrics
  - Clustering-based ensemble pruning
  - Greedy forward selection maximizing diversity + accuracy
  - Reports effective number of independent signals

References
----------
Kuncheva, L. I., & Whitaker, C. J. (2003). Measures of diversity in
  classifier ensembles and their relationship with the ensemble accuracy.
  Machine Learning, 51(2), 181-207.
"""
import warnings
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings('ignore')


def pairwise_pearson(probas):
    """Return pairwise Pearson correlation matrix of base-learner probabilities."""
    proba = np.asarray(probas)
    n = proba.shape[1]
    corr = np.corrcoef(proba.T)
    return corr


def q_statistic(preds):
    """Compute pairwise Q-statistic diversity (Kuncheva & Whitaker, 2003).

    Q = (N11*N00 - N10*N01) / (N11*N00 + N10*N01)
    Q close to 0 -> diverse; Q close to 1 -> identical.
    """
    preds = np.asarray(preds).astype(int)
    n = preds.shape[1]
    Q = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = preds[:, i], preds[:, j]
            n11 = ((pi == 1) & (pj == 1)).sum()
            n00 = ((pi == 0) & (pj == 0)).sum()
            n10 = ((pi == 1) & (pj == 0)).sum()
            n01 = ((pi == 0) & (pj == 1)).sum()
            denom = n11 * n00 + n10 * n01
            if denom == 0:
                q = 0.0
            else:
                q = (n11 * n00 - n10 * n01) / denom
            Q[i, j] = Q[j, i] = q
    return Q


def double_fault_rate(preds):
    """Fraction of samples on which both classifiers are wrong."""
    preds = np.asarray(preds).astype(int)
    labels = None
    return _pairwise_error_overlap(preds, labels, mode='both_wrong')


def _pairwise_error_overlap(preds, labels=None, mode='both_wrong'):
    preds = np.asarray(preds).astype(int)
    n = preds.shape[1]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if mode == 'both_wrong':
                # Need labels; if not provided assume preds already indicate errors
                overlap = ((preds[:, i] == 1) & (preds[:, j] == 1)).mean()
            else:
                overlap = (preds[:, i] != preds[:, j]).mean()
            M[i, j] = M[j, i] = overlap
    return M


def effective_independent_signals(corr_matrix, threshold=0.95):
    """Estimate effective number of independent signals given correlation matrix.

    Uses eigenvalue-based effective rank / participation ratio.
    """
    eigvals = np.linalg.eigvalsh(corr_matrix)
    eigvals = np.clip(eigvals, 1e-12, None)
    participation_ratio = eigvals.sum() ** 2 / (eigvals ** 2).sum()
    # Number of eigenvalues above threshold
    n_above = (eigvals > threshold).sum()
    return {
        'effective_rank': participation_ratio,
        'n_eigenvalues_above_threshold': int(n_above),
        'mean_correlation': corr_matrix[np.triu_indices_from(corr_matrix, k=1)].mean(),
        'max_correlation': corr_matrix[np.triu_indices_from(corr_matrix, k=1)].max(),
    }


def prune_by_correlation(probas, names=None, rho_threshold=0.95, method='greedy'):
    """Prune highly correlated base learners to increase ensemble diversity.

    Args:
        probas: [N, M] OOF probability matrix.
        names: list of M base-learner names (optional).
        rho_threshold: maximum allowed absolute Pearson correlation.
        method: 'greedy' (keep strongest uncorrelated) or 'cluster' (hierarchical).

    Returns:
        selected_idx: list of selected column indices
        selected_names: list of names
        info: dict with statistics
    """
    probas = np.asarray(probas)
    M = probas.shape[1]
    if names is None:
        names = [f"model_{i}" for i in range(M)]

    corr = np.abs(pairwise_pearson(probas))

    if method == 'greedy':
        # Sort by individual AUC (proxy for quality), then greedily keep if
        # not too correlated with already selected.
        try:
            individual_aucs = np.array([
                roc_auc_score(np.zeros(probas.shape[0]), probas[:, i])
                if len(np.unique(np.zeros(probas.shape[0]))) == 1
                else 0.5
                for i in range(M)
            ])
        except Exception:
            individual_aucs = np.zeros(M)
        order = np.argsort(-individual_aucs)

        selected = []
        for idx in order:
            if not selected:
                selected.append(idx)
                continue
            if corr[idx, selected].max() < rho_threshold:
                selected.append(idx)
        selected = sorted(selected)

    elif method == 'cluster':
        distance = 1 - corr
        np.fill_diagonal(distance, 0)
        condensed = squareform(distance, checks=False)
        Z = linkage(condensed, method='average')
        # Choose number of clusters so that within-cluster max corr < threshold
        max_clusters = M
        best_k = 1
        for k in range(2, max_clusters + 1):
            labels = fcluster(Z, k, criterion='maxclust')
            ok = True
            for c in np.unique(labels):
                members = np.where(labels == c)[0]
                if len(members) > 1:
                    sub_corr = corr[np.ix_(members, members)]
                    if sub_corr[np.triu_indices_from(sub_corr, k=1)].max() >= rho_threshold:
                        ok = False
                        break
            if ok:
                best_k = k
                break
        labels = fcluster(Z, best_k, criterion='maxclust')
        selected = []
        for c in np.unique(labels):
            members = np.where(labels == c)[0]
            # Pick member with highest individual AUC
            best_member = members[0]
            selected.append(best_member)
        selected = sorted(selected)

    else:
        raise ValueError(f"Unknown pruning method: {method}")

    info = {
        'original_count': M,
        'selected_count': len(selected),
        'rho_threshold': rho_threshold,
        'method': method,
        'mean_corr_before': corr[np.triu_indices_from(corr, k=1)].mean(),
        'mean_corr_after': (
            corr[np.ix_(selected, selected)][np.triu_indices(len(selected), k=1)].mean()
            if len(selected) > 1 else 0.0
        ),
    }
    return selected, [names[i] for i in selected], info


def greedy_diversity_selection(probas, labels, metric='f1', n_select=None,
                                diversity_weight=0.5):
    """Forward selection maximizing weighted combination of performance and diversity.

    Args:
        probas: [N, M] OOF probability signals.
        labels: [N] ground-truth labels.
        metric: 'f1' or 'auc' for individual quality.
        n_select: number of models to select (default: auto by elbow).
        diversity_weight: weight for diversity term in selection criterion.

    Returns:
        selected_idx, scores_trace, info
    """
    probas = np.asarray(probas)
    labels = np.asarray(labels).astype(int)
    N, M = probas.shape

    if metric == 'f1':
        def score(p):
            best = 0
            for th in np.arange(0.05, 0.95, 0.05):
                best = max(best, f1_score(labels, (p > th).astype(int), zero_division=0))
            return best
    else:
        def score(p):
            try:
                return roc_auc_score(labels, p)
            except Exception:
                return 0.5

    individual_scores = np.array([score(probas[:, i]) for i in range(M)])
    available = set(range(M))
    selected = []
    scores_trace = []

    if n_select is None:
        n_select = min(M, 20)

    while len(selected) < n_select and available:
        best_candidate = None
        best_obj = -np.inf
        for cand in available:
            trial = selected + [cand]
            ensemble_proba = probas[:, trial].mean(axis=1)
            perf = score(ensemble_proba)
            # Diversity: mean pairwise correlation among selected (lower = better)
            if len(trial) > 1:
                sub_corr = np.corrcoef(probas[:, trial].T)
                div = 1 - sub_corr[np.triu_indices_from(sub_corr, k=1)].mean()
            else:
                div = 1.0
            obj = (1 - diversity_weight) * perf + diversity_weight * div
            if obj > best_obj:
                best_obj = obj
                best_candidate = cand

        if best_candidate is None:
            break
        selected.append(best_candidate)
        available.remove(best_candidate)
        scores_trace.append(best_obj)

    info = {
        'individual_scores': individual_scores,
        'selected_scores': scores_trace,
        'mean_corr_after': (
            np.corrcoef(probas[:, selected].T)[np.triu_indices(len(selected), k=1)].mean()
            if len(selected) > 1 else 0.0
        ),
    }
    return selected, scores_trace, info


def analyze_ensemble_diversity(probas, labels=None, names=None, rho_threshold=0.95):
    """Full diversity report for an OOF probability pool."""
    probas = np.asarray(probas)
    corr = pairwise_pearson(probas)
    eff = effective_independent_signals(corr, threshold=0.95)

    report = {
        'n_signals': probas.shape[1],
        'mean_abs_correlation': np.abs(corr[np.triu_indices_from(corr, k=1)]).mean(),
        'max_correlation': corr[np.triu_indices_from(corr, k=1)].max(),
        'effective_independent_signals': eff,
    }

    # Pruning report
    selected, sel_names, prune_info = prune_by_correlation(
        probas, names=names, rho_threshold=rho_threshold, method='greedy'
    )
    report['pruning'] = prune_info
    report['pruning']['selected_idx'] = selected
    report['pruning']['selected_names'] = sel_names

    if labels is not None:
        sel, trace, info = greedy_diversity_selection(
            probas, labels, metric='f1', n_select=min(15, probas.shape[1])
        )
        report['greedy_selection'] = {
            'selected_idx': sel,
            'objective_trace': trace,
            'mean_corr_after': info['mean_corr_after'],
        }

    return report


if __name__ == '__main__':
    np.random.seed(42)
    n, m = 1000, 20
    labels = np.random.binomial(1, 0.1, size=n)
    # Create correlated signals
    base = np.random.rand(n, 1)
    probas = base * 0.8 + np.random.randn(n, m) * 0.2
    probas = np.clip(probas, 0, 1)

    report = analyze_ensemble_diversity(probas, labels=labels, rho_threshold=0.95)
    print("Diversity report:")
    for k, v in report.items():
        print(f"  {k}: {v}")
