"""
Conformal prediction calibration diagnostics for BSC-ETD selective classification.

Addresses reviewer concerns:
  - Empirical coverage 98.8% >> nominal 90% (overly conservative)
  - Ljung-Box time autocorrelation violates exchangeability
  - Alpha selection lacks sensitivity analysis

Provides:
  - Split conformal prediction sets for binary classification
  - Coverage calibration curve vs. nominal alpha
  - Recommendation for alpha to achieve target empirical coverage
  - Time-aware block calibration to assess exchangeability violation

References
----------
Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a
  Random World. Springer.
Angelopoulos, A. N., & Bates, S. (2023). Conformal Prediction: A Gentle
  Introduction. Foundations and Trends in Machine Learning.
"""
import warnings
import numpy as np
from scipy import stats

warnings.filterwarnings('ignore')


def split_conformal_scores(probas, labels):
    """Compute non-conformity scores for binary classification.

    For binary case, score = 1 - predicted probability of true class.
    Lower score -> more conformal (model is confident and correct).
    """
    probas = np.asarray(probas).astype(float)
    labels = np.asarray(labels).astype(int)
    n = len(labels)
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = 1.0 - probas[i, labels[i]]
    return scores


def calibrate_threshold(cal_scores, alpha=0.10):
    """Compute conformal quantile threshold q_hat on calibration scores."""
    cal_scores = np.asarray(cal_scores)
    n = len(cal_scores)
    # q_hat = ceil((n+1)*(1-alpha))/n quantile
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    q_hat = np.quantile(cal_scores, q_level, method='higher')
    return q_hat


def conformal_prediction_sets(probas, q_hat):
    """Return prediction sets for each sample.

    For binary classification, set contains class 0 if score_0 <= q_hat,
    and class 1 if score_1 <= q_hat.
    """
    probas = np.asarray(probas).astype(float)
    n = probas.shape[0]
    sets = []
    for i in range(n):
        s = []
        if 1.0 - probas[i, 0] <= q_hat:
            s.append(0)
        if 1.0 - probas[i, 1] <= q_hat:
            s.append(1)
        sets.append(s)
    return sets


def empirical_coverage(probas, labels, q_hat):
    """Fraction of samples whose true label is in the prediction set."""
    sets = conformal_prediction_sets(probas, q_hat)
    labels = np.asarray(labels).astype(int)
    covered = np.array([labels[i] in sets[i] for i in range(len(labels))])
    return covered.mean(), covered


def coverage_calibration_curve(cal_probas, cal_labels, test_probas, test_labels,
                               alphas=None):
    """Compute empirical coverage for a grid of nominal alpha values.

    Returns:
        dict with alphas, q_hats, empirical_coverages, abstention_rates
    """
    if alphas is None:
        alphas = np.array([0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30])

    cal_scores = split_conformal_scores(cal_probas, cal_labels)
    coverages = []
    abstentions = []
    q_hats = []
    for alpha in alphas:
        q = calibrate_threshold(cal_scores, alpha=alpha)
        q_hats.append(q)
        cov, _ = empirical_coverage(test_probas, test_labels, q)
        coverages.append(cov)
        # Abstention rate = fraction with empty or multi-label prediction sets
        sets = conformal_prediction_sets(test_probas, q)
        abst = np.mean([len(s) != 1 for s in sets])
        abstentions.append(abst)

    return {
        'alphas': alphas,
        'q_hats': np.array(q_hats),
        'empirical_coverages': np.array(coverages),
        'abstention_rates': np.array(abstentions),
    }


def recommend_alpha(target_coverage, cal_probas, cal_labels, test_probas, test_labels,
                    alphas=None):
    """Find nominal alpha whose empirical coverage is closest to target."""
    curve = coverage_calibration_curve(
        cal_probas, cal_labels, test_probas, test_labels, alphas=alphas
    )
    idx = np.argmin(np.abs(curve['empirical_coverages'] - target_coverage))
    return {
        'target_coverage': target_coverage,
        'recommended_alpha': curve['alphas'][idx],
        'expected_empirical_coverage': curve['empirical_coverages'][idx],
        'q_hat': curve['q_hats'][idx],
        'curve': curve,
    }


def block_coverage_test(probas, labels, fold_assignments, target_alpha=0.10):
    """Assess coverage stability across CV folds (proxy for exchangeability).

    If coverage varies greatly across folds, exchangeability may be violated.
    """
    probas = np.asarray(probas)
    labels = np.asarray(labels).astype(int)
    fold_assignments = np.asarray(fold_assignments)
    folds = np.unique(fold_assignments)

    per_fold = []
    for fold in folds:
        cal_mask = fold_assignments != fold
        test_mask = fold_assignments == fold
        cal_scores = split_conformal_scores(probas[cal_mask], labels[cal_mask])
        q = calibrate_threshold(cal_scores, alpha=target_alpha)
        cov, _ = empirical_coverage(probas[test_mask], labels[test_mask], q)
        per_fold.append({
            'fold': int(fold),
            'cal_size': int(cal_mask.sum()),
            'test_size': int(test_mask.sum()),
            'coverage': float(cov),
        })

    coverages = np.array([p['coverage'] for p in per_fold])
    return {
        'per_fold': per_fold,
        'mean_coverage': coverages.mean(),
        'std_coverage': coverages.std(),
        'min_coverage': coverages.min(),
        'max_coverage': coverages.max(),
    }


def ljung_box(residuals, lags=10):
    """Wrapper for Ljung-Box test on residual sequence."""
    if len(residuals) <= lags:
        return {'statistic': np.nan, 'pvalue': np.nan, 'lags': lags}
    stat, pvalue = stats.acorr_ljungbox(residuals, lags=lags, return_df=False)
    # Return last lag result
    return {
        'statistic': float(stat[-1]),
        'pvalue': float(pvalue[-1]),
        'lags': lags,
    }


if __name__ == '__main__':
    np.random.seed(42)
    n = 2000
    labels = np.random.binomial(1, 0.1, size=n)
    # Well-calibrated probabilities
    probas = np.column_stack([
        np.random.beta(2, 5, size=n),
        np.random.beta(5, 2, size=n)
    ])
    probas = probas / probas.sum(axis=1, keepdims=True)

    n_cal = 1000
    cal_p, test_p = probas[:n_cal], probas[n_cal:]
    cal_y, test_y = labels[:n_cal], labels[n_cal:]

    curve = coverage_calibration_curve(cal_p, cal_y, test_p, test_y)
    print("Alpha -> Empirical coverage:")
    for a, cov in zip(curve['alphas'], curve['empirical_coverages']):
        print(f"  alpha={a:.2f}: coverage={cov:.4f}")

    rec = recommend_alpha(0.90, cal_p, cal_y, test_p, test_y)
    print(f"\nTo reach 90% empirical coverage, use alpha={rec['recommended_alpha']:.2f}")

    # Block test
    folds = np.arange(n) % 5
    block = block_coverage_test(probas, labels, folds)
    print(f"\nBlock coverage: mean={block['mean_coverage']:.4f}, std={block['std_coverage']:.4f}")
