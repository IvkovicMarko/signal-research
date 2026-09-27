"""Corrections for having looked at the data more than once.

This module is the intellectual core of the project. Everything else measures
performance; this decides whether the measurement means anything.

The situation being corrected for: you test N candidate signals, all worthless. Each
produces a Sharpe ratio that is noise centred on zero with standard deviation roughly
1/sqrt(T). You report the best one. Its Sharpe is not zero -- it is close to
sigma * sqrt(2 ln N), because that is how the maximum of N draws from a normal
distribution behaves. With N = 40 and T = 480 months that is around 0.5 annualised
from luck alone, carrying a t-statistic near 2, which is "significant" by any
conventional single-test standard.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.57721566490153286


def expected_max_sharpe_under_null(n_trials: int, sharpe_std: float) -> float:
    """Expected maximum Sharpe ratio across `n_trials` worthless candidates.

    Uses the extreme-value approximation from Bailey and Lopez de Prado:

        E[max SR] ~= sharpe_std * [ (1 - g) * Z_inv(1 - 1/N)
                                    + g * Z_inv(1 - 1/(N * e)) ]

    where g is the Euler-Mascheroni constant and Z_inv is the inverse standard normal
    CDF.

    `sharpe_std` is the standard deviation of the Sharpe ratios *across trials*. When
    trials are independent and returns i.i.d. it is approximately 1/sqrt(T), but in
    practice estimate it from the observed spread of your own trial Sharpes:
    correlated trials produce a narrower spread and therefore a lower honest
    threshold.

    The leading behaviour is sharpe_std * sqrt(2 ln N), so the bar grows only with the
    square root of the log of the trial count. Testing a thousand signals instead of
    ten roughly doubles it. That slowness is why the effect is easy to underestimate.
    """
    if n_trials <= 1:
        return 0.0
    upper = stats.norm.ppf(1.0 - 1.0 / n_trials)
    lower = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sharpe_std * ((1.0 - EULER_MASCHERONI) * upper + EULER_MASCHERONI * lower))


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_periods: int,
    skew: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float:
    """Probability that the true Sharpe exceeds `benchmark_sharpe`.

        PSR = Z[ (SR - SR_b) * sqrt(T - 1)
                 / sqrt( 1 - g3 * SR + (g4 / 4) * SR^2 ) ]

    Z is the standard normal CDF, g3 is skewness and g4 is *excess* kurtosis. All
    Sharpe ratios are per-period.

    The denominator is doing real work. Negative skew and fat tails both inflate the
    naive Sharpe relative to its true value, which is precisely the profile of any
    strategy that sells insurance in some form.
    """
    if n_periods < 2:
        return np.nan

    variance = (
        1.0 - skew * observed_sharpe + 0.25 * excess_kurtosis * observed_sharpe**2
    )
    if variance <= 0:
        return np.nan

    z = (observed_sharpe - benchmark_sharpe) * np.sqrt(n_periods - 1) / np.sqrt(variance)
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_periods: int,
    sharpe_std: float,
    skew: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float:
    """Probabilistic Sharpe ratio evaluated against the best-of-N null benchmark.

    The composition is the point: it asks "is this better than the best I would expect
    from noise after this many attempts?" rather than "is this better than zero?".

    Above 0.95 is the usual bar. Expect most candidates to fail it.
    """
    benchmark = expected_max_sharpe_under_null(n_trials, sharpe_std)
    return probabilistic_sharpe_ratio(
        observed_sharpe, benchmark, n_periods, skew, excess_kurtosis
    )


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg step-up procedure. Returns a boolean array of rejections.

    Sort ascending, find the largest k with p_(k) <= alpha * k / m, and reject
    everything up to *and including* that one. Rejecting only the p-values below the
    line is a common and wrong variant.

    BH controls the false discovery rate, the expected fraction of rejections that are
    wrong. Bonferroni controls the family-wise error rate, the probability of any
    wrong rejection. In signal research, tolerating a known fraction of false
    positives is usually the right trade, so BH is the better fit and is far less
    conservative when many candidates are genuinely good.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = p_values.size
    rejected = np.zeros(m, dtype=bool)
    if m == 0:
        return rejected

    order = np.argsort(p_values, kind="mergesort")
    thresholds = alpha * np.arange(1, m + 1) / m
    passing = np.nonzero(p_values[order] <= thresholds)[0]
    if passing.size:
        rejected[order[: passing[-1] + 1]] = True
    return rejected


def bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Bonferroni correction. Reject where p <= alpha / m. Included for comparison."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return np.zeros(0, dtype=bool)
    return p_values <= alpha / p_values.size


def permutation_p_value(
    statistic: float,
    null_statistics: np.ndarray,
    alternative: str = "greater",
) -> float:
    """p-value of an observed statistic against an empirical null distribution.

        p = (1 + #{null at least as extreme}) / (1 + n_null)

    The plus-one in both places is not cosmetic. Without it, a statistic more extreme
    than every permutation gives p = 0, claiming more certainty than a finite number
    of permutations can support. With it the p-value is guaranteed valid under the
    null, and the smallest reportable value is 1 / (1 + n_null).
    """
    null_statistics = np.asarray(null_statistics, dtype=float)
    if alternative == "greater":
        at_least_as_extreme = np.sum(null_statistics >= statistic)
    elif alternative == "less":
        at_least_as_extreme = np.sum(null_statistics <= statistic)
    elif alternative == "two-sided":
        at_least_as_extreme = np.sum(np.abs(null_statistics) >= abs(statistic))
    else:
        raise ValueError("alternative must be 'greater', 'less' or 'two-sided'")

    return float((1 + at_least_as_extreme) / (1 + null_statistics.size))


def shuffle_cross_sections(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independently permute the asset ordering within each time period.

    This is the null-generating step for the permutation test. Shuffling *within* each
    cross-section destroys the signal's relationship to contemporaneous returns while
    preserving the marginal distribution of the signal, the number of assets, and the
    time-series structure of returns. Shuffling the whole panel would destroy all of
    that and produce a null that is too easy to beat.
    """
    return rng.permuted(np.asarray(signal, dtype=float), axis=1)
