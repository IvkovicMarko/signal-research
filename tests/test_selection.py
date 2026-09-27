import numpy as np
from scipy import stats

from src.selection import (
    benjamini_hochberg,
    bonferroni,
    deflated_sharpe_ratio,
    expected_max_sharpe_under_null,
    permutation_p_value,
    probabilistic_sharpe_ratio,
    shuffle_cross_sections,
)


# --- expected maximum under the null -------------------------------------------------

def test_single_trial_needs_no_deflation():
    assert expected_max_sharpe_under_null(1, sharpe_std=0.1) == 0.0


def test_threshold_increases_with_trial_count():
    values = [expected_max_sharpe_under_null(n, 0.1) for n in (2, 5, 20, 100, 1000)]
    assert values == sorted(values)


def test_threshold_scales_linearly_with_dispersion():
    a = expected_max_sharpe_under_null(50, 0.1)
    b = expected_max_sharpe_under_null(50, 0.2)
    np.testing.assert_allclose(b, 2.0 * a, rtol=1e-9)


def test_threshold_matches_simulated_maximum():
    """The whole premise of the project: the best of N worthless trials is not zero."""
    rng = np.random.default_rng(17)
    n_trials, sharpe_std = 50, 0.1
    simulated = np.mean(
        [rng.normal(0.0, sharpe_std, size=n_trials).max() for _ in range(20_000)]
    )
    analytic = expected_max_sharpe_under_null(n_trials, sharpe_std)
    assert abs(analytic - simulated) / simulated < 0.05


def test_threshold_tracks_but_sits_below_sqrt_two_log_n():
    """sqrt(2 ln N) is the leading term and an upper bound; the correction is a
    slowly vanishing O(ln ln N / sqrt(ln N)), so at realistic N the true expectation
    is roughly 10-15% lower."""
    n, s = 500, 0.1
    crude = s * np.sqrt(2 * np.log(n))
    value = expected_max_sharpe_under_null(n, s)
    assert 0.80 * crude < value < crude


# --- probabilistic and deflated Sharpe ------------------------------------------------

def test_psr_is_one_half_at_the_benchmark():
    np.testing.assert_allclose(
        probabilistic_sharpe_ratio(0.12, 0.12, n_periods=200), 0.5, atol=1e-12
    )


def test_psr_increases_with_observed_sharpe():
    low = probabilistic_sharpe_ratio(0.05, 0.0, n_periods=200)
    high = probabilistic_sharpe_ratio(0.20, 0.0, n_periods=200)
    assert 0.0 < low < high < 1.0


def test_psr_increases_with_sample_length():
    short = probabilistic_sharpe_ratio(0.10, 0.0, n_periods=50)
    long = probabilistic_sharpe_ratio(0.10, 0.0, n_periods=500)
    assert long > short


def test_negative_skew_and_fat_tails_reduce_psr():
    plain = probabilistic_sharpe_ratio(0.15, 0.0, 300, skew=0.0, excess_kurtosis=0.0)
    ugly = probabilistic_sharpe_ratio(0.15, 0.0, 300, skew=-1.5, excess_kurtosis=6.0)
    assert ugly < plain


def test_deflation_is_harsher_than_a_zero_benchmark():
    undeflated = probabilistic_sharpe_ratio(0.15, 0.0, n_periods=300)
    deflated = deflated_sharpe_ratio(0.15, n_trials=40, n_periods=300, sharpe_std=0.06)
    assert deflated < undeflated


def test_deflation_with_one_trial_equals_plain_psr():
    np.testing.assert_allclose(
        deflated_sharpe_ratio(0.15, n_trials=1, n_periods=300, sharpe_std=0.06),
        probabilistic_sharpe_ratio(0.15, 0.0, n_periods=300),
        rtol=1e-9,
    )


# --- multiple testing -----------------------------------------------------------------

def test_bonferroni_threshold():
    p = np.array([0.001, 0.02, 0.04, 0.9])
    np.testing.assert_array_equal(
        bonferroni(p, alpha=0.05), np.array([True, False, False, False])
    )


def test_bh_is_a_step_up_procedure():
    """The defining behaviour: 0.04 exceeds its own threshold but is still rejected,
    because a larger p-value further down the list clears the line."""
    p = np.array([0.001, 0.04, 0.045])
    np.testing.assert_array_equal(
        benjamini_hochberg(p, alpha=0.05), np.array([True, True, True])
    )


def test_bh_rejects_nothing_under_a_pure_null():
    p = np.array([0.3, 0.5, 0.7, 0.9, 0.95])
    assert not benjamini_hochberg(p, alpha=0.05).any()


def test_bh_is_at_least_as_powerful_as_bonferroni():
    rng = np.random.default_rng(21)
    p = np.concatenate([rng.uniform(0, 0.01, 10), rng.uniform(0, 1, 90)])
    assert benjamini_hochberg(p, 0.05).sum() >= bonferroni(p, 0.05).sum()


def test_bh_controls_false_discovery_rate():
    rng = np.random.default_rng(23)
    n_true_null, n_false_null, alpha = 90, 10, 0.10
    false_discovery_rates = []
    for _ in range(600):
        null_p = rng.uniform(0, 1, n_true_null)
        signal_p = 1.0 - stats.norm.cdf(rng.normal(4.0, 1.0, n_false_null))
        p = np.concatenate([null_p, signal_p])
        rejected = benjamini_hochberg(p, alpha)
        n_rejected = rejected.sum()
        false_discovery_rates.append(
            0.0 if n_rejected == 0 else rejected[:n_true_null].sum() / n_rejected
        )
    assert np.mean(false_discovery_rates) <= alpha + 0.02


# --- permutation testing --------------------------------------------------------------

def test_permutation_p_value_formula():
    null = np.array([1.0, 2.0, 3.0, 4.0])
    # Two of four null values are >= 2.5, so p = (1 + 2) / (1 + 4).
    np.testing.assert_allclose(permutation_p_value(2.5, null), 3.0 / 5.0)


def test_permutation_p_value_is_never_zero():
    null = np.zeros(999)
    assert permutation_p_value(10.0, null) == 1.0 / 1000.0


def test_permutation_p_values_are_uniform_under_the_null():
    """If the pipeline is calibrated, p-values on noise are uniform. If they are not,
    every significance claim downstream is wrong."""
    rng = np.random.default_rng(29)
    p_values = []
    for _ in range(3000):
        draws = rng.normal(size=200)
        p_values.append(permutation_p_value(draws[0], draws[1:]))
    p_values = np.array(p_values)

    assert abs(p_values.mean() - 0.5) < 0.02
    assert abs((p_values <= 0.05).mean() - 0.05) < 0.015
    assert stats.kstest(p_values, "uniform").pvalue > 0.01


def test_shuffle_permutes_within_rows_only():
    signal = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    out = shuffle_cross_sections(signal, np.random.default_rng(31))
    assert out.shape == signal.shape
    for row_in, row_out in zip(signal, out):
        np.testing.assert_array_equal(np.sort(row_in), np.sort(row_out))


def test_shuffle_actually_changes_the_ordering():
    rng = np.random.default_rng(33)
    signal = rng.normal(size=(100, 25))
    out = shuffle_cross_sections(signal, rng)
    assert not np.array_equal(signal, out)
