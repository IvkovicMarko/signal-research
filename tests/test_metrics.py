import numpy as np
import pandas as pd

from src.metrics import (
    ic_summary,
    information_coefficient,
    max_drawdown,
    newey_west_long_run_variance,
    sharpe_ratio,
    sharpe_standard_error,
)


def test_perfect_signal_has_ic_one():
    signals = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]] * 5)
    fwd = pd.DataFrame([[10.0, 20.0, 30.0, 40.0]] * 5)
    ic = information_coefficient(signals, fwd, method="spearman")
    np.testing.assert_allclose(ic.values, 1.0)


def test_reversed_signal_has_ic_minus_one():
    signals = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]] * 5)
    fwd = pd.DataFrame([[40.0, 30.0, 20.0, 10.0]] * 5)
    ic = information_coefficient(signals, fwd, method="spearman")
    np.testing.assert_allclose(ic.values, -1.0)


def test_ic_is_monotone_invariant_under_spearman():
    rng = np.random.default_rng(3)
    signals = pd.DataFrame(rng.normal(size=(40, 15)))
    fwd = pd.DataFrame(rng.normal(size=(40, 15)))
    base = information_coefficient(signals, fwd, method="spearman")
    squashed = information_coefficient(np.tanh(signals * 3), fwd, method="spearman")
    np.testing.assert_allclose(base.values, squashed.values, atol=1e-12)


def test_ic_summary_fields():
    ic = pd.Series([0.02, 0.04, -0.01, 0.03, 0.05])
    out = ic_summary(ic)
    assert out["n"] == 5
    np.testing.assert_allclose(out["mean"], ic.mean())
    np.testing.assert_allclose(out["ir"], ic.mean() / ic.std(ddof=1), rtol=1e-9)
    np.testing.assert_allclose(out["t_stat"], out["ir"] * np.sqrt(5), rtol=1e-9)


def test_sharpe_annualisation():
    rng = np.random.default_rng(11)
    monthly = pd.Series(rng.normal(0.01, 0.04, size=100_000))
    annual = sharpe_ratio(monthly, periods_per_year=12)
    expected = (0.01 / 0.04) * np.sqrt(12)
    np.testing.assert_allclose(annual, expected, rtol=0.05)


def test_sharpe_standard_error_matches_simulation():
    """The analytic SE must match the empirical spread of Sharpe estimates."""
    rng = np.random.default_rng(5)
    true_sr, n = 0.15, 250
    estimates = [
        sharpe_ratio(pd.Series(rng.normal(true_sr, 1.0, size=n)), periods_per_year=1)
        for _ in range(4000)
    ]
    empirical = np.std(estimates, ddof=1)
    analytic = sharpe_standard_error(true_sr, n)
    assert abs(empirical - analytic) / analytic < 0.10


def test_sharpe_standard_error_shrinks_with_sample_size():
    assert sharpe_standard_error(0.2, 1000) < sharpe_standard_error(0.2, 100)


def test_newey_west_equals_variance_for_white_noise():
    rng = np.random.default_rng(9)
    x = rng.normal(size=200_000)
    lrv = newey_west_long_run_variance(x, n_lags=6)
    np.testing.assert_allclose(lrv, 1.0, rtol=0.05)


def test_newey_west_exceeds_variance_for_positively_autocorrelated_series():
    """An AR(1) with phi=0.6 has long-run variance about (1+phi)/(1-phi) = 4x the innovation-scaled variance."""
    rng = np.random.default_rng(13)
    phi, n = 0.6, 200_000
    x = np.zeros(n)
    noise = rng.normal(size=n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]
    lrv = newey_west_long_run_variance(x, n_lags=30)
    assert lrv > 1.5 * np.var(x, ddof=1)


def test_max_drawdown():
    returns = pd.Series([0.10, -0.20, -0.10, 0.05])
    # Cumulative: 1.10, 0.88, 0.792, 0.8316. Peak 1.10, trough 0.792.
    np.testing.assert_allclose(max_drawdown(returns), 0.792 / 1.10 - 1.0, rtol=1e-9)


def test_max_drawdown_is_zero_for_monotone_gains():
    np.testing.assert_allclose(max_drawdown(pd.Series([0.01, 0.02, 0.03])), 0.0, atol=1e-12)
