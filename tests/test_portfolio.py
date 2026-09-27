import numpy as np
import pandas as pd

from src.portfolio import (
    cross_sectional_rank,
    run_backtest,
    to_dollar_neutral_weights,
    turnover,
)


def test_rank_range_and_ordering():
    signal = pd.DataFrame([[10.0, -5.0, 3.0, 7.0]])
    ranked = cross_sectional_rank(signal)
    assert ranked.values.min() >= -0.5 - 1e-12
    assert ranked.values.max() <= 0.5 + 1e-12
    # Ordering must be preserved: the largest input gets the largest rank.
    assert ranked.iloc[0].idxmax() == signal.iloc[0].idxmax()
    assert ranked.iloc[0].idxmin() == signal.iloc[0].idxmin()


def test_rank_ignores_nan():
    signal = pd.DataFrame([[1.0, np.nan, 3.0, 2.0]])
    ranked = cross_sectional_rank(signal)
    assert np.isnan(ranked.iloc[0, 1])
    assert ranked.iloc[0].notna().sum() == 3


def test_weights_are_dollar_neutral_and_unit_gross():
    rng = np.random.default_rng(2)
    signal = pd.DataFrame(rng.normal(size=(30, 12)))
    weights = to_dollar_neutral_weights(signal, gross_exposure=1.0)
    np.testing.assert_allclose(weights.sum(axis=1).values, 0.0, atol=1e-12)
    np.testing.assert_allclose(weights.abs().sum(axis=1).values, 1.0, atol=1e-12)


def test_weights_respect_gross_exposure():
    rng = np.random.default_rng(4)
    signal = pd.DataFrame(rng.normal(size=(10, 8)))
    weights = to_dollar_neutral_weights(signal, gross_exposure=2.0)
    np.testing.assert_allclose(weights.abs().sum(axis=1).values, 2.0, atol=1e-12)


def test_constant_signal_gives_zero_weights():
    signal = pd.DataFrame([[5.0, 5.0, 5.0]])
    weights = to_dollar_neutral_weights(signal)
    assert np.isfinite(weights.values).all()
    np.testing.assert_allclose(weights.values, 0.0, atol=1e-12)


def test_turnover_of_static_book_is_zero_after_first_period():
    weights = pd.DataFrame([[0.5, -0.5]] * 4)
    t = turnover(weights)
    np.testing.assert_allclose(t.iloc[0], 1.0)
    np.testing.assert_allclose(t.iloc[1:].values, 0.0, atol=1e-12)


def test_turnover_of_full_flip():
    weights = pd.DataFrame([[0.5, -0.5], [-0.5, 0.5]])
    np.testing.assert_allclose(turnover(weights).iloc[1], 2.0)


def test_backtest_uses_lagged_weights():
    """Weights from period 0 must earn the return of period 1, never period 0."""
    weights = pd.DataFrame({"a": [1.0, 0.0], "b": [0.0, 0.0]})
    returns = pd.DataFrame({"a": [100.0, 0.05], "b": [0.0, 0.0]})
    pnl = run_backtest(weights, returns, cost_per_unit_turnover=0.0)
    # If period-0 weights were applied to period-0 returns we would see 100.0.
    assert 100.0 not in pnl.values
    np.testing.assert_allclose(pnl.iloc[-1], 0.05)


def test_costs_reduce_pnl_by_turnover_times_rate():
    weights = pd.DataFrame([[0.5, -0.5], [-0.5, 0.5], [-0.5, 0.5]])
    returns = pd.DataFrame(np.zeros((3, 2)))
    rate = 0.001
    gross = run_backtest(weights, returns, cost_per_unit_turnover=0.0)
    net = run_backtest(weights, returns, cost_per_unit_turnover=rate)
    np.testing.assert_allclose((gross - net).values, (turnover(weights) * rate).values, atol=1e-12)


def test_backtest_output_length_matches_returns():
    rng = np.random.default_rng(8)
    weights = pd.DataFrame(rng.normal(size=(20, 5)))
    returns = pd.DataFrame(rng.normal(size=(20, 5)))
    assert len(run_backtest(weights, returns)) == 20
