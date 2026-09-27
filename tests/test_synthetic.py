"""The generator must actually produce the IC it claims, or nothing downstream means anything."""

import numpy as np
import pandas as pd
import pytest

from src.metrics import information_coefficient
from src.synthetic import (
    forward_returns,
    make_null_candidates,
    make_panel,
    standardise_cross_section,
)


def test_standardise_gives_zero_mean_unit_variance():
    frame = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 20)))
    out = standardise_cross_section(frame)
    np.testing.assert_allclose(out.mean(axis=1).values, 0.0, atol=1e-12)
    np.testing.assert_allclose(out.std(axis=1, ddof=0).values, 1.0, atol=1e-12)


def test_standardise_handles_constant_row():
    frame = pd.DataFrame([[3.0, 3.0, 3.0], [1.0, 2.0, 3.0]])
    out = standardise_cross_section(frame)
    assert np.isfinite(out.values).all(), "a constant row must not produce NaN or inf"
    np.testing.assert_allclose(out.iloc[0].values, 0.0)


def test_forward_returns_alignment():
    returns = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    fwd = forward_returns(returns)
    assert fwd["a"].iloc[0] == 2.0
    assert fwd["a"].iloc[1] == 3.0
    assert np.isnan(fwd["a"].iloc[2])


@pytest.mark.parametrize("true_ic", [0.0, 0.03, 0.08, 0.15])
def test_measured_ic_matches_ground_truth(true_ic):
    panel = make_panel(n_assets=60, n_periods=4000, ic=true_ic, seed=7)
    fwd = forward_returns(panel.returns)
    ic = information_coefficient(panel.signals, fwd, method="pearson")

    # Standard error of a mean of T cross-sectional correlations, each with
    # dispersion roughly 1/sqrt(n_assets).
    tolerance = 5.0 / np.sqrt(60 * 4000)
    assert abs(ic.mean() - true_ic) < max(tolerance, 0.01)


def test_panel_shapes_and_reproducibility():
    a = make_panel(n_assets=10, n_periods=100, ic=0.05, seed=42)
    b = make_panel(n_assets=10, n_periods=100, ic=0.05, seed=42)
    c = make_panel(n_assets=10, n_periods=100, ic=0.05, seed=43)

    assert a.signals.shape == (100, 10)
    assert a.returns.shape == (100, 10)
    pd.testing.assert_frame_equal(a.signals, b.signals)
    assert not a.signals.equals(c.signals), "different seeds must give different data"


def test_null_candidates_share_one_return_series():
    candidates, returns = make_null_candidates(
        n_assets=20, n_periods=200, n_candidates=15, seed=1
    )
    assert len(candidates) == 15
    assert returns.shape == (200, 20)
    for candidate in candidates:
        assert candidate.shape == returns.shape
    assert not candidates[0].equals(candidates[1])
