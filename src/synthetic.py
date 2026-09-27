"""Synthetic panel generator with a controllable ground-truth information coefficient.

Everything else in this project is validated against data produced here, because this
is the only place where the true answer is known.

The generating model, for asset i:

    r[t+1, i] = vol * ( ic * x[t, i] + sqrt(1 - ic^2) * e[t, i] )

where x[t, :] and e[t, :] are each standardised across assets within the period and e
is independent of x. Under this construction the cross-sectional correlation between
the signal x[t, :] and the forward return r[t+1, :] has expectation `ic`.

Two conventions matter and are easy to get wrong:

  * `signals.loc[t]` is information available at the close of period t.
  * `returns.loc[t]` is the return realised *over* period t.

So the signal at t is evaluated against the return at t+1. The `forward_returns`
helper enforces this; never shift by hand at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Panel:
    """A synthetic dataset together with the ground truth used to generate it."""

    signals: pd.DataFrame          # index = period, columns = asset
    returns: pd.DataFrame          # index = period, columns = asset
    true_ic: float


def _asset_names(n_assets: int) -> list[str]:
    return [f"a{i:03d}" for i in range(n_assets)]


def forward_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Align returns so that row t holds the return realised over period t+1.

    The final row becomes NaN. Keeping the NaN rather than silently truncating makes
    an accidental off-by-one visible instead of quietly biasing every result.
    """
    return returns.shift(-1)


def standardise_cross_section(frame: pd.DataFrame) -> pd.DataFrame:
    """Demean and scale each row to unit standard deviation.

    Rows with zero cross-sectional dispersion come back as all zeros rather than NaN
    or infinity.
    """
    dispersion = frame.std(axis=1, ddof=0)
    safe = dispersion.where(dispersion > 0, 1.0)
    return frame.sub(frame.mean(axis=1), axis=0).div(safe, axis=0)


def _standardised_normal(
    rng: np.random.Generator, n_periods: int, n_assets: int
) -> pd.DataFrame:
    raw = pd.DataFrame(
        rng.standard_normal((n_periods, n_assets)),
        index=pd.RangeIndex(n_periods),
        columns=_asset_names(n_assets),
    )
    return standardise_cross_section(raw)


def make_panel(
    n_assets: int,
    n_periods: int,
    ic: float,
    seed: int,
    vol: float = 0.06,
) -> Panel:
    """Generate one panel whose true cross-sectional IC equals `ic`.

    Args:
        n_assets: number of assets in the cross-section.
        n_periods: number of time periods.
        ic: the true information coefficient, in [0, 1). Zero produces pure noise.
        seed: seed for the random generator, so every experiment is reproducible.
        vol: per-period standard deviation of returns (0.06 is roughly monthly equity).
    """
    if not 0.0 <= ic < 1.0:
        raise ValueError("ic must lie in [0, 1)")

    rng = np.random.default_rng(seed)
    signals = _standardised_normal(rng, n_periods, n_assets)
    noise = _standardised_normal(rng, n_periods, n_assets)

    driven = vol * (ic * signals + np.sqrt(1.0 - ic**2) * noise)

    # Row 0 is unpredicted by construction: nothing precedes it. Rows 1.. are driven
    # by the signal one period earlier, which is exactly the relationship
    # `forward_returns` recovers.
    returns = driven.shift(1)
    returns.iloc[0] = vol * _standardised_normal(rng, 1, n_assets).iloc[0].to_numpy()

    return Panel(signals=signals, returns=returns, true_ic=ic)


def make_null_candidates(
    n_assets: int,
    n_periods: int,
    n_candidates: int,
    seed: int,
    vol: float = 0.06,
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Generate many candidate signals that are all worthless, against shared returns.

    Every candidate has a true IC of exactly zero, so anything the pipeline "finds"
    here is by construction a false discovery.

    All candidates are scored against the *same* realised return series. That is what
    a real researcher does when sweeping parameters, and it is what makes the trial
    statistics correlated rather than independent.
    """
    rng = np.random.default_rng(seed)
    returns = vol * _standardised_normal(rng, n_periods, n_assets)
    candidates = [
        _standardised_normal(rng, n_periods, n_assets) for _ in range(n_candidates)
    ]
    return candidates, returns
