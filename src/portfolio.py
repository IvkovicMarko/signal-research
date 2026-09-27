"""Turning a signal into a portfolio, and a portfolio into a net P&L series.

The gap between "my signal has positive IC" and "this makes money" lives here. A
signal with genuine predictive power can still lose after costs if it turns over fast
enough.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_rank(signal: pd.DataFrame) -> pd.DataFrame:
    """Convert each row to ranks rescaled to the interval [-0.5, 0.5].

    Ranking discards the magnitude of the signal and keeps only the ordering, which
    makes the resulting portfolio robust to outliers and to any monotone
    transformation of the raw signal. The cost is that genuine information in the
    magnitudes is thrown away.
    """
    ranks = signal.rank(axis=1)
    count = signal.notna().sum(axis=1)
    spread = (count - 1).where(count > 1, 1)
    scaled = ranks.sub(1.0, axis=0).div(spread, axis=0) - 0.5
    return scaled.where(count.gt(1), 0.0, axis=0).where(signal.notna())


def to_dollar_neutral_weights(
    signal: pd.DataFrame, gross_exposure: float = 1.0
) -> pd.DataFrame:
    """Convert a signal into portfolio weights that are dollar neutral.

    Two constraints per row:
        sum(w)   == 0                 (the long book funds the short book)
        sum(|w|) == gross_exposure

    Dollar neutrality strips out the market-wide move, so what remains measures the
    signal's ability to rank assets rather than its accidental market exposure. Note
    that dollar neutral is not the same as beta neutral.
    """
    centred = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
    gross = centred.abs().sum(axis=1)
    scale = gross_exposure / gross.where(gross > 0, 1.0)
    return centred.mul(scale, axis=0)


def turnover(weights: pd.DataFrame) -> pd.Series:
    """Per-period turnover, sum(|w[t] - w[t-1]|), with the first period as sum(|w[0]|).

    Between rebalances the weights drift with returns, so true turnover is the
    difference between the target weights and the *drifted* previous weights, not the
    previous target weights. The simplification is acceptable at monthly frequency
    with modest returns, but it does understate cost slightly.
    """
    weights = weights.fillna(0.0)
    previous = weights.shift(1).fillna(0.0)
    return (weights - previous).abs().sum(axis=1).rename("turnover")


def run_backtest(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_per_unit_turnover: float = 0.0,
) -> pd.Series:
    """Net P&L series for a set of weights.

        gross[t] = sum_i w[t-1, i] * r[t, i]
        net[t]   = gross[t] - cost_per_unit_turnover * turnover[t]

    The lag is the whole point: weights formed using information available at the end
    of period t-1 earn the return realised over period t. Getting this wrong produces
    a spectacular backtest and is the most common error in the genre.

    `cost_per_unit_turnover` is a fraction, not basis points: 0.001 means 10bp.
    """
    weights, returns = weights.align(returns, join="inner")
    weights = weights.fillna(0.0)
    returns = returns.fillna(0.0)

    gross = (weights.shift(1).fillna(0.0) * returns).sum(axis=1)
    if cost_per_unit_turnover:
        gross = gross - cost_per_unit_turnover * turnover(weights)
    return gross.rename("pnl")
