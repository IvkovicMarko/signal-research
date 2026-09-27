"""Evaluation metrics and, more importantly, their uncertainty.

A performance number without an error bar is not a result. Every statistic here has a
companion that says how precisely it is measured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def information_coefficient(
    signals: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """Cross-sectional correlation between signal and forward return, per period.

    Returns one number per period: the correlation taken *across assets* within that
    period. This is deliberately not a pooled correlation over the whole panel, which
    would be dominated by time-series variation in the return level rather than by the
    signal's ability to rank assets against each other.

    Spearman (rank) correlation is the default because return cross-sections have fat
    tails, and a single extreme observation can otherwise dominate a Pearson estimate.

    Periods with fewer than two usable assets, or with no dispersion on either side,
    are dropped rather than returned as NaN.
    """
    signals, fwd_returns = signals.align(fwd_returns, join="inner")
    usable = signals.notna() & fwd_returns.notna()

    left = signals.where(usable)
    right = fwd_returns.where(usable)
    if method == "spearman":
        left = left.rank(axis=1)
        right = right.rank(axis=1)
    elif method != "pearson":
        raise ValueError("method must be 'spearman' or 'pearson'")

    left = left.sub(left.mean(axis=1), axis=0)
    right = right.sub(right.mean(axis=1), axis=0)

    covariance = (left * right).sum(axis=1)
    dispersion = np.sqrt((left**2).sum(axis=1) * (right**2).sum(axis=1))

    keep = (usable.sum(axis=1) >= 2) & (dispersion > 0)
    return (covariance[keep] / dispersion[keep]).rename("ic")


def ic_summary(ic: pd.Series) -> dict[str, float]:
    """Summarise an IC time series.

    The t-statistic assumes IC observations are independent across periods. That
    assumption is optimistic when signals use overlapping lookback windows, and the
    direction of the error is unhelpful: it inflates the t-statistic.
    """
    ic = pd.Series(ic).dropna()
    n = int(ic.size)
    if n < 2:
        return {"mean": np.nan, "std": np.nan, "ir": np.nan, "t_stat": np.nan, "n": n}

    mean = float(ic.mean())
    std = float(ic.std(ddof=1))
    ir = mean / std if std > 0 else np.nan
    return {
        "mean": mean,
        "std": std,
        "ir": ir,
        "t_stat": ir * np.sqrt(n) if np.isfinite(ir) else np.nan,
        "n": n,
    }


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Annualised Sharpe ratio. Excess returns are assumed."""
    returns = pd.Series(returns).dropna()
    if returns.size < 2:
        return np.nan
    std = returns.std(ddof=1)
    if std == 0:
        return np.nan
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def sharpe_standard_error(sharpe_per_period: float, n_periods: int) -> float:
    """Standard error of a Sharpe ratio estimate, for i.i.d. normal returns.

        SE(SR) ~= sqrt( (1 + SR^2 / 2) / T )

    Input and output are both in *per-period* units. This is Lo (2002), and it follows
    from the delta method applied to the joint asymptotic distribution of the sample
    mean and variance.

    The practical consequence is the most useful fact in this file: with T = 120
    monthly observations the standard error of an annualised Sharpe is about 0.32, so
    "Sharpe 0.6 over ten years" has not distinguished itself from zero.
    """
    if n_periods < 2:
        return np.nan
    return float(np.sqrt((1.0 + 0.5 * sharpe_per_period**2) / n_periods))


def newey_west_long_run_variance(x: np.ndarray, n_lags: int) -> float:
    """Autocorrelation-consistent estimate of the long-run variance of a series.

        LRV = gamma_0 + 2 * sum_{k=1..L} w_k * gamma_k,    w_k = 1 - k / (L + 1)

    The Bartlett weights w_k are what guarantee the estimate stays non-negative.

    This matters because overlapping signals produce autocorrelated P&L, and the naive
    Sharpe standard error above assumes independence. Comparing the two shows how much
    apparent significance is an artefact of that assumption.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return np.nan

    centred = x - x.mean()
    total = float(centred @ centred) / n
    for lag in range(1, min(n_lags, n - 1) + 1):
        autocovariance = float(centred[lag:] @ centred[:-lag]) / n
        total += 2.0 * (1.0 - lag / (n_lags + 1.0)) * autocovariance
    return total


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline of the cumulative compounded return.

    Returned as a negative number, or zero if the series never declines.
    """
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return 0.0
    wealth = (1.0 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(min(drawdown.min(), 0.0))
