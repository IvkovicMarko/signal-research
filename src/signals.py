"""The candidate signal family for the real-data experiment.

Each constructor takes a return panel and returns a signal panel on the same index,
where row t uses only information available at the end of period t. Any lookahead
here invalidates everything downstream, so only `.rolling(...)` and `.shift(k)` for
positive k appear below.

The family is deliberately a parameter sweep rather than a handful of hand-picked
ideas. That is the honest version of what researchers actually do, and it is what
makes the multiple-testing correction in experiment 3 necessary rather than
decorative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def momentum(returns: pd.DataFrame, lookback: int, skip: int = 1) -> pd.DataFrame:
    """Cumulative log return over the trailing `lookback` periods, skipping the most
    recent `skip`.

    The skip is conventional and has a reason: the most recent month tends to reverse
    rather than continue, so including it dilutes a momentum signal.

    Log returns are summed rather than compounding simple returns because the sum is
    what the rolling window computes cleanly, and the monotone transformation does
    not change a rank-based signal at all.
    """
    log_returns = np.log1p(returns)
    trailing = log_returns.rolling(lookback, min_periods=lookback).sum()
    return trailing.shift(skip)


def reversal(returns: pd.DataFrame, lookback: int = 1) -> pd.DataFrame:
    """Negative of the trailing return: bet on short-horizon mean reversion."""
    return -momentum(returns, lookback=lookback, skip=0)


def volatility(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Negative trailing realised volatility, so that a high signal means low risk.

    Signed this way so a positive IC has the same interpretation as for the other
    signals. Keeping the sign convention consistent across the family matters once
    you are comparing dozens of them.
    """
    return -returns.rolling(lookback, min_periods=lookback).std(ddof=1)


MOMENTUM_LOOKBACKS = (2, 3, 6, 9, 12, 18, 24, 36)
MOMENTUM_SKIPS = (0, 1, 2)
REVERSAL_LOOKBACKS = (1, 2, 3, 6)
VOLATILITY_LOOKBACKS = (3, 6, 12, 24, 36, 60)


def build_candidate_family(returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Construct the full sweep of candidate signals.

    The number of entries is the trial count N used by `deflated_sharpe_ratio`, so do
    not quietly add candidates later without updating the correction.
    """
    family: dict[str, pd.DataFrame] = {}

    for lookback in MOMENTUM_LOOKBACKS:
        for skip in MOMENTUM_SKIPS:
            family[f"mom_{lookback}_{skip}"] = momentum(returns, lookback, skip)

    for lookback in REVERSAL_LOOKBACKS:
        family[f"rev_{lookback}"] = reversal(returns, lookback)

    for lookback in VOLATILITY_LOOKBACKS:
        family[f"lowvol_{lookback}"] = volatility(returns, lookback)

    return family
