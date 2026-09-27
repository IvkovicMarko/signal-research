"""Chronological cross-validation with purging and embargo.

Standard k-fold cross-validation is invalid on financial panels for two reasons:

  1. It trains on the future to predict the past, which no live system can do.
  2. Observations adjacent in time are nearly identical, so a random split places
     near-duplicates in both train and test. The model appears to generalise when it
     has in effect memorised.

The fix is to keep splits strictly chronological and to leave a gap (the embargo)
between the end of the training window and the start of the test window, wide enough
to cover any lookback or holding overlap in the signal.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


def walk_forward_splits(
    n_periods: int,
    n_test: int,
    embargo: int = 0,
    min_train: int = 1,
    expanding: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_index, test_index) pairs, ordered in time.

    Args:
        n_periods: total length of the series.
        n_test: length of each test window.
        embargo: number of periods discarded between the train and test windows.
        min_train: minimum training length before the first split is emitted.
        expanding: if True the training window grows from the start; if False it is a
            rolling window of fixed length `min_train`.

    Yields nothing at all if `n_periods` is too small to honour the constraints. That
    is intentional: silently returning a degenerate split is worse than returning none.
    """
    if n_test < 1 or min_train < 1 or embargo < 0:
        raise ValueError("n_test and min_train must be positive and embargo non-negative")

    test_start = min_train + embargo
    while test_start + n_test <= n_periods:
        train_end = test_start - embargo          # exclusive
        train_start = 0 if expanding else max(0, train_end - min_train)
        train = np.arange(train_start, train_end)
        if train.size >= min_train:
            yield train, np.arange(test_start, test_start + n_test)
        test_start += n_test
