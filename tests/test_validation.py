import numpy as np

from src.validation import walk_forward_splits


def all_splits(**kwargs):
    return list(walk_forward_splits(**kwargs))


def test_train_always_precedes_test():
    for train, test in all_splits(n_periods=200, n_test=20, embargo=5, min_train=40):
        assert train.max() < test.min()


def test_embargo_gap_is_respected():
    embargo = 7
    for train, test in all_splits(n_periods=300, n_test=25, embargo=embargo, min_train=50):
        assert test.min() - train.max() >= embargo + 1


def test_no_index_in_both_train_and_test():
    for train, test in all_splits(n_periods=200, n_test=20, embargo=3, min_train=40):
        assert len(np.intersect1d(train, test)) == 0


def test_test_windows_are_contiguous_and_correctly_sized():
    n_test = 20
    for _, test in all_splits(n_periods=200, n_test=n_test, embargo=2, min_train=40):
        assert len(test) == n_test
        np.testing.assert_array_equal(test, np.arange(test.min(), test.min() + n_test))


def test_test_windows_do_not_overlap():
    splits = all_splits(n_periods=400, n_test=30, embargo=5, min_train=60)
    seen = np.concatenate([test for _, test in splits])
    assert len(seen) == len(np.unique(seen))


def test_expanding_training_window_grows():
    splits = all_splits(n_periods=400, n_test=30, embargo=5, min_train=60, expanding=True)
    sizes = [len(train) for train, _ in splits]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_rolling_training_window_is_fixed_length():
    splits = all_splits(n_periods=400, n_test=30, embargo=5, min_train=60, expanding=False)
    assert {len(train) for train, _ in splits} == {60}


def test_yields_nothing_when_series_is_too_short():
    assert all_splits(n_periods=30, n_test=20, embargo=5, min_train=40) == []


def test_produces_multiple_splits_on_realistic_sizes():
    splits = all_splits(n_periods=700, n_test=60, embargo=12, min_train=120)
    assert len(splits) >= 5
