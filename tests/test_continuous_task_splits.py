"""Episode boundaries: width control and cross-task alignment.

Episode width was never a controlled variable -- it defaulted to the dataset's
validation window, so how many episodes a task yields was an accident of how
RelBench happened to split it. These tests pin the two options that make it one.
"""

import pandas as pd
import pytest
from relbench.base import Table

from experiments.continuous_learning.continuous_task import ContinuousWrapper


class _FakeDataset:
    def __init__(self, val_timestamp, test_timestamp):
        self.val_timestamp = val_timestamp
        self.test_timestamp = test_timestamp


class _FakeTask:
    """The minimal surface ContinuousWrapper touches."""

    def __init__(self, timestamps, val_timestamp, test_timestamp):
        self.dataset = _FakeDataset(val_timestamp, test_timestamp)
        df = pd.DataFrame({"entity": range(len(timestamps)),
                           "y": [float(i % 2) for i in range(len(timestamps))],
                           "timestamp": timestamps})
        self._table = Table(df=df, fkey_col_to_pkey_table={"entity": "e"},
                            pkey_col=None, time_col="timestamp")

    def get_table(self, split, mask_input_cols=False):
        return self._table


def _wrapper(start="2000-01-01", periods=400, freq="D",
             val="2001-01-01", test="2001-04-01"):
    ts = pd.date_range(start, periods=periods, freq=freq)
    return ContinuousWrapper(_FakeTask(ts, pd.Timestamp(val), pd.Timestamp(test)))


def test_default_width_is_the_validation_window():
    # The published behaviour, which must not move.
    w = _wrapper()
    explicit = w.get_splits(val_delta=pd.Timestamp("2001-04-01") - pd.Timestamp("2001-01-01"))
    assert w.get_splits() == explicit


def test_narrower_episodes_yield_more_of_them():
    w = _wrapper()
    wide = len(w.get_splits(val_delta=pd.Timedelta(days=90))) - 2
    narrow = len(w.get_splits(val_delta=pd.Timedelta(days=30))) - 2
    assert narrow > wide > 0


def test_boundaries_are_increasing_and_bracket_the_horizon():
    splits = _wrapper().get_splits(val_delta=pd.Timedelta(days=45))
    assert splits == sorted(splits)
    assert splits[-1] == pd.Timestamp("2001-04-01")  # test_timestamp
    assert splits[-2] == pd.Timestamp("2001-01-01")  # val_timestamp


def test_calendar_alignment_uses_a_fixed_stride_from_the_validation_horizon():
    splits = _wrapper().get_splits(
        val_delta=pd.Timedelta(days=30), align="calendar", min_rows_frac=0.0
    )
    interior = splits[1:-1]
    gaps = {(b - a).days for a, b in zip(interior, interior[1:])}
    assert gaps == {30}, "calendar mode must step by exactly val_delta"


def test_calendar_alignment_puts_two_tasks_on_a_shared_grid():
    # The point of the mode: two tasks on one database, given the same width, must
    # get the same boundaries so their episodes can be compared row for row. Under
    # data alignment each task walks its own observed timestamps and they diverge.
    long_task = _wrapper(start="2000-01-01", periods=400)
    short_task = _wrapper(start="2000-06-01", periods=200)
    delta = pd.Timedelta(days=30)

    a = set(long_task.get_splits(val_delta=delta, align="calendar", min_rows_frac=0.0)[1:-1])
    b = set(short_task.get_splits(val_delta=delta, align="calendar", min_rows_frac=0.0)[1:-1])
    assert b and b <= a, "the shorter task's boundaries must lie on the longer one's grid"


def test_data_alignment_does_not_share_a_grid():
    # The contrast that motivates the option: same width, different timestamp sets,
    # essentially no shared boundaries.
    long_task = _wrapper(start="2000-01-01", periods=400, freq="D")
    short_task = _wrapper(start="2000-01-01", periods=100, freq="3D")
    delta = pd.Timedelta(days=30)
    a = set(long_task.get_splits(val_delta=delta, min_rows_frac=0.0)[1:-1])
    b = set(short_task.get_splits(val_delta=delta, min_rows_frac=0.0)[1:-1])
    assert len(a & b) < min(len(a), len(b))


def test_row_filter_can_be_disabled():
    # The filter is per-task, so it can remove different boundaries for different
    # tasks and silently break an alignment that was otherwise exact.
    w = _wrapper()
    delta = pd.Timedelta(days=30)
    unfiltered = w.get_splits(val_delta=delta, align="calendar", min_rows_frac=0.0)
    filtered = w.get_splits(val_delta=delta, align="calendar", min_rows_frac=0.9)
    assert len(unfiltered) > len(filtered)


def test_unknown_alignment_is_rejected():
    with pytest.raises(ValueError, match="align"):
        _wrapper().get_splits(align="nonsense")
