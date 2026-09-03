import pandas as pd
import pytest
from relbench.base import Table

from experiments.continuous_learning.utils import subsample_val_table


def make_table(n: int) -> Table:
    return Table(
        df=pd.DataFrame(
            {
                "entity": range(n),
                "y": [float(i % 7) for i in range(n)],
                "timestamp": pd.date_range("2020-01-01", periods=n, freq="h"),
            }
        ),
        fkey_col_to_pkey_table={"entity": "customer"},
        pkey_col=None,
        time_col="timestamp",
    )


def test_no_cap_returns_the_table_untouched():
    table = make_table(100)
    out, seed = subsample_val_table(table, None, "rel-hm", "user-churn", 3)
    assert out is table and seed is None


def test_table_smaller_than_cap_is_untouched():
    table = make_table(50)
    out, seed = subsample_val_table(table, 25_000, "rel-hm", "user-churn", 3)
    assert out is table and seed is None


def test_cap_is_respected():
    out, seed = subsample_val_table(make_table(5_000), 500, "rel-hm", "user-churn", 3)
    assert len(out.df) == 500 and seed is not None


def test_subsample_is_deterministic_for_the_same_episode():
    a, sa = subsample_val_table(make_table(5_000), 500, "rel-hm", "user-churn", 3)
    b, sb = subsample_val_table(make_table(5_000), 500, "rel-hm", "user-churn", 3)
    assert sa == sb
    pd.testing.assert_frame_equal(a.df, b.df)


def test_different_episodes_get_different_subsamples():
    a, sa = subsample_val_table(make_table(5_000), 500, "rel-hm", "user-churn", 3)
    b, sb = subsample_val_table(make_table(5_000), 500, "rel-hm", "user-churn", 4)
    assert sa != sb
    assert not a.df["entity"].equals(b.df["entity"])


def test_different_tasks_get_different_subsamples():
    _, sa = subsample_val_table(make_table(5_000), 500, "rel-hm", "user-churn", 3)
    _, sb = subsample_val_table(make_table(5_000), 500, "rel-hm", "item-sales", 3)
    assert sa != sb


def test_seed_is_independent_of_any_trial_seed():
    # Every learning mode and every trial seed must select against the identical
    # evaluation set, so the only inputs are dataset, task and increment.
    import inspect

    params = inspect.signature(subsample_val_table).parameters
    assert "seed" not in params and "random_state" not in params
    assert set(params) == {
        "val_table", "max_rows", "dataset_name", "task_name", "increment",
    }


def test_subsample_stays_time_ordered():
    out, _ = subsample_val_table(make_table(5_000), 500, "rel-hm", "user-churn", 3)
    assert out.df["timestamp"].is_monotonic_increasing


def test_subsample_preserves_table_metadata():
    table = make_table(5_000)
    out, _ = subsample_val_table(table, 500, "rel-hm", "user-churn", 3)
    assert out.fkey_col_to_pkey_table == table.fkey_col_to_pkey_table
    assert out.time_col == table.time_col
    assert out.pkey_col == table.pkey_col


def test_subsample_spans_the_whole_window_not_just_the_start():
    # The reason we subsample the table rather than using limit_val_batches on an
    # unshuffled loader: coverage must stay temporally uniform.
    out, _ = subsample_val_table(make_table(10_000), 500, "rel-hm", "user-churn", 3)
    assert out.df["entity"].min() < 1_000
    assert out.df["entity"].max() > 9_000
