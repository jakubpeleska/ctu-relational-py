import numpy as np
import pandas as pd
import pytest
from relbench.base import TaskType

from redelex.tasks.mixins import TemporalTaskMixin
from redelex.tasks.task_impute import ImputeEntityStaticTask

from .helpers import (
    N_USERS,
    NAN_TARGET_CAT_ROWS,
    NAN_TARGET_NUM_ROWS,
    UserStaticBinaryTask,
    UserStaticMulticlassTask,
    UserStaticRegressionTask,
    UserTemporalBinaryTask,
)


class UserCatAsMulticlassTask(ImputeEntityStaticTask):
    """Multiclass task over the two-category target, to check label dtypes."""

    entity_col = "__PK__"
    entity_table = "users"
    target_col = "target_cat"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class UserStaticBadBinaryTask(ImputeEntityStaticTask):
    entity_col = "__PK__"
    entity_table = "users"
    target_col = "age"  # 20 distinct values
    task_type = TaskType.BINARY_CLASSIFICATION


def _all_split_frames(task):
    return pd.concat(
        [
            task.get_table(split, mask_input_cols=False).df
            for split in ["train", "val", "test"]
        ]
    )


def test_static_binary_nan_target_sentinel(synthetic_dataset):
    """Rows with a missing target keep the documented -1 sentinel label."""
    task = UserStaticBinaryTask(synthetic_dataset)
    df = _all_split_frames(task)

    assert len(df) == N_USERS
    assert set(df["target_cat"].unique()) <= {-1.0, 0.0, 1.0}

    nan_rows = df[df["__PK__"].isin(NAN_TARGET_CAT_ROWS)]
    assert (nan_rows["target_cat"] == -1.0).all()
    # "no" < "yes" after sorted factorization
    ok_rows = df[~df["__PK__"].isin(NAN_TARGET_CAT_ROWS)]
    expected = ok_rows["__PK__"].map(lambda pk: 1.0 if pk % 2 == 0 else 0.0)
    assert (ok_rows["target_cat"] == expected).all()


def test_static_multiclass_int_labels(synthetic_dataset):
    task = UserCatAsMulticlassTask(synthetic_dataset)
    df = _all_split_frames(task)

    assert pd.api.types.is_integer_dtype(df["target_cat"])
    assert set(df["target_cat"].unique()) <= {-1, 0, 1}


def test_static_regression_nan_preserved(synthetic_dataset):
    task = UserStaticRegressionTask(synthetic_dataset)
    df = _all_split_frames(task)

    nan_rows = df[df["__PK__"].isin(NAN_TARGET_NUM_ROWS)]
    assert nan_rows["target_num"].isna().all()
    assert df[~df["__PK__"].isin(NAN_TARGET_NUM_ROWS)]["target_num"].notna().all()


def test_labels_exclude_the_missing_sentinel(synthetic_dataset):
    """The -1 label used for missing targets is not one of the classes."""
    task = UserStaticMulticlassTask(synthetic_dataset)
    labels = _all_split_frames(task)["target_multi"]

    assert -1 in set(labels)  # the synthetic target does contain NaNs
    assert set(labels) - {-1} == {0, 1, 2}


def test_declared_num_classes_matches_the_encoded_labels(synthetic_dataset):
    """`num_classes` is declared per task, so it can disagree with the data.
    Where a task declares it, the labels must land in ``range(num_classes)``.
    """
    task = UserStaticMulticlassTask(synthetic_dataset)
    assert task.num_classes == 3

    labels = set(_all_split_frames(task)["target_multi"]) - {-1}
    assert labels == set(range(task.num_classes))


def test_num_classes_defaults_to_none(synthetic_dataset):
    """Nothing infers the count: a task that does not declare it reports None."""
    assert UserStaticRegressionTask(synthetic_dataset).num_classes is None
    assert UserCatAsMulticlassTask(synthetic_dataset).num_classes is None


def test_binary_task_validates_category_count(synthetic_dataset):
    task = UserStaticBadBinaryTask(synthetic_dataset)
    with pytest.raises(ValueError, match="exactly 2 categories"):
        task.get_table("train", mask_input_cols=False)


def test_static_split_partition(synthetic_dataset):
    task = UserStaticBinaryTask(synthetic_dataset)
    db = synthetic_dataset.get_db(upto_test_timestamp=False)

    splits = {s: task.make_split_range(db, s) for s in ["train", "val", "test"]}

    assert len(splits["train"]) == 16
    assert len(splits["val"]) == 2
    assert len(splits["test"]) == 2
    combined = splits["train"].append(splits["val"]).append(splits["test"])
    assert sorted(combined) == list(range(N_USERS))

    # Deterministic: same split on a second call.
    again = task.make_split_range(db, "train")
    assert sorted(splits["train"]) == sorted(again)


def test_make_modified_db_rejects_inplace_without_db(synthetic_dataset):
    """The dataset's database is lru_cached, so modifying it in place would
    corrupt every later get_table call. That combination is rejected."""
    task = UserStaticBinaryTask(synthetic_dataset)

    with pytest.raises(ValueError, match="in place"):
        task.make_modified_db(inplace=True)


def test_make_modified_db_does_not_mutate_cached_db(synthetic_dataset):
    """The default (copying) path must leave the cached database untouched."""
    task = UserStaticBinaryTask(synthetic_dataset)

    modified = task.make_modified_db()
    assert "target_cat" not in modified.table_dict["users"].df.columns

    cached = synthetic_dataset.get_db(upto_test_timestamp=False)
    assert "target_cat" in cached.table_dict["users"].df.columns

    # And the task still works afterwards.
    table = task.get_table("train", mask_input_cols=False)
    assert "target_cat" in table.df.columns


def test_make_modified_db_inplace_with_explicit_db(synthetic_dataset):
    """An explicitly passed database may be modified in place."""
    task = UserStaticBinaryTask(synthetic_dataset)
    db = task.make_modified_db()  # a private copy, target already dropped
    db.table_dict["users"].df["target_cat"] = "yes"

    same = task.make_modified_db(db=db, inplace=True)
    assert same is db
    assert "target_cat" not in db.table_dict["users"].df.columns


def test_make_modified_db_explicit_db_inplace(synthetic_dataset):
    task = UserStaticBinaryTask(synthetic_dataset)
    db = task.make_modified_db()  # a private copy with the target dropped

    # Repeated modification of the same object must not raise.
    again = task._make_modified_db(db)
    assert "target_cat" not in again.table_dict["users"].df.columns


def test_default_test_table_masks_columns(synthetic_dataset):
    """Regression: static task tables have time_col=None, and _mask_input_cols
    used to build df[[None, ...]] raising KeyError for every static task."""
    task = UserStaticBinaryTask(synthetic_dataset)
    table = task.get_table("test")  # default masks input columns

    assert list(table.df.columns) == ["__PK__"]
    assert "target_cat" not in table.df.columns


def test_get_table_with_explicit_db_bypasses_cache(synthetic_dataset, tmp_path):
    task = UserStaticBinaryTask(synthetic_dataset, cache_dir=str(tmp_path))
    cached = task.get_table("train", mask_input_cols=False)
    assert (tmp_path / "train.parquet").exists()

    import copy

    db = copy.deepcopy(synthetic_dataset.get_db(upto_test_timestamp=False))
    db.table_dict["users"].df["target_cat"] = ["no", "yes"] * (N_USERS // 2)

    table = task.get_table("train", mask_input_cols=False, db=db)
    # The table must reflect the provided db, not the cached parquet.
    assert not table.df["target_cat"].equals(cached.df["target_cat"])


def test_temporal_task_splits(synthetic_dataset):
    task = UserTemporalBinaryTask(synthetic_dataset)

    frames = {
        s: task.get_table(s, mask_input_cols=False).df for s in ["train", "val", "test"]
    }

    train_max = frames["train"]["reg_date"].max()
    val_min, val_max = frames["val"]["reg_date"].min(), frames["val"]["reg_date"].max()
    test_min = frames["test"]["reg_date"].min()

    assert train_max < pd.Timestamp("2020-01-01")
    assert val_min >= pd.Timestamp("2020-01-01")
    assert val_max < pd.Timestamp("2020-07-01")
    assert test_min >= pd.Timestamp("2020-07-01")

    pks = pd.concat([f["__PK__"] for f in frames.values()])
    assert pks.is_unique  # no entity appears in two splits


def test_temporal_unknown_split_raises(synthetic_dataset):
    """Regression: an unknown split silently fell through to the test range in
    ImputeEntityTemporalTask and raised UnboundLocalError in TemporalTaskMixin."""
    task = UserTemporalBinaryTask(synthetic_dataset)
    db = synthetic_dataset.get_db(upto_test_timestamp=False)

    with pytest.raises(ValueError, match="Unknown split"):
        task.make_split_range(db, "Val")

    with pytest.raises(ValueError, match="Unknown split"):
        TemporalTaskMixin.make_split_range(task, db, "bogus")


def test_temporal_timedelta_guard(synthetic_dataset):
    class TooLargeDelta(UserTemporalBinaryTask):
        timedelta = pd.Timedelta(days=365)

    with pytest.raises(ValueError, match="timedelta"):
        TooLargeDelta(synthetic_dataset)


def test_evaluate_length_mismatch(synthetic_dataset):
    task = UserStaticRegressionTask(synthetic_dataset)
    target_table = task.get_table("test", mask_input_cols=False)
    with pytest.raises(ValueError, match="length"):
        task.evaluate(np.zeros(len(target_table.df) + 1), target_table, metrics=[])


def test_evaluate_reports_empty_default_metrics(synthetic_dataset):
    """`metrics` is declared per task. A task that declares an empty list gets a
    clear error rather than an empty result dict."""

    class NoMetricsTask(UserStaticRegressionTask):
        metrics = []

    task = NoMetricsTask(synthetic_dataset)
    target_table = task.get_table("test", mask_input_cols=False)
    with pytest.raises(NotImplementedError, match="Default metrics are not defined"):
        task.evaluate(np.zeros(len(target_table.df)), target_table)


def test_evaluate_with_explicit_metrics(synthetic_dataset):
    task = UserStaticRegressionTask(synthetic_dataset)
    target_table = task.get_table("test", mask_input_cols=False)

    def mae(target, pred):
        return float(np.abs(target - pred).mean())

    scores = task.evaluate(np.zeros(len(target_table.df)), target_table, metrics=[mae])
    assert set(scores) == {"mae"} and scores["mae"] >= 0.0
