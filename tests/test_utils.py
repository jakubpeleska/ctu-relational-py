import numpy as np
import pandas as pd
import pytest
import torch
from relbench.base import Database, Table
from torch_frame import stype
from torch_frame.datasets import FakeDataset

# TODO(stage 5): switch to `from redelex.utils import ...` once the package
# re-exports its public API.
from redelex.utils.datetime import NAT_UNIX_TIME, convert_timedelta, to_unix_time
from redelex.utils.merge import merge_tf


def test_to_unix_time_epoch_values():
    ser = pd.Series(pd.to_datetime(["1970-01-01", "1970-01-02", "2020-01-01"]))
    out = to_unix_time(ser)
    np.testing.assert_array_equal(out, [0, 86400, 1577836800])


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_to_unix_time_units(unit):
    ser = pd.Series(pd.to_datetime(["1970-01-02", "2020-01-01"])).astype(
        f"datetime64[{unit}]"
    )
    np.testing.assert_array_equal(to_unix_time(ser), [86400, 1577836800])


def test_to_unix_time_nat_sentinel():
    """Regression: NaT used to silently become an accidental division artifact;
    it now maps to the documented earliest-representable-time sentinel."""
    ser = pd.Series(pd.to_datetime(["2020-01-01", None, "1970-01-01"]))
    with pytest.warns(UserWarning, match="missing timestamps"):
        out = to_unix_time(ser)
    np.testing.assert_array_equal(out, [1577836800, NAT_UNIX_TIME, 0])


def test_to_unix_time_rejects_non_datetime():
    with pytest.raises(ValueError, match="datetime64"):
        to_unix_time(pd.Series([1, 2, 3]))


def test_convert_timedelta():
    df = pd.DataFrame(
        {
            "pk": [0, 1],
            "td": pd.to_timedelta([1, 2], unit="D"),
            "x": [1.0, 2.0],
        }
    )
    db = Database(
        table_dict={
            "t": Table(df=df, fkey_col_to_pkey_table={}, pkey_col="pk", time_col=None)
        }
    )
    convert_timedelta(db)

    out = db.table_dict["t"].df
    assert pd.api.types.is_datetime64_any_dtype(out["td"])
    assert out["td"].tolist() == [
        pd.Timestamp("1900-01-02"),
        pd.Timestamp("1900-01-03"),
    ]
    assert out["x"].tolist() == [1.0, 2.0]


# ---------------------------------------------------------------------------
# merge_tf
# ---------------------------------------------------------------------------


def _tf(stypes, num_rows=10):
    return FakeDataset(num_rows=num_rows, stypes=stypes).materialize().tensor_frame


def test_merge_tf_overlapping_cols_raise():
    left = _tf([stype.numerical])
    right = _tf([stype.numerical])
    with pytest.raises(ValueError, match="overlap"):
        merge_tf(left, right, torch.tensor([0]), torch.tensor([0]))


def test_merge_tf_left_join_num_cat():
    left = _tf([stype.numerical, stype.categorical])
    right = _tf([stype.numerical, stype.categorical])
    left_idx = torch.tensor([0, 2, 4])
    right_idx = torch.tensor([1, 3, 5])

    out = merge_tf(left, right, left_idx, right_idx, right_prefix="r_")

    assert out.num_rows == left.num_rows
    assert out.col_names_dict[stype.numerical] == [
        "num_1",
        "num_2",
        "num_3",
        "r_num_1",
        "r_num_2",
        "r_num_3",
    ]

    num_feat = out.feat_dict[stype.numerical]
    cat_feat = out.feat_dict[stype.categorical]

    # Left columns are unchanged.
    torch.testing.assert_close(num_feat[:, :3], left.feat_dict[stype.numerical])
    torch.testing.assert_close(cat_feat[:, :2], left.feat_dict[stype.categorical])

    # Matched rows carry the right frame's values...
    torch.testing.assert_close(
        num_feat[left_idx, 3:], right.feat_dict[stype.numerical][right_idx]
    )
    torch.testing.assert_close(
        cat_feat[left_idx, 2:], right.feat_dict[stype.categorical][right_idx]
    )

    # ...and unmatched rows are filled with NaN / -1.
    unmatched = torch.tensor([i for i in range(left.num_rows) if i not in {0, 2, 4}])
    assert num_feat[unmatched, 3:].isnan().all()
    assert (cat_feat[unmatched, 2:] == -1).all()

    # y comes from the left frame.
    torch.testing.assert_close(out.y, left.y)


def test_merge_tf_timestamp():
    left = _tf([stype.numerical])
    right = _tf([stype.timestamp])
    left_idx = torch.tensor([1, 3])
    right_idx = torch.tensor([0, 2])

    out = merge_tf(left, right, left_idx, right_idx)

    ts = out.feat_dict[stype.timestamp]
    assert ts.shape == (10, 3, 7)
    torch.testing.assert_close(ts[left_idx], right.feat_dict[stype.timestamp][right_idx])
    unmatched = torch.tensor([i for i in range(10) if i not in {1, 3}])
    assert (ts[unmatched] == -1).all()


def test_merge_tf_embedding():
    left = _tf([stype.numerical])
    right = _tf([stype.embedding])
    left_idx = torch.tensor([0, 5])
    right_idx = torch.tensor([2, 4])

    out = merge_tf(left, right, left_idx, right_idx)

    emb = out.feat_dict[stype.embedding]
    assert emb.num_rows == 10
    assert emb.num_cols == len(right.col_names_dict[stype.embedding])
    torch.testing.assert_close(
        emb.values[left_idx], right.feat_dict[stype.embedding].values[right_idx]
    )
