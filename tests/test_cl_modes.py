"""Mode registry and buffer-to-table reconstruction.

The alignment test here pins a defect that made DER++ measure an indexing bug
rather than a literature baseline: 93% of replayed exemplars on rel-f1 were
distilled against a different row's stored logit.
"""

import numpy as np
import pandas as pd
import pytest
from relbench.base import Table

from experiments.continuous_learning.cl_modes import (
    DEFAULT_ROSTER,
    MODES,
    buffer_to_table,
    resolve_mode,
)
from redelex.continual.replay import ReservoirBuffer


def _template():
    return Table(
        df=pd.DataFrame({"driverId": [0], "position": [0.0],
                         "timestamp": pd.to_datetime(["2020-01-01"])}),
        fkey_col_to_pkey_table={"driverId": "drivers"},
        pkey_col=None, time_col="timestamp",
    )


def _buffer_with(entities, times, targets, logits):
    buf = ReservoirBuffer(capacity=1000, seed=0)
    buf.add(entities, timestamps=times, targets=targets, logits=logits)
    return buf


def test_roster_names_all_resolve():
    for name in DEFAULT_ROSTER:
        assert resolve_mode(name).name == name


def test_old_regime_names_still_resolve_as_aliases():
    # published runs must stay comparable
    assert resolve_mode("ft_full").name == "joint"
    assert resolve_mode("ft_newonly").name == "naive"


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown learning mode"):
        resolve_mode("ft_upsample_v2")


def test_only_increment_modes_need_a_previous_timestamp():
    assert not resolve_mode("from_scratch").needs_prev_timestamp
    assert not resolve_mode("joint").needs_prev_timestamp
    assert resolve_mode("naive").needs_prev_timestamp
    assert resolve_mode("er").needs_prev_timestamp


def test_only_stateful_modes_need_chain_state():
    assert not resolve_mode("naive").needs_chain_state
    for name in ("er", "der_pp", "ewc", "freeze_extend"):
        assert resolve_mode(name).needs_chain_state


# --- the alignment regression ------------------------------------------------


def test_returned_permutation_reorders_logits_onto_their_own_rows():
    # The real situation: one entity recurs at several timestamps, each row with
    # its own stored logit. Re-joining on the entity id collapsed to one logit
    # per entity and handed most exemplars somebody else's teacher signal.
    entities = np.array([7, 7, 7, 8, 8, 9])
    times = np.array([300, 100, 200, 200, 100, 100])
    targets = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    logits = np.array([-9.0, -2.0, 5.0, 3.0, -3.0, 0.5])
    buf = _buffer_with(entities, times, targets, logits)

    table, order = buffer_to_table(buf, "driverId", "timestamp", "position", _template())
    reordered = buf.logits[order]

    # every row's logit must still belong to the row it sits next to
    for row, logit in zip(table.df.itertuples(index=False), reordered):
        matches = [
            lg for e, t, lg in zip(entities, times, logits)
            if e == row.driverId and pd.Timestamp(t, unit="s") == row.timestamp
        ]
        assert logit in matches


def test_alignment_survives_timestamp_ties():
    # sort_values defaults to a non-stable quicksort, and these tables have heavy
    # ties, so recomputing the sort at the call site would disagree and silently
    # reshuffle the logits. The permutation must come from the same sort.
    n = 200
    entities = np.arange(n) % 7          # entities recur
    times = np.zeros(n, dtype=np.int64)  # every row ties
    targets = np.arange(n, dtype=float)
    logits = -np.arange(n, dtype=float)
    buf = _buffer_with(entities, times, targets, logits)

    table, order = buffer_to_table(buf, "driverId", "timestamp", "position", _template())
    # target and logit were built from the same index, so they must stay paired
    np.testing.assert_allclose(table.df["position"].to_numpy(), buf.targets[order])
    np.testing.assert_allclose(-table.df["position"].to_numpy(), buf.logits[order])


def test_table_is_time_ordered_and_keeps_metadata():
    buf = _buffer_with(np.array([1, 2, 3]), np.array([300, 100, 200]),
                       np.array([0.0, 1.0, 0.0]), np.array([1.0, 2.0, 3.0]))
    table, order = buffer_to_table(buf, "driverId", "timestamp", "position", _template())
    assert table.df["timestamp"].is_monotonic_increasing
    assert table.fkey_col_to_pkey_table == {"driverId": "drivers"}
    assert sorted(order.tolist()) == [0, 1, 2]


def test_empty_buffer_is_rejected():
    with pytest.raises(ValueError, match="empty buffer"):
        buffer_to_table(ReservoirBuffer(capacity=5, seed=0),
                        "driverId", "timestamp", "position", _template())
