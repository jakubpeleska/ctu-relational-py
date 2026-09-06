r"""Turn per-checkpoint predictions into the continual-learning evaluation matrix.

``experiments/continuous_learning/run_predictions.py`` scores every finished
checkpoint over the *whole* task timeline and writes one wide CSV per task:
the task table (train + val + test rows, sorted by time) plus one column per
MLflow run, named ``f"{increment}_{run_id}"``. Several seeds run per increment,
so several columns normally share an increment.

This script reduces that table to ``R``, the evaluation matrix of the CL
literature: ``R[i, j]`` is the score of the model trained through episode ``i``,
evaluated on episode ``j``. The lower triangle is retention, the diagonal is
plasticity, and the upper triangle is what the model already knew. **Row 0 is
exactly the drift curve**: one model, frozen after the first episode, watched as
the distribution moves away from it.

Sign convention -- read this before interpreting any number
-----------------------------------------------------------
Every cell of ``R`` is **higher-is-better**, always, for every task type.
Regression is therefore scored as *negative* MAE, not MAE. This is deliberate:
:mod:`redelex.continual.metrics` sign-corrects backward transfer and forgetting
via a ``higher_is_better`` flag, and mixing directions inside one matrix would
silently invert the sign of every conclusion drawn from it -- forgetting would
read as transfer and vice versa. Because ``R`` is normalised here, every call
into ``metrics`` from this module uses ``higher_is_better=True``.

Two decay averages are reported, and they are different objects
---------------------------------------------------------------
``first_model_episode_decay_avg`` collapses row 0 of ``R`` -- the frozen first
increment, per episode. ``final_model_decay_avg`` is the number the published
plots report: the *final* increment scored per unique task timestamp from
``val_timestamp`` on, row-count weighted. See :func:`final_model_decay_avg`.

Everything except :func:`main` is a pure function over DataFrames and arrays, so
the reduction is testable without any fixture on disk.

Usage:
    .venv/bin/python scripts/build_evaluation_matrix.py \
        --predictions data/pelesjak_cl_ft_full/rel-f1_driver-position_predictions.csv \
        --dataset rel-f1 --task driver-position --aggregate mean

    Pass ``--val-delta-days`` whenever the run itself was swept with
    ``--val_delta_days``: the episode grid must be the run's own grid.
"""

import argparse
import json
import re
import sys
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from redelex.continual.metrics import (
    average_accuracy,
    backward_transfer,
    evaluation_matrix_from_predictions,
    exp_decay_avg,
    forward_transfer,
    per_episode_forgetting,
)

# `drift.py` may or may not exist yet -- it is being added on a parallel branch.
# The row-to-episode assignment is a three-line searchsorted, so this module
# carries its own copy rather than blocking on that module landing.
try:  # pragma: no cover - exercised by whichever branch of the import runs
    from redelex.continual.drift import episode_of_row as _shared_episode_of_row
except ImportError:  # pragma: no cover
    _shared_episode_of_row = None


__all__ = [
    "checkpoint_column_candidates",
    "select_checkpoint_columns",
    "checkpoint_columns_by_increment",
    "apply_checkpoint_aggregation",
    "assign_episodes",
    "episode_boundaries_from_splits",
    "metric_for_task_type",
    "roc_auc",
    "negative_mae",
    "build_matrix",
    "final_model_decay_avg",
    "summarise",
]

# Prediction columns are named "{increment}_{run_id}" by run_predictions.py. The
# increment must be an integer, which is what separates prediction columns from
# the task table's own columns (`driverId`, `date`, `position`, ...) and from
# runs whose increment was missing and got logged as "unknown".
_CHECKPOINT_RE = re.compile(r"^(\d+)_(.+)$")

# Suffix given to a column synthesised by averaging an increment's seeds. It is
# excluded when parsing so that re-running the selection over an already
# aggregated frame does not mistake "3_mean" for a run named "mean".
_MEAN_SUFFIX = "mean"

AGGREGATES = ("first", "mean")


# --- checkpoint column selection --------------------------------------------


def checkpoint_column_candidates(
    columns: Sequence, data_cols: Optional[Iterable] = None
) -> List:
    r"""The columns that may name a checkpoint, with the task's own columns removed.

    The ``"{increment}_{run_id}"`` pattern is not a safe filter on its own.
    ``run_predictions.py`` seeds the predictions CSV with the *whole* task table
    -- entity id, timestamp, target, and anything else the task carries -- so a
    data column named like ``"2_driverId"`` parses as increment 2. It would then
    be sorted in among that increment's real seeds and, whenever it sorts first,
    silently become the checkpoint whose scores fill a row of ``R``: a column of
    entity ids scored as if it were a model. No regex can tell the two apart,
    which is why the known data columns are dropped by exact name first.

    Args:
        columns: Column names of the predictions table.
        data_cols: Names of the task table's own columns, e.g.
            ``wrapper.full_table.df.columns``. Compared as strings, so a CSV read
            back with non-string column labels still matches. ``None`` drops
            nothing, leaving the caller responsible for the exclusion.

    Returns:
        The names in ``columns`` that are not data columns, in the original order
        and as the original objects, so the result still indexes the frame.
    """
    if data_cols is None:
        known = set()
    else:
        known = {str(column) for column in data_cols}
    return [column for column in columns if str(column) not in known]


def checkpoint_columns_by_increment(columns: Sequence[str]) -> Dict[int, List[str]]:
    r"""Group prediction columns by the increment their checkpoint was trained through.

    Args:
        columns: Column names of the predictions table, prediction columns and
            data columns mixed together.

    Returns:
        Mapping of integer increment to the run columns for that increment,
        sorted by name so the order does not depend on CSV column order.
    """
    groups: Dict[int, List[str]] = {}
    for column in columns:
        match = _CHECKPOINT_RE.match(str(column))
        if match is None:
            continue
        if match.group(2) == _MEAN_SUFFIX:
            continue
        groups.setdefault(int(match.group(1)), []).append(str(column))
    return {increment: sorted(runs) for increment, runs in sorted(groups.items())}


def select_checkpoint_columns(
    columns: Sequence[str], aggregate: str = "first"
) -> List[str]:
    r"""One column name per increment, ordered by increment.

    Args:
        columns: Column names of the predictions table. Names that do not look
            like ``"{increment}_{run_id}"`` are ignored, so the task table's own
            data columns can be passed straight through.
        aggregate: ``"first"`` picks the alphabetically first run of each
            increment, which is deterministic across re-runs because MLflow run
            ids are stable. ``"mean"`` names a synthetic ``"{increment}_mean"``
            column that averages the increment's seeds;
            :func:`apply_checkpoint_aggregation` is what actually creates it.

    Returns:
        Column names ordered by increment, one per increment.

    Raises:
        ValueError: If ``aggregate`` is not a supported strategy.
    """
    if aggregate not in AGGREGATES:
        raise ValueError(f"`aggregate` must be one of {AGGREGATES}, got {aggregate!r}")

    groups = checkpoint_columns_by_increment(columns)
    if aggregate == "first":
        return [runs[0] for runs in groups.values()]
    return [f"{increment}_{_MEAN_SUFFIX}" for increment in groups]


def apply_checkpoint_aggregation(
    predictions_df: pd.DataFrame, aggregate: str = "first"
) -> Tuple[pd.DataFrame, List[str]]:
    r"""Materialise the columns :func:`select_checkpoint_columns` names.

    Args:
        predictions_df: Wide predictions table.
        aggregate: ``"first"`` or ``"mean"``; see :func:`select_checkpoint_columns`.

    Returns:
        ``(frame, columns)`` where ``frame`` holds every name in ``columns``.
        For ``"first"`` the input frame is returned unchanged; for ``"mean"`` a
        shallow copy carrying the extra ``"{increment}_mean"`` columns is
        returned, so the caller's frame is never mutated.

    Raises:
        ValueError: If the table holds no prediction columns at all, which
            almost always means the wrong CSV was passed.
    """
    columns = select_checkpoint_columns(predictions_df.columns, aggregate=aggregate)
    if not columns:
        raise ValueError(
            "no '{increment}_{run_id}' prediction columns found; "
            f"got columns {list(predictions_df.columns)[:8]}"
        )
    if aggregate == "first":
        return predictions_df, columns

    groups = checkpoint_columns_by_increment(predictions_df.columns)
    frame = predictions_df.copy()
    for name, (_, runs) in zip(columns, groups.items(), strict=True):
        frame[name] = frame[runs].astype(float).mean(axis=1)
    return frame, columns


# --- row to episode assignment ----------------------------------------------


def _as_ordinal(values) -> np.ndarray:
    """Times or boundaries on a common numeric axis, so searchsorted can mix them."""
    index = pd.Index(values)
    if pd.api.types.is_numeric_dtype(index):
        return np.asarray(index, dtype=float)
    # Nanoseconds since the epoch; NaT lands far below every real boundary and
    # so falls outside every episode, which is the behaviour we want.
    return pd.DatetimeIndex(index).asi8.astype(float)


def _episode_of_row_local(times, boundaries) -> np.ndarray:
    """Fallback for `redelex.continual.drift.episode_of_row`; see :func:`assign_episodes`."""
    edges = _as_ordinal(boundaries)
    if edges.size < 2:
        raise ValueError(f"need at least 2 boundaries to form an episode, got {edges.size}")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("`boundaries` must be strictly increasing")

    stamps = _as_ordinal(times)
    # side="right" puts a row landing exactly on a boundary into the episode the
    # boundary opens, matching ContinuousWrapper.get_table's [start, end) windows.
    episodes = np.searchsorted(edges, stamps, side="right") - 1
    # Rows before the first boundary or at/after the last one belong to no
    # episode. metrics.evaluation_matrix_from_predictions drops negatives.
    episodes[episodes >= edges.size - 1] = -1
    episodes[stamps < edges[0]] = -1
    return episodes.astype(int)


def assign_episodes(times, boundaries) -> np.ndarray:
    r"""Episode index of every row, or ``-1`` for rows outside every episode.

    Episode ``j`` spans ``[boundaries[j], boundaries[j + 1])``, so
    ``len(boundaries) - 1`` episodes are defined.

    Prefers :func:`redelex.continual.drift.episode_of_row` when that module is
    importable, so the two stay in step, and falls back to a local searchsorted
    otherwise.

    Args:
        times: Timestamp (or numeric time) of each row.
        boundaries: Strictly increasing episode edges.

    Returns:
        Integer array of episode indices, one per row.
    """
    if _shared_episode_of_row is not None:
        try:
            return np.asarray(_shared_episode_of_row(times, boundaries), dtype=int)
        except TypeError:
            # The shared helper landed with a different signature than assumed;
            # the local copy is equivalent, so degrade rather than crash.
            pass
    return _episode_of_row_local(times, boundaries)


def episode_boundaries_from_splits(splits: Sequence) -> List:
    r"""Evaluation-episode edges implied by ``ContinuousWrapper.get_splits()``.

    ``get_splits()`` returns ``[t_first, ..., val_timestamp, test_timestamp]``
    and ``continuous_learning.py`` runs increments ``i = 1 .. len(splits) - 2``,
    where increment ``i`` trains on everything before ``splits[i]`` and is
    validated on ``[splits[i], splits[i + 1])``. So the first window,
    ``[splits[0], splits[1])``, is training data for increment 1 and is never an
    evaluation episode: dropping it lines episode ``j`` up with increment
    ``j + 1``, which is the alignment ``R`` needs.

    The RelBench **test** split is deliberately not an episode
    ---------------------------------------------------------
    The returned list ends at ``test_timestamp``, and :func:`assign_episodes`
    labels every row at or after the last boundary ``-1``, so nothing on or after
    ``test_timestamp`` reaches ``R``. That is a choice, not an oversight, and the
    rejected alternative was to append a boundary past ``test_timestamp`` and let
    the test split be a final column. Those rows are still scored, by
    :func:`final_model_decay_avg` -- see below.

    Args:
        splits: The list returned by ``ContinuousWrapper.get_splits()``.

    Returns:
        Boundaries defining ``len(splits) - 2`` evaluation episodes, ending at
        ``splits[-1]`` (``test_timestamp``), which closes the final episode
        rather than opening one.

    Raises:
        ValueError: If fewer than three splits are given, which leaves no
            episode to evaluate.
    """
    if len(splits) < 3:
        raise ValueError(
            f"need at least 3 splits to form one evaluation episode, got {len(splits)}"
        )
    # Why the test split gets no column: every column of `R` needs a checkpoint
    # trained *through* that episode to fill its diagonal, and the protocol trains
    # no increment on the validation window -- the last increment is validated on
    # it. A test column would therefore have no diagonal cell, and
    # `average_accuracy`, `backward_transfer` and `per_episode_forgetting` all
    # read the diagonal; `R` would also stop being square, which `build_matrix`
    # rejects outright. The compute `run_predictions.py` spends on those rows is
    # not wasted: `final_model_decay_avg` scores the final checkpoint from the
    # start of its own episode onwards with no upper bound, which is exactly the
    # notebook's `[val_timestamp, Timestamp.max)` window and covers the test split.
    return list(splits[1:])


# --- metrics ----------------------------------------------------------------


def roc_auc(y_true, y_score) -> float:
    r"""Area under the ROC curve, ties averaged, computed from rank statistics.

    Equivalent to ``sklearn.metrics.roc_auc_score`` for binary targets but
    without the dependency, since this script only needs the one metric.

    Args:
        y_true: Binary ground truth, values in ``{0, 1}``.
        y_score: Predicted score, any monotone transform of the probability.

    Returns:
        The AUC, or ``nan`` when the window holds a single class and the metric
        is undefined. ``nan`` is what an empty episode already yields, so the
        two unusable cases look alike downstream.

    Raises:
        ValueError: If ``y_true`` holds values other than 0 and 1.
    """
    truth = np.asarray(y_true, dtype=float)
    scores = np.asarray(y_score, dtype=float)
    if truth.shape != scores.shape:
        raise ValueError(f"shape mismatch: y_true {truth.shape}, y_score {scores.shape}")

    labels = np.unique(truth)
    if not np.all(np.isin(labels, (0.0, 1.0))):
        raise ValueError(f"`y_true` must be binary 0/1, found values {labels[:5]}")

    positive = truth == 1.0
    n_pos = int(positive.sum())
    n_neg = int(truth.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    _, inverse, counts = np.unique(scores[order], return_inverse=True, return_counts=True)
    # Mid-rank within each tied group, so tied pairs count as half a win --
    # the same convention sklearn uses.
    ends = np.cumsum(counts)
    starts = ends - counts
    ranks_sorted = ((starts + 1) + ends) / 2.0
    ranks = np.empty(scores.size, dtype=float)
    ranks[order] = ranks_sorted[inverse]

    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def negative_mae(y_true, y_pred) -> float:
    r"""Mean absolute error, negated so that higher is better.

    The negation is the whole point: see the sign convention in the module
    docstring. A perfect regressor scores ``0.0`` and every real one scores
    below it.
    """
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if truth.shape != pred.shape:
        raise ValueError(f"shape mismatch: y_true {truth.shape}, y_pred {pred.shape}")
    if truth.size == 0:
        return float("nan")
    return -float(np.mean(np.abs(truth - pred)))


# Keyed by `relbench.base.TaskType` values. Every entry must be
# higher-is-better; adding a raw error metric here would break every sign in
# `summarise`.
_METRIC_FNS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "binary_classification": roc_auc,
    "regression": negative_mae,
}

_TASK_TYPE_ALIASES = {
    "binary": "binary_classification",
    "classification": "binary_classification",
}


def metric_for_task_type(task_type) -> Callable[[np.ndarray, np.ndarray], float]:
    r"""The higher-is-better scorer for a task type.

    Args:
        task_type: A ``relbench.base.TaskType`` or its string value, e.g.
            ``"binary_classification"`` or ``"regression"``.

    Returns:
        ``metric_fn(y_true, y_pred) -> float``, higher meaning better.

    Raises:
        ValueError: For a task type with no higher-is-better scorer defined.
            Multiclass and link prediction are deliberately absent: the
            continual-learning grid is binary and regression only, and guessing
            a scorer would be worse than failing.
    """
    name = str(getattr(task_type, "value", task_type)).lower()
    name = _TASK_TYPE_ALIASES.get(name, name)
    if name not in _METRIC_FNS:
        raise ValueError(
            f"no higher-is-better metric for task type {task_type!r}; "
            f"supported: {sorted(_METRIC_FNS)}"
        )
    return _METRIC_FNS[name]


# --- the matrix and its summary ---------------------------------------------


def build_matrix(
    predictions_df: pd.DataFrame,
    splits: Sequence,
    target_col: str,
    time_col: str,
    task_type,
    aggregate: str = "first",
    data_cols: Optional[Iterable] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Reduce the wide predictions table to the evaluation matrix ``R``.

    Args:
        predictions_df: Table written by ``run_predictions.py``: the task rows
            plus one ``"{increment}_{run_id}"`` column per checkpoint.
        splits: Episode boundaries, already aligned with the checkpoints --
            episode ``j`` is ``[splits[j], splits[j + 1])``. Pass
            ``episode_boundaries_from_splits(wrapper.get_splits())`` to get this
            from the experiment's own splits.
        target_col: Ground-truth column of the task table.
        time_col: Timestamp column the episodes are cut on.
        task_type: Task type, resolved by :func:`metric_for_task_type`.
        aggregate: How to collapse an increment's seeds; see
            :func:`select_checkpoint_columns`.
        data_cols: The task table's own columns, which ``run_predictions.py``
            copies into the predictions CSV verbatim. Pass
            ``wrapper.full_table.df.columns``: it is the only way a data column
            named like ``"2_driverId"`` can be kept out of ``R`` -- see
            :func:`checkpoint_column_candidates`. When omitted, only
            ``target_col`` and ``time_col`` are known to be data, which is
            enough only for a table whose remaining columns cannot parse as
            ``"{increment}_{run_id}"``.

    Returns:
        ``(R, episodes)``. ``R`` is square, ``n_episodes`` on a side, every cell
        higher-is-better, with ``nan`` where an episode caught no rows. ``R[0]``
        is the drift curve. ``episodes`` is the per-row episode index, kept so
        callers can count rows per episode without redoing the assignment.

    Raises:
        ValueError: If a required column is missing, or if the number of
            checkpoints does not match the number of episodes. The counts must
            agree exactly -- a mismatch would slide rows against columns and
            quietly turn retention into transfer.
    """
    for column in (target_col, time_col):
        if column not in predictions_df.columns:
            raise ValueError(f"column {column!r} not in the predictions table")

    metric_fn = metric_for_task_type(task_type)

    # Drop every column known to be data before parsing names, so a data column
    # that happens to look like "2_something" cannot be mistaken for a checkpoint.
    # `target_col` and `time_col` are always known; the rest of the task table is
    # only known when the caller passes it, hence the union rather than a default.
    known_data = {str(target_col), str(time_col)}
    if data_cols is not None:
        known_data.update(str(column) for column in data_cols)
    candidates = checkpoint_column_candidates(predictions_df.columns, known_data)
    frame, columns = apply_checkpoint_aggregation(
        predictions_df[candidates], aggregate=aggregate
    )

    n_episodes = len(splits) - 1
    if len(columns) != n_episodes:
        raise ValueError(
            f"{len(columns)} checkpoint(s) but {n_episodes} episode(s); "
            "R must be square and checkpoint i must correspond to episode i"
        )

    episodes = assign_episodes(predictions_df[time_col], splits)
    matrix = evaluation_matrix_from_predictions(
        predictions=frame,
        episode_of_row=episodes,
        checkpoint_columns=columns,
        target=predictions_df[target_col],
        metric_fn=metric_fn,
    )
    return matrix, episodes


def _diagonal_free_baseline(matrix: np.ndarray) -> np.ndarray:
    """Mean score on episode ``j`` over the checkpoints trained before it.

    Lopez-Paz & Ranzato's forward transfer compares against an independently
    initialised model, which this experiment never trains. The closest thing
    ``R`` contains on its own is column ``j`` above the diagonal: every model
    that had not yet been trained through episode ``j``. Comparing the most
    recent of those (``R[j-1, j]``, what `forward_transfer` uses) against the
    mean of the *older* ones asks whether accumulating more history helps on an
    unseen future episode -- a within-matrix proxy, not the published FWT.

    Returns:
        Length-``n`` vector, ``nan`` at indices 0 and 1 where the proxy has no
        rows to average, so a matrix of fewer than three episodes yields no
        baseline at all.
    """
    n = matrix.shape[0]
    baseline = np.full(n, np.nan, dtype=float)
    # Row j-1 must be left OUT of column j's baseline: it is the very cell
    # `forward_transfer` subtracts the baseline from, so averaging it into its own
    # baseline pulls the difference toward zero -- and at j = 1, where row 0 is the
    # only row above the diagonal, the difference is *identically* zero. A
    # two-episode task (rel-f1/driver-top3 is one) would then report
    # `forward_transfer = 0.0000` for every mode, which reads as "no transfer"
    # rather than "undefined". So the proxy starts at j = 2 and indices 0 and 1
    # stay nan: undefined, not fabricated.
    for j in range(2, n):
        baseline[j] = float(np.mean(matrix[: j - 1, j]))
    return baseline


def final_model_decay_avg(
    times,
    y_true,
    y_pred,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    start,
    end=None,
    decay: float = 0.0,
) -> float:
    r"""The published decay metric: one model, scored per task timestamp, from ``start`` on.

    This reproduces ``calculate_metric_for_split`` in
    ``notebooks/process-data-continuous-learning.ipynb``, which is what every
    published continual-learning number was computed with: take **one**
    checkpoint -- the final increment's -- score it separately on every *unique
    task timestamp* in ``[start, end)``, and collapse those per-timestamp scores
    with :func:`~redelex.continual.metrics.exp_decay_avg`, weighting each by how
    many rows that timestamp carries.

    It is *not* ``exp_decay_avg(R[0, :])``, which this module reports separately
    as ``first_model_episode_decay_avg``. The two differ in four ways, and used
    to share the name ``drift_curve_avg``:

    1. **Which model.** Row 0 of ``R`` is the *first* increment's checkpoint,
       frozen; the published metric uses the *final* one, the deployed model.
    2. **Which windows.** Row 0 is scored per *episode*; the published metric is
       scored per *unique task timestamp*, of which an episode holds many.
    3. **Which span.** Row 0 covers the whole episode grid; the published metric
       starts at the final model's own window (``val_timestamp``) and runs to
       ``Timestamp.max``, so it includes the RelBench test split -- which
       :func:`episode_boundaries_from_splits` deliberately leaves out of ``R``.
    4. **Which weights.** Row 0 is collapsed unweighted; the published metric
       weights each window by its row count, so a timestamp with three rows
       counts three times as much as one with a single row.

    Args:
        times: Timestamp (or numeric time) of each row, aligned with ``y_true``.
        y_true: Ground truth per row.
        y_pred: One checkpoint's prediction per row.
        metric_fn: Called as ``metric_fn(y_true, y_pred)`` on one window's rows.
            Must be higher-is-better; see the module docstring.
        start: Inclusive lower bound of the window, normally ``val_timestamp``.
        end: Exclusive upper bound. ``None`` means unbounded, which is what the
            notebook's ``pd.Timestamp.max`` amounts to.
        decay: Passed to :func:`~redelex.continual.metrics.exp_decay_avg`; ``0``
            is the plain row-count-weighted mean the published plots used.

    Returns:
        The weighted average, or ``nan`` when no window in the span is scorable.

    Raises:
        ValueError: If the three per-row inputs disagree in length, or if no row
            falls in ``[start, end)`` at all -- an empty span means the wrong
            ``start`` was passed, not a model with nothing to say.
    """
    stamps = pd.Index(times)
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if not (len(stamps) == truth.size == pred.size):
        raise ValueError(
            f"length mismatch: times {len(stamps)}, y_true {truth.size}, "
            f"y_pred {pred.size}"
        )

    mask = np.asarray(stamps >= start)
    if end is not None:
        mask &= np.asarray(stamps < end)
    if not mask.any():
        raise ValueError(f"no rows in [{start!r}, {end!r}); check `start`")

    window_of_row = stamps.to_numpy()[mask]
    window_truth = truth[mask]
    window_pred = pred[mask]

    scores: List[float] = []
    counts: List[int] = []
    for window in np.unique(window_of_row):  # np.unique sorts, so oldest first
        rows = window_of_row == window
        score = float(metric_fn(window_truth[rows], window_pred[rows]))
        if np.isnan(score):
            # A single-timestamp window very often holds one class only, where
            # ROC-AUC simply has no value. The notebook's torchmetrics BinaryAUROC
            # returns 0.0 there and averages it in as though the model had ranked
            # every pair backwards; propagating the nan instead would erase the
            # metric for almost every binary task. Dropping the window is the only
            # reading that neither fabricates a score nor loses the rest. Note this
            # also shifts later windows one place earlier in the decay schedule,
            # which is a no-op at the published `decay=0`.
            continue
        scores.append(score)
        counts.append(int(rows.sum()))

    if not scores:
        return float("nan")
    return exp_decay_avg(scores, counts=counts, decay=decay)


def summarise(
    R,
    baseline: Optional[Sequence[float]] = None,
    decay: float = 0.0,
    final_model_decay: Optional[float] = None,
) -> Dict[str, object]:
    r"""The four CL metrics plus the drift curve, for a higher-is-better ``R``.

    ``R`` must already follow the module's sign convention (higher is better in
    every cell), which is what :func:`build_matrix` produces; every call below
    therefore passes ``higher_is_better=True``.

    Args:
        R: Square evaluation matrix.
        baseline: Per-episode score of an independently initialised model, for
            forward transfer. Defaults to the diagonal-free within-matrix proxy
            described in :func:`_diagonal_free_baseline`.
        decay: Passed to :func:`~redelex.continual.metrics.exp_decay_avg` when
            collapsing row 0 of ``R`` to a single number. ``0`` is a plain mean.
        final_model_decay: The published decay metric from
            :func:`final_model_decay_avg`, which needs the per-row predictions
            and so cannot be recomputed from ``R``. Reported as ``nan`` when the
            caller has no predictions to hand.

    Returns:
        Dict with ``n_episodes``, ``average_accuracy``, ``backward_transfer``,
        ``forward_transfer``, ``forward_transfer_baseline``,
        ``per_episode_forgetting``, ``drift_curve`` (row 0 of ``R``),
        ``first_model_episode_decay_avg`` and ``final_model_decay_avg``. ``nan``
        cells propagate rather than being dropped, so an incomplete matrix is
        visible instead of silently averaged away.

    ``first_model_episode_decay_avg`` was called ``drift_curve_avg`` and is a
    within-``R`` quantity: the first increment's per-episode scores, collapsed.
    It is **not** the number the published plots report -- that is
    ``final_model_decay_avg``, and :func:`final_model_decay_avg` lists the four
    ways the two differ.
    """
    matrix = np.asarray(R, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"`R` must be a square 2-D matrix, got shape {matrix.shape}")

    n = matrix.shape[0]
    if baseline is None:
        base = _diagonal_free_baseline(matrix)
        # The within-matrix proxy is undefined for episode 1 -- its only earlier
        # model is the one forward transfer scores -- so episode 1 is dropped from
        # the average by handing `forward_transfer` the sub-matrix from episode 1
        # on: its `R[j-1, j]` and `base[j]` then range over episodes 2.. of the
        # original. Below three episodes nothing is left, and the answer is `nan`
        # rather than the 0.0 an all-inclusive baseline used to manufacture.
        fwt = (
            forward_transfer(matrix[1:, 1:], base[1:], higher_is_better=True)
            if n >= 3
            else float("nan")
        )
    else:
        base = np.asarray(baseline, dtype=float)
        fwt = forward_transfer(matrix, base, higher_is_better=True)
    drift_curve = matrix[0, :]

    return {
        "n_episodes": int(n),
        "average_accuracy": average_accuracy(matrix),
        "backward_transfer": backward_transfer(matrix, higher_is_better=True),
        "forward_transfer": fwt,
        "forward_transfer_baseline": [float(v) for v in base],
        "per_episode_forgetting": [
            float(v) for v in per_episode_forgetting(matrix, higher_is_better=True)
        ],
        "drift_curve": [float(v) for v in drift_curve],
        "first_model_episode_decay_avg": exp_decay_avg(drift_curve, decay=decay),
        "final_model_decay_avg": (
            float("nan") if final_model_decay is None else float(final_model_decay)
        ),
    }


# --- CLI ---------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--predictions", required=True, help="CSV written by run_predictions.py."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--target-col",
        default=None,
        help="Ground-truth column. Defaults to the task's own target column.",
    )
    parser.add_argument(
        "--time-col",
        default=None,
        help="Timestamp column. Defaults to the task's own time column.",
    )
    parser.add_argument(
        "--aggregate",
        default="first",
        choices=list(AGGREGATES),
        help="How to collapse the seeds of one increment into one checkpoint.",
    )
    parser.add_argument(
        "--val-delta-days",
        type=float,
        default=None,
        help=(
            "Episode width in days. Must be whatever the run passed to "
            "continuous_learning.py's --val_delta_days; the boundaries derived "
            "from it become the columns of R."
        ),
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.0,
        help="Decay for both decay averages; 0 is a plain (weighted) mean.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write R (CSV) and the summary (JSON) next to this path prefix.",
    )
    return parser.parse_args(argv)


def _load_task_context(
    dataset: str, task_name: str, val_delta_days: Optional[float] = None
):
    r"""Task metadata, episode boundaries and the task table's own columns, imported lazily.

    relbench and the experiment package pull in torch and the dataset cache, so
    importing them at module scope would make the pure functions above
    untestable without the full environment.

    Args:
        dataset: RelBench dataset name, e.g. ``"rel-f1"``.
        task_name: RelBench task name, e.g. ``"driver-position"``.
        val_delta_days: Episode width in days, forwarded to
            ``ContinuousWrapper.get_splits`` as ``val_delta``. ``None`` keeps the
            wrapper's default, the dataset's own validation window.

    Returns:
        ``(task, splits, table_columns)``.
    """
    from relbench.tasks import get_task

    from experiments.continuous_learning.continuous_task import ContinuousWrapper

    task = get_task(dataset, task_name)
    wrapper = ContinuousWrapper(task)
    # The width has to be the run's own width, not get_splits' default. A run
    # swept with --val_delta_days cut the timeline into different windows than the
    # default does, and nothing downstream would notice: `build_matrix` only checks
    # that the *number* of episodes equals the number of checkpoints, so a matching
    # count over a mismatched partition passes, and R's rows (checkpoints from one
    # partition) would be scored against columns (episodes from another).
    val_delta = None if val_delta_days is None else pd.Timedelta(days=val_delta_days)
    # The full table's columns are exactly the non-prediction columns of the CSV:
    # run_predictions.py seeds the file with `wrapped_task.full_table.df` and only
    # ever appends checkpoint columns to it.
    return (
        task,
        wrapper.get_splits(val_delta=val_delta),
        list(wrapper.full_table.df.columns),
    )


def main(argv=None) -> int:
    args = parse_args(argv)

    predictions = pd.read_csv(args.predictions)
    task, raw_splits, table_cols = _load_task_context(
        args.dataset, args.task, val_delta_days=args.val_delta_days
    )

    target_col = args.target_col or task.target_col
    time_col = args.time_col or task.time_col
    predictions[time_col] = pd.to_datetime(predictions[time_col])

    # Everything the task table carries -- entity id included -- is data, never a
    # checkpoint. The overridable --target-col/--time-col are added because they
    # may name a column the task table itself does not list.
    data_cols = set(map(str, table_cols)) | {str(target_col), str(time_col)}

    boundaries = episode_boundaries_from_splits(raw_splits)
    candidates = checkpoint_column_candidates(predictions.columns, data_cols)
    increments = sorted(checkpoint_columns_by_increment(candidates))
    if not increments:
        raise SystemExit(
            "no '{increment}_{run_id}' prediction columns in "
            f"{args.predictions}; got columns {list(predictions.columns)[:8]}"
        )

    # Episode j is aligned to increment j + 1, so the increments present must be
    # the prefix 1..k for any truncation to be meaningful. Only a *tail* gap can
    # be dropped. A head or interior gap must not be: run_predictions.py queries
    # MLflow with no `order_by`, and MLflow defaults to `start_time DESC`, so an
    # interrupted predict job leaves the NEWEST increments in the CSV. Chopping
    # the boundary list from the right would then score increment 3 against
    # episode 0 and print a "drift curve" that improves over time -- from models
    # that never saw the early episodes. Refuse rather than guess.
    if increments != list(range(1, len(increments) + 1)):
        missing = sorted(set(range(1, max(increments) + 1)) - set(increments))
        raise SystemExit(
            f"increments {increments} in {args.predictions} are not the prefix "
            f"1..{len(increments)}: increment(s) {missing} are missing. Episode j "
            "is aligned to increment j + 1, so the remaining columns cannot be "
            "matched to episodes. Re-run predictions for the missing increments, "
            "or drop the later columns to leave a prefix."
        )

    frame, columns = apply_checkpoint_aggregation(
        predictions[candidates], aggregate=args.aggregate
    )
    # A chain that stopped early leaves fewer checkpoints than episodes, and the
    # check above has already established they are the leading ones. Keep the
    # episodes they line up with rather than failing, but say so loudly -- the
    # tail of the drift curve is being dropped.
    if len(columns) < len(boundaries) - 1:
        print(
            f"WARNING: only {len(columns)} increment(s) finished but "
            f"{len(boundaries) - 1} episode(s) exist; truncating to the "
            "episodes the checkpoints cover.",
            file=sys.stderr,
        )
        boundaries = boundaries[: len(columns) + 1]

    matrix, episodes = build_matrix(
        predictions,
        boundaries,
        target_col=target_col,
        time_col=time_col,
        task_type=task.task_type,
        aggregate=args.aggregate,
        data_cols=data_cols,
    )
    # The deployed model's decay curve, over its own window and everything after
    # it -- including the RelBench test split, which R has no column for.
    # `boundaries[-2]` opens the last episode kept above, which is `val_timestamp`
    # whenever nothing was truncated.
    summary = summarise(
        matrix,
        decay=args.decay,
        final_model_decay=final_model_decay_avg(
            predictions[time_col],
            predictions[target_col],
            frame[columns[-1]],
            metric_for_task_type(task.task_type),
            start=boundaries[-2],
            decay=args.decay,
        ),
    )

    metric_name = "roc_auc" if metric_for_task_type(task.task_type) is roc_auc else "-mae"
    counts = [int(np.sum(episodes == j)) for j in range(matrix.shape[0])]
    print(f"{args.dataset}/{args.task}  metric={metric_name} (higher is better)")
    print(f"rows per episode: {counts}\n")

    frame = pd.DataFrame(
        matrix,
        index=[f"after_ep{i}" for i in range(matrix.shape[0])],
        columns=[f"ep{j}" for j in range(matrix.shape[0])],
    )
    print(frame.round(4).to_string())
    print()
    for key in (
        "average_accuracy",
        "backward_transfer",
        "forward_transfer",
        "first_model_episode_decay_avg",
        "final_model_decay_avg",
    ):
        print(f"{key:>30}: {summary[key]:.4f}")
    print(f"{'drift_curve':>30}: {np.round(summary['drift_curve'], 4).tolist()}")
    print(f"{'forgetting':>30}: {np.round(summary['per_episode_forgetting'], 4).tolist()}")

    if np.isnan(matrix).any():
        print("\nWARNING: R holds nan cells (empty episode, or AUC on one class).")

    if args.out:
        frame.to_csv(f"{args.out}_matrix.csv")
        with open(f"{args.out}_summary.json", "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"\nWrote {args.out}_matrix.csv and {args.out}_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
