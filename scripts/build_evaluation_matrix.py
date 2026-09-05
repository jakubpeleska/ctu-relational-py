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

Everything except :func:`main` is a pure function over DataFrames and arrays, so
the reduction is testable without any fixture on disk.

Usage:
    .venv/bin/python scripts/build_evaluation_matrix.py \
        --predictions data/pelesjak_cl_ft_full/rel-f1_driver-position_predictions.csv \
        --dataset rel-f1 --task driver-position --aggregate mean
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

    Args:
        splits: The list returned by ``ContinuousWrapper.get_splits()``.

    Returns:
        Boundaries defining ``len(splits) - 2`` evaluation episodes.

    Raises:
        ValueError: If fewer than three splits are given, which leaves no
            episode to evaluate.
    """
    if len(splits) < 3:
        raise ValueError(
            f"need at least 3 splits to form one evaluation episode, got {len(splits)}"
        )
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
    recent of those (``R[j-1, j]``, what `forward_transfer` uses) against their
    mean asks whether accumulating more history helps on an unseen future
    episode -- a within-matrix proxy, not the published FWT.
    """
    n = matrix.shape[0]
    baseline = np.empty(n, dtype=float)
    # Index 0 is never read by `forward_transfer`; fill it with the episode's own
    # diagonal so the vector is finite and printable.
    baseline[0] = matrix[0, 0]
    for j in range(1, n):
        baseline[j] = float(np.mean(matrix[:j, j]))
    return baseline


def summarise(
    R, baseline: Optional[Sequence[float]] = None, decay: float = 0.0
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
            collapsing the drift curve to a single number. ``0`` is a plain mean.

    Returns:
        Dict with ``n_episodes``, ``average_accuracy``, ``backward_transfer``,
        ``forward_transfer``, ``forward_transfer_baseline``,
        ``per_episode_forgetting``, ``drift_curve`` (row 0 of ``R``) and
        ``drift_curve_avg``. ``nan`` cells propagate rather than being dropped,
        so an incomplete matrix is visible instead of silently averaged away.
    """
    matrix = np.asarray(R, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"`R` must be a square 2-D matrix, got shape {matrix.shape}")

    if baseline is None:
        base = _diagonal_free_baseline(matrix)
    else:
        base = np.asarray(baseline, dtype=float)
    drift_curve = matrix[0, :]

    return {
        "n_episodes": int(matrix.shape[0]),
        "average_accuracy": average_accuracy(matrix),
        "backward_transfer": backward_transfer(matrix, higher_is_better=True),
        "forward_transfer": forward_transfer(matrix, base, higher_is_better=True),
        "forward_transfer_baseline": [float(v) for v in base],
        "per_episode_forgetting": [
            float(v) for v in per_episode_forgetting(matrix, higher_is_better=True)
        ],
        "drift_curve": [float(v) for v in drift_curve],
        "drift_curve_avg": exp_decay_avg(drift_curve, decay=decay),
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
        "--decay",
        type=float,
        default=0.0,
        help="Decay for the drift-curve average; 0 is a plain mean.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write R (CSV) and the summary (JSON) next to this path prefix.",
    )
    return parser.parse_args(argv)


def _load_task_context(dataset: str, task_name: str):
    """Task metadata, episode boundaries and the task table's own columns, imported lazily.

    relbench and the experiment package pull in torch and the dataset cache, so
    importing them at module scope would make the pure functions above
    untestable without the full environment.
    """
    from relbench.tasks import get_task

    from experiments.continuous_learning.continuous_task import ContinuousWrapper

    task = get_task(dataset, task_name)
    wrapper = ContinuousWrapper(task)
    # The full table's columns are exactly the non-prediction columns of the CSV:
    # run_predictions.py seeds the file with `wrapped_task.full_table.df` and only
    # ever appends checkpoint columns to it.
    return task, wrapper.get_splits(), list(wrapper.full_table.df.columns)


def main(argv=None) -> int:
    args = parse_args(argv)

    predictions = pd.read_csv(args.predictions)
    task, raw_splits, table_cols = _load_task_context(args.dataset, args.task)

    target_col = args.target_col or task.target_col
    time_col = args.time_col or task.time_col
    predictions[time_col] = pd.to_datetime(predictions[time_col])

    # Everything the task table carries -- entity id included -- is data, never a
    # checkpoint. The overridable --target-col/--time-col are added because they
    # may name a column the task table itself does not list.
    data_cols = set(map(str, table_cols)) | {str(target_col), str(time_col)}

    boundaries = episode_boundaries_from_splits(raw_splits)
    columns = select_checkpoint_columns(
        checkpoint_column_candidates(predictions.columns, data_cols),
        aggregate=args.aggregate,
    )
    # A chain that stopped early leaves fewer checkpoints than episodes. Keep the
    # leading episodes those checkpoints line up with rather than failing, but say
    # so loudly -- the tail of the drift curve is being dropped.
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
    summary = summarise(matrix, decay=args.decay)

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
        "drift_curve_avg",
    ):
        print(f"{key:>20}: {summary[key]:.4f}")
    print(f"{'drift_curve':>20}: {np.round(summary['drift_curve'], 4).tolist()}")
    print(f"{'forgetting':>20}: {np.round(summary['per_episode_forgetting'], 4).tolist()}")

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
