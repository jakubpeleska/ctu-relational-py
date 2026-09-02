r"""Metrics for evaluating a model across a sequence of learning episodes.

Two families live here.

**Continual-learning metrics** operate on an *evaluation matrix* ``R``, the
standard object in the CL literature (Lopez-Paz & Ranzato, 2017): ``R[i, j]`` is
the score of the model obtained after training on episode ``i``, evaluated on
episode ``j``. The diagonal is "how well did I learn this episode", the lower
triangle is "how much did I keep", the upper triangle is "what did I already
know". :func:`evaluation_matrix_from_predictions` builds ``R`` from the
per-checkpoint predictions that ``run_predictions.py`` writes.

**The decay metric** (:func:`exp_decay_avg`) summarises a sequence of per-window
scores while down-weighting windows further from the deployment horizon. It is a
*forward* decay over a fixed known horizon, in contrast to the *backward* decay
over an open stream used by Gama et al. (2013) and Hidalgo et al. (2019).

Every function takes ``higher_is_better`` where the direction matters, because
tasks here mix ROC-AUC (higher) with MAE (lower).
"""

from typing import Callable, Optional, Sequence

import numpy as np

__all__ = [
    "exp_decay_avg",
    "average_accuracy",
    "backward_transfer",
    "forward_transfer",
    "per_episode_forgetting",
    "evaluation_matrix_from_predictions",
]


def exp_decay_avg(
    metrics: Sequence[float],
    counts: Optional[Sequence[float]] = None,
    decay: float = 0.0,
) -> float:
    r"""Exponentially decayed weighted mean of per-window scores.

    Window ``k`` receives weight ``(1 - decay) ** k`` multiplied by ``counts[k]``,
    so index 0 is weighted most. ``decay=0`` reduces to the plain count-weighted
    mean.

    Args:
        metrics: Per-window scores, ordered oldest window first.
        counts: Optional per-window row counts, to weight windows by size.
        decay: Decay rate in ``[0, 1)``. ``0`` disables decay.

    Returns:
        The weighted average.

    Raises:
        ValueError: If ``metrics`` is empty, lengths disagree, ``decay`` is
            outside ``[0, 1)``, or the weights sum to zero.
    """
    values = np.asarray(metrics, dtype=float)
    if values.size == 0:
        raise ValueError("`metrics` must contain at least one value")
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"`decay` must be in [0, 1), got {decay}")

    if counts is None:
        weights_counts = np.ones_like(values)
    else:
        weights_counts = np.asarray(counts, dtype=float)
        if weights_counts.shape != values.shape:
            raise ValueError(
                f"`counts` has shape {weights_counts.shape}, expected {values.shape}"
            )
        if np.any(weights_counts < 0):
            raise ValueError("`counts` must be non-negative")

    weights = np.power(1.0 - decay, np.arange(values.size)) * weights_counts
    total = weights.sum()
    if total <= 0:
        raise ValueError("weights sum to zero; check `counts` and `decay`")
    return float(np.average(values, weights=weights))


def _as_matrix(R) -> np.ndarray:
    matrix = np.asarray(R, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"`R` must be a square 2-D matrix, got shape {matrix.shape}")
    if matrix.shape[0] < 1:
        raise ValueError("`R` must have at least one episode")
    return matrix


def average_accuracy(R) -> float:
    r"""Mean score of the final model over every episode seen so far.

    This is ``ACC`` in Lopez-Paz & Ranzato: the mean of the last row of ``R``.
    Reported in the task's own metric, so the direction follows that metric.
    """
    matrix = _as_matrix(R)
    return float(np.mean(matrix[-1, :]))


def backward_transfer(R, higher_is_better: bool = True) -> float:
    r"""How much learning later episodes changed performance on earlier ones.

    ``BWT = mean_{j < T} (R[T, j] - R[j, j])``, sign-corrected so that a
    **negative** value always means forgetting regardless of metric direction.

    Returns ``0.0`` for a single-episode matrix, where the quantity is undefined.
    """
    matrix = _as_matrix(R)
    n = matrix.shape[0]
    if n < 2:
        return 0.0
    last = n - 1
    deltas = [matrix[last, j] - matrix[j, j] for j in range(last)]
    signed = np.asarray(deltas) if higher_is_better else -np.asarray(deltas)
    return float(np.mean(signed))


def forward_transfer(R, baseline: Sequence[float], higher_is_better: bool = True) -> float:
    r"""How much prior episodes helped on episodes not yet trained on.

    ``FWT = mean_{j > 0} (R[j - 1, j] - baseline[j])``, sign-corrected so that a
    **positive** value always means useful transfer. ``baseline[j]`` is the score
    of an independently initialised model on episode ``j``.

    Returns ``0.0`` for a single-episode matrix.
    """
    matrix = _as_matrix(R)
    n = matrix.shape[0]
    base = np.asarray(baseline, dtype=float)
    if base.shape != (n,):
        raise ValueError(f"`baseline` must have length {n}, got {base.shape}")
    if n < 2:
        return 0.0
    deltas = [matrix[j - 1, j] - base[j] for j in range(1, n)]
    signed = np.asarray(deltas) if higher_is_better else -np.asarray(deltas)
    return float(np.mean(signed))


def per_episode_forgetting(R, higher_is_better: bool = True) -> np.ndarray:
    r"""Forgetting on each episode: its best past score minus its final score.

    ``f_j = max_{i in [j, T)} R[i, j] - R[T, j]`` for ``j < T``, sign-corrected so
    that **larger is worse** for both metric directions. Negative values mean the
    episode ended better than it ever was before, i.e. backward transfer.

    Returns:
        Array of length ``T`` (number of episodes minus one), one entry per
        episode excluding the last. Empty for a single-episode matrix.
    """
    matrix = _as_matrix(R)
    n = matrix.shape[0]
    if n < 2:
        return np.empty(0, dtype=float)
    last = n - 1
    work = matrix if higher_is_better else -matrix
    return np.asarray(
        [float(np.max(work[j:last, j]) - work[last, j]) for j in range(last)],
        dtype=float,
    )


def evaluation_matrix_from_predictions(
    predictions,
    episode_of_row,
    checkpoint_columns: Sequence[str],
    target: Sequence[float],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    r"""Assemble ``R`` from per-checkpoint predictions over the full timeline.

    ``run_predictions.py`` scores every saved checkpoint over the whole task
    table, writing one column per checkpoint. Grouping those rows by the episode
    they fall in turns that wide table into the evaluation matrix.

    Args:
        predictions: Mapping of column name to per-row predictions, or anything
            indexable by the names in ``checkpoint_columns`` (a DataFrame works).
        episode_of_row: Episode index for each row; rows with a negative index
            are ignored.
        checkpoint_columns: Column names ordered by training episode, one per
            episode.
        target: Ground-truth value per row.
        metric_fn: Called as ``metric_fn(y_true, y_pred)`` for one episode's rows.

    Returns:
        Square matrix ``R`` of shape ``(len(checkpoint_columns),) * 2``. Cells
        whose episode has no rows are ``nan``.
    """
    episodes = np.asarray(episode_of_row)
    y_true = np.asarray(target, dtype=float)
    if episodes.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"`episode_of_row` has {episodes.shape[0]} rows but `target` has {y_true.shape[0]}"
        )

    n = len(checkpoint_columns)
    matrix = np.full((n, n), np.nan, dtype=float)
    for i, column in enumerate(checkpoint_columns):
        preds = np.asarray(predictions[column], dtype=float)
        if preds.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"column {column!r} has {preds.shape[0]} rows, expected {y_true.shape[0]}"
            )
        for j in range(n):
            mask = episodes == j
            if not np.any(mask):
                continue
            matrix[i, j] = float(metric_fn(y_true[mask], preds[mask]))
    return matrix
