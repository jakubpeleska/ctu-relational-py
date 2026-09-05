r"""Per-episode statistics of the target distribution, and how far it drifts.

The benchmark paper claims that "temporal concept drift occurs in the majority of
predictive tasks", but the evidence behind that sentence was a notebook plotting
*cumulative* target statistics: each point summarised everything up to time
``t``, so a cumulative mean necessarily flattens out and a genuinely abrupt shift
in one window is diluted by every window before it. The follow-up claim is
stronger and more specific -- drift is task-**heterogeneous**, some tasks are
essentially stationary while others move sharply -- and that claim can only be
supported by *per-episode* statistics on the exact same episode boundaries the
continual-learning runs use. Hence this module rather than another notebook.

Everything here measures the **target marginal** ``P(y)``. That is a deliberate
and limited choice, spelled out again in :func:`target_drift`: a moving ``P(y)``
is *prior probability shift*, which is neither necessary nor sufficient for the
*concept drift* ``P(y | x)`` that actually breaks a fitted model. The two are
routinely conflated; the names in this module are chosen so that a plot made from
it cannot be captioned as evidence of the wrong one.

The pipeline is three steps:

1. :func:`episode_of_row` labels every row with the episode it falls in, using
   the same ``splits`` list that ``ContinuousWrapper.get_splits()`` produces.
2. :func:`per_episode_target_stats` reduces the targets to one row per episode.
3. :func:`target_drift` turns those rows into a single scale-free series that is
   comparable across tasks with different units, :func:`class_prior_drift` does
   the same for a multiclass prior -- which has no single signed shift, so it is
   summarised by the Jensen-Shannon divergence of its class-frequency vector --
   and :func:`population_stability_index` compares two *samples* directly when
   the whole shape of the distribution matters, not just its first moment.
"""

from typing import Sequence, Union

import numpy as np
import pandas as pd

__all__ = [
    "episode_of_row",
    "per_episode_target_stats",
    "target_drift",
    "class_prior_divergence",
    "class_prior_drift",
    "population_stability_index",
]

REGRESSION = "regression"
BINARY_CLASSIFICATION = "binary_classification"
MULTICLASS_CLASSIFICATION = "multiclass_classification"

_SUPPORTED_TASK_TYPES = (REGRESSION, BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION)

# Class-frequency columns are prefixed so that a stats frame stays self-describing
# once it is merged with frames from other tasks or written to disk.
FREQ_PREFIX = "freq_"


def _normalise_task_type(task_type) -> str:
    r"""Accept either a plain string or a ``relbench`` ``TaskType`` enum member."""
    # TaskType is a plain Enum, not a StrEnum, so `str(TaskType.REGRESSION)` gives
    # "TaskType.REGRESSION"; the value is the string this module documents.
    value = getattr(task_type, "value", task_type)
    if not isinstance(value, str):
        raise ValueError(f"`task_type` must be a string, got {type(task_type).__name__}")
    value = value.strip().lower()
    if value not in _SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"unsupported `task_type` {value!r}; expected one of {_SUPPORTED_TASK_TYPES}"
        )
    return value


def _to_ordinal(values, name: str) -> np.ndarray:
    r"""Map datetime-like or numeric values onto a common comparable axis.

    Timestamps become integer nanoseconds since the epoch so that a
    :class:`~pandas.DatetimeIndex`, a datetime ``Series`` and a list of
    ``Timestamp`` objects all reduce to the same array and can be compared
    against each other with :func:`numpy.searchsorted`. Integer input stays
    integral for the same reason: nanosecond epochs are far above the 53-bit
    mantissa of a ``float64``, so rounding them would move rows across episode
    boundaries.
    """
    array = values if isinstance(values, (pd.Series, pd.Index)) else np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"`{name}` must be 1-D, got shape {array.shape}")
    if len(array) == 0:
        return np.empty(0, dtype=np.int64)

    index = array if isinstance(array, pd.Index) else pd.Index(array)
    if isinstance(index, pd.DatetimeIndex):
        return index.asi8.astype(np.int64)
    if pd.api.types.is_numeric_dtype(index) and not pd.api.types.is_bool_dtype(index):
        values = index.to_numpy()
        # Integers must not be routed through float64: its mantissa holds only 53
        # bits, so two integers above 2**53 that straddle an episode boundary
        # round to the same float and the row lands in the wrong window, silently
        # breaking the half-open [start, end) contract of `episode_of_row`.
        # Nanosecond epochs are ~2**62, i.e. squarely in that range.
        if values.dtype.kind == "i":
            return values.astype(np.int64, copy=False)
        if values.dtype.kind == "u":
            # An unsigned value above the int64 range would wrap to a negative one
            # under the cast, reordering the axis instead of merely blurring it.
            if values.max() > np.iinfo(np.int64).max:
                raise ValueError(
                    f"`{name}` holds unsigned values above the int64 range, "
                    "which cannot be ordered on a signed axis"
                )
            return values.astype(np.int64)
        return values.astype(np.float64)
    raise ValueError(f"`{name}` must be datetime-like or numeric, got dtype {index.dtype}")


def episode_of_row(timestamps, splits) -> np.ndarray:
    r"""Label every row with the episode whose window contains its timestamp.

    Episode ``i`` covers the half-open window ``[splits[i], splits[i + 1])``, so a
    row landing exactly on a boundary belongs to the episode that *starts* there.
    ``splits`` is the list returned by ``ContinuousWrapper.get_splits()``, which
    has one more entry than there are episodes.

    Rows before ``splits[0]`` or at/after ``splits[-1]`` get ``-1``: they belong
    to no episode and every downstream function drops them, rather than being
    silently folded into the first or last window.

    Args:
        timestamps: Row timestamps. A ``DatetimeIndex``, a datetime ``Series``, a
            ``datetime64`` array, a list of ``Timestamp`` objects, or plain
            numbers. Need not be sorted.
        splits: Strictly increasing episode boundaries, at least two of them.

    Returns:
        Array of ``int64`` episode indices, one per row, in ``[-1, len(splits) - 1)``.

    Raises:
        ValueError: If ``splits`` has fewer than two entries, is not strictly
            increasing, or the inputs are neither datetime-like nor numeric.
    """
    edges = _to_ordinal(splits, "splits")
    if edges.size < 2:
        raise ValueError(f"`splits` must contain at least two boundaries, got {edges.size}")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("`splits` must be strictly increasing")

    times = _to_ordinal(timestamps, "timestamps")
    if times.size == 0:
        return np.empty(0, dtype=np.int64)

    # side="right" puts a value equal to a boundary into the window that starts at
    # it, which is exactly the half-open [start, end) convention above.
    episodes = np.searchsorted(edges, times, side="right").astype(np.int64) - 1
    # searchsorted cannot distinguish "before the first edge" (-> -1 already) from
    # "at or after the last edge" (-> len(edges) - 1), so clamp the tail by hand.
    episodes[episodes >= edges.size - 1] = -1
    return episodes


def _validated_pairs(targets, episodes):
    r"""Align targets with episode labels and drop rows belonging to no episode."""
    values = np.asarray(targets)
    if values.ndim != 1:
        raise ValueError(f"`targets` must be 1-D, got shape {values.shape}")
    labels = np.asarray(episodes)
    if labels.ndim != 1:
        raise ValueError(f"`episodes` must be 1-D, got shape {labels.shape}")
    if labels.shape[0] != values.shape[0]:
        raise ValueError(
            f"`episodes` has {labels.shape[0]} rows but `targets` has {values.shape[0]}"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        labels = labels.astype(np.int64)
    keep = labels >= 0
    return values[keep], labels[keep].astype(np.int64)


def per_episode_target_stats(targets, episodes, task_type) -> pd.DataFrame:
    r"""Summarise the target distribution separately within each episode.

    Unlike the cumulative curves this replaces, every row depends only on the rows
    of its own episode, so an abrupt shift shows up as an abrupt step.

    Args:
        targets: Target value per row.
        episodes: Episode index per row, as returned by :func:`episode_of_row`.
            Rows labelled ``-1`` are excluded.
        task_type: ``"regression"``, ``"binary_classification"`` or
            ``"multiclass_classification"`` (a ``relbench`` ``TaskType`` member is
            also accepted).

    Returns:
        DataFrame indexed by episode, ascending, with columns

        * regression -- ``n``, ``mean``, ``std``, ``min``, ``max``, ``median``.
          ``std`` is the *population* standard deviation (``ddof=0``) so that a
          one-row episode reports ``0.0`` instead of ``NaN``, and so that it is a
          usable denominator in :func:`target_drift`.
        * binary classification -- ``n``, ``positive_rate``.
        * multiclass classification -- ``n`` plus one ``freq_<class>`` column per
          class observed anywhere in the input; frequencies sum to 1 per row and a
          class missing from an episode is ``0.0``.

        Episodes with no rows are **absent** from the index rather than present as
        ``NaN`` rows, so ``len(stats)`` is the number of non-empty episodes and no
        caller has to guess whether a ``NaN`` means "empty" or "degenerate".

    Raises:
        ValueError: If the inputs disagree in length, the task type is unknown, or
            a binary target holds values other than 0 and 1.
    """
    kind = _normalise_task_type(task_type)
    values, labels = _validated_pairs(targets, episodes)

    if kind == MULTICLASS_CLASSIFICATION:
        if values.size == 0:
            return pd.DataFrame(
                {"n": pd.Series(dtype=np.int64)},
                index=pd.Index([], dtype=np.int64, name="episode"),
            )
        counts = pd.crosstab(
            pd.Series(labels, name="episode"), pd.Series(values, name="class")
        )
        totals = counts.sum(axis=1)
        stats = counts.div(totals, axis=0)
        stats.columns = [f"{FREQ_PREFIX}{column}" for column in counts.columns]
        stats.insert(0, "n", totals.astype(np.int64))
        stats.index = stats.index.astype(np.int64)
        stats.index.name = "episode"
        stats.columns.name = None
        return stats.sort_index()

    numeric = values.astype(np.float64)
    if kind == BINARY_CLASSIFICATION and not np.all(np.isin(numeric, (0.0, 1.0))):
        offending = np.unique(numeric[~np.isin(numeric, (0.0, 1.0))])[:5]
        raise ValueError(f"binary `targets` must be 0 or 1, found {offending.tolist()}")

    frame = pd.DataFrame({"episode": labels, "target": numeric})
    grouped = frame.groupby("episode")["target"]
    if kind == BINARY_CLASSIFICATION:
        stats = pd.DataFrame(
            {"n": grouped.size().astype(np.int64), "positive_rate": grouped.mean()}
        )
    else:
        stats = pd.DataFrame(
            {
                "n": grouped.size().astype(np.int64),
                "mean": grouped.mean(),
                "std": grouped.std(ddof=0),
                "min": grouped.min(),
                "max": grouped.max(),
                "median": grouped.median(),
            }
        )
    stats.index = stats.index.astype(np.int64)
    stats.index.name = "episode"
    return stats.sort_index()


def target_drift(stats: pd.DataFrame, kind) -> pd.Series:
    r"""Per-episode change of the target marginal relative to the first episode.

    **This measures ``P(y)``, not ``P(y | x)``.** It quantifies *non-stationarity
    of the target marginal* -- prior probability shift -- and says nothing about
    whether the input-to-target relationship changed. A task can score zero here
    and still be devastating for a fitted model (``P(y | x)`` rotates while
    ``P(y)`` stays put), and it can score high while remaining perfectly learnable
    (the label prior drifts but the concept does not). Treat the output as
    evidence that the *data* moved, and never as evidence of concept drift; the
    conflation of the two is exactly what this module refuses to repeat.

    Both variants are referenced to episode 0 and normalised so that magnitudes
    are comparable across tasks whose targets have different units:

    * regression -- standardised mean shift ``(mean_i - mean_0) / std_0``, i.e.
      the move of the mean expressed in first-episode standard deviations.
    * binary classification -- ``positive_rate_i - positive_rate_0``, already
      unit-free and bounded in ``[-1, 1]``.

    The sign is retained: negative means the target moved *down* relative to the
    first episode. Take :func:`abs` for a magnitude-only ranking of tasks.

    A multiclass prior has no single signed shift, so this function rejects it;
    :func:`class_prior_drift` is its counterpart there. Note that
    :func:`population_stability_index` is *not* the tool for that job, however
    natural it looks: PSI consumes a **sample** and bins it, so handing it two
    class-frequency *vectors* compares them as two tiny samples of numbers.
    ``{.6, .3, .1}`` and ``{.1, .3, .6}`` are the same multiset, so a total
    reversal of the class priors scores exactly ``0.0``. PSI over the raw class
    **labels** of the two episodes does work, and is what to reach for when the
    per-row labels are still at hand rather than only their frequencies.

    Args:
        stats: Frame from :func:`per_episode_target_stats`. Its first row is the
            reference, so it must be the earliest episode present (it is, since
            that function sorts by episode).
        kind: ``"regression"`` or ``"binary_classification"``.

    Returns:
        Series indexed like ``stats``, whose first entry is ``0.0`` by
        construction.

    Raises:
        ValueError: If ``stats`` is empty, lacks the columns the ``kind``
            requires, or ``kind`` is multiclass -- use :func:`class_prior_drift`
            for that case.
    """
    measure = _normalise_task_type(kind)
    if measure == MULTICLASS_CLASSIFICATION:
        raise ValueError(
            "multiclass targets have no single signed mean shift; measure the "
            "movement of the class prior with `class_prior_drift` instead "
            "(`population_stability_index` bins samples, not prior vectors, and "
            "reports 0.0 for a reversed prior)"
        )
    if len(stats) == 0:
        raise ValueError("`stats` is empty; nothing to reference the drift against")

    if measure == REGRESSION:
        missing = {"mean", "std"} - set(stats.columns)
        if missing:
            raise ValueError(f"`stats` is missing regression columns {sorted(missing)}")
        means = stats["mean"].astype(float)
        reference_std = float(stats["std"].iloc[0])
        # A first episode with zero (or non-finite) spread gives no scale to divide
        # by. Falling back to 1.0 keeps a constant target at exactly zero drift and
        # keeps the series finite; the values are then a raw shift, not a
        # standardised one, so they are no longer comparable across tasks.
        scale = reference_std if np.isfinite(reference_std) and reference_std > 0 else 1.0
        drift = (means - means.iloc[0]) / scale
        drift.name = "standardised_mean_shift"
        return drift

    if "positive_rate" not in stats.columns:
        raise ValueError("`stats` is missing the `positive_rate` column")
    rates = stats["positive_rate"].astype(float)
    drift = rates - rates.iloc[0]
    drift.name = "positive_rate_shift"
    return drift


def _as_prior(weights, name: str) -> np.ndarray:
    r"""Normalise a vector of non-negative weights into a probability vector.

    Raw counts are accepted as well as frequencies, so a ``freq_*`` row and a
    ``value_counts()`` of the same episode give the same answer.
    """
    vector = np.asarray(weights, dtype=float).ravel()
    if vector.size == 0:
        raise ValueError(f"`{name}` must contain at least one class")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"`{name}` must be finite, got {vector.tolist()[:5]}")
    if np.any(vector < 0):
        raise ValueError(f"`{name}` must be non-negative, got {vector.tolist()[:5]}")
    total = vector.sum()
    if total <= 0:
        raise ValueError(f"`{name}` sums to {total}; a class prior needs positive mass")
    return vector / total


def _kl_bits(p: np.ndarray, q: np.ndarray) -> float:
    r"""``KL(p || q)`` in bits, skipping the zeros of ``p``.

    ``0 * log(0 / q)`` is ``0`` by continuity, and this is only ever called with
    ``q`` the mixture of ``p`` with something else, so ``q`` is strictly positive
    wherever ``p`` is and no term can be infinite.
    """
    support = p > 0
    return float(np.sum(p[support] * np.log2(p[support] / q[support])))


def class_prior_divergence(reference, current) -> float:
    r"""Jensen-Shannon divergence between two class-frequency vectors, in bits.

    ``JSD(r, c) = (KL(r || m) + KL(c || m)) / 2`` with ``m = (r + c) / 2``. This
    is the measure to use on the ``freq_*`` columns of
    :func:`per_episode_target_stats`, where :func:`population_stability_index`
    is actively wrong: PSI bins a *sample*, so it reads a prior vector as a
    handful of numbers and scores the total reversal ``{.6, .3, .1}`` ->
    ``{.1, .3, .6}`` as ``0.0``, the two vectors being the same multiset.

    Jensen-Shannon rather than the symmetric KL (Jeffreys) divergence that PSI
    approximates, because a class that is absent from one of the two episodes --
    routine early or late in a stream -- makes Jeffreys infinite and drags any
    average over episodes to infinity with it. Mixing with ``m`` bounds every
    term, so the result is always finite and, in bits, lands in ``[0, 1]``:
    ``0`` for identical priors, ``1`` for priors with disjoint support.

    It is a divergence, not a signed shift: it says *how far* the prior moved,
    never in which direction, because "direction" is not defined for more than
    two unordered classes. And like everything here it describes ``P(y)`` only,
    never ``P(y | x)``.

    Args:
        reference: Baseline class weights, e.g. the ``freq_*`` row of the first
            episode. Counts are fine; they are normalised to sum to 1.
        current: Class weights to compare, over the **same classes in the same
            order** -- the ``freq_*`` columns of one stats frame guarantee that.

    Returns:
        The divergence in bits, in ``[0, 1]``. Exactly ``0.0`` for two identical
        priors.

    Raises:
        ValueError: If the two vectors differ in length, are empty, hold
            negative or non-finite weights, or sum to zero.
    """
    ref = _as_prior(reference, "reference")
    cur = _as_prior(current, "current")
    if ref.size != cur.size:
        raise ValueError(
            f"`reference` has {ref.size} classes but `current` has {cur.size}; "
            "the two vectors must cover the same classes in the same order"
        )

    mixture = 0.5 * (ref + cur)
    divergence = 0.5 * (_kl_bits(ref, mixture) + _kl_bits(cur, mixture))
    # The quantity is non-negative by construction, but two priors that differ
    # only by rounding can leave a tiny negative sum; clamping keeps the
    # documented range instead of leaking a -1e-17 into a plot.
    return float(max(divergence, 0.0))


def class_prior_drift(stats: pd.DataFrame) -> pd.Series:
    r"""Per-episode movement of a multiclass prior relative to the first episode.

    The multiclass counterpart of :func:`target_drift`: same shape of output,
    same reference episode, but the value is the unsigned
    :func:`class_prior_divergence` of that episode's class frequencies against
    the first episode's, since a prior over unordered classes has no direction to
    report. Being bounded in ``[0, 1]`` it is comparable across tasks with
    different numbers of classes.

    Args:
        stats: Frame from :func:`per_episode_target_stats` for a multiclass task,
            i.e. one carrying ``freq_*`` columns.

    Returns:
        Series indexed like ``stats``, whose first entry is ``0.0`` by
        construction, named ``class_prior_js_divergence``.

    Raises:
        ValueError: If ``stats`` is empty or carries no ``freq_*`` columns (the
            frame of a regression or binary task, most likely).
    """
    if len(stats) == 0:
        raise ValueError("`stats` is empty; nothing to reference the drift against")
    columns = [column for column in stats.columns if str(column).startswith(FREQ_PREFIX)]
    if not columns:
        raise ValueError(
            f"`stats` has no `{FREQ_PREFIX}*` columns; `class_prior_drift` needs the "
            "frame of a multiclass task"
        )

    frequencies = stats[columns].to_numpy(dtype=float)
    reference = frequencies[0]
    values = [class_prior_divergence(reference, row) for row in frequencies]
    return pd.Series(values, index=stats.index, name="class_prior_js_divergence")


def _interior_edges(ref: np.ndarray, cur: np.ndarray, bins: int) -> np.ndarray:
    r"""Interior bin edges for a PSI comparison, cut from ``ref`` where possible.

    Quantile edges are the textbook choice, and they silently collapse on a lumpy
    reference: every decile of ``[0] * 95 + [1] * 5`` is ``0.0``, so a single
    interior edge survives deduplication and the open top bin ``[0, +inf)``
    swallows the reference's zeros *and* its ones *and* everything in the current
    sample. Two samples whose class balance could not differ more then report
    identical shares and PSI reads ``0.0``. Imbalanced binary and other
    small-cardinality targets are the main thing this module compares, so this
    path has to be correct rather than merely non-crashing.

    The fallback bins on distinct **values** instead, taken from both samples
    rather than from ``ref`` alone: a constant reference offers no edge of its
    own, which is why ``psi([3] * 10, [9] * 10)`` used to read ``0.0`` while
    ``psi([3] * 10, [-9999] * 10)`` read ``27.6`` -- the same move, scored by
    which side of the reference it happened on.

    Args:
        ref: Finite reference sample.
        cur: Finite current sample.
        bins: Requested number of bins, at least 1.

    Returns:
        Strictly increasing interior edges: at most ``bins - 1`` of them, and
        possibly none, which happens only when both samples hold a single shared
        value and one bin really is the whole story.
    """
    quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    interior = np.unique(np.quantile(ref, quantiles))
    # Counting edges is not enough: an edge sitting at the reference minimum splits
    # nothing, because every reference value lands on or above it. At bins=2 -- the
    # value a caller would naturally pass for a binary target -- a single such edge
    # satisfies a count-only guard, so the degenerate case slips straight through
    # and PSI reports 0.0 for a 5% -> 100% shift.
    if interior.size >= quantiles.size and (
        interior.size == 0 or interior[0] > ref.min()
    ):
        return interior  # the reference supports the resolution that was asked for

    distinct = np.unique(np.concatenate((ref, cur)))
    if distinct.size <= bins:
        # One bin per distinct value: a midpoint between every pair of adjacent
        # values separates everything either sample actually takes. Written as
        # `a + (b - a) / 2` rather than `(a + b) / 2` so that two large values
        # cannot overflow to inf on the way.
        return np.unique(distinct[:-1] + (distinct[1:] - distinct[:-1]) / 2.0)
    # More distinct values than bins, but a reference too lumpy to cut into
    # quantiles -- a heavily tied reference against a spread-out current sample.
    # Quantiling the distinct values keeps the requested resolution: they are
    # strictly increasing, so these edges cannot collapse the way the ones above
    # did.
    return np.unique(np.quantile(distinct, quantiles))


def population_stability_index(
    reference, current, bins: Union[int, Sequence[float]] = 10, epsilon: float = 1e-6
) -> float:
    r"""Population Stability Index between two samples of the same quantity.

    ``PSI = sum_b (c_b - r_b) * ln(c_b / r_b)`` over bins ``b``, where ``r`` and
    ``c`` are the reference and current proportions falling in each bin. It is the
    symmetrised KL divergence (Jeffreys divergence) of the two binned
    distributions, and unlike :func:`target_drift` it reacts to a change in *any*
    part of the distribution, not only its mean.

    Conventional reading, from credit-risk monitoring where the measure comes
    from: ``< 0.1`` no meaningful shift, ``0.1 - 0.25`` moderate shift worth
    investigating, ``> 0.25`` major shift. Those thresholds are rules of thumb
    tuned for ~10 bins and thousands of rows, not test statistics.

    Like everything else in this module it compares *marginals*. Applied to
    targets it detects prior shift; applied to a feature it detects covariate
    shift; neither is concept drift.

    Args:
        reference: Baseline sample, e.g. the first episode's targets.
        current: Sample to compare against the baseline.
        bins: Number of quantile bins cut from ``reference`` (default 10, the
            usual decile binning), or an explicit sequence of interior bin edges.
            A reference too lumpy to supply that many distinct quantiles -- any
            binary or small-cardinality target -- is binned on the distinct
            *values* of both samples instead, because collapsed quantile edges
            otherwise hide a total change of class balance behind one open bin.
            Passing shared explicit edges makes the result exactly symmetric in
            its two arguments.
        epsilon: Floor applied to both proportions. Without it an empty bin gives
            ``ln(0)`` and the index is ``inf``, which destroys any average taken
            over tasks; with it an empty bin contributes a large but finite term.

    Returns:
        The index, always finite and ``>= 0``. Exactly ``0.0`` when both samples
        put identical proportions in every bin.

    Raises:
        ValueError: If either sample is empty after dropping non-finite values,
            ``bins`` is smaller than 1, or ``epsilon`` is not positive.
    """
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        raise ValueError("`reference` and `current` must both hold finite values")
    if epsilon <= 0:
        raise ValueError(f"`epsilon` must be positive, got {epsilon}")

    if isinstance(bins, (int, np.integer)):
        if bins < 1:
            raise ValueError(f"`bins` must be at least 1, got {bins}")
        interior = _interior_edges(ref, cur, int(bins))
    else:
        interior = np.unique(np.asarray(bins, dtype=float).ravel())
        if not np.all(np.isfinite(interior)):
            raise ValueError("explicit bin edges must be finite")

    # Open the outer bins so that values outside the reference range still land
    # somewhere instead of being dropped, which would understate the shift.
    edges = np.concatenate(([-np.inf], interior, [np.inf]))
    ref_share = np.histogram(ref, bins=edges)[0] / ref.size
    cur_share = np.histogram(cur, bins=edges)[0] / cur.size

    ref_share = np.clip(ref_share, epsilon, None)
    cur_share = np.clip(cur_share, epsilon, None)
    # `(c - r) * (log c - log r)` rather than the textbook `(c - r) * log(c / r)`:
    # the two are the same quantity, but swapping the samples negates both factors
    # of this form exactly in IEEE arithmetic, so shared explicit edges give a
    # bitwise symmetric result rather than one that agrees to a few ulps.
    return float(np.sum((cur_share - ref_share) * (np.log(cur_share) - np.log(ref_share))))
