import numpy as np
import pandas as pd
import pytest

from redelex.continual.drift import (
    class_prior_divergence,
    class_prior_drift,
    episode_of_row,
    per_episode_target_stats,
    population_stability_index,
    target_drift,
)

# Three episodes: [Jan, Feb), [Feb, Mar), [Mar, Apr). Shaped like the list that
# ContinuousWrapper.get_splits() returns: boundaries, so one more than episodes.
SPLITS = [
    pd.Timestamp("2020-01-01"),
    pd.Timestamp("2020-02-01"),
    pd.Timestamp("2020-03-01"),
    pd.Timestamp("2020-04-01"),
]


# --- episode_of_row ---------------------------------------------------------


def test_episode_of_row_assigns_interior_rows():
    stamps = pd.to_datetime(["2020-01-15", "2020-02-15", "2020-03-15"])
    np.testing.assert_array_equal(episode_of_row(stamps, SPLITS), [0, 1, 2])


def test_episode_of_row_windows_are_half_open_at_both_ends():
    # a row exactly on splits[i] opens episode i; a row exactly on splits[i+1]
    # already belongs to episode i+1, never to both
    on_boundaries = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])
    np.testing.assert_array_equal(episode_of_row(on_boundaries, SPLITS), [0, 1, 2])


def test_episode_of_row_last_boundary_is_excluded():
    # splits[-1] closes the final window, so nothing may land on or after it
    np.testing.assert_array_equal(
        episode_of_row(pd.to_datetime(["2020-03-31", "2020-04-01", "2020-05-01"]), SPLITS),
        [2, -1, -1],
    )


def test_episode_of_row_marks_rows_before_the_first_split():
    stamps = pd.to_datetime(["2019-12-31", "2019-01-01", "2020-01-01"])
    np.testing.assert_array_equal(episode_of_row(stamps, SPLITS), [-1, -1, 0])


def test_episode_of_row_accepts_series_index_and_array_alike():
    stamps = pd.to_datetime(["2020-01-15", "2020-02-15", "2020-04-15"])
    expected = [0, 1, -1]
    np.testing.assert_array_equal(episode_of_row(pd.Series(stamps), SPLITS), expected)
    np.testing.assert_array_equal(
        episode_of_row(pd.DatetimeIndex(stamps), SPLITS), expected
    )
    np.testing.assert_array_equal(episode_of_row(stamps.to_numpy(), SPLITS), expected)
    np.testing.assert_array_equal(episode_of_row(list(stamps), SPLITS), expected)


def test_episode_of_row_accepts_a_datetime_index_of_splits():
    stamps = pd.to_datetime(["2020-01-15", "2020-02-15"])
    np.testing.assert_array_equal(episode_of_row(stamps, pd.DatetimeIndex(SPLITS)), [0, 1])


def test_episode_of_row_works_on_plain_numbers():
    np.testing.assert_array_equal(
        episode_of_row([-1, 0, 2, 3, 5, 9], [0, 3, 5]), [-1, 0, 0, 1, -1, -1]
    )


def test_episode_of_row_keeps_integer_precision_above_2_53():
    # float64 carries a 53-bit mantissa, so routing integers through it rounds
    # neighbours together: `base` and `base + 1` collapse onto the same float and
    # the row lands on the wrong side of the boundary that separates them. The
    # half-open contract has to survive integers of any magnitude.
    base = 1 << 53
    np.testing.assert_array_equal(
        episode_of_row([base, base + 1, base + 8], [0, base + 1, base + 9]), [0, 1, 1]
    )


def test_episode_of_row_is_exact_on_nanosecond_epochs():
    # the realistic form of the case above: `.value` of a Timestamp is ~2**60, so
    # floats there are 256 ns apart and a row one nanosecond before a boundary
    # would be rounded into the next episode
    boundary = pd.Timestamp("2020-02-01").value
    splits = [pd.Timestamp("2020-01-01").value, boundary, pd.Timestamp("2020-03-01").value]
    np.testing.assert_array_equal(
        episode_of_row([boundary - 1, boundary, boundary + 1], splits), [0, 1, 1]
    )


def test_episode_of_row_matches_between_timestamps_and_their_int64_epochs():
    stamps = pd.to_datetime(["2020-01-15", "2020-02-01", "2020-03-31", "2020-04-01"])
    np.testing.assert_array_equal(
        episode_of_row(stamps, SPLITS),
        episode_of_row(stamps.asi8, [split.value for split in SPLITS]),
    )


def test_episode_of_row_rejects_unsigned_values_beyond_int64():
    # an unsafe cast would wrap these to negative numbers, reordering the axis
    # rather than merely blurring it
    huge = np.array([2**63 + 1, 2**63 + 5], dtype=np.uint64)
    with pytest.raises(ValueError, match="int64 range"):
        episode_of_row(huge, [0, 10])


def test_episode_of_row_does_not_require_sorted_timestamps():
    # results are per row, so a shuffled input must come back shuffled the same way
    stamps = pd.to_datetime(["2020-03-15", "2020-01-15", "2020-02-15"])
    np.testing.assert_array_equal(episode_of_row(stamps, SPLITS), [2, 0, 1])


def test_episode_of_row_returns_integers_and_matching_length():
    out = episode_of_row(pd.to_datetime(["2020-01-15", "2020-05-15"]), SPLITS)
    assert np.issubdtype(out.dtype, np.integer)
    assert out.shape == (2,)


def test_episode_of_row_handles_empty_input():
    assert episode_of_row(pd.DatetimeIndex([]), SPLITS).shape == (0,)


def test_episode_of_row_rejects_single_boundary():
    with pytest.raises(ValueError, match="at least two"):
        episode_of_row(pd.to_datetime(["2020-01-15"]), [SPLITS[0]])


def test_episode_of_row_rejects_unsorted_splits():
    with pytest.raises(ValueError, match="strictly increasing"):
        episode_of_row([1, 2], [0, 5, 3])


def test_episode_of_row_rejects_duplicate_splits():
    # a zero-width window would be unreachable and silently swallow an episode
    with pytest.raises(ValueError, match="strictly increasing"):
        episode_of_row([1, 2], [0, 3, 3, 5])


def test_episode_of_row_rejects_non_temporal_values():
    with pytest.raises(ValueError, match="datetime-like or numeric"):
        episode_of_row(["a", "b"], [0, 1, 2])


# --- per_episode_target_stats ----------------------------------------------

# Episode 0 holds 1, 2, 3; episode 2 holds 10, 12; episode 1 is deliberately empty.
REG_TARGETS = np.array([1.0, 2.0, 3.0, 10.0, 12.0])
REG_EPISODES = np.array([0, 0, 0, 2, 2])


def test_per_episode_target_stats_regression_matches_hand_computation():
    stats = per_episode_target_stats(REG_TARGETS, REG_EPISODES, "regression")
    assert list(stats.columns) == ["n", "mean", "std", "min", "max", "median"]
    assert stats.loc[0, "n"] == 3
    assert stats.loc[0, "mean"] == pytest.approx(2.0)
    assert stats.loc[0, "std"] == pytest.approx(np.sqrt(2.0 / 3.0))  # population, ddof=0
    assert stats.loc[0, "min"] == pytest.approx(1.0)
    assert stats.loc[0, "max"] == pytest.approx(3.0)
    assert stats.loc[0, "median"] == pytest.approx(2.0)
    assert stats.loc[2, "mean"] == pytest.approx(11.0)
    assert stats.loc[2, "std"] == pytest.approx(1.0)


def test_per_episode_target_stats_omits_empty_episodes_entirely():
    # episode 1 has no rows: it must be absent, not a row of NaNs that a mean
    # over episodes would silently swallow
    stats = per_episode_target_stats(REG_TARGETS, REG_EPISODES, "regression")
    assert list(stats.index) == [0, 2]
    assert not stats.isna().any().any()


def test_per_episode_target_stats_excludes_unassigned_rows():
    targets = np.append(REG_TARGETS, 1000.0)
    episodes = np.append(REG_EPISODES, -1)
    stats = per_episode_target_stats(targets, episodes, "regression")
    assert stats["n"].sum() == 5
    assert stats["max"].max() == pytest.approx(12.0)


def test_per_episode_target_stats_single_row_episode_has_zero_std():
    stats = per_episode_target_stats([4.0], [0], "regression")
    assert stats.loc[0, "std"] == 0.0


def test_per_episode_target_stats_index_is_sorted_named_and_integral():
    stats = per_episode_target_stats([1.0, 2.0, 3.0], [2, 0, 1], "regression")
    assert stats.index.name == "episode"
    assert list(stats.index) == [0, 1, 2]
    assert np.issubdtype(stats.index.dtype, np.integer)
    assert np.issubdtype(stats["n"].dtype, np.integer)


def test_per_episode_target_stats_binary_positive_rate():
    stats = per_episode_target_stats(
        [1, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1], "binary_classification"
    )
    assert list(stats.columns) == ["n", "positive_rate"]
    assert stats.loc[0, "positive_rate"] == pytest.approx(2 / 3)
    assert stats.loc[1, "positive_rate"] == pytest.approx(0.2)
    assert list(stats["n"]) == [3, 5]


def test_per_episode_target_stats_binary_rejects_other_values():
    with pytest.raises(ValueError, match="0 or 1"):
        per_episode_target_stats([0, 1, 2], [0, 0, 0], "binary_classification")


def test_per_episode_target_stats_multiclass_frequencies_sum_to_one():
    stats = per_episode_target_stats(
        ["a", "b", "a", "c", "c"], [0, 0, 0, 1, 1], "multiclass_classification"
    )
    assert list(stats.columns) == ["n", "freq_a", "freq_b", "freq_c"]
    assert stats.loc[0, "freq_a"] == pytest.approx(2 / 3)
    assert stats.loc[0, "freq_b"] == pytest.approx(1 / 3)
    np.testing.assert_allclose(
        stats[["freq_a", "freq_b", "freq_c"]].sum(axis=1), [1.0, 1.0]
    )


def test_per_episode_target_stats_multiclass_absent_class_is_zero_not_nan():
    stats = per_episode_target_stats(
        [0, 0, 1, 2], [0, 0, 0, 1], "multiclass_classification"
    )
    # class 2 never appears in episode 0, class 0 never in episode 1
    assert stats.loc[0, "freq_2"] == 0.0
    assert stats.loc[1, "freq_0"] == 0.0
    assert not stats.isna().any().any()


def test_per_episode_target_stats_rejects_length_mismatch():
    with pytest.raises(ValueError, match="rows"):
        per_episode_target_stats([1.0, 2.0], [0], "regression")


def test_per_episode_target_stats_rejects_unknown_task_type():
    with pytest.raises(ValueError, match="unsupported"):
        per_episode_target_stats([1.0], [0], "link_prediction")


def test_per_episode_target_stats_accepts_relbench_task_type_enum():
    from relbench.base import TaskType

    stats = per_episode_target_stats(REG_TARGETS, REG_EPISODES, TaskType.REGRESSION)
    assert stats.loc[0, "mean"] == pytest.approx(2.0)


# --- target_drift -----------------------------------------------------------


def test_target_drift_is_zero_for_a_constant_target():
    # no movement at all in P(y): the whole point of the measure is that this
    # reads exactly zero rather than 0/0
    stats = per_episode_target_stats([5.0] * 6, [0, 0, 1, 1, 2, 2], "regression")
    drift = target_drift(stats, "regression")
    np.testing.assert_allclose(drift.to_numpy(), [0.0, 0.0, 0.0])


def test_target_drift_regression_is_measured_in_first_episode_sigmas():
    # episode 0 is [-1, 1]: mean 0, population std exactly 1, so the shift of a
    # later mean is that mean itself
    stats = per_episode_target_stats(
        [-1.0, 1.0, 2.0, 4.0, -3.0, -1.0], [0, 0, 1, 1, 2, 2], "regression"
    )
    drift = target_drift(stats, "regression")
    np.testing.assert_allclose(drift.to_numpy(), [0.0, 3.0, -2.0])


def test_target_drift_regression_scales_by_the_reference_spread():
    # same absolute shift, four times the reference spread -> a quarter the drift
    wide = per_episode_target_stats([-4.0, 4.0, 3.0, 5.0], [0, 0, 1, 1], "regression")
    assert target_drift(wide, "regression").iloc[1] == pytest.approx(1.0)


def test_target_drift_first_entry_is_the_reference_and_is_zero():
    stats = per_episode_target_stats(REG_TARGETS, REG_EPISODES, "regression")
    drift = target_drift(stats, "regression")
    assert drift.iloc[0] == 0.0
    assert list(drift.index) == list(stats.index)


def test_target_drift_binary_is_a_signed_change_in_positive_rate():
    stats = per_episode_target_stats(
        [1, 0, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 2, 2], "binary_classification"
    )
    # rates are 0.5, 0.75, 0.0
    np.testing.assert_allclose(
        target_drift(stats, "binary_classification").to_numpy(), [0.0, 0.25, -0.5]
    )


def test_target_drift_rejects_multiclass_and_names_a_measure_that_works():
    # the error used to send callers to `population_stability_index` on the
    # `freq_*` columns, which reports 0.0 for a reversed prior (see
    # test_psi_cannot_compare_prior_vectors); it must point at `class_prior_drift`
    stats = per_episode_target_stats([0, 1, 2], [0, 0, 1], "multiclass_classification")
    with pytest.raises(ValueError, match="class_prior_drift"):
        target_drift(stats, "multiclass_classification")


def test_target_drift_rejects_mismatched_stats_columns():
    binary = per_episode_target_stats([1, 0], [0, 0], "binary_classification")
    with pytest.raises(ValueError, match="regression columns"):
        target_drift(binary, "regression")


def test_target_drift_rejects_empty_stats():
    empty = per_episode_target_stats(np.array([]), np.array([]), "regression")
    with pytest.raises(ValueError, match="empty"):
        target_drift(empty, "regression")


# --- population_stability_index ---------------------------------------------


def _normal(shift, seed=0, size=4000):
    return np.random.default_rng(seed).normal(size=size) + shift


def test_psi_is_zero_for_the_same_sample():
    sample = _normal(0.0)
    assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-12)


def test_psi_is_near_zero_for_two_draws_from_one_distribution():
    assert population_stability_index(_normal(0.0, seed=1), _normal(0.0, seed=2)) < 0.1


def test_psi_grows_monotonically_as_the_distributions_separate():
    reference = _normal(0.0, seed=3)
    values = [
        population_stability_index(reference, reference + shift)
        for shift in (0.0, 0.25, 0.5, 1.0, 2.0)
    ]
    assert values == sorted(values)
    assert all(later > earlier for earlier, later in zip(values, values[1:]))


def test_psi_crosses_the_conventional_thresholds_in_the_expected_order():
    reference = _normal(0.0, seed=4)
    assert population_stability_index(reference, reference + 0.1) < 0.1
    assert population_stability_index(reference, reference + 1.0) > 0.25


def test_psi_is_roughly_symmetric_in_its_arguments():
    # bins are cut from the reference, so the two directions differ slightly;
    # the divergence itself is symmetric, hence "roughly"
    a, b = _normal(0.0, seed=5), _normal(0.75, seed=6)
    assert population_stability_index(a, b) == pytest.approx(
        population_stability_index(b, a), rel=0.2
    )


def test_psi_is_exactly_symmetric_with_shared_explicit_bins():
    # exact, not approximate: each term is `(c - r) * (log c - log r)`, and
    # swapping the samples negates both factors, which is exact in IEEE
    # arithmetic. Asserting equality rather than approx keeps this test
    # deterministic instead of dependent on how the terms happen to round.
    a, b = _normal(0.0, seed=7), _normal(0.75, seed=8)
    edges = [-1.0, 0.0, 1.0]
    assert population_stability_index(a, b, bins=edges) == population_stability_index(
        b, a, bins=edges
    )


def test_psi_stays_finite_when_a_bin_is_empty():
    # reference has nothing in the middle bin, current has nothing anywhere else:
    # every term would be log(0) without the epsilon floor
    reference = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    current = np.full(6, 0.5)
    value = population_stability_index(reference, current, bins=[0.25, 0.75])
    assert np.isfinite(value)
    assert value > 0.25


def test_psi_stays_finite_for_completely_disjoint_samples():
    reference = _normal(0.0, seed=9)
    value = population_stability_index(reference, reference + 1000.0)
    assert np.isfinite(value)
    assert value > 1.0


def test_psi_counts_values_outside_the_reference_range():
    # the outer bins are open, so a current sample beyond the reference support
    # must still be counted rather than dropped
    reference = np.linspace(0.0, 1.0, 100)
    inside = population_stability_index(reference, np.linspace(0.0, 1.0, 100))
    outside = population_stability_index(reference, np.linspace(5.0, 6.0, 100))
    assert outside > inside


def test_psi_ignores_nan_values():
    reference = _normal(0.0, seed=10)
    current = np.concatenate([reference, np.full(50, np.nan)])
    assert population_stability_index(reference, current) == pytest.approx(0.0, abs=1e-12)


def test_psi_bin_count_changes_resolution_not_direction():
    reference = _normal(0.0, seed=11)
    for bins in (5, 10, 20):
        assert population_stability_index(reference, reference + 1.0, bins=bins) > 0.25


def test_psi_flags_a_reversed_imbalanced_binary_reference():
    # every decile of a 5%-positive reference is 0.0, so quantile edges collapse
    # to a single interior edge and the open top bin swallows the zeros, the ones
    # and the whole current sample alike: this used to read exactly 0.0, i.e. "no
    # shift" for a positive rate going 5% -> 100%
    value = population_stability_index([0.0] * 95 + [1.0] * 5, [1.0] * 100)
    assert np.isfinite(value)
    assert value > 0.25  # "major shift" by the conventional reading


def test_psi_bins_a_binary_reference_between_its_two_labels():
    # pins the fallback: two bins split at 0.5, so the shares are (0.95, 0.05)
    # against (0, 1) and the value is the hand-computed sum
    epsilon = 1e-6
    empty_bin = (epsilon - 0.95) * (np.log(epsilon) - np.log(0.95))
    full_bin = (1.0 - 0.05) * (np.log(1.0) - np.log(0.05))
    value = population_stability_index(
        [0.0] * 95 + [1.0] * 5, [1.0] * 100, epsilon=epsilon
    )
    assert value == pytest.approx(empty_bin + full_bin)


def test_psi_is_direction_independent_for_a_constant_reference():
    # a constant reference offers no quantile edge of its own, so the comparison
    # used to depend on which side the current sample sat: moving 3 -> 9 scored
    # 0.0 while 3 -> -9999 scored 27.6
    up = population_stability_index([3.0] * 10, [9.0] * 10)
    down = population_stability_index([3.0] * 10, [-9999.0] * 10)
    assert up > 0.25
    assert up == pytest.approx(down)
    assert up == pytest.approx(population_stability_index([9.0] * 10, [3.0] * 10))


def test_psi_on_class_labels_sees_a_reversed_prior():
    # PSI does work on raw labels, because there it is binning a sample; 3 classes
    # get 3 bins and a 60/30/10 -> 10/30/60 reversal is a major shift
    value = population_stability_index(
        [0] * 60 + [1] * 30 + [2] * 10, [0] * 10 + [1] * 30 + [2] * 60
    )
    assert value == pytest.approx(np.log(6.0), rel=1e-5)  # 2 * 0.5 * ln 6
    assert value > 0.25


def test_psi_is_zero_for_identical_low_cardinality_samples():
    # the fallback must not invent drift where there is none
    imbalanced = [0.0] * 95 + [1.0] * 5
    assert population_stability_index(imbalanced, list(imbalanced)) == 0.0
    assert population_stability_index([3.0] * 10, [3.0] * 10) == 0.0


def test_psi_scales_with_the_size_of_a_binary_move():
    reference = [0] * 50 + [1] * 50
    small = population_stability_index(reference, [0] * 40 + [1] * 60)
    large = population_stability_index(reference, [0] * 10 + [1] * 90)
    assert 0.0 < small < large


def test_psi_keeps_reference_quantile_bins_when_they_do_not_degenerate():
    # the fallback is for degenerate cuts only: a well-spread reference must give
    # exactly what its own deciles give
    reference = np.linspace(0.0, 1.0, 100)
    current = np.linspace(0.3, 1.4, 100)
    deciles = list(np.quantile(reference, np.linspace(0.0, 1.0, 11)[1:-1]))
    assert population_stability_index(reference, current) == population_stability_index(
        reference, current, bins=deciles
    )


def test_psi_handles_a_lumpy_reference_against_a_spread_current():
    # 90% ties plus a continuous tail: too lumpy for deciles, too many distinct
    # values for one bin each. Still zero against itself, still major on a shift.
    rng = np.random.default_rng(21)
    lumpy = np.concatenate([np.zeros(900), rng.normal(size=100)])
    assert population_stability_index(lumpy, lumpy) == 0.0
    assert population_stability_index(lumpy, lumpy + 5.0) > 0.25


def test_psi_is_deterministic_across_repeated_calls():
    # every path must be bit-reproducible: a plot of these values is compared
    # across runs, and the binning must not depend on set or dict ordering
    cases = [
        (np.linspace(0.0, 1.0, 50), np.linspace(0.2, 1.2, 50), 10),
        ([0.0] * 95 + [1.0] * 5, [1.0] * 100, 10),
        ([3.0] * 10, [9.0] * 10, 4),
        ([0, 1, 2, 2, 1, 0], [2, 2, 2, 1, 0, 0], 10),
    ]
    for reference, current, bins in cases:
        first = population_stability_index(reference, current, bins=bins)
        assert all(
            population_stability_index(reference, current, bins=bins) == first
            for _ in range(5)
        )


def test_psi_rejects_empty_samples():
    with pytest.raises(ValueError, match="finite values"):
        population_stability_index([], [1.0, 2.0])


def test_psi_rejects_non_positive_bin_counts():
    with pytest.raises(ValueError, match="bins"):
        population_stability_index([1.0, 2.0], [1.0, 2.0], bins=0)


def test_psi_rejects_non_positive_epsilon():
    with pytest.raises(ValueError, match="epsilon"):
        population_stability_index([1.0, 2.0], [1.0, 2.0], epsilon=0.0)


# --- class_prior_divergence / class_prior_drift ------------------------------


def test_psi_cannot_compare_prior_vectors():
    # PSI bins a *sample*, so two prior vectors are read as two tiny samples of
    # numbers and {.6, .3, .1} vs {.1, .3, .6} is the same multiset: a total
    # reversal of the class priors scores exactly 0.0. This is why the multiclass
    # remedy is `class_prior_drift`, not PSI on the `freq_*` columns.
    assert population_stability_index([0.6, 0.3, 0.1], [0.1, 0.3, 0.6]) == 0.0
    assert class_prior_divergence([0.6, 0.3, 0.1], [0.1, 0.3, 0.6]) > 0.25


def test_class_prior_divergence_is_zero_for_identical_priors():
    assert class_prior_divergence([0.6, 0.3, 0.1], [0.6, 0.3, 0.1]) == 0.0
    assert class_prior_divergence([0.5, 0.5], [0.5, 0.5]) == 0.0


def test_class_prior_divergence_matches_hand_computation():
    # m = (.35, .3, .35); both KL terms equal .6 log2(.6/.35) + .1 log2(.1/.35)
    reference, current = [0.6, 0.3, 0.1], [0.1, 0.3, 0.6]
    mixture = [0.35, 0.3, 0.35]
    # the two KL terms are equal for this pair, so their mean is either one alone
    expected = sum(p * np.log2(p / m) for p, m in zip(reference, mixture))
    assert class_prior_divergence(reference, current) == pytest.approx(expected)


def test_class_prior_divergence_is_one_bit_for_disjoint_support():
    assert class_prior_divergence([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


def test_class_prior_divergence_is_symmetric():
    a, b = [0.7, 0.2, 0.1], [0.2, 0.2, 0.6]
    assert class_prior_divergence(a, b) == pytest.approx(class_prior_divergence(b, a))


def test_class_prior_divergence_stays_finite_when_a_class_is_missing():
    # the symmetric KL that PSI approximates would be infinite here, which would
    # drag any average over episodes to infinity
    value = class_prior_divergence([0.5, 0.5, 0.0], [0.0, 0.0, 1.0])
    assert np.isfinite(value)
    assert 0.0 <= value <= 1.0


def test_class_prior_divergence_normalises_counts():
    assert class_prior_divergence([6, 3, 1], [1, 3, 6]) == pytest.approx(
        class_prior_divergence([0.6, 0.3, 0.1], [0.1, 0.3, 0.6])
    )


def test_class_prior_divergence_stays_within_zero_and_one_bit():
    rng = np.random.default_rng(3)
    for _ in range(50):
        a = rng.dirichlet(np.ones(5))
        b = rng.dirichlet(np.ones(5))
        assert 0.0 <= class_prior_divergence(a, b) <= 1.0


def test_class_prior_divergence_grows_as_the_priors_separate():
    reference = [0.5, 0.5]
    values = [class_prior_divergence(reference, [p, 1.0 - p]) for p in (0.5, 0.4, 0.2, 0.0)]
    assert all(later > earlier for earlier, later in zip(values, values[1:]))


def test_class_prior_divergence_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same classes"):
        class_prior_divergence([0.5, 0.5], [0.3, 0.3, 0.4])


def test_class_prior_divergence_rejects_degenerate_vectors():
    with pytest.raises(ValueError, match="non-negative"):
        class_prior_divergence([0.5, -0.5], [0.5, 0.5])
    with pytest.raises(ValueError, match="positive mass"):
        class_prior_divergence([0.0, 0.0], [0.5, 0.5])
    with pytest.raises(ValueError, match="at least one class"):
        class_prior_divergence([], [0.5, 0.5])
    with pytest.raises(ValueError, match="finite"):
        class_prior_divergence([np.nan, 1.0], [0.5, 0.5])


def test_class_prior_drift_is_zero_at_the_reference_episode():
    stats = per_episode_target_stats(
        [0, 1, 2] * 4, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], "multiclass_classification"
    )
    drift = class_prior_drift(stats)
    assert drift.iloc[0] == 0.0
    assert list(drift.index) == list(stats.index)
    assert drift.name == "class_prior_js_divergence"
    # the two episodes hold the same balanced prior, so nothing moved
    np.testing.assert_allclose(drift.to_numpy(), [0.0, 0.0])


def test_class_prior_drift_sees_a_reversed_prior():
    # episode 0 is 3/2/1 over classes a/b/c, episode 1 is 1/2/3: PSI on the
    # `freq_*` columns reports 0.0 for exactly this frame
    targets = ["a"] * 3 + ["b"] * 2 + ["c"] + ["a"] + ["b"] * 2 + ["c"] * 3
    stats = per_episode_target_stats(
        targets, [0] * 6 + [1] * 6, "multiclass_classification"
    )
    frequencies = stats[["freq_a", "freq_b", "freq_c"]]
    assert population_stability_index(frequencies.iloc[0], frequencies.iloc[1]) == 0.0
    drift = class_prior_drift(stats)
    assert drift.iloc[1] == pytest.approx(
        class_prior_divergence(frequencies.iloc[0], frequencies.iloc[1])
    )
    assert drift.iloc[1] > 0.0


def test_class_prior_drift_rejects_a_frame_without_frequencies():
    binary = per_episode_target_stats([1, 0], [0, 0], "binary_classification")
    with pytest.raises(ValueError, match="freq_"):
        class_prior_drift(binary)


def test_class_prior_drift_rejects_empty_stats():
    empty = per_episode_target_stats(
        np.array([]), np.array([]), "multiclass_classification"
    )
    with pytest.raises(ValueError, match="empty"):
        class_prior_drift(empty)


# --- end-to-end -------------------------------------------------------------


def test_pipeline_flags_a_shifted_task_and_clears_a_stationary_one():
    # two tasks on the same timeline: one whose positive rate marches upward and
    # one that stays put. Drift must separate them, which is the heterogeneity
    # claim this module exists to support.
    stamps = pd.to_datetime(["2020-01-10"] * 4 + ["2020-02-10"] * 4 + ["2020-03-10"] * 4)
    episodes = episode_of_row(stamps, SPLITS)
    shifting = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    stationary = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0]

    shifting_drift = target_drift(
        per_episode_target_stats(shifting, episodes, "binary_classification"),
        "binary_classification",
    )
    stationary_drift = target_drift(
        per_episode_target_stats(stationary, episodes, "binary_classification"),
        "binary_classification",
    )
    assert shifting_drift.iloc[-1] == pytest.approx(1.0)
    np.testing.assert_allclose(stationary_drift.to_numpy(), [0.0, 0.0, 0.0])


def test_multiclass_pipeline_flags_a_shifted_task_and_clears_a_stationary_one():
    # the multiclass counterpart of the test above: `target_drift` has no signed
    # answer here, so the heterogeneity claim rests on `class_prior_drift`
    stamps = pd.to_datetime(["2020-01-10"] * 6 + ["2020-02-10"] * 6 + ["2020-03-10"] * 6)
    episodes = episode_of_row(stamps, SPLITS)
    # class "a" drains into class "c" across the three episodes
    shifting = list("aaaabc") + list("aabbcc") + list("abcccc")
    stationary = list("aabbcc") * 3

    shifting_drift = class_prior_drift(
        per_episode_target_stats(shifting, episodes, "multiclass_classification")
    )
    stationary_drift = class_prior_drift(
        per_episode_target_stats(stationary, episodes, "multiclass_classification")
    )
    assert shifting_drift.iloc[0] == 0.0
    # the prior keeps moving away from episode 0, and ends far from the task that
    # never moved at all
    assert shifting_drift.iloc[1] < shifting_drift.iloc[2]
    assert shifting_drift.iloc[2] > 0.2
    np.testing.assert_allclose(stationary_drift.to_numpy(), [0.0, 0.0, 0.0])


# --- regression: bins=2 reproduced the degeneracy the fallback exists to fix ---


@pytest.mark.parametrize("bins", [2, 3, 5, 10])
def test_psi_detects_an_imbalanced_binary_shift_at_every_bin_count(bins):
    # bins=2 is the value a caller would naturally reach for with a binary target,
    # and it was the one case the degeneracy guard missed: a single quantile edge
    # sitting at the reference minimum satisfies a count-only check while splitting
    # nothing, so a 5% -> 100% positive-rate shift scored 0.0.
    reference = [0.0] * 95 + [1.0] * 5
    current = [1.0] * 100
    assert population_stability_index(reference, current, bins=bins) > 0.25


@pytest.mark.parametrize("bins", [2, 3, 10])
def test_psi_is_direction_independent_for_constant_samples(bins):
    # The old guard made the score depend on which side of the reference the shift
    # landed: moving 3 -> 9 read as no shift while 3 -> -9999 read as a huge one.
    up = population_stability_index([3.0] * 10, [9.0] * 10, bins=bins)
    down = population_stability_index([9.0] * 10, [3.0] * 10, bins=bins)
    assert up > 0.25
    assert up == pytest.approx(down)


def test_psi_still_uses_reference_quantiles_when_they_are_not_degenerate():
    # The fallback must not fire on well-behaved continuous data, or every
    # previously correct score would shift.
    rng = np.random.default_rng(0)
    reference = rng.normal(size=2000)
    current = rng.normal(size=2000)
    assert population_stability_index(reference, current) < 0.1
