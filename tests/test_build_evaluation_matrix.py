import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from relbench.base import TaskType

import scripts.build_evaluation_matrix as bem
from scripts.build_evaluation_matrix import (
    _diagonal_free_baseline,
    _episode_of_row_local,
    apply_checkpoint_aggregation,
    assign_episodes,
    build_matrix,
    checkpoint_column_candidates,
    checkpoint_columns_by_increment,
    episode_boundaries_from_splits,
    final_model_decay_avg,
    metric_for_task_type,
    negative_mae,
    roc_auc,
    select_checkpoint_columns,
    summarise,
)

# Three episodes cut on day 0/10/20/30. Rows sit strictly inside their episode,
# never on a boundary, so these fixtures do not depend on whether the row-to-
# episode helper puts a boundary row in the window it opens or the one it closes.
BOUNDARIES = [pd.Timestamp("2020-01-01") + pd.Timedelta(days=d) for d in (0, 10, 20, 30)]
ROW_DAYS = [1, 2, 11, 12, 21, 22]
EPISODE_OF_ROW = [0, 0, 1, 1, 2, 2]

# R for the three checkpoints of `_regression_table` when each one predicts its
# own episode's target exactly: the negated distance matrix. Written out once
# because several tests assert that a stray column did not perturb it.
CLEAN_REGRESSION_R = [
    [0.0, -10.0, -20.0],
    [-10.0, 0.0, -10.0],
    [-20.0, -10.0, 0.0],
]

# `main` re-derives the episode boundaries from raw splits by dropping the first
# window, so a raw split list needs one extra entry in front of BOUNDARIES.
RAW_SPLITS = [BOUNDARIES[0] - pd.Timedelta(days=10)] + BOUNDARIES

# Four episodes, for the increment-gap tests. Three is too few: {1, 2, 4} needs a
# fourth episode to be an *interior* gap rather than an out-of-range increment.
BOUNDARIES_4 = [
    pd.Timestamp("2020-01-01") + pd.Timedelta(days=d) for d in (0, 10, 20, 30, 40)
]
ROW_DAYS_4 = [1, 2, 11, 12, 21, 22, 31, 32]
RAW_SPLITS_4 = [BOUNDARIES_4[0] - pd.Timedelta(days=10)] + BOUNDARIES_4


def _regression_table(constant_predictions):
    """Three two-row episodes with targets 10, 20, 30 and constant-prediction models.

    A model that always predicts ``v`` scores ``-|target - v|`` on every episode,
    so every expected cell of R can be written down by hand.
    """
    df = pd.DataFrame(
        {
            "time": [pd.Timestamp("2020-01-01") + pd.Timedelta(days=d) for d in ROW_DAYS],
            "y": [10.0, 10.0, 20.0, 20.0, 30.0, 30.0],
        }
    )
    for name, value in constant_predictions.items():
        df[name] = float(value)
    return df


def _regression_table_4(constant_predictions):
    """`_regression_table` with a fourth episode, targets 10/20/30/40."""
    df = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2020-01-01") + pd.Timedelta(days=d) for d in ROW_DAYS_4
            ],
            "y": [10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0],
        }
    )
    for name, value in constant_predictions.items():
        df[name] = float(value)
    return df


def _binary_table():
    """Three four-row episodes, each labelled [0, 0, 1, 1].

    Each checkpoint separates its own episode perfectly (AUC 1.0), is uninformative
    on the neighbour it never trained through (constant score, AUC 0.5), and is
    exactly wrong on the far episode (AUC 0.0).
    """
    days = [1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24]
    perfect = [0.1, 0.2, 0.8, 0.9]
    flat = [0.5, 0.5, 0.5, 0.5]
    inverted = [0.9, 0.8, 0.2, 0.1]
    return pd.DataFrame(
        {
            "time": [pd.Timestamp("2020-01-01") + pd.Timedelta(days=d) for d in days],
            "y": [0.0, 0.0, 1.0, 1.0] * 3,
            "1_aaa": perfect + flat + inverted,
            "2_bbb": flat + perfect + flat,
            "3_ccc": inverted + flat + perfect,
        }
    )


# --- checkpoint column selection --------------------------------------------


def test_checkpoint_column_candidates_drops_the_task_tables_own_columns():
    columns = ["driverId", "date", "position", "1_aaa", "2_bbb"]
    data_cols = ["driverId", "date", "position"]
    assert checkpoint_column_candidates(columns, data_cols) == ["1_aaa", "2_bbb"]


def test_checkpoint_column_candidates_drops_a_data_column_that_parses_as_a_checkpoint():
    # the whole reason the filter exists: the regex would happily accept this one
    columns = ["2_driverId", "1_aaa", "2_bbb"]
    assert checkpoint_column_candidates(columns, ["2_driverId"]) == ["1_aaa", "2_bbb"]


def test_checkpoint_column_candidates_keeps_everything_when_no_data_cols_are_known():
    columns = ["driverId", "1_aaa"]
    assert checkpoint_column_candidates(columns) == columns


def test_checkpoint_column_candidates_matches_non_string_labels_by_their_text():
    # a CSV round-trip can hand back integer-like labels; comparing as text keeps
    # them matchable while the original objects are returned, so the result can
    # still index the frame
    assert checkpoint_column_candidates([2020, "1_aaa"], ["2020"]) == ["1_aaa"]


def test_select_checkpoint_columns_orders_by_integer_not_string():
    # a lexical sort would put "10_..." before "2_...", inverting the timeline
    columns = ["10_jjj", "2_bbb", "1_aaa"]
    assert select_checkpoint_columns(columns) == ["1_aaa", "2_bbb", "10_jjj"]


def test_select_checkpoint_columns_ignores_non_prediction_columns():
    columns = ["driverId", "date", "position", "1_aaa", "2_bbb", "unknown_zzz"]
    # "unknown_zzz" is a run whose increment was not logged; it has no place in R
    assert select_checkpoint_columns(columns) == ["1_aaa", "2_bbb"]


def test_select_checkpoint_columns_returns_empty_for_a_table_with_no_runs():
    assert select_checkpoint_columns(["driverId", "date", "position"]) == []


def test_select_checkpoint_columns_first_is_deterministic_across_column_order():
    shuffled = ["1_ccc", "1_aaa", "1_bbb"]
    assert select_checkpoint_columns(shuffled) == ["1_aaa"]
    assert select_checkpoint_columns(list(reversed(shuffled))) == ["1_aaa"]


def test_select_checkpoint_columns_mean_names_one_column_per_increment():
    columns = ["1_aaa", "1_bbb", "2_ccc", "2_ddd", "2_eee"]
    assert select_checkpoint_columns(columns, aggregate="mean") == ["1_mean", "2_mean"]


def test_select_checkpoint_columns_rejects_unknown_aggregate():
    with pytest.raises(ValueError, match="aggregate"):
        select_checkpoint_columns(["1_aaa"], aggregate="median")


def test_checkpoint_columns_by_increment_groups_the_seeds():
    groups = checkpoint_columns_by_increment(["2_ddd", "1_bbb", "1_aaa", "id"])
    assert groups == {1: ["1_aaa", "1_bbb"], 2: ["2_ddd"]}


def test_apply_checkpoint_aggregation_averages_the_seeds():
    df = pd.DataFrame({"id": [0, 1], "1_aaa": [0.0, 10.0], "1_bbb": [4.0, 20.0]})
    frame, columns = apply_checkpoint_aggregation(df, aggregate="mean")
    assert columns == ["1_mean"]
    np.testing.assert_allclose(frame["1_mean"], [2.0, 15.0])


def test_apply_checkpoint_aggregation_does_not_mutate_the_caller_frame():
    df = pd.DataFrame({"1_aaa": [0.0], "1_bbb": [4.0]})
    apply_checkpoint_aggregation(df, aggregate="mean")
    assert "1_mean" not in df.columns


def test_apply_checkpoint_aggregation_first_leaves_the_frame_alone():
    df = pd.DataFrame({"1_aaa": [0.0], "1_bbb": [4.0]})
    frame, columns = apply_checkpoint_aggregation(df, aggregate="first")
    assert columns == ["1_aaa"]
    assert frame is df


def test_apply_checkpoint_aggregation_does_not_treat_its_own_mean_as_a_seed():
    # "1_mean" parses as increment 1 / run "mean". Re-running with aggregate="mean"
    # cannot show the guard working -- averaging a mean back into its own inputs is
    # arithmetically a no-op, so that assertion could never fail. The observable
    # case is "first", where "1_mean" sorts ahead of a run id like "zzz" and would
    # be handed back as though it were one of the seeds.
    df = pd.DataFrame({"1_zzz": [0.0], "1_yyy": [4.0]})
    once, _ = apply_checkpoint_aggregation(df, aggregate="mean")
    _, columns = apply_checkpoint_aggregation(once, aggregate="first")
    assert columns == ["1_yyy"]


def test_apply_checkpoint_aggregation_rejects_a_table_without_predictions():
    with pytest.raises(ValueError, match="prediction columns"):
        apply_checkpoint_aggregation(pd.DataFrame({"driverId": [1], "y": [2]}))


# --- row to episode assignment ----------------------------------------------


def test_episode_of_row_local_assigns_half_open_windows():
    times = [pd.Timestamp("2020-01-01") + pd.Timedelta(days=d) for d in ROW_DAYS]
    np.testing.assert_array_equal(_episode_of_row_local(times, BOUNDARIES), EPISODE_OF_ROW)


def test_episode_of_row_local_puts_a_boundary_row_in_the_window_it_opens():
    # matches ContinuousWrapper.get_table, which selects [start, end)
    np.testing.assert_array_equal(
        _episode_of_row_local(BOUNDARIES[:3], BOUNDARIES), [0, 1, 2]
    )


def test_episode_of_row_local_excludes_rows_outside_every_episode():
    before = BOUNDARIES[0] - pd.Timedelta(days=1)
    after = BOUNDARIES[-1]  # the last boundary closes the final episode
    np.testing.assert_array_equal(
        _episode_of_row_local([before, after, pd.NaT], BOUNDARIES), [-1, -1, -1]
    )


def test_episode_of_row_local_accepts_a_numeric_time_axis():
    np.testing.assert_array_equal(_episode_of_row_local([0.5, 5.0], [0, 1, 10]), [0, 1])


def test_episode_of_row_local_rejects_unsorted_boundaries():
    with pytest.raises(ValueError, match="increasing"):
        _episode_of_row_local([1.0], [0, 10, 5])


def test_episode_of_row_local_rejects_a_single_boundary():
    with pytest.raises(ValueError, match="at least 2"):
        _episode_of_row_local([1.0], [0])


def test_assign_episodes_matches_the_local_implementation():
    # assign_episodes prefers redelex.continual.drift.episode_of_row and falls back
    # to the local copy; the two must label every row identically, or the matrix
    # would depend on whether that module happens to be importable
    times = [
        pd.Timestamp("2020-01-01") + pd.Timedelta(days=d)
        for d in ROW_DAYS + [-1, 30]  # one row before every episode, one after
    ]
    episodes = assign_episodes(times, BOUNDARIES)

    assert episodes.dtype == np.dtype(int)
    np.testing.assert_array_equal(episodes, EPISODE_OF_ROW + [-1, -1])
    np.testing.assert_array_equal(episodes, _episode_of_row_local(times, BOUNDARIES))


def test_episode_boundaries_from_splits_drops_the_initial_training_window():
    # increment i trains on everything before splits[i] and is scored on
    # [splits[i], splits[i+1]), so episode j lines up with increment j+1
    splits = [0, 10, 20, 30]
    assert episode_boundaries_from_splits(splits) == [10, 20, 30]


def test_episode_boundaries_from_splits_rejects_too_few_splits():
    with pytest.raises(ValueError, match="at least 3"):
        episode_boundaries_from_splits([0, 10])


# --- metrics ----------------------------------------------------------------


def test_roc_auc_is_one_for_a_perfect_ranking():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_roc_auc_is_zero_for_an_exactly_inverted_ranking():
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)


def test_roc_auc_is_half_for_constant_scores():
    # every pair is tied, and a tie counts as half a win
    assert roc_auc([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_roc_auc_averages_ranks_within_a_tie_group():
    # 3 of the 4 pairs are ordered correctly, the fourth is tied -> 3.5/4
    assert roc_auc([0, 0, 1, 1], [0.1, 0.5, 0.5, 0.9]) == pytest.approx(0.875)


def test_roc_auc_is_nan_when_the_window_holds_one_class():
    assert np.isnan(roc_auc([1, 1, 1], [0.1, 0.5, 0.9]))


def test_roc_auc_rejects_non_binary_targets():
    with pytest.raises(ValueError, match="binary"):
        roc_auc([0, 1, 2], [0.1, 0.5, 0.9])


def test_negative_mae_is_zero_for_a_perfect_regressor_and_negative_otherwise():
    assert negative_mae([1.0, 3.0], [1.0, 3.0]) == pytest.approx(0.0)
    assert negative_mae([1.0, 3.0], [2.0, 3.0]) == pytest.approx(-0.5)


def test_negative_mae_is_higher_for_the_better_model():
    # the sign flip is the whole point: a smaller error must score higher
    close = negative_mae([10.0, 10.0], [11.0, 11.0])
    far = negative_mae([10.0, 10.0], [20.0, 20.0])
    assert close > far


def test_metric_for_task_type_accepts_the_relbench_enum():
    assert metric_for_task_type(TaskType.BINARY_CLASSIFICATION) is roc_auc
    assert metric_for_task_type(TaskType.REGRESSION) is negative_mae


def test_metric_for_task_type_accepts_the_plain_string():
    assert metric_for_task_type("regression") is negative_mae
    assert metric_for_task_type("binary") is roc_auc


def test_metric_for_task_type_rejects_a_type_with_no_defined_direction():
    with pytest.raises(ValueError, match="no higher-is-better metric"):
        metric_for_task_type(TaskType.MULTICLASS_CLASSIFICATION)


# --- build_matrix -----------------------------------------------------------


def test_build_matrix_regression_matches_the_hand_computed_values():
    # checkpoint i predicts episode i's target exactly and is off by 10 per
    # episode of distance, so R is the negated distance matrix
    df = _regression_table({"1_aaa": 10.0, "2_bbb": 20.0, "3_ccc": 30.0})
    matrix, episodes = build_matrix(df, BOUNDARIES, "y", "time", TaskType.REGRESSION)

    np.testing.assert_array_equal(episodes, EPISODE_OF_ROW)
    np.testing.assert_allclose(
        matrix,
        [
            [0.0, -10.0, -20.0],
            [-10.0, 0.0, -10.0],
            [-20.0, -10.0, 0.0],
        ],
    )


def test_build_matrix_regression_scores_own_episode_highest():
    df = _regression_table({"1_aaa": 10.0, "2_bbb": 20.0, "3_ccc": 30.0})
    matrix, _ = build_matrix(df, BOUNDARIES, "y", "time", TaskType.REGRESSION)

    assert matrix.shape == (3, 3)
    # higher is better, so the diagonal -- own episode -- must dominate its row
    for i in range(3):
        assert matrix[i, i] == np.max(matrix[i, :])
    off_diagonal = matrix[~np.eye(3, dtype=bool)]
    assert np.all(off_diagonal < 0)


def test_build_matrix_binary_scores_own_episode_highest():
    df = _binary_table()
    matrix, _ = build_matrix(df, BOUNDARIES, "y", "time", TaskType.BINARY_CLASSIFICATION)

    np.testing.assert_allclose(
        matrix,
        [
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.5],
            [0.0, 0.5, 1.0],
        ],
    )
    # the same higher-is-better assertion as the regression case, unchanged --
    # which is exactly what the negated-MAE convention buys
    for i in range(3):
        assert matrix[i, i] == np.max(matrix[i, :])


def test_build_matrix_row_zero_is_the_drift_curve():
    df = _binary_table()
    matrix, _ = build_matrix(df, BOUNDARIES, "y", "time", TaskType.BINARY_CLASSIFICATION)
    # one frozen model watched as the distribution moves away from it
    np.testing.assert_allclose(matrix[0, :], [1.0, 0.5, 0.0])
    assert summarise(matrix)["drift_curve"] == pytest.approx([1.0, 0.5, 0.0])


def test_build_matrix_mean_aggregation_averages_the_seeds():
    # neither seed of increment 1 predicts 10, but their mean does
    df = _regression_table({"1_aaa": 8.0, "1_bbb": 12.0, "2_ccc": 20.0, "3_ddd": 30.0})
    matrix, _ = build_matrix(
        df, BOUNDARIES, "y", "time", TaskType.REGRESSION, aggregate="mean"
    )
    np.testing.assert_allclose(matrix[0, :], [0.0, -10.0, -20.0])


def test_build_matrix_first_aggregation_uses_a_single_seed():
    df = _regression_table({"1_aaa": 8.0, "1_bbb": 12.0, "2_ccc": 20.0, "3_ddd": 30.0})
    matrix, _ = build_matrix(df, BOUNDARIES, "y", "time", TaskType.REGRESSION)
    # "1_aaa" alone is off by 2 on its own episode, unlike the seed mean
    np.testing.assert_allclose(matrix[0, :], [-2.0, -12.0, -22.0])


def test_build_matrix_ignores_the_tasks_own_columns():
    df = _regression_table({"1_aaa": 10.0, "2_bbb": 20.0, "3_ccc": 30.0})
    df["driverId"] = 7
    df["position"] = 3.0
    matrix, _ = build_matrix(
        df,
        BOUNDARIES,
        "y",
        "time",
        TaskType.REGRESSION,
        data_cols=["time", "y", "driverId", "position"],
    )
    np.testing.assert_allclose(matrix, CLEAN_REGRESSION_R)


def test_build_matrix_excludes_a_data_column_that_shadows_a_real_checkpoint():
    # run_predictions.py copies the entity column straight out of the task table,
    # so a task whose entity column is named like this parses as increment 2 --
    # and sorts before the real run, making a column of entity ids the checkpoint
    # whose scores fill row 1 of R
    df = _regression_table({"1_run1": 10.0, "2_run2": 20.0, "3_run3": 30.0})
    df["2_driverId"] = 7.0
    # the premise, checked against the real selector: unfiltered, "2_driverId" is
    # what "first" would hand back for increment 2
    assert select_checkpoint_columns(df.columns) == ["1_run1", "2_driverId", "3_run3"]

    matrix, _ = build_matrix(
        df,
        BOUNDARIES,
        "y",
        "time",
        TaskType.REGRESSION,
        data_cols=["time", "y", "2_driverId"],
    )
    np.testing.assert_allclose(matrix, CLEAN_REGRESSION_R)


def test_build_matrix_excludes_a_data_column_that_invents_an_increment():
    # the same column under an increment no checkpoint used would instead add a
    # fourth "checkpoint" to a three-episode run, which build_matrix rejects
    df = _regression_table({"1_run1": 10.0, "2_run2": 20.0, "3_run3": 30.0})
    df["4_driverId"] = 7.0

    matrix, _ = build_matrix(
        df,
        BOUNDARIES,
        "y",
        "time",
        TaskType.REGRESSION,
        data_cols=["time", "y", "4_driverId"],
    )
    np.testing.assert_allclose(matrix, CLEAN_REGRESSION_R)


def test_build_matrix_excludes_the_target_column_even_without_data_cols():
    # target_col and time_col are known to be data whether or not the caller
    # passes the task table; here the target itself parses as increment 2 and
    # sorts first, so leaking it in would score a checkpoint against itself
    df = _regression_table({"1_run1": 10.0, "2_run2": 20.0, "3_run3": 30.0})
    df = df.rename(columns={"y": "2_position"})
    # unfiltered, the target column itself is what "first" picks for increment 2
    assert select_checkpoint_columns(df.columns) == ["1_run1", "2_position", "3_run3"]

    matrix, _ = build_matrix(df, BOUNDARIES, "2_position", "time", TaskType.REGRESSION)
    np.testing.assert_allclose(matrix, CLEAN_REGRESSION_R)


def test_build_matrix_rejects_a_checkpoint_episode_count_mismatch():
    # a mismatch would slide rows against columns and read retention as transfer
    df = _regression_table({"1_aaa": 10.0, "2_bbb": 20.0})
    with pytest.raises(ValueError, match="checkpoint"):
        build_matrix(df, BOUNDARIES, "y", "time", TaskType.REGRESSION)


def test_build_matrix_rejects_a_missing_column():
    df = _regression_table({"1_aaa": 10.0, "2_bbb": 20.0, "3_ccc": 30.0})
    with pytest.raises(ValueError, match="not in the predictions table"):
        build_matrix(df, BOUNDARIES, "target", "time", TaskType.REGRESSION)


# --- summarise --------------------------------------------------------------


def test_summarise_returns_finite_values_on_a_well_formed_matrix():
    df = _binary_table()
    matrix, _ = build_matrix(df, BOUNDARIES, "y", "time", TaskType.BINARY_CLASSIFICATION)
    summary = summarise(matrix)

    assert summary["n_episodes"] == 3
    for key in (
        "average_accuracy",
        "backward_transfer",
        "forward_transfer",
        "first_model_episode_decay_avg",
    ):
        assert np.isfinite(summary[key]), key
    assert np.all(np.isfinite(summary["per_episode_forgetting"]))
    assert np.all(np.isfinite(summary["drift_curve"]))
    # episodes 0 and 1 have no within-matrix baseline; the rest must be real
    assert np.all(np.isnan(summary["forward_transfer_baseline"][:2]))
    assert np.all(np.isfinite(summary["forward_transfer_baseline"][2:]))


def test_summarise_reports_forgetting_as_negative_backward_transfer():
    # episode 0 decays 0.9 -> 0.7 -> 0.6 while later episodes are learned
    forgetting = np.array([[0.9, 0.5, 0.5], [0.7, 0.8, 0.5], [0.6, 0.75, 0.85]])
    summary = summarise(forgetting)
    assert summary["backward_transfer"] < 0
    assert summary["per_episode_forgetting"] == pytest.approx([0.30, 0.05])


def test_summarise_average_accuracy_is_the_final_rows_mean():
    # the first row's mean (0.7) and the whole matrix's mean (0.55) both differ
    # from the last row's (0.4), so a mutation to either would be caught
    matrix = np.array([[0.9, 0.5], [0.6, 0.2]])
    assert summarise(matrix)["average_accuracy"] == pytest.approx(0.4)


def test_summarise_accepts_an_explicit_forward_transfer_baseline():
    matrix = np.array([[0.9, 0.5], [0.6, 0.8]])
    # R[0, 1] = 0.5 against a fresh model's 0.4 is 0.1 of useful transfer
    summary = summarise(matrix, baseline=[0.0, 0.4])
    assert summary["forward_transfer"] == pytest.approx(0.1)
    assert summary["forward_transfer_baseline"] == pytest.approx([0.0, 0.4])


def test_diagonal_free_baseline_excludes_the_row_forward_transfer_scores():
    # R[j-1, j] is the cell forward_transfer subtracts the baseline from, so it
    # must stay out of that baseline; R[2, 3] is made wild to prove it is not
    # averaged in
    matrix = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.6],
            [0.3, 0.4, 0.5, 9.0],
            [0.4, 0.5, 0.6, 0.7],
        ]
    )
    baseline = _diagonal_free_baseline(matrix)
    assert baseline.shape == (4,)
    assert baseline[2] == pytest.approx(0.3)  # R[0, 2] alone, not with R[1, 2]
    assert baseline[3] == pytest.approx((0.4 + 0.6) / 2)  # rows 0 and 1, not row 2


def test_diagonal_free_baseline_is_undefined_below_episode_two():
    matrix = np.array([[0.9, 0.5, 0.4], [0.6, 0.8, 0.2], [0.1, 0.2, 0.7]])
    baseline = _diagonal_free_baseline(matrix)
    # episode 1's only earlier model is the one forward transfer scores, so the
    # proxy has nothing to average; episode 0 has no earlier model at all
    assert np.isnan(baseline[0])
    assert np.isnan(baseline[1])
    assert np.isfinite(baseline[2])


def test_diagonal_free_baseline_is_all_nan_below_three_episodes():
    assert np.all(np.isnan(_diagonal_free_baseline(np.array([[0.9, 0.5], [0.6, 0.8]]))))


def test_summarise_forward_transfer_is_nan_for_a_two_episode_matrix():
    # rel-f1/driver-top3 is a two-episode task: the within-matrix proxy is
    # undefined there, and reporting 0.0 would read as "no transfer" instead
    matrix = np.array([[0.9, 0.5], [0.6, 0.8]])
    assert np.isnan(summarise(matrix)["forward_transfer"])


def test_summarise_forward_transfer_compares_the_latest_model_to_the_older_ones():
    # the only defined comparison in a 3x3 is episode 2: R[1, 2] against R[0, 2]
    matrix = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.9], [0.7, 0.8, 0.9]])
    assert summarise(matrix)["forward_transfer"] == pytest.approx(0.6)


def test_summarise_keeps_every_episode_when_the_baseline_is_explicit():
    # an explicit baseline is a real independently-initialised model, so episode 1
    # is a genuine comparison and must not be dropped from the average
    matrix = np.array([[0.1, 0.6, 0.3], [0.4, 0.5, 0.8], [0.7, 0.8, 0.9]])
    summary = summarise(matrix, baseline=[0.0, 0.1, 0.2])
    # mean(R[0, 1] - 0.1, R[1, 2] - 0.2) = mean(0.5, 0.6); dropping episode 1
    # would leave 0.6 alone
    assert summary["forward_transfer"] == pytest.approx(0.55)


def test_summarise_first_model_episode_decay_avg_weights_the_earliest_episode_most():
    matrix = np.array([[1.0, 0.0], [0.5, 0.5]])
    summary = summarise(matrix, decay=0.5)
    assert summary["first_model_episode_decay_avg"] == pytest.approx(1.0 / 1.5)
    # the old name promised the published decay metric and delivered this one
    assert "drift_curve_avg" not in summary


def test_summarise_final_model_decay_avg_is_passed_through_not_derived():
    # it cannot be derived: it needs the per-row predictions, which R has lost
    matrix = np.array([[1.0, 0.0], [0.5, 0.5]])
    assert summarise(matrix, final_model_decay=0.25)[
        "final_model_decay_avg"
    ] == pytest.approx(0.25)
    assert np.isnan(summarise(matrix)["final_model_decay_avg"])


def test_summarise_rejects_a_non_square_matrix():
    with pytest.raises(ValueError, match="square"):
        summarise(np.zeros((2, 3)))


# --- main --------------------------------------------------------------------


def _fake_task_context(table_cols, raw_splits=None, recorder=None):
    """Stand-in for `_load_task_context`, so `main` runs without relbench or a cache."""
    task = SimpleNamespace(
        target_col="y", time_col="time", task_type=TaskType.REGRESSION
    )
    splits = RAW_SPLITS if raw_splits is None else raw_splits

    def _context(dataset, task_name, val_delta_days=None):
        if recorder is not None:
            recorder["val_delta_days"] = val_delta_days
        return task, splits, list(table_cols)

    return _context


def _write_predictions(tmp_path, df):
    csv_path = tmp_path / "predictions.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_main_writes_the_matrix_and_summary_without_the_entity_column(
    tmp_path, monkeypatch
):
    # end-to-end proof that the CLI keeps the task table out of R: "2_driverId"
    # parses as increment 2 and sorts before the real run, so if main did not pass
    # the task table's columns down, every cell of row 1 would move
    df = _regression_table({"1_run1": 10.0, "2_run2": 20.0, "3_run3": 30.0})
    df["2_driverId"] = 7.0
    csv_path = _write_predictions(tmp_path, df)
    monkeypatch.setattr(
        bem, "_load_task_context", _fake_task_context(["time", "y", "2_driverId"])
    )

    out = tmp_path / "run"
    exit_code = bem.main(
        [
            "--predictions",
            str(csv_path),
            "--dataset",
            "rel-f1",
            "--task",
            "driver-position",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    written = pd.read_csv(f"{out}_matrix.csv", index_col=0)
    assert list(written.columns) == ["ep0", "ep1", "ep2"]
    np.testing.assert_allclose(written.to_numpy(), CLEAN_REGRESSION_R)

    summary = json.loads(Path(f"{out}_summary.json").read_text())
    assert summary["n_episodes"] == 3
    assert summary["drift_curve"] == pytest.approx([0.0, -10.0, -20.0])


def test_main_truncates_to_the_episodes_the_finished_checkpoints_cover(
    tmp_path, monkeypatch, capsys
):
    # a chain that stopped early: two checkpoints, three episodes. Keeping the
    # leading episodes is right, but the dropped tail of the drift curve must be
    # announced rather than inferred from the matrix being small.
    df = _regression_table({"1_run1": 10.0, "2_run2": 20.0})
    csv_path = _write_predictions(tmp_path, df)
    monkeypatch.setattr(bem, "_load_task_context", _fake_task_context(["time", "y"]))

    out = tmp_path / "run"
    exit_code = bem.main(
        [
            "--predictions",
            str(csv_path),
            "--dataset",
            "rel-f1",
            "--task",
            "driver-position",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    assert "truncating" in capsys.readouterr().err
    written = pd.read_csv(f"{out}_matrix.csv", index_col=0)
    np.testing.assert_allclose(written.to_numpy(), [[0.0, -10.0], [-10.0, 0.0]])


# --- the published decay metric ----------------------------------------------

_W0 = pd.Timestamp("2020-02-01")
_W1 = pd.Timestamp("2020-02-08")


def test_final_model_decay_avg_weights_windows_by_their_row_count():
    # one row at the first timestamp, three at the second: the count-weighted mean
    # is -3.0 where an unweighted mean over the two windows would be -2.0
    times = [_W0, _W1, _W1, _W1]
    y_true = [10.0, 20.0, 20.0, 20.0]
    y_pred = [10.0, 16.0, 16.0, 16.0]
    value = final_model_decay_avg(times, y_true, y_pred, negative_mae, start=_W0)
    assert value == pytest.approx(-3.0)


def test_final_model_decay_avg_scores_each_timestamp_as_its_own_window():
    # decay is applied per window, so the two timestamps cannot have been pooled:
    # weights 1 and 0.5 over scores 0.0 and -4.0
    times = [_W0, _W1]
    value = final_model_decay_avg(
        times, [10.0, 20.0], [10.0, 16.0], negative_mae, start=_W0, decay=0.5
    )
    assert value == pytest.approx(-4.0 * 0.5 / 1.5)


def test_final_model_decay_avg_ignores_rows_before_start():
    # the first window is where the model is worst; starting after it must not
    # merely down-weight it but drop it
    times = [_W0, _W1]
    value = final_model_decay_avg(
        times, [10.0, 20.0], [99.0, 16.0], negative_mae, start=_W1
    )
    assert value == pytest.approx(-4.0)


def test_final_model_decay_avg_end_is_exclusive():
    times = [_W0, _W1]
    value = final_model_decay_avg(
        times, [10.0, 20.0], [12.0, 99.0], negative_mae, start=_W0, end=_W1
    )
    assert value == pytest.approx(-2.0)


def test_final_model_decay_avg_drops_a_window_whose_metric_is_undefined():
    # a single-timestamp window very often holds one class, where ROC-AUC has no
    # value; propagating that nan would erase the metric for every binary task
    times = [_W0] * 4 + [_W1] * 2
    y_true = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    y_pred = [0.1, 0.2, 0.8, 0.9, 0.3, 0.4]
    value = final_model_decay_avg(times, y_true, y_pred, roc_auc, start=_W0)
    assert value == pytest.approx(1.0)


def test_final_model_decay_avg_is_nan_when_no_window_is_scorable():
    times = [_W0, _W1]
    assert np.isnan(
        final_model_decay_avg(times, [1.0, 1.0], [0.3, 0.4], roc_auc, start=_W0)
    )


def test_final_model_decay_avg_rejects_an_empty_span():
    with pytest.raises(ValueError, match="no rows"):
        final_model_decay_avg(
            [_W0], [10.0], [10.0], negative_mae, start=_W1
        )


def test_final_model_decay_avg_rejects_ragged_inputs():
    with pytest.raises(ValueError, match="length mismatch"):
        final_model_decay_avg([_W0, _W1], [10.0], [10.0, 10.0], negative_mae, start=_W0)


def test_final_model_decay_avg_is_not_the_first_model_episode_average():
    # the two used to share the name `drift_curve_avg`. Here the first increment
    # drifts badly across the episode grid while the final one is exact on its own
    # window, so the two numbers cannot be confused
    df = _regression_table({"1_run1": 10.0, "2_run2": 20.0, "3_run3": 30.0})
    matrix, _ = build_matrix(df, BOUNDARIES, "y", "time", TaskType.REGRESSION)
    first_model = summarise(matrix)["first_model_episode_decay_avg"]
    final_model = final_model_decay_avg(
        df["time"], df["y"], df["3_run3"], negative_mae, start=BOUNDARIES[-2]
    )
    assert first_model == pytest.approx(-10.0)
    assert final_model == pytest.approx(0.0)


# --- what happens to the RelBench test split ---------------------------------


def test_test_split_rows_are_outside_r_but_inside_the_decay_metric():
    # `episode_boundaries_from_splits` stops AT test_timestamp, so nothing on or
    # after it forms a column of R -- there is no checkpoint trained through the
    # validation window to fill such a column's diagonal. Those rows are not
    # thrown away: `final_model_decay_avg` runs from the final episode's start
    # with no upper bound, which is where they are scored.
    boundaries = episode_boundaries_from_splits(RAW_SPLITS)
    assert boundaries[-1] == RAW_SPLITS[-1]

    test_timestamp = RAW_SPLITS[-1]
    times = [test_timestamp, test_timestamp + pd.Timedelta(days=1)]
    np.testing.assert_array_equal(assign_episodes(times, boundaries), [-1, -1])

    value = final_model_decay_avg(
        times, [1.0, 3.0], [1.0, 1.0], negative_mae, start=boundaries[-2]
    )
    assert value == pytest.approx(-1.0)


# --- episode width -----------------------------------------------------------


def test_load_task_context_passes_the_episode_width_to_get_splits(monkeypatch):
    # a --val_delta_days sweep cuts the timeline into different windows; taking
    # get_splits' default here would build R's columns from one partition and its
    # rows from another, and only the counts are checked downstream
    import relbench.tasks

    from experiments.continuous_learning import continuous_task

    recorded = {}

    class _RecordingWrapper:
        def __init__(self, task):
            self.full_table = SimpleNamespace(df=pd.DataFrame(columns=["time", "y"]))

        def get_splits(self, val_delta=None, **kwargs):
            recorded["val_delta"] = val_delta
            return list(RAW_SPLITS)

    monkeypatch.setattr(relbench.tasks, "get_task", lambda dataset, task_name: object())
    monkeypatch.setattr(continuous_task, "ContinuousWrapper", _RecordingWrapper)

    _, splits, table_cols = bem._load_task_context(
        "rel-f1", "driver-position", val_delta_days=7
    )
    assert recorded["val_delta"] == pd.Timedelta(days=7)
    assert splits == list(RAW_SPLITS)
    assert table_cols == ["time", "y"]


def test_load_task_context_keeps_the_wrappers_default_width_when_unset(monkeypatch):
    import relbench.tasks

    from experiments.continuous_learning import continuous_task

    recorded = {}

    class _RecordingWrapper:
        def __init__(self, task):
            self.full_table = SimpleNamespace(df=pd.DataFrame(columns=["time", "y"]))

        def get_splits(self, val_delta=None, **kwargs):
            recorded["val_delta"] = val_delta
            return list(RAW_SPLITS)

    monkeypatch.setattr(relbench.tasks, "get_task", lambda dataset, task_name: object())
    monkeypatch.setattr(continuous_task, "ContinuousWrapper", _RecordingWrapper)

    bem._load_task_context("rel-f1", "driver-position")
    assert recorded["val_delta"] is None


def test_main_forwards_val_delta_days_to_the_split_builder(tmp_path, monkeypatch):
    df = _regression_table({"1_run1": 10.0, "2_run2": 20.0, "3_run3": 30.0})
    csv_path = _write_predictions(tmp_path, df)
    recorded = {}
    monkeypatch.setattr(
        bem, "_load_task_context", _fake_task_context(["time", "y"], recorder=recorded)
    )

    exit_code = bem.main(
        [
            "--predictions",
            str(csv_path),
            "--dataset",
            "rel-f1",
            "--task",
            "driver-position",
            "--val-delta-days",
            "7",
        ]
    )

    assert exit_code == 0
    assert recorded["val_delta_days"] == pytest.approx(7.0)


# --- increment gaps ----------------------------------------------------------


def _run_main_on(tmp_path, monkeypatch, df, table_cols, raw_splits, extra_args=()):
    csv_path = _write_predictions(tmp_path, df)
    monkeypatch.setattr(
        bem, "_load_task_context", _fake_task_context(table_cols, raw_splits=raw_splits)
    )
    return bem.main(
        [
            "--predictions",
            str(csv_path),
            "--dataset",
            "rel-f1",
            "--task",
            "driver-position",
            *extra_args,
        ]
    )


def test_main_refuses_an_interior_increment_gap(tmp_path, monkeypatch):
    # increments {1, 2, 4} over four episodes. Truncating to the first three
    # boundaries would score increment 4 against episode 2 and report numbers that
    # look plausible: the old code exited 0 here.
    df = _regression_table_4({"1_run1": 10.0, "2_run2": 20.0, "4_run4": 40.0})
    with pytest.raises(SystemExit) as excinfo:
        _run_main_on(tmp_path, monkeypatch, df, ["time", "y"], RAW_SPLITS_4)
    assert "[3]" in str(excinfo.value)


def test_main_refuses_a_head_increment_gap(tmp_path, monkeypatch):
    # the likely shape in practice: run_predictions.py queries MLflow with no
    # order_by, MLflow defaults to start_time DESC, so an interrupted predict job
    # leaves the NEWEST increments behind. Truncating would produce a "drift curve"
    # that improves over time, from models that never saw the early episodes.
    df = _regression_table_4({"3_run3": 30.0, "4_run4": 40.0})
    with pytest.raises(SystemExit) as excinfo:
        _run_main_on(tmp_path, monkeypatch, df, ["time", "y"], RAW_SPLITS_4)
    assert "[1, 2]" in str(excinfo.value)


def test_main_still_truncates_a_genuine_tail_gap(tmp_path, monkeypatch, capsys):
    # the case the truncation was written for, over four episodes: a prefix of
    # increments really does line up with the leading episodes
    df = _regression_table_4({"1_run1": 10.0, "2_run2": 20.0})
    exit_code = _run_main_on(tmp_path, monkeypatch, df, ["time", "y"], RAW_SPLITS_4)
    assert exit_code == 0
    assert "truncating" in capsys.readouterr().err


def test_main_refuses_a_table_with_no_prediction_columns(tmp_path, monkeypatch):
    df = _regression_table({})
    with pytest.raises(SystemExit) as excinfo:
        _run_main_on(tmp_path, monkeypatch, df, ["time", "y"], RAW_SPLITS)
    assert "prediction columns" in str(excinfo.value)


def test_main_summary_carries_both_decay_averages(tmp_path, monkeypatch):
    df = _regression_table({"1_run1": 10.0, "2_run2": 20.0, "3_run3": 30.0})
    csv_path = _write_predictions(tmp_path, df)
    monkeypatch.setattr(bem, "_load_task_context", _fake_task_context(["time", "y"]))

    out = tmp_path / "run"
    exit_code = bem.main(
        [
            "--predictions",
            str(csv_path),
            "--dataset",
            "rel-f1",
            "--task",
            "driver-position",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    summary = json.loads(Path(f"{out}_summary.json").read_text())
    assert "drift_curve_avg" not in summary
    # the frozen first increment averaged over the episode grid, against the final
    # increment scored per timestamp from its own window on -- not the same number
    assert summary["first_model_episode_decay_avg"] == pytest.approx(-10.0)
    assert summary["final_model_decay_avg"] == pytest.approx(0.0)
