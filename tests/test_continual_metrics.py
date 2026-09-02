import numpy as np
import pandas as pd
import pytest

from redelex.continual import (
    average_accuracy,
    backward_transfer,
    evaluation_matrix_from_predictions,
    exp_decay_avg,
    forward_transfer,
    per_episode_forgetting,
)


# --- exp_decay_avg ----------------------------------------------------------


def test_exp_decay_avg_zero_decay_is_plain_mean():
    assert exp_decay_avg([1.0, 2.0, 3.0], decay=0.0) == pytest.approx(2.0)


def test_exp_decay_avg_weights_earliest_window_most():
    # weights are (1-decay)**k, so index 0 dominates and pulls the mean toward it
    out = exp_decay_avg([1.0, 0.0], decay=0.5)
    assert out == pytest.approx(1.0 / 1.5)


def test_exp_decay_avg_matches_closed_form():
    values = [4.0, 2.0, 1.0]
    decay = 0.3
    weights = np.power(1 - decay, np.arange(3))
    expected = np.sum(np.array(values) * weights) / weights.sum()
    assert exp_decay_avg(values, decay=decay) == pytest.approx(expected)


def test_exp_decay_avg_counts_weight_windows_by_size():
    # a window holding 3x the rows counts 3x, absent any decay
    assert exp_decay_avg([1.0, 0.0], counts=[3, 1], decay=0.0) == pytest.approx(0.75)


def test_exp_decay_avg_counts_and_decay_compose():
    values, counts, decay = [1.0, 0.0], [1, 3], 0.5
    weights = np.power(1 - decay, np.arange(2)) * np.array(counts)
    expected = np.sum(np.array(values) * weights) / weights.sum()
    assert exp_decay_avg(values, counts=counts, decay=decay) == pytest.approx(expected)


def test_exp_decay_avg_reproduces_notebook_implementation():
    # the original lived only in notebooks/process-data-continuous-learning.ipynb
    def notebook_version(metrics, counts=None, decay=0):
        counts = np.ones_like(metrics) if counts is None else np.array(counts)
        weights = (np.power(1 - decay, np.arange(len(metrics))) * counts) / counts.sum()
        return np.average(metrics, weights=weights)

    metrics = np.array([3.1, 2.4, 2.9, 1.8])
    counts = np.array([100, 250, 80, 400])
    for decay in (0.0, 0.3, 0.9):
        assert exp_decay_avg(metrics, counts=counts, decay=decay) == pytest.approx(
            notebook_version(metrics, counts=counts, decay=decay)
        )


@pytest.mark.parametrize("decay", [-0.1, 1.0, 1.5])
def test_exp_decay_avg_rejects_out_of_range_decay(decay):
    with pytest.raises(ValueError, match="decay"):
        exp_decay_avg([1.0, 2.0], decay=decay)


def test_exp_decay_avg_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        exp_decay_avg([])


def test_exp_decay_avg_rejects_mismatched_counts():
    with pytest.raises(ValueError, match="counts"):
        exp_decay_avg([1.0, 2.0], counts=[1.0])


# --- evaluation matrix metrics ---------------------------------------------

# Model after episode i (row) scored on episode j (column). Higher is better.
# Episode 0 degrades 0.9 -> 0.7 -> 0.6 as training continues: clear forgetting.
FORGETTING = np.array(
    [
        [0.90, 0.50, 0.50],
        [0.70, 0.80, 0.50],
        [0.60, 0.75, 0.85],
    ]
)


def test_average_accuracy_is_mean_of_final_row():
    assert average_accuracy(FORGETTING) == pytest.approx((0.60 + 0.75 + 0.85) / 3)


def test_backward_transfer_is_negative_when_forgetting():
    # (0.60-0.90) and (0.75-0.80) -> mean -0.175
    assert backward_transfer(FORGETTING) == pytest.approx(-0.175)


def test_backward_transfer_positive_when_later_episodes_help():
    improving = np.array([[0.5, 0.0], [0.8, 0.6]])
    assert backward_transfer(improving) > 0


def test_backward_transfer_sign_is_direction_corrected():
    # same numbers read as an error metric: rising error is still forgetting
    errors = np.array([[1.0, 9.0], [3.0, 2.0]])
    assert backward_transfer(errors, higher_is_better=True) > 0
    assert backward_transfer(errors, higher_is_better=False) < 0


def test_backward_transfer_undefined_for_single_episode():
    assert backward_transfer(np.array([[0.7]])) == 0.0


def test_per_episode_forgetting_uses_best_past_score():
    # episode 0 peaked at 0.90 and ended at 0.60; episode 1 peaked 0.80 ended 0.75
    np.testing.assert_allclose(
        per_episode_forgetting(FORGETTING), [0.30, 0.05], atol=1e-12
    )


def test_per_episode_forgetting_negative_means_improvement():
    improving = np.array([[0.5, 0.0], [0.8, 0.6]])
    assert per_episode_forgetting(improving)[0] < 0


def test_per_episode_forgetting_direction_corrected_for_error_metrics():
    errors = np.array([[1.0, 9.0], [3.0, 2.0]])
    # error grew 1.0 -> 3.0, which is forgetting, so the value must be positive
    assert per_episode_forgetting(errors, higher_is_better=False)[0] == pytest.approx(2.0)


def test_per_episode_forgetting_empty_for_single_episode():
    assert per_episode_forgetting(np.array([[0.7]])).size == 0


def test_forward_transfer_positive_when_history_helps():
    # R[0,1]=0.50 vs baseline 0.40, R[1,2]=0.50 vs baseline 0.45
    baseline = [0.0, 0.40, 0.45]
    assert forward_transfer(FORGETTING, baseline) == pytest.approx((0.10 + 0.05) / 2)


def test_forward_transfer_direction_corrected():
    errors = np.array([[1.0, 5.0], [3.0, 2.0]])
    baseline = [0.0, 8.0]
    # the model was better (lower error) than baseline, so transfer is positive
    assert forward_transfer(errors, baseline, higher_is_better=False) == pytest.approx(3.0)


def test_forward_transfer_rejects_wrong_baseline_length():
    with pytest.raises(ValueError, match="baseline"):
        forward_transfer(FORGETTING, [0.1, 0.2])


@pytest.mark.parametrize(
    "bad", [np.zeros((2, 3)), np.zeros(3)], ids=["non-square", "one-dim"]
)
def test_matrix_metrics_reject_bad_shapes(bad):
    with pytest.raises(ValueError, match="square"):
        average_accuracy(bad)


# --- evaluation_matrix_from_predictions -------------------------------------


def _neg_abs_error(y_true, y_pred):
    return -float(np.mean(np.abs(y_true - y_pred)))


def test_evaluation_matrix_from_predictions_groups_rows_by_episode():
    df = pd.DataFrame(
        {
            "target": [1.0, 1.0, 2.0, 2.0],
            "ckpt_0": [1.0, 1.0, 0.0, 0.0],  # perfect on ep 0, off by 2 on ep 1
            "ckpt_1": [0.0, 0.0, 2.0, 2.0],  # off by 1 on ep 0, perfect on ep 1
        }
    )
    matrix = evaluation_matrix_from_predictions(
        predictions=df,
        episode_of_row=[0, 0, 1, 1],
        checkpoint_columns=["ckpt_0", "ckpt_1"],
        target=df["target"],
        metric_fn=_neg_abs_error,
    )
    np.testing.assert_allclose(matrix, [[0.0, -2.0], [-1.0, 0.0]])


def test_evaluation_matrix_from_predictions_marks_empty_episodes_nan():
    df = pd.DataFrame({"target": [1.0, 2.0], "a": [1.0, 2.0], "b": [1.0, 2.0]})
    matrix = evaluation_matrix_from_predictions(
        predictions=df,
        episode_of_row=[0, 0],  # nothing lands in episode 1
        checkpoint_columns=["a", "b"],
        target=df["target"],
        metric_fn=_neg_abs_error,
    )
    assert np.all(np.isnan(matrix[:, 1]))
    assert not np.any(np.isnan(matrix[:, 0]))


def test_evaluation_matrix_from_predictions_ignores_unassigned_rows():
    df = pd.DataFrame({"target": [1.0, 5.0], "a": [1.0, 99.0]})
    matrix = evaluation_matrix_from_predictions(
        predictions=df,
        episode_of_row=[0, -1],  # second row belongs to no episode
        checkpoint_columns=["a"],
        target=df["target"],
        metric_fn=_neg_abs_error,
    )
    np.testing.assert_allclose(matrix, [[0.0]])


def test_evaluation_matrix_from_predictions_rejects_length_mismatch():
    df = pd.DataFrame({"target": [1.0, 2.0], "a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="rows"):
        evaluation_matrix_from_predictions(
            predictions=df,
            episode_of_row=[0],
            checkpoint_columns=["a"],
            target=df["target"],
            metric_fn=_neg_abs_error,
        )
