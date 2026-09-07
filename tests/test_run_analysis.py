"""Tests for `scripts/run_analysis.py`, the end-to-end analysis driver.

Everything here runs from synthetic CSVs on disk: no MLflow, no relbench, no
GPU, no network. The fixtures are built so that every expected number can be
written down by hand -- a model that always predicts ``v`` scores ``-|y - v|``
on a regression episode whose target is constant -- because a metric test whose
expectation is computed by the code under test cannot fail.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.run_analysis as ra
from scripts.run_analysis import (
    AGG_SEED_PREFIX,
    DEFAULT_MODES,
    DRIFT_MODE,
    REFERENCE_MODES,
    TIDY_COLUMNS,
    ModeResult,
    analyse_mode,
    chain_id_for,
    cross_mode_refusal,
    drift_rows,
    experiment_name,
    format_exclusions,
    format_summary_table,
    increment_gap_message,
    load_task_spec,
    order_modes,
    parse_splits,
    pooled_protocol,
    prediction_command,
    predictions_csv_path,
    protocol_verdict,
    replicate_column_sets,
    scalar_summary,
    score_metric_name,
    tidy_frame,
)

DATASET = "rel-fake"
TASK = "fake-position"
PREFIX = "smoke_cl"

# Three episodes cut on day 0/10/20/30. Rows sit strictly inside their episode so
# the fixtures do not depend on which side of a boundary a boundary row lands.
BOUNDARIES = [pd.Timestamp("2020-01-01") + pd.Timedelta(days=d) for d in (0, 10, 20, 30)]
# `episode_boundaries_from_splits` drops the first window (it is increment 1's
# training data), so a raw split list needs one extra entry in front.
RAW_SPLITS = [BOUNDARIES[0] - pd.Timedelta(days=10)] + BOUNDARIES
ROW_DAYS = [1, 2, 11, 12, 21, 22]
TARGETS = [10.0, 10.0, 20.0, 20.0, 30.0, 30.0]

TABLE_COLUMNS = ["driverId", "date", "y"]

BASE_SPEC = {
    "dataset": DATASET,
    "task": TASK,
    "task_type": "regression",
    "target_col": "y",
    "time_col": "date",
    "table_columns": TABLE_COLUMNS,
    "splits": RAW_SPLITS,
}

AGREEING_PROTOCOL_ENTRY = {
    "n_runs": 2,
    "max_training_steps": ["2000"],
    "val_check_interval": ["100"],
    "val_max_rows": ["25000"],
    "val_delta_days": ["<missing>"],
}


# --- fixture builders -------------------------------------------------------


def regression_frame(predictions, extra_columns=None):
    """Six rows, three constant-target episodes, one column per checkpoint.

    Args:
        predictions: ``{increment: [value_per_trial, ...]}``. Column names are
            ``"{increment}_run{increment}{a,b,...}"``.
        extra_columns: Additional literal columns, e.g. a data column whose name
            happens to parse as a checkpoint.
    """
    frame = pd.DataFrame(
        {
            "driverId": [1, 2, 1, 2, 1, 2],
            "date": [pd.Timestamp("2020-01-01") + pd.Timedelta(days=d) for d in ROW_DAYS],
            "y": list(TARGETS),
        }
    )
    for increment, values in sorted(predictions.items()):
        for slot, value in enumerate(values):
            frame[f"{increment}_run{increment}{chr(97 + slot)}"] = float(value)
    for name, values in (extra_columns or {}).items():
        frame[name] = values
    return frame


def sidecar_for(predictions, overrides=None):
    """A seed record whose protocol block agrees across every increment."""
    protocol = {}
    for increment, values in sorted(predictions.items()):
        entry = dict(AGREEING_PROTOCOL_ENTRY)
        entry["n_runs"] = len(values)
        entry.update((overrides or {}).get(increment, {}))
        protocol[str(increment)] = entry
    return {
        "seed": 42,
        "dataset_name": DATASET,
        "task_name": TASK,
        "protocol": protocol,
        "columns": {},
    }


# Constant predictions per (mode, increment, trial). Chosen so no two modes
# score alike and every cell is an exact integer distance.
MODE_PREDICTIONS = {
    "from_scratch": {1: [10.0, 11.0], 2: [20.0, 21.0], 3: [30.0, 31.0]},
    "joint": {1: [10.0, 10.5], 2: [19.0, 20.0], 3: [29.0, 30.0]},
    "naive": {1: [10.0, 12.0], 2: [22.0, 20.0], 3: [33.0, 30.0]},
    "er": {1: [10.0, 11.0], 2: [20.5, 20.0], 3: [30.5, 30.0]},
}


def write_mode(root, mode, predictions=None, protocol_overrides=None, sidecar=True):
    """Write one mode's predictions CSV (and seed record) under ``root/data``."""
    predictions = MODE_PREDICTIONS[mode] if predictions is None else predictions
    directory = Path(root) / "data" / experiment_name(PREFIX, mode)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{DATASET}_{TASK}_predictions.csv"
    regression_frame(predictions).to_csv(csv_path, index=False)
    if sidecar:
        csv_path.with_suffix(".seeds.json").write_text(
            json.dumps(sidecar_for(predictions, protocol_overrides), indent=2)
        )
    return csv_path


def write_spec(root):
    path = Path(root) / "spec.json"
    spec = dict(BASE_SPEC, splits=[str(edge) for edge in RAW_SPLITS])
    path.write_text(json.dumps(spec, indent=2))
    return path


@pytest.fixture()
def grid(tmp_path):
    """A complete four-mode grid plus its task spec."""
    for mode in MODE_PREDICTIONS:
        write_mode(tmp_path, mode)
    write_spec(tmp_path)
    return tmp_path


def cli(root, *extra, modes=tuple(MODE_PREDICTIONS)):
    """The from-csv CLI invocation for a fixture root."""
    return [
        "--dataset", DATASET,
        "--task", TASK,
        "--modes", *modes,
        "--from-csv",
        "--data-root", str(Path(root) / "data"),
        "--experiment-prefix", PREFIX,
        "--task-spec", str(Path(root) / "spec.json"),
        "--out", str(Path(root) / "out"),
        *extra,
    ]


# --- naming and paths -------------------------------------------------------


def test_order_modes_puts_the_reference_frame_first_and_dedupes():
    ordered = order_modes(["lwf", "naive", "er", "from_scratch", "joint", "er"])
    assert ordered[:3] == ["from_scratch", "joint", "naive"]
    assert ordered[3:] == ["er", "lwf"]


def test_order_modes_keeps_reference_order_even_when_input_reverses_it():
    assert order_modes(["naive", "joint", "from_scratch"]) == list(REFERENCE_MODES)


def test_default_modes_match_the_launcher_roster():
    # The two lists are declared separately (run_analysis must stay importable
    # without the experiment package), so nothing but this test stops them
    # drifting apart and silently analysing a different set than was launched.
    import scripts.run_grid as run_grid

    assert list(DEFAULT_MODES) == list(run_grid.DEFAULT_MODES)


def test_experiment_name_matches_run_grid_convention():
    assert experiment_name("pelesjak_cl_v2", "der_pp") == "pelesjak_cl_v2_der_pp"


def test_predictions_csv_path_matches_run_predictions_layout():
    path = predictions_csv_path("data", "pelesjak_cl_v2_er", "rel-f1", "driver-position")
    assert path == Path("data/pelesjak_cl_v2_er/rel-f1_driver-position_predictions.csv")


def test_chain_id_keeps_the_model_save_dir_verbatim():
    # continuous_learning.py builds the chain id from the *raw* --model_save_dir
    # argument, before making it absolute, so a relative path must survive.
    assert chain_id_for("rel-f1", "driver-dnf", "ewc", "logs/grid/x/models") == (
        "rel-f1/driver-dnf/ewc/logs/grid/x/models"
    )


def test_prediction_command_pins_the_chain_and_is_strict_by_default():
    command = prediction_command(
        "rel-f1", "driver-dnf", "exp", out_dir="data/exp", seed=7,
        chain_id="rel-f1/driver-dnf/ewc/m", mlflow_uri="http://host:2222", python="py",
    )
    assert command[:3] == ["py", "-u", str(ra.RUN_PREDICTIONS)]
    assert "--chain_id=rel-f1/driver-dnf/ewc/m" in command
    assert "--strict-protocol" in command
    assert "--seed=7" in command
    assert "--mlflow_uri=http://host:2222" in command


def test_prediction_command_omits_optional_flags_when_unset():
    command = prediction_command(
        "rel-f1", "driver-dnf", "exp", out_dir="d", seed=1, python="py",
        strict_protocol=False,
    )
    assert not any(item.startswith("--chain_id") for item in command)
    assert not any(item.startswith("--mlflow_uri") for item in command)
    assert "--strict-protocol" not in command


# --- task spec --------------------------------------------------------------


def test_parse_splits_reads_iso_strings_and_numbers():
    stamps = parse_splits(["2020-01-01", "2020-01-11", "2020-01-21"])
    assert stamps == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-11"),
                      pd.Timestamp("2020-01-21")]
    assert parse_splits([0, 1.5, 3]) == [0.0, 1.5, 3.0]


def test_parse_splits_rejects_too_few_boundaries():
    with pytest.raises(ValueError, match="at least 3 boundaries"):
        parse_splits(["2020-01-01", "2020-01-11"])


def test_parse_splits_rejects_non_increasing_boundaries():
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_splits(["2020-01-01", "2020-01-21", "2020-01-11"])


def test_load_task_spec_roundtrips_and_parses_splits(tmp_path):
    spec = load_task_spec(write_spec(tmp_path))
    assert spec["target_col"] == "y"
    assert spec["task_type"] == "regression"
    assert spec["splits"] == RAW_SPLITS


@pytest.mark.parametrize("missing", ["task_type", "target_col", "time_col", "splits",
                                     "table_columns"])
def test_load_task_spec_refuses_to_guess_a_missing_key(tmp_path, missing):
    payload = dict(BASE_SPEC, splits=[str(edge) for edge in RAW_SPLITS])
    payload.pop(missing)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=missing):
        load_task_spec(path)


def test_score_metric_name_follows_the_task_type():
    assert score_metric_name("regression") == "neg_mae"
    assert score_metric_name("binary_classification") == "roc_auc"


# --- per-mode validation helpers --------------------------------------------


def test_increment_gap_message_accepts_a_contiguous_prefix():
    assert increment_gap_message([1, 2, 3]) is None
    assert increment_gap_message([3, 1, 2]) is None


def test_increment_gap_message_names_an_interior_hole():
    message = increment_gap_message([1, 3, 4])
    assert message is not None
    assert "[2]" in message


def test_increment_gap_message_rejects_a_head_gap():
    # MLflow returns newest-first, so an interrupted predict pass leaves the
    # *last* increments behind. Accepting {2, 3} would score increment 2 against
    # episode 0 and print a drift curve that improves over time.
    message = increment_gap_message([2, 3])
    assert message is not None
    assert "[1]" in message


def test_increment_gap_message_reports_an_empty_selection():
    assert "no checkpoint columns" in increment_gap_message([])


def test_pooled_protocol_pools_across_increments():
    summary = {
        "1": {"max_training_steps": ["2000"], "val_max_rows": ["25000"]},
        "2": {"max_training_steps": ["50"], "val_max_rows": ["25000"]},
    }
    pooled = pooled_protocol(summary, ("max_training_steps", "val_max_rows"))
    assert pooled["max_training_steps"] == ["2000", "50"]
    assert pooled["val_max_rows"] == ["25000"]


def test_pooled_protocol_is_empty_without_a_record():
    assert pooled_protocol(None) == {}
    assert pooled_protocol({}) == {}


def test_protocol_verdict_reports_unverified_without_a_sidecar():
    pooled, conflict, verified = protocol_verdict(None)
    assert (pooled, conflict, verified) == ({}, None, False)


def test_protocol_verdict_accepts_an_agreeing_chain():
    pooled, conflict, verified = protocol_verdict(sidecar_for(MODE_PREDICTIONS["er"]))
    assert conflict is None
    assert verified is True
    assert pooled["max_training_steps"] == ["2000"]


def test_protocol_verdict_names_the_disagreeing_param():
    sidecar = sidecar_for(
        MODE_PREDICTIONS["er"], overrides={2: {"val_max_rows": ["1000"]}}
    )
    _, conflict, verified = protocol_verdict(sidecar)
    assert verified is True
    assert conflict is not None
    assert "val_max_rows" in conflict


# --- replicate slots --------------------------------------------------------


def test_replicate_column_sets_is_empty_when_disabled():
    groups = {1: ["1_a", "1_b"], 2: ["2_a", "2_b"]}
    assert replicate_column_sets(groups, 0) == []
    assert replicate_column_sets({}, -1) == []


def test_replicate_column_sets_takes_one_column_per_increment_in_order():
    groups = {2: ["2_a", "2_b"], 1: ["1_a", "1_b"]}
    assert replicate_column_sets(groups, -1) == [
        ("trial0", ["1_a", "2_a"]),
        ("trial1", ["1_b", "2_b"]),
    ]


def test_replicate_column_sets_caps_at_the_thinnest_increment():
    # A slot that existed for increment 1 but not increment 2 would leave R with
    # more rows than columns.
    groups = {1: ["1_a", "1_b", "1_c"], 2: ["2_a"]}
    assert replicate_column_sets(groups, -1) == [("trial0", ["1_a", "2_a"])]
    assert replicate_column_sets(groups, 5) == [("trial0", ["1_a", "2_a"])]


# --- analyse_mode -----------------------------------------------------------


def analyse(predictions, spec=None, **kwargs):
    """Analyse one in-memory mode with an agreeing protocol unless told otherwise."""
    kwargs.setdefault("sidecar", sidecar_for(predictions))
    return analyse_mode(
        "er",
        regression_frame(predictions),
        spec or dict(BASE_SPEC),
        dataset=DATASET,
        task=TASK,
        **kwargs,
    )


def values_of(result, metric_name, seed):
    """The rows of one metric for one seed label, ordered by episode."""
    frame = tidy_frame(result.rows)
    rows = frame[(frame["metric_name"] == metric_name) & (frame["seed"] == seed)]
    return rows.sort_values(["train_episode", "episode"])["value"].tolist()


def test_analyse_mode_builds_the_hand_computed_matrix():
    # One trial per increment predicting 10 / 20 / 30 on episodes whose targets
    # are 10 / 20 / 30: R[i, j] = -|10 * (j - i)| * 1, the negated distance.
    result = analyse({1: [10.0], 2: [20.0], 3: [30.0]}, replicates=1)
    assert result.included, result.reason
    assert values_of(result, "R", "trial0") == [
        0.0, -10.0, -20.0,
        -10.0, 0.0, -10.0,
        -20.0, -10.0, 0.0,
    ]
    assert values_of(result, "average_accuracy", "trial0") == [-10.0]
    assert values_of(result, "backward_transfer", "trial0") == [-15.0]
    assert values_of(result, "drift_curve", "trial0") == [0.0, -10.0, -20.0]
    assert values_of(result, "final_model_score", "trial0") == [-20.0, -10.0, 0.0]
    # Forgetting is undefined for the last episode: nothing trained after it.
    assert values_of(result, "per_episode_forgetting", "trial0") == [20.0, 10.0]


def test_analyse_mode_separates_the_trial_slots():
    result = analyse({1: [10.0, 11.0], 2: [20.0, 21.0], 3: [30.0, 31.0]}, replicates=-1)
    assert result.n_replicates == 2
    assert values_of(result, "average_accuracy", "trial0") == [-10.0]
    # Slot 1 always predicts one unit high: it is off by 21, 11 and 1.
    assert values_of(result, "average_accuracy", "trial1") == [-11.0]


def test_analyse_mode_emits_the_aggregate_row_under_its_own_seed_label():
    result = analyse({1: [10.0], 2: [20.0], 3: [30.0]}, replicates=0, aggregate="first")
    seeds = {row["seed"] for row in result.rows}
    assert seeds == {f"{AGG_SEED_PREFIX}first"}
    assert result.n_replicates == 0


def test_analyse_mode_excludes_an_increment_gap():
    result = analyse({1: [10.0], 3: [30.0]})
    assert not result.included
    assert "[2]" in result.reason
    assert result.rows == []


def test_analyse_mode_excludes_a_protocol_conflict():
    predictions = {1: [10.0], 2: [20.0], 3: [30.0]}
    result = analyse(
        predictions,
        sidecar=sidecar_for(predictions, overrides={3: {"max_training_steps": ["50"]}}),
    )
    assert not result.included
    assert "max_training_steps" in result.reason
    assert result.rows == []


def test_analyse_mode_records_the_prediction_seed_from_the_sidecar():
    predictions = {1: [10.0], 2: [20.0], 3: [30.0]}
    sidecar = dict(sidecar_for(predictions), seed=7)
    assert analyse(predictions, sidecar=sidecar).predict_seed == 7
    assert analyse(predictions, sidecar=None).predict_seed is None


def test_analyse_mode_includes_but_flags_a_missing_protocol_record():
    result = analyse({1: [10.0], 2: [20.0], 3: [30.0]}, sidecar=None)
    assert result.included
    assert result.protocol_verified is False


def test_analyse_mode_excludes_a_missing_protocol_record_when_required():
    result = analyse({1: [10.0], 2: [20.0], 3: [30.0]}, sidecar=None,
                     require_protocol=True)
    assert not result.included
    assert "--require-protocol" in result.reason


def test_analyse_mode_excludes_more_checkpoints_than_episodes():
    # Four increments against a three-episode grid: the grid is not the one this
    # chain ran on, so nothing can be lined up.
    result = analyse({1: [10.0], 2: [20.0], 3: [30.0], 4: [40.0]})
    assert not result.included
    assert "episode grid" in result.reason


def test_analyse_mode_excludes_a_missing_target_column():
    spec = dict(BASE_SPEC, target_col="not_a_column")
    result = analyse({1: [10.0], 2: [20.0], 3: [30.0]}, spec=spec)
    assert not result.included
    # The exact up-front message, not whatever `build_matrix` would have raised
    # further down: the guard has to fire before any work is done.
    assert result.reason == "column 'not_a_column' is not in the predictions CSV"


def test_analyse_mode_excludes_a_missing_time_column():
    # The time column is read before the matrix is built, so without the guard
    # this is a bare KeyError that takes the whole run down instead of excluding
    # one mode.
    spec = dict(BASE_SPEC, time_col="nope")
    result = analyse({1: [10.0], 2: [20.0], 3: [30.0]}, spec=spec)
    assert not result.included
    assert result.reason == "column 'nope' is not in the predictions CSV"


def test_analyse_mode_truncates_the_grid_when_a_chain_stops_early():
    # Two increments against the three-episode grid: keep the two episodes the
    # checkpoints line up with, and report that count.
    result = analyse({1: [10.0], 2: [20.0]}, replicates=1)
    assert result.included, result.reason
    assert result.n_episodes == 2
    assert values_of(result, "R", "trial0") == [0.0, -10.0, -10.0, 0.0]


def test_max_episodes_recomputes_scalars_rather_than_slicing_them():
    predictions = {1: [10.0], 2: [20.0], 3: [30.0]}
    full = analyse(predictions, replicates=1)
    cut = analyse(predictions, replicates=1, max_episodes=2)
    assert values_of(full, "average_accuracy", "trial0") == [-10.0]
    # Over two episodes the deployed model is the second checkpoint, scoring
    # -10 and 0, so ACC is -5 -- not the mean of the full run's first two cells.
    assert values_of(cut, "average_accuracy", "trial0") == [-5.0]
    assert cut.n_episodes == 2
    assert values_of(cut, "R", "trial0") == [0.0, -10.0, -10.0, 0.0]


def test_max_episodes_rejects_a_nonsensical_cut():
    with pytest.raises(ValueError, match="at least 1"):
        analyse({1: [10.0], 2: [20.0]}, max_episodes=0)


def test_analyse_mode_ignores_a_data_column_that_parses_as_a_checkpoint():
    # run_predictions.py copies the whole task table into the CSV, so a task with
    # a column named "2_driverId" would otherwise sort in among increment 2's
    # seeds and become a row of R: a column of entity ids scored as a model.
    predictions = {1: [10.0], 2: [20.0], 3: [30.0]}
    spec = dict(BASE_SPEC, table_columns=TABLE_COLUMNS + ["2_driverId"])
    frame = regression_frame(predictions, extra_columns={"2_driverId": [7] * 6})
    result = analyse_mode(
        "er", frame, spec, dataset=DATASET, task=TASK,
        sidecar=sidecar_for(predictions), replicates=1,
    )
    assert result.included, result.reason
    assert result.n_episodes == 3
    assert values_of(result, "R", "trial0") == [
        0.0, -10.0, -20.0, -10.0, 0.0, -10.0, -20.0, -10.0, 0.0,
    ]


def test_analyse_mode_does_not_mutate_the_caller_frame():
    predictions = {1: [10.0], 2: [20.0], 3: [30.0]}
    frame = regression_frame(predictions)
    frame["date"] = frame["date"].astype(str)
    before = frame["date"].tolist()
    result = analyse_mode(
        "er", frame, dict(BASE_SPEC), dataset=DATASET, task=TASK,
        sidecar=sidecar_for(predictions),
    )
    assert result.included, result.reason
    assert frame["date"].dtype == object
    assert frame["date"].tolist() == before


def test_analyse_mode_scores_a_binary_task_with_roc_auc():
    frame = pd.DataFrame(
        {
            "driverId": [1, 2, 1, 2, 1, 2],
            "date": [pd.Timestamp("2020-01-01") + pd.Timedelta(days=d) for d in ROW_DAYS],
            "y": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            # Ranks the positive above the negative in every episode: AUC 1.0.
            "1_a": [0.1, 0.9, 0.1, 0.9, 0.1, 0.9],
            "2_b": [0.1, 0.9, 0.1, 0.9, 0.1, 0.9],
            "3_c": [0.9, 0.1, 0.9, 0.1, 0.9, 0.1],
        }
    )
    spec = dict(BASE_SPEC, task_type="binary_classification")
    result = analyse_mode(
        "er", frame, spec, dataset=DATASET, task=TASK, replicates=1,
        sidecar=sidecar_for({1: [0], 2: [0], 3: [0]}),
    )
    assert result.included, result.reason
    assert values_of(result, "R", "trial0") == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                0.0, 0.0, 0.0]
    assert {row["score_metric"] for row in result.rows} == {"roc_auc"}


# --- drift ------------------------------------------------------------------


def test_drift_rows_describe_the_full_grid_not_a_truncated_chain():
    frame = regression_frame({1: [10.0]})
    rows = drift_rows(frame, dict(BASE_SPEC), DATASET, TASK)
    table = tidy_frame(rows)
    assert set(table["mode"]) == {DRIFT_MODE}
    counts = table[table["metric_name"] == "episode_n_rows"]
    assert counts["value"].tolist() == [2.0, 2.0, 2.0]
    means = table[table["metric_name"] == "target_mean"]
    assert means["value"].tolist() == [10.0, 20.0, 30.0]


def test_drift_rows_report_the_signed_shift_from_the_first_episode():
    rows = drift_rows(regression_frame({1: [10.0]}), dict(BASE_SPEC), DATASET, TASK)
    table = tidy_frame(rows)
    drift = table[table["metric_name"] == "target_drift"]
    # Episode 0 has zero spread, so drift.py falls back to a scale of 1.0 and the
    # values are the raw shift of the mean: 0, +10, +20.
    assert drift["value"].tolist() == [0.0, 10.0, 20.0]


def test_drift_rows_use_the_positive_rate_for_a_binary_task():
    frame = pd.DataFrame(
        {
            "driverId": [1, 2, 1, 2, 1, 2],
            "date": [pd.Timestamp("2020-01-01") + pd.Timedelta(days=d) for d in ROW_DAYS],
            "y": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        }
    )
    table = tidy_frame(
        drift_rows(frame, dict(BASE_SPEC, task_type="binary_classification"),
                   DATASET, TASK)
    )
    rates = table[table["metric_name"] == "target_positive_rate"]
    assert rates["value"].tolist() == [0.0, 0.5, 1.0]
    drift = table[table["metric_name"] == "target_drift"]
    assert drift["value"].tolist() == [0.0, 0.5, 1.0]


# --- the tidy table ---------------------------------------------------------


def test_tidy_frame_has_the_documented_columns_and_nullable_indices():
    result = analyse({1: [10.0], 2: [20.0], 3: [30.0]}, replicates=1)
    table = tidy_frame(result.rows)
    assert list(table.columns) == list(TIDY_COLUMNS)
    scalars = table[table["metric_name"] == "average_accuracy"]
    # A scalar metric must leave both index columns empty: a 0 there would read
    # as episode 0.
    assert scalars["episode"].isna().all()
    assert scalars["train_episode"].isna().all()
    cells = table[table["metric_name"] == "R"]
    assert cells["train_episode"].notna().all()
    assert str(table["episode"].dtype) == "Int64"


# --- cross-mode refusal -----------------------------------------------------


def included_result(mode, n_episodes=3, protocol=None, predict_seed=42):
    return ModeResult(
        mode, True, "", rows=[], n_episodes=n_episodes,
        protocol=protocol or {"max_training_steps": ["2000"]},
        protocol_verified=True, predict_seed=predict_seed,
    )


def test_cross_mode_refusal_passes_a_coherent_grid():
    assert cross_mode_refusal([included_result("joint"), included_result("er")]) is None


def test_cross_mode_refusal_ignores_a_single_mode():
    assert cross_mode_refusal([included_result("er", n_episodes=2)]) is None


def test_cross_mode_refusal_names_a_protocol_disagreement():
    message = cross_mode_refusal(
        [
            included_result("joint"),
            included_result("er", protocol={"max_training_steps": ["50"]}),
        ]
    )
    assert message is not None
    assert "max_training_steps" in message
    assert "er" in message


def test_cross_mode_refusal_names_different_prediction_seeds():
    message = cross_mode_refusal(
        [included_result("joint"), included_result("er", predict_seed=7)]
    )
    assert message is not None
    assert "prediction seeds" in message
    assert "er=7" in message


def test_cross_mode_refusal_tolerates_an_unrecorded_prediction_seed():
    # Unknown is not the same as different: an old CSV with no sidecar must not
    # be reported as disagreeing with one that has a seed.
    assert cross_mode_refusal(
        [included_result("joint"), included_result("er", predict_seed=None)]
    ) is None


def test_cross_mode_refusal_names_unequal_episode_counts():
    message = cross_mode_refusal(
        [included_result("joint", n_episodes=11), included_result("er", n_episodes=7)]
    )
    assert message is not None
    assert "er=7" in message
    assert "--truncate-to-common" in message


def test_cross_mode_refusal_skips_excluded_modes():
    excluded = ModeResult("naive", False, "nope", n_episodes=0)
    assert cross_mode_refusal([included_result("joint"), excluded]) is None


# --- the printed summary ----------------------------------------------------


def summary_table(**kwargs):
    rows = []
    for mode, predictions in MODE_PREDICTIONS.items():
        result = analyse_mode(
            mode, regression_frame(predictions), dict(BASE_SPEC),
            dataset=DATASET, task=TASK, sidecar=sidecar_for(predictions), **kwargs
        )
        assert result.included, result.reason
        rows.extend(result.rows)
    return tidy_frame(rows)


def test_scalar_summary_prefers_the_trial_replicates():
    summary = scalar_summary(summary_table(replicates=-1))
    row = summary[
        (summary["mode"] == "from_scratch") & (summary["metric_name"] == "average_accuracy")
    ].iloc[0]
    assert row["source"] == "trials"
    # Two trials only: the aggregate column must not be pooled in as a third.
    assert row["n"] == 2
    assert row["mean"] == pytest.approx(-10.5)
    assert row["sd"] == pytest.approx(np.std([-10.0, -11.0], ddof=1))


def test_scalar_summary_ignores_a_lone_replicate():
    # One trial gives no spread, so the aggregate column is what gets reported.
    # The two disagree here by construction: `agg:mean` averages both trials'
    # predictions (11 / 21 / 31) while trial0 is the first of them (10 / 20 / 30).
    predictions = {1: [10.0, 12.0], 2: [20.0, 22.0], 3: [30.0, 32.0]}
    result = analyse_mode(
        "er", regression_frame(predictions), dict(BASE_SPEC), dataset=DATASET,
        task=TASK, sidecar=sidecar_for(predictions), aggregate="mean", replicates=1,
    )
    assert result.included, result.reason
    summary = scalar_summary(tidy_frame(result.rows))
    row = summary[summary["metric_name"] == "average_accuracy"].iloc[0]
    assert row["source"] == "aggregate"
    assert row["n"] == 1
    assert row["mean"] == pytest.approx(-11.0)


def test_scalar_summary_falls_back_to_the_aggregate_without_replicates():
    summary = scalar_summary(summary_table(replicates=0))
    row = summary[
        (summary["mode"] == "from_scratch") & (summary["metric_name"] == "average_accuracy")
    ].iloc[0]
    assert row["source"] == "aggregate"
    assert row["n"] == 1
    assert np.isnan(row["sd"])


def test_format_summary_table_leads_with_the_reference_frame():
    lines = format_summary_table(summary_table(replicates=-1), DATASET, TASK, "neg_mae")
    body = [line.split()[0] for line in lines if line and not line.startswith("-")]
    modes = [token for token in body if token in MODE_PREDICTIONS]
    assert modes == ["from_scratch", "joint", "naive", "er"]
    assert any("higher is better" in line for line in lines)


def mode_line(lines, mode):
    """The one data row of the printed table belonging to a mode."""
    matches = [line for line in lines if line.split()[:1] == [mode]]
    assert len(matches) == 1, matches
    return matches[0]


def test_format_summary_table_shows_a_spread_only_when_one_was_measured():
    with_trials = format_summary_table(
        summary_table(replicates=-1), DATASET, TASK, "neg_mae"
    )
    without = format_summary_table(summary_table(replicates=0), DATASET, TASK, "neg_mae")
    assert "+-" in mode_line(with_trials, "from_scratch")
    assert "+-" not in mode_line(without, "from_scratch")
    assert "no trial replicates available" in "\n".join(without)


def test_format_exclusions_names_every_missing_mode():
    results = [
        ModeResult("joint", True, ""),
        ModeResult("naive", False, "increments are not a prefix"),
    ]
    lines = format_exclusions(results, ["joint", "naive", "lwf"])
    text = "\n".join(lines)
    assert "naive: increments are not a prefix" in text
    assert "lwf: no result" in text
    assert "EXCLUDED 2 of 3" in text


def test_format_exclusions_is_silent_when_nothing_was_excluded():
    assert format_exclusions([ModeResult("joint", True, "")], ["joint"]) == []


# --- the CLI ----------------------------------------------------------------


def read_results(root):
    return pd.read_csv(Path(root) / "out_results.csv")


def test_main_from_csv_writes_the_tidy_table_and_the_metadata(grid, capsys):
    assert ra.main(cli(grid)) == 0
    printed = capsys.readouterr().out
    assert "from_scratch" in printed
    assert "neg_mae" in printed

    table = read_results(grid)
    assert list(table.columns) == list(TIDY_COLUMNS)
    assert set(table["mode"]) == set(MODE_PREDICTIONS) | {DRIFT_MODE}
    # The reference bounds behave as they must: `joint` retains more than `naive`.
    final = table[(table["metric_name"] == "average_accuracy")
                  & (table["seed"] == "trial0")]
    scores = dict(zip(final["mode"], final["value"]))
    assert scores["joint"] > scores["naive"]

    meta = json.loads((Path(grid) / "out_meta.json").read_text())
    assert meta["score_metric"] == "neg_mae"
    assert meta["modes"]["er"]["included"] is True
    assert meta["modes"]["er"]["n_episodes"] == 3
    assert meta["modes"]["er"]["predict_seed"] == 42
    assert len(meta["episode_grid"]) == 4


def test_main_excludes_a_missing_mode_and_says_so(grid, capsys):
    code = ra.main(cli(grid, modes=list(MODE_PREDICTIONS) + ["lwf"]))
    assert code == 1
    errors = capsys.readouterr().err
    assert "EXCLUDED" in errors
    assert "lwf" in errors
    # The table is still written for the modes that survived.
    assert set(read_results(grid)["mode"]) == set(MODE_PREDICTIONS) | {DRIFT_MODE}


def test_main_refuses_when_modes_used_different_protocols(grid, capsys):
    path = Path(grid) / "data" / experiment_name(PREFIX, "naive")
    sidecar = path / f"{DATASET}_{TASK}_predictions.seeds.json"
    payload = json.loads(sidecar.read_text())
    for entry in payload["protocol"].values():
        entry["max_training_steps"] = ["50"]
    sidecar.write_text(json.dumps(payload))

    assert ra.main(cli(grid)) == 2
    errors = capsys.readouterr().err
    assert "REFUSING" in errors
    assert "max_training_steps" in errors
    assert not (Path(grid) / "out_results.csv").exists()


def test_main_refuses_when_modes_were_scored_with_different_seeds(grid, capsys):
    sidecar = (Path(grid) / "data" / experiment_name(PREFIX, "naive")
               / f"{DATASET}_{TASK}_predictions.seeds.json")
    payload = json.loads(sidecar.read_text())
    payload["seed"] = 7
    sidecar.write_text(json.dumps(payload))

    assert ra.main(cli(grid)) == 2
    errors = capsys.readouterr().err
    assert "prediction seeds" in errors
    assert not (Path(grid) / "out_results.csv").exists()


def test_main_refuses_when_modes_cover_different_episode_counts(grid, capsys):
    csv_path = (Path(grid) / "data" / experiment_name(PREFIX, "er")
                / f"{DATASET}_{TASK}_predictions.csv")
    frame = pd.read_csv(csv_path)
    frame.drop(columns=[c for c in frame.columns if c.startswith("3_")]).to_csv(
        csv_path, index=False
    )

    assert ra.main(cli(grid)) == 2
    errors = capsys.readouterr().err
    assert "different numbers of episodes" in errors
    assert not (Path(grid) / "out_results.csv").exists()


def test_main_truncate_to_common_rebuilds_every_mode(grid, capsys):
    csv_path = (Path(grid) / "data" / experiment_name(PREFIX, "er")
                / f"{DATASET}_{TASK}_predictions.csv")
    frame = pd.read_csv(csv_path)
    frame.drop(columns=[c for c in frame.columns if c.startswith("3_")]).to_csv(
        csv_path, index=False
    )

    assert ra.main(cli(grid, "--truncate-to-common")) == 0
    assert "rebuilding every mode over the first 2" in capsys.readouterr().err

    table = read_results(grid)
    episodes = table[table["metric_name"] == "n_episodes"]
    assert set(episodes["value"]) == {2.0}
    # `from_scratch` predicts 10 then 20 in slot 0, so over two episodes the
    # deployed model scores -10 and 0 and ACC is -5. The full-length answer is
    # -10, so a sliced table would show that instead.
    accuracy = table[(table["metric_name"] == "average_accuracy")
                     & (table["mode"] == "from_scratch")
                     & (table["seed"] == "trial0")]
    assert accuracy["value"].tolist() == [-5.0]
    # Drift still describes the whole task, not the truncated chains.
    counts = table[table["metric_name"] == "episode_n_rows"]
    assert len(counts) == 3


def test_main_refuses_when_no_mode_is_usable(tmp_path, capsys):
    write_spec(tmp_path)
    (tmp_path / "data").mkdir()
    assert ra.main(cli(tmp_path)) == 2
    assert "no requested mode" in capsys.readouterr().err


def test_main_dry_run_prints_the_commands_and_runs_nothing(grid, capsys, monkeypatch):
    def explode(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("a dry run must not launch a prediction pass")

    monkeypatch.setattr(ra.subprocess, "call", explode)
    argv = [
        "--dataset", DATASET, "--task", TASK, "--modes", "lwf",
        "--data-root", str(Path(grid) / "data"), "--experiment-prefix", PREFIX,
        "--task-spec", str(Path(grid) / "spec.json"), "--dry-run",
    ]
    assert ra.main(argv) == 0
    printed = capsys.readouterr().out
    assert "run_predictions.py" in printed
    assert "--chain_id=rel-fake/fake-position/lwf/" in printed


def test_main_reuses_an_existing_csv_instead_of_rerunning_predictions(grid, monkeypatch):
    def explode(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("the CSV already exists; predictions must be reused")

    monkeypatch.setattr(ra.subprocess, "call", explode)
    argv = [
        "--dataset", DATASET, "--task", TASK, "--modes", *MODE_PREDICTIONS,
        "--data-root", str(Path(grid) / "data"), "--experiment-prefix", PREFIX,
        "--task-spec", str(Path(grid) / "spec.json"), "--out", str(Path(grid) / "out"),
    ]
    assert ra.main(argv) == 0


def test_main_runs_predictions_for_a_mode_with_no_csv(grid, monkeypatch):
    launched = []

    def fake_call(command, **kwargs):
        launched.append(command)
        write_mode(grid, "lwf", predictions=MODE_PREDICTIONS["er"])
        return 0

    monkeypatch.setattr(ra.subprocess, "call", fake_call)
    argv = [
        "--dataset", DATASET, "--task", TASK, "--modes", *MODE_PREDICTIONS, "lwf",
        "--data-root", str(Path(grid) / "data"), "--experiment-prefix", PREFIX,
        "--task-spec", str(Path(grid) / "spec.json"), "--out", str(Path(grid) / "out"),
    ]
    assert ra.main(argv) == 0
    assert len(launched) == 1
    assert "--mlflow_experiment=smoke_cl_lwf" in launched[0]
    assert "lwf" in set(read_results(grid)["mode"])


def test_main_excludes_a_mode_whose_prediction_pass_failed(grid, monkeypatch, capsys):
    monkeypatch.setattr(ra.subprocess, "call", lambda command, **kwargs: 3)
    argv = [
        "--dataset", DATASET, "--task", TASK, "--modes", *MODE_PREDICTIONS, "lwf",
        "--data-root", str(Path(grid) / "data"), "--experiment-prefix", PREFIX,
        "--task-spec", str(Path(grid) / "spec.json"), "--out", str(Path(grid) / "out"),
    ]
    assert ra.main(argv) == 1
    assert "exited 3" in capsys.readouterr().err


def test_main_write_task_spec_asks_relbench_once_and_exits(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_spec(dataset, task, val_delta_days=None):
        calls.append((dataset, task, val_delta_days))
        return dict(BASE_SPEC, splits=list(RAW_SPLITS))

    monkeypatch.setattr(ra, "task_spec_from_relbench", fake_spec)
    target = tmp_path / "nested" / "spec.json"
    code = ra.main([
        "--dataset", DATASET, "--task", TASK,
        "--write-task-spec", str(target), "--val-delta-days", "30",
    ])
    assert code == 0
    assert calls == [(DATASET, TASK, 30.0)]
    written = json.loads(target.read_text())
    assert written["target_col"] == "y"
    assert "Wrote" in capsys.readouterr().out
