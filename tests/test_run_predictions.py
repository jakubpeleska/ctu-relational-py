"""Tests for the prediction pass and the port-fidelity gate.

Both modules are covered here because they share one concern: a number is only
comparable to another number when the thing that produced it was held fixed.
`run_predictions.py` holds the neighbour sample fixed across checkpoints;
`compare_to_published.py` refuses to compare across validation protocols.

Nothing here touches MLflow, the network or a GPU: the frames are hand-built and
the graph is a 600-node synthetic.
"""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

import scripts.compare_to_published as ctp
from experiments.continuous_learning.run_predictions import (
    DEFAULT_SEED,
    MISSING,
    UNKNOWN_INCREMENT,
    build_run_filter,
    format_conflict_message,
    format_protocol_report,
    generate_all_predictions_df,
    increment_label,
    load_sidecar,
    new_sidecar,
    order_runs_by_start_time,
    parse_args,
    prediction_columns,
    protocol_conflicts,
    protocol_summary,
    seed_all,
    seed_conflict_message,
    sidecar_path,
    unrecorded_columns,
    write_sidecar,
)
from scripts.build_evaluation_matrix import checkpoint_columns_by_increment


# --- fixtures -----------------------------------------------------------------


def runs_frame(specs):
    """A runs frame shaped like `get_experiment_runs_df` output.

    Each spec is a dict of param columns; `_run_id` and `_start_time` are filled
    in when absent so the frame carries the two attribute columns the pass reads.
    """
    rows = []
    for i, spec in enumerate(specs):
        row = {"_run_id": f"run{i}", "_start_time": 1_000 + i}
        row.update(spec)
        rows.append(row)
    return pd.DataFrame(rows)


def real_chain(n_increments=3, seeds=5, **overrides):
    """A well-formed chain: `seeds` runs per increment, one protocol throughout."""
    protocol = {
        "max_training_steps": "2000",
        "val_check_interval": "100",
        "val_max_rows": "25000",
        "chain_id": "rel-f1/driver-position/ft_full/models",
    }
    protocol.update(overrides)
    return [
        {"increment": str(inc), **protocol}
        for inc in range(1, n_increments + 1)
        for _ in range(seeds)
    ]


# --- run selection ------------------------------------------------------------


def test_filter_without_chain_id_selects_every_chain():
    where = build_run_filter("rel-f1", "driver-position")
    assert "params.dataset_name = 'rel-f1'" in where
    assert "params.task_name = 'driver-position'" in where
    assert "attributes.status = 'FINISHED'" in where
    assert "chain_id" not in where


def test_filter_pins_the_chain_when_one_is_given():
    where = build_run_filter("rel-f1", "driver-position", chain_id="chain-a")
    assert "params.chain_id = 'chain-a'" in where


def test_filter_rejects_a_value_that_would_break_out_of_the_quoted_literal():
    with pytest.raises(ValueError, match="quote"):
        build_run_filter("rel-f1", "driver-position", chain_id="a' or '1' = '1")


def test_filter_rejects_an_empty_value():
    with pytest.raises(ValueError, match="non-empty"):
        build_run_filter("rel-f1", "")


def test_runs_are_reordered_oldest_first():
    # MLflow answers start_time DESC, which is the order this arrives in.
    df = pd.DataFrame(
        {"_run_id": ["newest", "middle", "oldest"], "_start_time": [300, 200, 100]}
    )
    assert list(order_runs_by_start_time(df)["_run_id"]) == ["oldest", "middle", "newest"]


def test_reordering_puts_runs_without_a_start_time_last():
    df = pd.DataFrame({"_run_id": ["a", "b", "c"], "_start_time": [np.nan, 200, 100]})
    assert list(order_runs_by_start_time(df)["_run_id"]) == ["c", "b", "a"]


def test_reordering_tolerates_a_frame_without_start_times():
    df = pd.DataFrame({"_run_id": ["a", "b"]})
    assert list(order_runs_by_start_time(df)["_run_id"]) == ["a", "b"]
    assert order_runs_by_start_time(pd.DataFrame()).empty


# --- increment labels ---------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [("3", "3"), (3, "3"), (3.0, "3"), ("3.0", "3"), (" 7 ", "7")],
)
def test_increment_label_normalises_to_a_bare_integer(value, expected):
    assert increment_label(value) == expected


@pytest.mark.parametrize("value", [None, float("nan"), "", "   "])
def test_increment_label_marks_a_missing_increment(value):
    assert increment_label(value) == UNKNOWN_INCREMENT


def test_float_increment_still_yields_a_column_the_matrix_builder_can_parse():
    # A param column promoted to float gives 3.0; "3.0_run" fails the builder's
    # ^(\d+)_ pattern and the checkpoint would vanish from R without a word.
    column = f"{increment_label(3.0)}_abc123"
    assert checkpoint_columns_by_increment([column]) == {3: [column]}


def test_unknown_increment_never_becomes_a_row_of_R():
    column = f"{increment_label(None)}_abc123"
    assert checkpoint_columns_by_increment([column]) == {}


# --- protocol reporting -------------------------------------------------------


def test_summary_counts_the_runs_of_each_increment():
    summary = protocol_summary(runs_frame(real_chain(n_increments=3, seeds=5)))
    assert [entry["n_runs"] for entry in summary.values()] == [5, 5, 5]
    assert list(summary) == ["1", "2", "3"]


def test_summary_reports_one_value_per_param_for_a_clean_chain():
    summary = protocol_summary(runs_frame(real_chain()))
    assert summary["1"]["max_training_steps"] == ["2000"]
    assert summary["1"]["val_max_rows"] == ["25000"]


def test_summary_marks_a_param_no_run_logged():
    summary = protocol_summary(runs_frame(real_chain()))
    assert summary["1"]["val_delta_days"] == [MISSING]


def test_summary_exposes_a_smoke_chain_merged_into_the_selection():
    # Two increments of a 50-step smoke chain in the same experiment: the run
    # count of increments 1-2 doubles and both protocols show up side by side.
    merged = real_chain(n_increments=3, seeds=5) + real_chain(
        n_increments=2,
        seeds=1,
        max_training_steps="50",
        val_check_interval="10",
        chain_id="smoke",
    )
    summary = protocol_summary(runs_frame(merged))
    assert [entry["n_runs"] for entry in summary.values()] == [6, 6, 5]
    assert summary["1"]["max_training_steps"] == ["2000", "50"]


def test_summary_orders_numeric_increments_and_puts_unknown_last():
    frame = runs_frame([{"increment": "10"}, {"increment": None}, {"increment": "2"}])
    assert list(protocol_summary(frame)) == ["2", "10", UNKNOWN_INCREMENT]


def test_summary_reads_the_same_value_written_as_string_and_as_float():
    # A frame built from runs that disagree on which params exist promotes the
    # column to float; 2000.0 and "2000" are the same protocol.
    frame = runs_frame(
        [
            {"increment": "1", "max_training_steps": 2000.0},
            {"increment": "1", "max_training_steps": "2000"},
        ]
    )
    assert protocol_summary(frame)["1"]["max_training_steps"] == ["2000"]


def test_summary_of_an_empty_frame_is_empty():
    assert protocol_summary(pd.DataFrame()) == {}


def test_a_clean_chain_reports_no_conflict():
    assert protocol_conflicts(protocol_summary(runs_frame(real_chain()))) == {}


def test_conflicts_pool_across_increments_not_only_within_one():
    # The smoke chain stops at increment 2, so increment 3 alone never mixes
    # protocols; only pooling shows that R would span two of them.
    merged = real_chain(n_increments=3, seeds=5) + real_chain(
        n_increments=2, seeds=1, max_training_steps="50", chain_id="smoke"
    )
    conflicts = protocol_conflicts(protocol_summary(runs_frame(merged)))
    assert conflicts["max_training_steps"] == ["2000", "50"]


def test_conflicts_catch_a_param_only_some_runs_logged():
    # Half the runs predating val_max_rows is not evidence that the halves agree.
    mixed = real_chain(n_increments=1, seeds=2) + [
        {"increment": "1", "max_training_steps": "2000", "val_check_interval": "100"}
    ]
    conflicts = protocol_conflicts(protocol_summary(runs_frame(mixed)))
    assert conflicts["val_max_rows"] == sorted(["25000", MISSING])
    assert MISSING in conflicts["val_max_rows"]


def test_report_has_a_header_and_one_line_per_increment():
    lines = format_protocol_report(protocol_summary(runs_frame(real_chain())))
    assert len(lines) == 4
    assert "increment" in lines[0] and "max_training_steps" in lines[0]
    assert "2000" in lines[1] and "25000" in lines[1]


def test_report_shows_both_protocols_of_a_mixed_increment():
    merged = real_chain(n_increments=1, seeds=1) + real_chain(
        n_increments=1, seeds=1, max_training_steps="50", chain_id="smoke"
    )
    lines = format_protocol_report(protocol_summary(runs_frame(merged)))
    assert "2000|50" in lines[1]


def test_report_of_an_empty_selection_says_so():
    assert format_protocol_report({}) == ["(no runs selected)"]


def test_conflict_message_names_the_param_the_values_and_the_remedy():
    message = format_conflict_message({"max_training_steps": ["2000", "50"]})
    assert "max_training_steps" in message and "2000" in message and "50" in message
    assert "--chain_id" in message


def test_no_conflict_produces_no_message():
    assert format_conflict_message({}) == ""


# --- seeding ------------------------------------------------------------------


def test_seed_all_returns_the_seed_it_installed():
    assert seed_all(7) == 7


def test_the_same_seed_reproduces_the_same_draws():
    seed_all(11)
    first = torch.rand(4)
    seed_all(11)
    assert torch.equal(first, torch.rand(4))


def test_a_different_seed_gives_different_draws():
    seed_all(11)
    first = torch.rand(4)
    seed_all(12)
    assert not torch.equal(first, torch.rand(4))


def test_numpy_is_seeded_too():
    seed_all(11)
    first = np.random.rand(4)
    seed_all(11)
    assert np.array_equal(first, np.random.rand(4))


@pytest.mark.parametrize(
    "bad,message",
    [
        (-1, r"must be in \[0, 2\*\*32\)"),
        (2**32, r"must be in \[0, 2\*\*32\)"),
        (1.5, "must be an integer"),
        ("42", "must be an integer"),
        (True, "must be an integer"),
        (None, "must be an integer"),
    ],
)
def test_seed_all_rejects_an_unusable_seed(bad, message):
    # Matched on the message, not merely on ValueError: numpy raises its own
    # ValueError for a negative seed, so a bare `raises` would pass even with
    # this function's own range check deleted -- after `random.seed` had
    # already moved.
    with pytest.raises(ValueError, match=message):
        seed_all(bad)


# --- the property the seeding exists for --------------------------------------


def neighbour_graph():
    """A small temporal bipartite graph whose neighbour sampling is stochastic."""
    generator = torch.Generator().manual_seed(0)
    data = HeteroData()
    n_a, n_b = 200, 400
    data["a"].num_nodes = n_a
    data["b"].num_nodes = n_b
    data["a"].time = torch.arange(n_a) * 10
    data["b"].time = torch.arange(n_b) * 5
    src = torch.randint(0, n_b, (4_000,), generator=generator)
    dst = torch.randint(0, n_a, (4_000,), generator=generator)
    data["b", "to", "a"].edge_index = torch.stack([src, dst])
    data["a", "rev_to", "b"].edge_index = torch.stack([dst, src])
    return data


def full_loader(data):
    """The loader the pass builds: unshuffled, temporal, fixed input nodes."""
    return NeighborLoader(
        data,
        num_neighbors=[8, 4],
        time_attr="time",
        input_nodes=("a", torch.arange(64)),
        input_time=torch.full((64,), 1_500),
        batch_size=16,
        temporal_strategy="uniform",
        shuffle=False,
    )


def sampled_neighbours(loader):
    """The neighbour rows each batch pulled in -- the thing that must not move."""
    return [tuple(sorted(batch["b"].n_id.tolist())) for batch in loader]


def test_one_loader_iterated_twice_samples_different_neighbours():
    # The defect this module's seeding fixes. If this ever stops holding,
    # NeighborLoader has become deterministic and the reseeding is belt and
    # braces -- but until then, an unseeded pass scores each checkpoint on a
    # different subgraph, and BWT is a difference of two such scores.
    loader = full_loader(neighbour_graph())
    assert sampled_neighbours(loader) != sampled_neighbours(loader)


def test_reseeding_before_each_pass_gives_every_checkpoint_the_same_subgraph():
    # Exactly what the pass does: one loader, reused, reseeded immediately
    # before each checkpoint's forward pass.
    loader = full_loader(neighbour_graph())
    seed_all(DEFAULT_SEED)
    first = sampled_neighbours(loader)
    seed_all(DEFAULT_SEED)
    assert sampled_neighbours(loader) == first


def test_reseeding_reproduces_the_sample_across_freshly_built_loaders():
    # A resumed pass rebuilds the loader; the sample must still line up with the
    # columns an earlier pass wrote under the same seed.
    data = neighbour_graph()
    seed_all(DEFAULT_SEED)
    first = sampled_neighbours(full_loader(data))
    seed_all(DEFAULT_SEED)
    assert sampled_neighbours(full_loader(data)) == first


def test_a_different_seed_samples_a_different_subgraph():
    data = neighbour_graph()
    seed_all(1)
    first = sampled_neighbours(full_loader(data))
    seed_all(2)
    assert sampled_neighbours(full_loader(data)) != first


# --- the seed record ----------------------------------------------------------


def test_sidecar_sits_next_to_the_csv_without_being_a_column_of_it():
    path = sidecar_path("/data/exp/rel-f1_driver-position_predictions.csv")
    assert path.name == "rel-f1_driver-position_predictions.seeds.json"
    assert str(path.parent) == "/data/exp"


def test_sidecar_roundtrips(tmp_path):
    path = sidecar_path(tmp_path / "p.csv")
    payload = new_sidecar(42, "rel-f1", "driver-position", "exp", "chain-a", {"1": {}})
    payload["columns"]["1_run0"] = 42
    write_sidecar(path, payload)
    assert load_sidecar(path) == payload


def test_a_fresh_record_records_the_seed_and_no_columns():
    payload = new_sidecar(7, "rel-f1", "driver-position")
    assert payload["seed"] == 7 and payload["columns"] == {}


def test_a_missing_record_reads_as_none(tmp_path):
    assert load_sidecar(tmp_path / "absent.seeds.json") is None


def test_a_corrupt_record_is_reported_rather_than_raised(tmp_path, capsys):
    path = tmp_path / "broken.seeds.json"
    path.write_text("{not json")
    assert load_sidecar(path) is None
    assert "WARNING" in capsys.readouterr().out


def test_a_record_that_is_not_an_object_reads_as_none(tmp_path):
    path = tmp_path / "list.seeds.json"
    path.write_text(json.dumps([1, 2, 3]))
    assert load_sidecar(path) is None


def test_no_seed_conflict_when_the_seeds_agree():
    assert seed_conflict_message({"seed": 42}, 42) is None


def test_no_seed_conflict_without_a_record():
    assert seed_conflict_message(None, 42) is None


def test_a_changed_seed_is_a_conflict():
    message = seed_conflict_message({"seed": 42}, 7)
    assert "42" in message and "7" in message


def test_a_record_predating_seed_tracking_is_not_treated_as_a_conflict():
    assert seed_conflict_message({"columns": {}}, 42) is None


def test_prediction_columns_are_the_ones_the_task_table_does_not_own():
    columns = ["driverId", "date", "position", "1_runA", "2_runB"]
    table = ["driverId", "date", "position"]
    assert prediction_columns(columns, table) == ["1_runA", "2_runB"]


def test_columns_written_before_the_seed_was_tracked_are_flagged():
    columns = ["driverId", "1_old", "1_new"]
    table = ["driverId"]
    record = {"columns": {"1_new": 42}}
    assert unrecorded_columns(columns, table, record) == ["1_old"]


def test_every_prediction_column_is_unrecorded_without_a_record():
    assert unrecorded_columns(["driverId", "1_old"], ["driverId"], None) == ["1_old"]


# --- the CLI ------------------------------------------------------------------


def test_cli_defaults_to_the_documented_seed_and_every_chain():
    args = parse_args([])
    assert args.seed == DEFAULT_SEED
    assert args.chain_id is None
    assert args.strict_protocol is False
    assert args.allow_seed_change is False


@pytest.mark.parametrize("flag", ["--chain-id", "--chain_id"])
def test_cli_accepts_the_chain_flag_in_either_spelling(flag):
    assert parse_args([flag, "chain-a"]).chain_id == "chain-a"


def test_cli_takes_a_seed():
    assert parse_args(["--seed", "7"]).seed == 7


def test_cli_can_make_a_protocol_mismatch_fatal():
    assert parse_args(["--strict-protocol"]).strict_protocol is True


def test_the_pass_exposes_the_reproducibility_controls():
    import inspect

    params = inspect.signature(generate_all_predictions_df).parameters
    assert params["seed"].default == DEFAULT_SEED
    assert params["chain_id"].default is None
    assert params["strict_protocol"].default is False


# --- compare_to_published: the protocol gate ----------------------------------


def published_side(**overrides):
    """Runs as the paper produced them: 500 validations on the full window."""
    protocol = {"max_training_steps": "2000", "val_check_interval": "4"}
    protocol.update(overrides)
    return runs_frame(
        [{"increment": str(inc), **protocol} for inc in (1, 2) for _ in range(3)]
    )


def candidate_side(**overrides):
    """Runs as this branch produces them: 20 validations on a 25k subsample."""
    protocol = {
        "max_training_steps": "2000",
        "val_check_interval": "100",
        "val_max_rows": "25000",
        "chain_id": "chain-a",
    }
    protocol.update(overrides)
    return runs_frame(
        [{"increment": str(inc), **protocol} for inc in (1, 2) for _ in range(3)]
    )


def test_distinct_params_marks_a_param_the_side_never_logged():
    assert ctp.distinct_params(published_side())["val_max_rows"] == [ctp.MISSING]


def test_distinct_params_reads_string_and_float_spellings_alike():
    frame = runs_frame([{"max_training_steps": 2000.0}, {"max_training_steps": "2000"}])
    assert ctp.distinct_params(frame)["max_training_steps"] == ["2000"]


def test_identical_protocols_match_on_every_param():
    comparison = ctp.protocol_comparison(candidate_side(), candidate_side())
    assert {entry["status"] for entry in comparison} == {"match"}
    assert ctp.protocol_blockers(comparison) == []


def test_the_published_and_current_protocols_do_not_match():
    comparison = ctp.protocol_comparison(published_side(), candidate_side())
    status = {entry["param"]: entry["status"] for entry in comparison}
    assert status["val_check_interval"] == "differs"
    assert status["val_max_rows"] == "differs"
    assert status["max_training_steps"] == "match"


def test_a_param_neither_side_logged_blocks_rather_than_passing_silently():
    # Silence is not agreement: an unlogged val_max_rows on both sides leaves
    # the subsample size unknown, not equal.
    bare = runs_frame([{"increment": "1", "max_training_steps": "2000"}])
    status = {
        entry["param"]: entry["status"]
        for entry in ctp.protocol_comparison(bare, bare)
    }
    assert status["val_max_rows"] == "unverified"
    assert status["max_training_steps"] == "match"


def test_a_side_that_mixes_protocols_internally_is_ambiguous():
    mixed = pd.concat(
        [candidate_side(), candidate_side(val_check_interval="50")], ignore_index=True
    )
    status = {
        entry["param"]: entry["status"]
        for entry in ctp.protocol_comparison(mixed, candidate_side())
    }
    assert status["val_check_interval"] == "ambiguous"


def test_blockers_are_exactly_the_non_matching_params():
    comparison = ctp.protocol_comparison(published_side(), candidate_side())
    blocked = {entry["param"] for entry in ctp.protocol_blockers(comparison)}
    assert blocked == {"val_check_interval", "val_max_rows"}


def test_blockers_include_a_param_neither_side_logged():
    # "unverified" and "ambiguous" have to block as hard as "differs": neither
    # establishes that the two sides validated the same way, which is the whole
    # precondition for comparing a best-of-N statistic.
    bare = runs_frame([{"increment": "1", "max_training_steps": "2000"}])
    comparison = ctp.protocol_comparison(bare, bare)
    blocked = {entry["param"] for entry in ctp.protocol_blockers(comparison)}
    assert blocked == {"val_check_interval", "val_max_rows"}


def test_blockers_include_a_side_that_mixes_protocols_internally():
    mixed = pd.concat(
        [candidate_side(), candidate_side(val_check_interval="50")], ignore_index=True
    )
    comparison = ctp.protocol_comparison(mixed, candidate_side())
    blocked = {entry["param"] for entry in ctp.protocol_blockers(comparison)}
    assert "val_check_interval" in blocked


def test_the_protocol_table_prints_both_sides_and_the_verdict():
    lines = ctp.format_protocol_comparison(
        ctp.protocol_comparison(published_side(), candidate_side())
    )
    body = "\n".join(lines)
    assert "val_max_rows" in body and "25000" in body and "differs" in body


def test_the_validation_budget_is_the_number_of_best_of_n_draws():
    published = {"max_training_steps": ["2000"], "val_check_interval": ["4"]}
    candidate = {"max_training_steps": ["2000"], "val_check_interval": ["100"]}
    assert ctp.validation_budget(published) == 500
    assert ctp.validation_budget(candidate) == 20


@pytest.mark.parametrize(
    "values",
    [
        {"max_training_steps": [MISSING], "val_check_interval": ["100"]},
        {"max_training_steps": ["2000"], "val_check_interval": [MISSING]},
        {"max_training_steps": ["2000"], "val_check_interval": ["0"]},
        {"max_training_steps": ["2000", "50"], "val_check_interval": ["100"]},
    ],
)
def test_an_unusable_budget_is_none(values):
    assert ctp.validation_budget(values) is None


def test_the_bias_note_names_both_budgets():
    note = ctp.format_selection_bias_note(published_side(), candidate_side())
    assert "500" in note and "20" in note


def test_no_bias_note_when_the_budgets_agree():
    assert ctp.format_selection_bias_note(candidate_side(), candidate_side()) is None


def test_na_strategy_is_flagged_when_only_one_side_predates_the_pin():
    note = ctp.na_strategy_note(published_side(), candidate_side())
    assert "na_strategy" in note and "b3bf3f7" in note and "MEAN" in note


def test_na_strategy_is_quiet_when_both_sides_are_on_the_same_side_of_the_pin():
    assert ctp.na_strategy_note(candidate_side(), candidate_side()) is None
    assert ctp.na_strategy_note(published_side(), published_side()) is None


def test_na_strategy_is_compared_directly_when_it_is_ever_logged():
    left = candidate_side(na_strategy="MEAN")
    right = candidate_side(na_strategy="ZEROS")
    note = ctp.na_strategy_note(left, right)
    assert "MEAN" in note and "ZEROS" in note


def test_na_strategy_is_quiet_when_the_logged_values_agree():
    both = candidate_side(na_strategy="MEAN")
    assert ctp.na_strategy_note(both, both) is None


def test_by_episode_groups_the_metric_by_increment():
    frame = runs_frame(
        [
            {"increment": "1", "best_val_mae": "1.0"},
            {"increment": "1", "best_val_mae": "3.0"},
            {"increment": "2", "best_val_mae": "5.0"},
        ]
    )
    assert ctp.by_episode(frame, "best_val_mae") == {1: [1.0, 3.0], 2: [5.0]}


def test_by_episode_drops_a_run_whose_metric_is_missing():
    # A nan reaching np.mean turns the whole episode into nan, and the gap line
    # then prints "nan%" instead of failing.
    frame = runs_frame(
        [
            {"increment": "1", "best_val_mae": 1.0},
            {"increment": "1", "best_val_mae": np.nan},
        ]
    )
    assert ctp.by_episode(frame, "best_val_mae") == {1: [1.0]}


def test_by_episode_drops_a_run_with_no_increment():
    frame = runs_frame(
        [
            {"increment": np.nan, "best_val_mae": 1.0},
            {"increment": "2", "best_val_mae": 2.0},
        ]
    )
    assert ctp.by_episode(frame, "best_val_mae") == {2: [2.0]}


def test_by_episode_accepts_a_float_increment():
    frame = runs_frame([{"increment": 1.0, "best_val_mae": 2.0}])
    assert ctp.by_episode(frame, "best_val_mae") == {1: [2.0]}


# --- compare_to_published: the gate end to end --------------------------------


def run_main(monkeypatch, published, candidate, argv=()):
    """`main` with MLflow replaced by two hand-built frames."""
    monkeypatch.setattr(ctp, "get_potato_client", lambda uri: None)
    monkeypatch.setattr(
        ctp,
        "load",
        lambda client, experiment, dataset, task: (
            published if experiment == "pub" else candidate
        ),
    )
    where = ["--dataset", "rel-f1", "--task", "driver-position"]
    sides = ["--published", "pub", "--candidate", "cand"]
    return ctp.main([*where, *sides, "--metric", "best_val_mae", *argv])


def scored(frame, values_by_increment):
    """Attach a metric column so the episode table has something to compare."""
    frame = frame.copy()
    frame["best_val_mae"] = [values_by_increment[int(inc)] for inc in frame["increment"]]
    return frame


def test_mismatched_protocols_refuse_to_compare(monkeypatch, capsys):
    code = run_main(
        monkeypatch,
        scored(published_side(), {1: 1.0, 2: 2.0}),
        scored(candidate_side(), {1: 1.0, 2: 2.0}),
    )
    out = capsys.readouterr().out
    assert code == 3
    assert "REFUSING TO COMPARE" in out
    assert "val_max_rows" in out
    # The episode table must not be printed: identical numbers under two
    # protocols are exactly the false PASS this gate exists to prevent.
    assert "rel gap" not in out


def test_an_unverifiable_protocol_also_refuses(monkeypatch, capsys):
    # Neither side logs val_max_rows. That is not agreement -- one of them may
    # have validated on the full window and the other on a subsample.
    bare = runs_frame(
        [
            {"increment": "1", "max_training_steps": "2000", "val_check_interval": "100"},
            {"increment": "2", "max_training_steps": "2000", "val_check_interval": "100"},
        ]
    )
    code = run_main(
        monkeypatch,
        scored(bare, {1: 1.0, 2: 2.0}),
        scored(bare, {1: 1.0, 2: 2.0}),
    )
    out = capsys.readouterr().out
    assert code == 3
    assert "unverified" in out and "rel gap" not in out


def test_matching_protocols_compare_and_read_as_consistent(monkeypatch, capsys):
    code = run_main(
        monkeypatch,
        scored(candidate_side(), {1: 1.0, 2: 2.0}),
        scored(candidate_side(), {1: 1.0, 2: 2.5}),
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "CONSISTENT" in out and "rel gap" in out
    # The wording may not claim more than a best-of-N comparison supports.
    assert "PASS" not in out
    assert "evidence for the port, not proof" in out


def test_a_real_episode_1_gap_reads_as_inconsistent(monkeypatch, capsys):
    code = run_main(
        monkeypatch,
        scored(candidate_side(), {1: 1.0, 2: 2.0}),
        scored(candidate_side(), {1: 1.5, 2: 2.0}),
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "INCONSISTENT" in out
    assert "FAIL" not in out


def test_the_refusal_can_be_overridden_but_says_the_numbers_do_not_count(
    monkeypatch, capsys
):
    code = run_main(
        monkeypatch,
        scored(published_side(), {1: 1.0, 2: 2.0}),
        scored(candidate_side(), {1: 1.0, 2: 2.0}),
        argv=("--allow-protocol-mismatch",),
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "WARNING" in out and "cannot establish port fidelity" in out


def test_the_na_strategy_caveat_reaches_the_report(monkeypatch, capsys):
    run_main(
        monkeypatch,
        scored(published_side(), {1: 1.0, 2: 2.0}),
        scored(candidate_side(), {1: 1.0, 2: 2.0}),
    )
    assert "na_strategy" in capsys.readouterr().out


def test_a_missing_episode_1_is_inconclusive(monkeypatch, capsys):
    published = scored(candidate_side(), {1: 1.0, 2: 2.0})
    candidate = scored(candidate_side(), {1: 1.0, 2: 2.0})
    candidate = candidate[candidate["increment"] != "1"]
    code = run_main(monkeypatch, published, candidate)
    assert code == 2
    assert "INCONCLUSIVE" in capsys.readouterr().out
