r"""Tests for the preflight gate.

Every check has at least one test that fails the check and one that passes it.
A check whose test only ever exercises the happy path cannot tell a working
gate from a gate that returns PASS unconditionally, and this repo has shipped
exactly that kind of test before.

The `check_smoke` tests carry the most weight: `test_smoke_fails_without_na_strategy`
rebuilds the historical NaN bug -- the library-default numerical encoder, with no
`na_strategy` -- and asserts the check refuses it. If that test ever passes for
the wrong reason, the whole preflight is theatre.
"""

import json
import re
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.preflight as pf
from scripts.preflight import (
    FAIL,
    PASS,
    SKIP,
    WARN,
    CheckResult,
    chain_id_for,
    check_datasets,
    check_disk,
    check_git,
    check_interpreter,
    check_mlflow,
    check_plan,
    check_resume_quorum,
    check_smoke,
    check_threads,
    check_torch_build,
    exit_code,
    experiment_names,
    human_bytes,
    inspect_dataset,
    local_store_path,
    probe_tracking_store,
    project_grid,
    render_table,
    resolve_pairs,
    smoke_verdict,
)


# ---------------------------------------------------------------------------
# CheckResult and formatting helpers
# ---------------------------------------------------------------------------


def test_check_result_rejects_an_unknown_status():
    with pytest.raises(ValueError, match="unknown status"):
        CheckResult("x", "BROKEN", "reason")


def test_check_result_rejects_a_multiline_reason():
    # The table gives each check exactly one row; a newline would silently
    # desynchronise every row after it.
    with pytest.raises(ValueError, match="single line"):
        CheckResult("x", PASS, "line one\nline two")


def test_check_result_failed_is_true_only_for_fail():
    assert CheckResult("x", FAIL, "r").failed
    assert not CheckResult("x", WARN, "r").failed
    assert not CheckResult("x", SKIP, "r").failed


def test_check_result_to_dict_round_trips_through_json():
    payload = CheckResult("x", PASS, "fine", {"n": 1}).to_dict()
    assert json.loads(json.dumps(payload))["details"]["n"] == 1


@pytest.mark.parametrize(
    "value,expected",
    [(0, "0 B"), (512, "512 B"), (1024, "1.0 KiB"), (1024**3, "1.0 GiB"),
     (-1024**2, "-1.0 MiB")],
)
def test_human_bytes(value, expected):
    assert human_bytes(value) == expected


def test_render_table_rejects_an_empty_result_set():
    with pytest.raises(ValueError, match="must not be empty"):
        render_table([])


def test_render_table_gives_every_check_exactly_one_row():
    results = [CheckResult("a", PASS, "ok"), CheckResult("longname", FAIL, "bad")]
    lines = render_table(results).splitlines()
    assert len(lines) == 4  # header, rule, two checks
    assert lines[-1].startswith("longname")
    assert "bad" in lines[-1]


def test_exit_code_fails_on_fail_and_only_fails_on_warn_when_strict():
    warned = [CheckResult("a", PASS, "ok"), CheckResult("b", WARN, "hm")]
    assert exit_code(warned) == 0
    assert exit_code(warned, strict=True) == 1
    assert exit_code([CheckResult("a", FAIL, "no")]) == 1
    assert exit_code([CheckResult("a", SKIP, "n/a")]) == 0


# ---------------------------------------------------------------------------
# 7. Code state
# ---------------------------------------------------------------------------


def _git_runner(head="a" * 40, branch="main", status="", head_rc=0):
    def run(args):
        if args[:2] == ["git", "rev-parse"] and args[2] == "HEAD":
            return SimpleNamespace(returncode=head_rc, stdout=head + "\n")
        if args[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(returncode=0, stdout=branch + "\n")
        return SimpleNamespace(returncode=0, stdout=status)

    return run


def test_git_passes_on_a_clean_tree_and_reports_the_hash():
    result = check_git(runner=_git_runner(head="deadbeef" + "0" * 32))
    assert result.status == PASS
    assert "deadbeef" in result.reason
    assert result.details["commit"] == "deadbeef" + "0" * 32


def test_git_fails_on_uncommitted_tracked_changes():
    result = check_git(runner=_git_runner(status=" M redelex/nn/models.py\n"))
    assert result.status == FAIL
    assert result.details["dirty_tracked"] == ["redelex/nn/models.py"]


def test_git_only_warns_on_untracked_files():
    # Scratch files do not change what the grid executes.
    result = check_git(runner=_git_runner(status="?? notes/scratch.md\n"))
    assert result.status == WARN
    assert result.details["untracked"] == ["notes/scratch.md"]
    assert result.details["dirty_tracked"] == []


def test_git_fails_when_the_directory_is_not_a_repository():
    assert check_git(runner=_git_runner(head_rc=128)).status == FAIL


def test_git_fails_when_git_is_missing():
    def explode(args):
        raise OSError("git: command not found")

    assert check_git(runner=explode).status == FAIL


def test_git_against_the_real_repository_reports_a_real_commit():
    result = check_git()
    assert re.fullmatch(r"[0-9a-f]{40}", result.details["commit"])


# ---------------------------------------------------------------------------
# 1. Interpreter and build
# ---------------------------------------------------------------------------


def test_interpreter_passes_only_for_the_venv_python(tmp_path):
    venv = tmp_path / "python"
    venv.write_text("#!/bin/sh\n")
    assert check_interpreter(str(venv), venv_python=venv).status == PASS
    other = tmp_path / "other"
    other.write_text("#!/bin/sh\n")
    result = check_interpreter(str(other), venv_python=venv)
    assert result.status == FAIL
    assert str(venv) in result.reason


def test_interpreter_resolves_symlinks_before_comparing(tmp_path):
    # `uv run` and direnv both hand out symlinked interpreters; comparing the
    # literal strings would reject a perfectly correct one.
    venv = tmp_path / "python"
    venv.write_text("#!/bin/sh\n")
    link = tmp_path / "link"
    link.symlink_to(venv)
    assert check_interpreter(str(link), venv_python=venv).status == PASS


def _fake_torch(cuda_version="12.8", available=True, count=2, name="A100"):
    return SimpleNamespace(
        __version__=f"2.9.1+cu{(cuda_version or '').replace('.', '')}",
        version=SimpleNamespace(cuda=cuda_version),
        cuda=SimpleNamespace(
            is_available=lambda: available,
            device_count=lambda: count,
            get_device_name=lambda i: f"{name}-{i}",
        ),
    )


def test_torch_fails_when_a_cpu_build_is_used_for_a_gpu_run():
    # The `uv run` failure: the venv gets re-synced with the cpu dependency group.
    result = check_torch_build(expect_gpu=True, torch_module=_fake_torch(None, False, 0))
    assert result.status == FAIL
    assert "CPU-only build" in result.reason


def test_torch_warns_about_a_cpu_build_even_without_an_expectation():
    result = check_torch_build(expect_gpu=None, torch_module=_fake_torch(None, False, 0))
    assert result.status == WARN


def test_torch_fails_when_a_cuda_build_cannot_see_its_devices():
    # The potato container failure: device nodes present, open() returns EPERM.
    result = check_torch_build(expect_gpu=True, torch_module=_fake_torch(available=False))
    assert result.status == FAIL
    assert "is_available() is False" in result.reason


def test_torch_warns_when_gpus_are_visible_but_a_cpu_run_was_asked_for():
    assert check_torch_build(expect_gpu=False, torch_module=_fake_torch()).status == WARN


def test_torch_passes_and_names_the_devices():
    result = check_torch_build(expect_gpu=True, torch_module=_fake_torch())
    assert result.status == PASS
    assert "A100-0" in result.reason and "A100-1" in result.reason
    assert result.details["device_count"] == 2


def test_torch_treats_a_raising_driver_as_unavailable():
    def boom():
        raise RuntimeError("no CUDA driver")

    broken = _fake_torch()
    broken.cuda.is_available = boom
    assert check_torch_build(expect_gpu=True, torch_module=broken).status == FAIL


def test_torch_against_the_real_module_reports_this_build():
    import torch

    result = check_torch_build(expect_gpu=None)
    assert result.details["torch_version"] == torch.__version__


# ---------------------------------------------------------------------------
# 2. Thread settings
# ---------------------------------------------------------------------------


BASE_THREADS = dict(
    torch_threads=1, cpus_per_trial=2, cpus_per_job=4, jobs=4, cpu_count=64,
    omp_num_threads="1", observed_torch_threads=1,
)


@pytest.mark.parametrize("field", sorted(BASE_THREADS.keys() - {"omp_num_threads",
                                                               "observed_torch_threads"}))
@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_threads_rejects_non_positive_counts(field, bad):
    kwargs = dict(BASE_THREADS, **{field: bad})
    with pytest.raises(ValueError, match=field):
        check_threads(**kwargs)


def test_threads_passes_on_a_sane_gpu_layout():
    result = check_threads(**BASE_THREADS)
    assert result.status == PASS
    assert result.details["concurrent_trials"] == 8  # 4 jobs x (4 // 2) trials


def test_threads_fails_when_the_jobs_alone_exceed_the_cores():
    result = check_threads(**dict(BASE_THREADS, jobs=4, cpus_per_job=16, cpu_count=64))
    assert result.status == PASS  # 4 x 16 == 64 exactly, still fits
    result = check_threads(**dict(BASE_THREADS, jobs=4, cpus_per_job=17, cpu_count=64))
    assert result.status == FAIL
    assert "68 cores requested" in result.reason


def test_threads_fails_on_the_historical_cpu_oversubscription():
    # `torch.set_num_threads(1)` sat inside a cuda branch, so CPU trials kept the
    # interpreter default of one thread per core -- here, 64 threads per trial.
    result = check_threads(**dict(
        BASE_THREADS, torch_threads=64, cpus_per_trial=1, cpus_per_job=1, jobs=4,
        cpu_count=64,
    ))
    assert result.status == FAIL
    assert "256 > 64 cores" in result.reason


def test_threads_warns_when_omp_num_threads_is_unset():
    # torch's thread count does not bound OpenMP inside BLAS.
    result = check_threads(**dict(BASE_THREADS, omp_num_threads=None))
    assert result.status == WARN
    assert "OMP_NUM_THREADS unset" in result.reason


def test_threads_warns_when_omp_num_threads_is_not_a_number():
    assert check_threads(**dict(BASE_THREADS, omp_num_threads="all")).status == WARN


def test_threads_warns_when_omp_threads_oversubscribe():
    result = check_threads(**dict(BASE_THREADS, omp_num_threads="16"))
    assert result.status == WARN
    assert "16 x 8 trials" in result.reason


# ---------------------------------------------------------------------------
# 3. Datasets
# ---------------------------------------------------------------------------


def _make_caches(tmp_path, dataset="rel-f1", tables=3, materialized=None,
                 tasks=("driver-dnf",), schema=True):
    relbench = tmp_path / "relbench"
    graph = tmp_path / "graph"
    db_dir = relbench / dataset / "db"
    db_dir.mkdir(parents=True)
    for i in range(tables):
        (db_dir / f"t{i}.parquet").write_bytes(b"x" * 10)
    for task in tasks:
        (relbench / dataset / "tasks" / task).mkdir(parents=True)
    n_mat = tables if materialized is None else materialized
    mat = graph / dataset / "materialized"
    mat.mkdir(parents=True)
    for i in range(n_mat):
        (mat / f"t{i}.pt").write_bytes(b"y" * 100)
    if schema:
        (graph / dataset / "attribute-schema.json").write_text("{}")
    return relbench, graph


def test_datasets_pass_when_everything_is_cached(tmp_path):
    relbench, graph = _make_caches(tmp_path)
    result = check_datasets([("rel-f1", "driver-dnf")], relbench, graph)
    assert result.status == PASS
    assert result.details["datasets"][0]["problems"] == []
    assert result.details["total_bytes"] == 3 * 10 + 3 * 100 + 2  # +2 for the schema


def test_datasets_fail_when_the_relbench_cache_is_absent(tmp_path):
    relbench, graph = _make_caches(tmp_path)
    result = check_datasets([("rel-hm", "user-churn")], relbench, graph)
    assert result.status == FAIL
    assert "no relbench db cache" in result.reason


def test_datasets_fail_when_the_task_cache_is_absent(tmp_path):
    relbench, graph = _make_caches(tmp_path, tasks=("driver-dnf",))
    result = check_datasets([("rel-f1", "driver-top3")], relbench, graph)
    assert result.status == FAIL
    assert "driver-top3" in result.reason


def test_datasets_fail_on_a_partially_materialised_graph(tmp_path):
    # A partial cache is the case that otherwise surfaces as the first trial
    # silently paying a multi-hour rebuild.
    relbench, graph = _make_caches(tmp_path, tables=5, materialized=3)
    result = check_datasets([("rel-f1", "driver-dnf")], relbench, graph)
    assert result.status == FAIL
    assert "3 of 5 tables" in result.reason


def test_datasets_fail_when_no_graph_cache_exists_at_all(tmp_path):
    relbench, graph = _make_caches(tmp_path, materialized=0)
    assert check_datasets([("rel-f1", "driver-dnf")], relbench, graph).status == FAIL


def test_datasets_warn_when_only_the_attribute_schema_is_missing(tmp_path):
    relbench, graph = _make_caches(tmp_path, schema=False)
    result = check_datasets([("rel-f1", "driver-dnf")], relbench, graph)
    assert result.status == WARN
    assert "attribute-schema.json" in result.reason


def test_datasets_skip_when_no_pairs_were_requested(tmp_path):
    assert check_datasets([], tmp_path, tmp_path).status == SKIP


def test_inspect_dataset_reports_sizes(tmp_path):
    relbench, graph = _make_caches(tmp_path, tables=2)
    report = inspect_dataset("rel-f1", ["driver-dnf"], relbench, graph)
    assert report["relbench_bytes"] == 2 * 10
    assert report["graph_bytes"] == 2 * 100 + 2
    assert report["n_tables"] == 2 and report["n_materialized"] == 2


# ---------------------------------------------------------------------------
# 4. Plan and disk
# ---------------------------------------------------------------------------


def test_project_grid_rejects_bad_inputs():
    with pytest.raises(ValueError, match="num_samples"):
        project_grid([("rel-f1", "driver-dnf")], ["naive"], 0)
    with pytest.raises(ValueError, match="modes"):
        project_grid([("rel-f1", "driver-dnf")], [], 3)


def test_project_grid_multiplies_episodes_seeds_and_modes():
    projection = project_grid([("rel-f1", "driver-dnf")], ["naive"], 3)
    expected = 11 * 3 * (pf.CHECKPOINT_BYTES + pf.CL_STATE_BYTES_DEFAULT)
    assert projection["total_bytes"] == expected
    assert projection["trial_episodes"] == 11 * 3 * 1
    assert projection["chains"] == 1


def test_project_grid_charges_ewc_for_its_two_model_copies():
    naive = project_grid([("rel-f1", "driver-dnf")], ["naive"], 3)["total_bytes"]
    ewc = project_grid([("rel-f1", "driver-dnf")], ["ewc"], 3)["total_bytes"]
    # EWC stores a parameter anchor plus a Fisher diagonal: two full models.
    assert ewc > 2 * naive


def test_project_grid_flags_a_pair_with_no_measured_episode_count():
    projection = project_grid([("rel-f1", "made-up-task")], ["naive"], 3)
    assert projection["unknown_pairs"] == ["rel-f1:made-up-task"]
    assert projection["total_bytes"] == 0


def test_project_grid_flags_chains_too_short_to_show_forgetting():
    projection = project_grid([("rel-f1", "driver-top3")], ["naive"], 3)
    assert projection["short_pairs"] == ["rel-f1:driver-top3 (2)"]


def test_project_grid_scales_gpu_hours_with_modes_and_seeds():
    # rel-hm rather than rel-f1: the headline figure is rounded to one decimal,
    # and a ratio taken between two ~0.3 values is mostly rounding.
    pair = [("rel-hm", "user-churn")]
    small = project_grid(pair, ["naive"], 1)["gpu_hours"]
    big = project_grid(pair, ["naive", "ewc"], 2)["gpu_hours"]
    assert big == pytest.approx(4 * small, rel=0.02)
    # Half of rel-hm's episodes, one of seven modes, one of five seeds.
    assert small == pytest.approx(389.0 * 0.5 / 7 / 5, rel=0.01)


def test_plan_fails_on_an_uncostable_pair():
    result = check_plan(project_grid([("rel-f1", "nope")], ["naive"], 3))
    assert result.status == FAIL
    assert "rel-f1:nope" in result.reason


def test_plan_warns_on_a_two_episode_chain():
    result = check_plan(project_grid([("rel-f1", "driver-top3")], ["naive"], 3))
    assert result.status == WARN
    assert "forgetting" in result.reason


def test_plan_passes_and_summarises():
    result = check_plan(project_grid([("rel-f1", "driver-dnf")], ["naive"], 3))
    assert result.status == PASS
    assert "1 chains" in result.reason and "A100-h" in result.reason


def test_plan_skips_with_no_pairs():
    assert check_plan(project_grid([], ["naive"], 3)).status == SKIP


def test_disk_rejects_headroom_below_one(tmp_path):
    with pytest.raises(ValueError, match="headroom"):
        check_disk({"total_bytes": 1}, tmp_path, headroom=0.9)


def test_disk_fails_when_the_projection_does_not_fit(tmp_path):
    result = check_disk({"total_bytes": 100}, tmp_path, headroom=1.25, free_bytes=120)
    assert result.status == FAIL
    assert result.details["needed_bytes"] == 125


def test_disk_warns_when_the_grid_would_take_over_half_the_filesystem(tmp_path):
    result = check_disk({"total_bytes": 100}, tmp_path, headroom=1.0, free_bytes=150)
    assert result.status == WARN


def test_disk_passes_with_room_to_spare(tmp_path):
    result = check_disk({"total_bytes": 100}, tmp_path, headroom=1.0, free_bytes=10_000)
    assert result.status == PASS


def test_disk_measures_the_nearest_existing_ancestor_and_creates_nothing(tmp_path):
    missing = tmp_path / "logs" / "grid" / "deep"
    result = check_disk({"total_bytes": 1}, missing, free_bytes=10**9)
    assert result.details["measured_at"] == str(tmp_path.resolve())
    assert not missing.exists()


def test_disk_uses_real_free_space_when_none_is_injected(tmp_path):
    result = check_disk({"total_bytes": 1}, tmp_path)
    assert result.details["free_bytes"] > 0


# ---------------------------------------------------------------------------
# 5. MLflow
# ---------------------------------------------------------------------------


def test_experiment_names_mirror_the_launcher():
    assert experiment_names("p", ["ewc", "naive"]) == ["p_ewc", "p_naive"]
    with pytest.raises(ValueError, match="prefix"):
        experiment_names("", ["ewc"])


def test_chain_id_matches_what_the_experiment_actually_logs():
    # Anchored to a chain_id read back off the tracking server, so a change to
    # either side of the format shows up here.
    assert chain_id_for("rel-f1", "driver-top3", "ewc", Path("logs/smoke-v2")) == (
        "rel-f1/driver-top3/ewc/logs/smoke-v2/rel-f1_driver-top3_ewc/models"
    )


def test_local_store_path_follows_sqlalchemy_slash_counting():
    assert local_store_path("sqlite:///mlruns.db") == Path("mlruns.db")
    assert local_store_path("sqlite:////var/mlruns.db") == Path("/var/mlruns.db")
    assert local_store_path("file:///tmp/mlruns") == Path("/tmp/mlruns")
    assert local_store_path("./mlruns") == Path("./mlruns")
    assert local_store_path("http://host:2222") is None
    assert local_store_path("https://host") is None


def test_probe_refuses_to_create_a_missing_local_store(tmp_path):
    """The pollution this check exists to prevent.

    Building an MLflow client against a missing sqlite file creates it, after
    which the store is trivially "reachable" and holds no colliding runs -- a
    preflight that always passes.
    """
    missing = tmp_path / "mlruns.db"
    message = probe_tracking_store(f"sqlite:///{missing}")
    assert message is not None and "will not create it" in message
    assert not missing.exists()


def test_probe_accepts_a_local_store_that_exists(tmp_path):
    store = tmp_path / "mlruns.db"
    store.write_bytes(b"")
    assert probe_tracking_store(f"sqlite:///{store}") is None


def test_probe_reports_a_missing_host():
    assert "cannot parse a host" in probe_tracking_store("http://")


def test_probe_succeeds_against_a_listening_socket():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        assert probe_tracking_store(f"http://127.0.0.1:{port}", timeout=2.0) is None


def test_probe_reports_a_closed_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    message = probe_tracking_store(f"http://127.0.0.1:{port}", timeout=2.0)
    assert message is not None and "unreachable" in message


def test_mlflow_fails_rather_than_creating_a_missing_store(tmp_path):
    missing = tmp_path / "mlruns.db"
    result = check_mlflow(f"sqlite:///{missing}", ["p_ewc"])
    assert result.status == FAIL
    assert not missing.exists()


class _FakeRuns(list):
    token = None


class _FakeClient:
    """Minimal stand-in for MlflowClient; records the filters it was given."""

    def __init__(self, experiments=None, runs=None, raises=None):
        self.experiments = experiments or {}
        self.runs = runs or {}
        self.raises = raises
        self.filters = []

    def get_experiment_by_name(self, name):
        if self.raises:
            raise self.raises
        if name not in self.experiments:
            return None
        return SimpleNamespace(experiment_id=self.experiments[name])

    def search_runs(self, experiment_ids, filter_string=None, max_results=None):
        self.filters.append(filter_string)
        return _FakeRuns(self.runs.get(experiment_ids[0], []))


def _run(increment):
    return SimpleNamespace(data=SimpleNamespace(params={"increment": str(increment)}))


def test_mlflow_skips_without_experiment_names():
    assert check_mlflow("http://x", [], client=_FakeClient()).status == SKIP


def test_mlflow_passes_when_every_target_name_is_free():
    result = check_mlflow("http://x", ["p_ewc"], client=_FakeClient())
    assert result.status == PASS
    assert result.details["experiments"][0]["exists"] is False


def test_mlflow_passes_when_an_experiment_exists_but_holds_no_runs():
    client = _FakeClient(experiments={"p_ewc": "7"}, runs={"7": []})
    assert check_mlflow("http://x", ["p_ewc"], client=client).status == PASS


def test_mlflow_refuses_a_collision_with_an_occupied_experiment():
    # The published grid: pelesjak_cl_from_scratch, experiment 92, whose
    # checkpoints live on a machine this host cannot reach.
    client = _FakeClient(experiments={"pelesjak_cl_from_scratch": "92"},
                         runs={"92": [_run(1)] * 684})
    result = check_mlflow("http://x", ["pelesjak_cl_from_scratch"], client=client)
    assert result.status == FAIL
    assert "92" in result.reason and "684" in result.reason


def test_mlflow_accepts_an_occupied_experiment_when_resuming():
    client = _FakeClient(experiments={"p_ewc": "7"}, runs={"7": [_run(1)]})
    result = check_mlflow("http://x", ["p_ewc"], resume=True, client=client)
    assert result.status == PASS
    assert "--resume will continue" in result.reason


def test_mlflow_warns_when_resume_has_nothing_to_resume_from():
    result = check_mlflow("http://x", ["p_ewc"], resume=True, client=_FakeClient())
    assert result.status == WARN
    assert "episode 1" in result.reason


def test_mlflow_fails_when_the_server_errors():
    client = _FakeClient(raises=RuntimeError("connection reset"))
    result = check_mlflow("http://x", ["p_ewc"], client=client)
    assert result.status == FAIL
    assert "connection reset" in result.reason


# ---------------------------------------------------------------------------
# The resume quorum
# ---------------------------------------------------------------------------


RESUME_KWARGS = dict(
    pairs=[("rel-f1", "driver-top3")], modes=["ewc"], prefix="p",
    out_root=Path("logs/smoke-v2"),
)


def test_resume_skips_without_pairs():
    result = check_resume_quorum(
        [], ["ewc"], "p", 3, Path("logs/grid"), client=_FakeClient()
    )
    assert result.status == SKIP


def test_resume_passes_when_the_seed_count_matches_the_history():
    client = _FakeClient(experiments={"p_ewc": "9"},
                         runs={"9": [_run(1), _run(1), _run(2), _run(2)]})
    result = check_resume_quorum(num_samples=2, client=client, **RESUME_KWARGS)
    assert result.status == PASS
    assert result.details["chains_resumable"] == 1


def test_resume_fails_when_the_quorum_can_never_be_met():
    # The historical bug: a chain run with 2 seeds, relaunched asking for 3.
    # Every past increment then looks incomplete and the chain restarts at 1.
    client = _FakeClient(experiments={"p_ewc": "9"},
                         runs={"9": [_run(1), _run(1), _run(2), _run(2)]})
    result = check_resume_quorum(num_samples=3, client=client, **RESUME_KWARGS)
    assert result.status == FAIL
    assert "increment 2 has 2 trial(s)" in result.reason


def test_resume_fails_when_the_quorum_would_be_met_by_a_partial_increment():
    # The mirror image: asking for fewer seeds than were run makes a half
    # finished increment look complete.
    client = _FakeClient(experiments={"p_ewc": "9"},
                         runs={"9": [_run(1)] * 5 + [_run(2)] * 5})
    result = check_resume_quorum(num_samples=3, client=client, **RESUME_KWARGS)
    assert result.status == FAIL


def test_resume_warns_when_no_run_matches_the_chain_id():
    client = _FakeClient(experiments={"p_ewc": "9"}, runs={"9": []})
    result = check_resume_quorum(num_samples=3, client=client, **RESUME_KWARGS)
    assert result.status == WARN
    assert "episode 1" in result.reason


def test_resume_filters_by_the_exact_chain_id():
    # Without the chain_id filter, unrelated runs in the same experiment --
    # including the published grid -- can satisfy the quorum.
    client = _FakeClient(experiments={"p_ewc": "9"}, runs={"9": [_run(1)]})
    check_resume_quorum(num_samples=1, client=client, **RESUME_KWARGS)
    assert len(client.filters) == 1
    expected = chain_id_for("rel-f1", "driver-top3", "ewc", Path("logs/smoke-v2"))
    assert f"params.chain_id = '{expected}'" in client.filters[0]
    assert "attributes.status = 'FINISHED'" in client.filters[0]


def test_resume_ignores_a_mode_whose_experiment_does_not_exist():
    result = check_resume_quorum(num_samples=3, client=_FakeClient(), **RESUME_KWARGS)
    assert result.status == WARN
    assert result.details["chains_with_runs"] == 0


def test_resume_fails_when_the_server_errors():
    client = _FakeClient(raises=RuntimeError("boom"))
    result = check_resume_quorum(num_samples=3, client=client, **RESUME_KWARGS)
    assert result.status == FAIL


# ---------------------------------------------------------------------------
# 6. The numerical smoke test
# ---------------------------------------------------------------------------


def test_smoke_rejects_a_non_positive_step_count():
    with pytest.raises(ValueError, match="steps"):
        check_smoke("synthetic", steps=0)


def test_smoke_passes_on_the_real_model():
    result = check_smoke("synthetic", steps=4)
    assert result.status == PASS
    assert result.details["non_finite_parameters"] == []
    assert all(v == v for v in result.details["losses"])


def test_the_synthetic_graph_actually_exercises_the_missing_value_path():
    # A smoke test that never sees a missing numerical cell cannot detect the
    # bug it exists for. Prove the fixture does.
    result = check_smoke("synthetic", steps=4)
    assert result.details["batches_with_missing_numericals"] == 4


def test_smoke_fails_without_na_strategy():
    r"""The historical NaN grid, rebuilt.

    With torch_frame's default numerical encoder a column holding any missing
    cell produces a NaN gradient; Adam writes a NaN parameter on the first step.
    The forward pass nan_to_num's its output so the loss and the logged metric
    both stay plausible -- only the parameters give it away.
    """
    from relbench.modeling.nn import HeteroEncoder

    from experiments.continuous_learning.models import HeterogeneousSAGE

    def unpinned_factory(data, col_stats, width):
        model = HeterogeneousSAGE(
            data, col_stats, gnn_channels=width, gnn_layers=2, out_channels=1
        )
        model.encoder = HeteroEncoder(
            channels=width,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats,
        )
        return model

    result = check_smoke("synthetic", steps=4, model_factory=unpinned_factory)
    assert result.status == FAIL
    assert "non-finite" in result.reason
    assert any("numerical" in name
               for name in result.details["non_finite_parameters"])


VERDICT = dict(source="synthetic", entity="users", steps=4, channels=16)


def test_smoke_verdict_rejects_a_run_that_took_no_step():
    # A smoke that never stepped has nothing to report and must not read as a pass.
    with pytest.raises(ValueError, match="losses"):
        smoke_verdict(non_finite=[], losses=[], batches_with_missing=1, **VERDICT)


def test_smoke_verdict_fails_on_a_non_finite_parameter():
    result = smoke_verdict(
        non_finite=["encoder.encoders.users.encoder.encoder_dict.numerical.weight"],
        losses=[1.0, 1.0], batches_with_missing=4, **VERDICT,
    )
    assert result.status == FAIL
    assert "numerical.weight" in result.reason


def test_smoke_verdict_fails_on_a_non_finite_loss_even_with_finite_parameters():
    # Reachable when the NaN enters through a path carrying no gradient to any
    # parameter: the weights look clean and only the loss gives it away.
    result = smoke_verdict(
        non_finite=[], losses=[1.0, float("nan")], batches_with_missing=4, **VERDICT
    )
    assert result.status == FAIL
    assert "loss went non-finite" in result.reason


def test_smoke_verdict_fails_on_an_infinite_loss():
    result = smoke_verdict(
        non_finite=[], losses=[float("inf")], batches_with_missing=4, **VERDICT
    )
    assert result.status == FAIL


def test_smoke_verdict_refuses_to_claim_a_pass_it_did_not_earn():
    # Finite throughout, but no batch held a missing cell: this run could not
    # have detected the failure the check exists for, so it is not a PASS.
    result = smoke_verdict(
        non_finite=[], losses=[1.0, 0.5], batches_with_missing=0, **VERDICT
    )
    assert result.status == WARN
    assert "na_strategy was never exercised" in result.reason


def test_smoke_verdict_passes_when_finite_and_covered():
    result = smoke_verdict(
        non_finite=[], losses=[1.0, 0.5], batches_with_missing=4, **VERDICT
    )
    assert result.status == PASS
    assert "4/4 batches" in result.reason


def test_smoke_warns_when_the_graph_holds_no_missing_numerical_cells(monkeypatch):
    """The zero-coverage verdict, driven through the real check.

    Fills every missing numerical cell in the synthetic graph, so the run is
    genuinely healthy and genuinely uninformative.
    """
    import torch
    import torch_frame

    original = pf._synthetic_graph

    def clean_graph():
        data, col_stats, entity = original()
        for node_type in data.node_types:
            feats = data[node_type].tf.feat_dict.get(torch_frame.numerical)
            if feats is not None:
                torch.nan_to_num_(feats)
        return data, col_stats, entity

    monkeypatch.setattr(pf, "_synthetic_graph", clean_graph)
    result = check_smoke("synthetic", steps=3)
    assert result.status == WARN
    assert result.details["batches_with_missing_numericals"] == 0
    assert result.details["non_finite_parameters"] == []


def test_smoke_fails_when_the_graph_cannot_be_built(tmp_path):
    result = check_smoke("rel-does-not-exist", steps=2, graph_cache=tmp_path)
    assert result.status == FAIL
    assert "could not build" in result.reason


def test_smoke_fails_when_a_training_step_raises():
    class Exploding:
        def parameters(self):
            import torch

            return [torch.nn.Parameter(torch.zeros(1))]

        def named_parameters(self):
            return []

        def __call__(self, *args, **kwargs):
            raise RuntimeError("shape mismatch")

    result = check_smoke("synthetic", steps=2,
                         model_factory=lambda *a: Exploding())
    assert result.status == FAIL
    assert "shape mismatch" in result.reason


def test_the_refusing_text_embedder_raises_instead_of_downloading():
    # Left to default, make_pkey_fkey_graph would construct the 480 MB glove
    # embedder. A cache miss must be loud, not a download.
    with pytest.raises(RuntimeError, match="must not download"):
        pf._RefusingTextEmbedder()(["some text"])


REAL_CACHE = Path(pf.REPO) / ".cache" / "rel-f1" / "materialized"
requires_rel_f1 = pytest.mark.skipif(
    not REAL_CACHE.is_dir() or not any(REAL_CACHE.glob("*.pt")),
    reason="rel-f1 materialised cache not present on this host",
)


@requires_rel_f1
def test_cached_graph_seeds_from_a_table_that_has_missing_cells():
    # A seed table with no missing cells would leave the entity encoder itself
    # untested: only its neighbours' encoders would see the na_strategy path.
    import torch
    import torch_frame

    data, _stats, entity = pf._cached_graph("rel-f1", Path(pf.REPO) / ".cache")
    assert hasattr(data[entity], "time")
    numeric = data[entity].tf.feat_dict.get(torch_frame.numerical)
    assert numeric is not None and bool(torch.isnan(numeric).any())


@requires_rel_f1
def test_smoke_runs_on_a_real_cached_dataset_and_writes_nothing():
    """The cached-graph path, and the promise that preflight never modifies.

    A cache half-written by a preflight would be indistinguishable from a real
    one at launch time, which is the failure the dataset check exists to catch.
    """
    root = Path(pf.REPO) / ".cache" / "rel-f1"
    before = {
        str(path): path.stat().st_mtime_ns
        for path in sorted(root.rglob("*")) if path.is_file()
    }
    result = check_smoke("rel-f1", steps=3)
    after = {
        str(path): path.stat().st_mtime_ns
        for path in sorted(root.rglob("*")) if path.is_file()
    }
    assert after == before
    assert result.status == PASS
    assert result.details["source"] == "rel-f1"
    # The real dataset must also reach the missing-value path, or the check is
    # weaker on real data than on the synthetic fixture.
    assert result.details["batches_with_missing_numericals"] > 0


def test_pick_smoke_dataset_prefers_the_smallest_fully_cached_dataset(tmp_path):
    relbench = tmp_path / "relbench"
    graph = tmp_path / "graph"
    for name, size in (("rel-big", 5000), ("rel-small", 10)):
        db = relbench / name / "db"
        db.mkdir(parents=True)
        (db / "t0.parquet").write_bytes(b"x" * size)
        mat = graph / name / "materialized"
        mat.mkdir(parents=True)
        (mat / "t0.pt").write_bytes(b"y")
        (graph / name / "attribute-schema.json").write_text("{}")
    picked = pf.pick_smoke_dataset(
        [("rel-big", "t"), ("rel-small", "t")], relbench, graph
    )
    assert picked == "rel-small"


def test_pick_smoke_dataset_rejects_a_partially_materialised_dataset(tmp_path):
    relbench, graph = _make_caches(tmp_path, tables=4, materialized=2)
    assert pf.pick_smoke_dataset([("rel-f1", "driver-dnf")], relbench, graph) is None


def test_pick_smoke_dataset_ignores_a_missing_task_cache(tmp_path):
    # The smoke needs the graph, not the labels.
    relbench, graph = _make_caches(tmp_path, tasks=())
    assert pf.pick_smoke_dataset([("rel-f1", "driver-dnf")], relbench, graph) == "rel-f1"


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def test_resolve_pairs_from_explicit_pairs():
    args = pf.parse_args(["--pairs", "rel-f1:driver-dnf", "rel-hm:user-churn"])
    assert resolve_pairs(args) == [("rel-f1", "driver-dnf"), ("rel-hm", "user-churn")]


def test_resolve_pairs_rejects_a_malformed_pair():
    args = pf.parse_args(["--pairs", "rel-f1"])
    with pytest.raises(ValueError, match="dataset:task"):
        resolve_pairs(args)


def test_resolve_pairs_from_a_tier_matches_the_launcher():
    args = pf.parse_args(["--tier", "A"])
    assert resolve_pairs(args) == pf.run_grid.TIERS["A"]


def test_resolve_pairs_is_empty_without_either_flag():
    assert resolve_pairs(pf.parse_args([])) == []


def test_expect_gpu_and_expect_cpu_are_three_valued():
    assert pf.parse_args([]).expect_gpu is None
    assert pf.parse_args(["--expect-gpu"]).expect_gpu is True
    assert pf.parse_args(["--expect-cpu"]).expect_gpu is False


def test_main_returns_two_on_a_usage_error(capsys):
    assert pf.main(["--pairs", "not-a-pair", "--only", "plan"]) == 2
    assert "dataset:task" in capsys.readouterr().err


def test_main_prints_the_commit_above_the_table(capsys):
    assert pf.main(["--only", "git"]) in (0, 1)
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("commit ")
    assert "CHECK" in out


def test_main_json_output_is_machine_readable(capsys):
    code = pf.main(["--only", "git", "interpreter", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["exit_code"] == code
    assert payload["ok"] == (code == 0)
    assert re.fullmatch(r"[0-9a-f]{40}", payload["commit"])
    assert {c["name"] for c in payload["checks"]} == {"git", "interpreter"}


def test_main_keeps_third_party_chatter_off_the_json_document(capsys, monkeypatch):
    """relbench and MLflow print to stdout while the checks run.

    Without the redirect, that chatter lands ahead of the document and every
    consumer of --json gets a parse error instead of a result.
    """
    def noisy_check(*args, **kwargs):
        print("Loading Database object from /home/x/.cache/relbench/rel-f1/db...")
        return CheckResult("git", PASS, "quiet", {"commit": "0" * 40})

    monkeypatch.setattr(pf, "check_git", noisy_check)
    pf.main(["--only", "git", "--json"])
    captured = capsys.readouterr()
    assert json.loads(captured.out)["checks"][0]["name"] == "git"
    assert "Loading Database object" in captured.err


def test_main_keeps_third_party_chatter_off_the_table_too(capsys, monkeypatch):
    monkeypatch.setattr(
        pf, "check_git",
        lambda *a, **k: (print("noise"), CheckResult("git", PASS, "quiet"))[1],
    )
    pf.main(["--only", "git"])
    captured = capsys.readouterr()
    assert "noise" not in captured.out
    assert "noise" in captured.err


def test_main_exits_non_zero_when_a_check_fails(capsys, monkeypatch):
    monkeypatch.setattr(
        pf, "check_git", lambda *a, **k: CheckResult("git", FAIL, "synthetic failure")
    )
    assert pf.main(["--only", "git"]) == 1
    assert "DO NOT LAUNCH" in capsys.readouterr().out


def test_main_strict_turns_a_warning_into_a_refusal(capsys, monkeypatch):
    monkeypatch.setattr(
        pf, "check_git", lambda *a, **k: CheckResult("git", WARN, "synthetic warning")
    )
    assert pf.main(["--only", "git"]) == 0
    assert pf.main(["--only", "git", "--strict"]) == 1


def test_skip_removes_a_check(capsys):
    pf.main(["--only", "git", "interpreter", "--skip", "git", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert [c["name"] for c in payload["checks"]] == ["interpreter"]


def test_every_check_name_is_reachable_from_the_runner():
    # A slug accepted by --only that no branch in run_checks answers to would
    # silently produce a shorter table than the operator asked for.
    for name in pf.CHECK_NAMES:
        args = pf.parse_args([
            "--only", name, "--pairs", "rel-f1:driver-dnf",
            "--mlflow-uri", "sqlite:///no/such/store.db", "--smoke-dataset", "synthetic",
            "--relbench-cache", "/nonexistent", "--graph-cache", "/nonexistent",
        ])
        results = pf.run_checks(args)
        assert [r.name for r in results] == [name], name
