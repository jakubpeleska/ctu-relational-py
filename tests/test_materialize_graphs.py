r"""Tests for the dataset-preparation script.

The interesting behaviour here is not the materialisation itself -- that is
torch_frame's -- but the decisions around it: which datasets may be downloaded,
which caches count as finished, and when a dataset is too expensive to start.
Those are the parts that silently do the wrong thing.
"""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import torch

import scripts.materialize_graphs as mg

REPO_ROOT = Path(__file__).resolve().parent.parent


def write_frame(path: Path, truncate: int = 0) -> Path:
    r"""Write a torch checkpoint, optionally chopping bytes off the end.

    Args:
        path: Destination file.
        truncate: Bytes to remove, simulating a write killed part-way.

    Returns:
        Path: `path`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(({"a": torch.zeros(8)}, {}), path)
    if truncate:
        with open(path, "r+b") as handle:
            handle.truncate(path.stat().st_size - truncate)
    return path


def make_tree(root: Path, name: str, tables: list[str], materialised: list[str],
              schema: bool = True, truncate: list[str] = ()) -> tuple[Path, Path]:
    r"""Build a fake RelBench cache and materialised cache for one dataset.

    Args:
        root: Scratch directory.
        name: Dataset name.
        tables: Parquet stems to create in the database directory.
        materialised: Tensor-frame stems to create in the materialised cache.
        schema: Whether to create ``attribute-schema.json``.
        truncate: Subset of `materialised` to leave truncated.

    Returns:
        tuple[Path, Path]: ``(cache_dir, relbench_cache)``.
    """
    cache_dir, relbench_cache = root / "cache", root / "relbench"
    db = mg.db_dir(relbench_cache, name)
    if tables:
        db.mkdir(parents=True, exist_ok=True)
        for table in tables:
            (db / f"{table}.parquet").write_bytes(b"")
    mat = mg.materialized_dir(cache_dir, name)
    mat.mkdir(parents=True, exist_ok=True)
    for table in materialised:
        write_frame(mat / f"{table}.pt", truncate=64 if table in truncate else 0)
    if schema:
        mg.schema_path(cache_dir, name).write_text("{}")
    return cache_dir, relbench_cache


# --------------------------------------------------------------------------
# Download policy -- the two traps this script exists to avoid
# --------------------------------------------------------------------------


def test_rel_stack_is_never_downloaded():
    # relbench pins a SHA256 the server no longer serves, and download=True
    # deletes the good local db.zip before raising.
    assert mg.download_decision("rel-stack", db_present=True) == (False, "cached")
    assert mg.download_decision("rel-stack", db_present=False) == (False, "from-raw")


def test_rel_amazon_uses_the_prepared_download():
    # make_db() 404s on a file UCSD removed, so the prepared path is the only one.
    assert mg.download_decision("rel-amazon", db_present=False) == (True, "fetch")


def test_no_forbidden_dataset_is_ever_downloaded():
    r"""The policy flag, not the call site, decides. Check every combination."""
    for name, policy in mg.DOWNLOAD_POLICY.items():
        for db_present in (True, False):
            download, _ = mg.download_decision(name, db_present)
            if not policy.prepared:
                assert download is False, f"{name} would be downloaded"


def test_present_database_is_never_re_downloaded():
    # A download with the db already extracted re-verifies a hash at best and
    # re-transfers 6 GB at worst.
    for name in mg.DOWNLOAD_POLICY:
        assert mg.download_decision(name, db_present=True) == (False, "cached")


def test_unknown_dataset_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="no download policy"):
        mg.download_decision("rel-nonesuch", db_present=False)


def test_every_default_dataset_has_a_policy_and_an_estimate():
    for name in mg.DEFAULT_DATASETS:
        assert name in mg.DOWNLOAD_POLICY
        assert name in mg.ESTIMATES


# --------------------------------------------------------------------------
# Completeness
# --------------------------------------------------------------------------


def test_expected_tables_are_the_parquet_stems(tmp_path):
    cache_dir, relbench_cache = make_tree(
        tmp_path, "rel-x", ["b", "a", "c"], materialised=[]
    )
    assert mg.expected_tables(mg.db_dir(relbench_cache, "rel-x")) == ["a", "b", "c"]


def test_expected_tables_is_none_when_the_database_is_absent(tmp_path):
    # None means "unknown", which is different from "no tables"; the caller has
    # to be able to tell a missing download from an empty database.
    assert mg.expected_tables(tmp_path / "nope") is None
    (tmp_path / "empty").mkdir()
    assert mg.expected_tables(tmp_path / "empty") is None


def test_truncated_frames_are_detected(tmp_path):
    mat = tmp_path / "materialized"
    write_frame(mat / "good.pt")
    write_frame(mat / "bad.pt", truncate=64)
    assert mg.corrupt_tables(mat) == ["bad"]


def test_a_truncated_frame_really_is_unloadable(tmp_path):
    r"""The integrity probe must agree with what torch actually does.

    A probe that flagged loadable files, or missed unloadable ones, would either
    throw away an hour of work or let the next run die on it.
    """
    mat = tmp_path / "materialized"
    good = write_frame(mat / "good.pt")
    bad = write_frame(mat / "bad.pt", truncate=64)
    torch.load(good, weights_only=True)  # must not raise
    with pytest.raises(Exception):
        torch.load(bad, weights_only=True)
    with pytest.raises(zipfile.BadZipFile):
        zipfile.ZipFile(bad).namelist()


@pytest.mark.parametrize(
    "tables, materialised, schema, truncate, expected",
    [
        (["a", "b"], ["a", "b"], True, (), "complete"),
        (["a", "b"], ["a", "b"], False, (), "needs-schema"),
        (["a", "b"], ["a"], True, (), "partial"),
        (["a", "b"], [], True, (), "missing"),
        (["a", "b"], ["a", "b", "ghost"], True, (), "stale"),
        (["a", "b"], ["a", "b"], True, ("b",), "corrupt"),
        ([], [], True, (), "needs-download"),
        ([], ["a"], True, (), "unverifiable"),
    ],
)
def test_status_classification(tmp_path, tables, materialised, schema, truncate,
                               expected):
    cache_dir, relbench_cache = make_tree(
        tmp_path, "rel-x", tables, materialised, schema=schema, truncate=truncate
    )
    state = mg.inspect("rel-x", cache_dir, relbench_cache)
    assert state.status == expected
    assert state.complete is (expected == "complete")


def test_classify_rejects_inconsistent_listings():
    with pytest.raises(ValueError, match="listings disagree"):
        mg.classify(True, ["a"], ["a"], ["b"], True)


def test_receipt_is_not_the_skip_predicate(tmp_path):
    r"""A cache built by an older script, or copied in, still counts as done."""
    cache_dir, relbench_cache = make_tree(tmp_path, "rel-x", ["a"], ["a"])
    state = mg.inspect("rel-x", cache_dir, relbench_cache)
    assert state.complete and state.receipt == {}


def test_receipt_is_read_when_present(tmp_path):
    cache_dir, relbench_cache = make_tree(tmp_path, "rel-x", ["a"], ["a"])
    receipt = mg.materialized_dir(cache_dir, "rel-x") / mg.RECEIPT_NAME
    receipt.write_text(json.dumps({"wall_s": 12.5, "peak_rss_bytes": 1024}))
    state = mg.inspect("rel-x", cache_dir, relbench_cache)
    assert state.receipt["wall_s"] == 12.5
    # A receipt is not a tensor frame; it must not be counted as a table.
    assert state.materialised == ["a"]


def test_a_corrupt_receipt_does_not_break_inspection(tmp_path):
    cache_dir, relbench_cache = make_tree(tmp_path, "rel-x", ["a"], ["a"])
    (mg.materialized_dir(cache_dir, "rel-x") / mg.RECEIPT_NAME).write_text("{oops")
    assert mg.inspect("rel-x", cache_dir, relbench_cache).status == "complete"


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------


def test_deadline_prefers_whichever_limit_is_earlier(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_END_TIME", "1000")
    assert mg.job_deadline(None, now=0) == 1000
    assert mg.job_deadline(500, now=0) == 500       # explicit budget is tighter
    assert mg.job_deadline(5000, now=0) == 1000     # scheduler is tighter


def test_deadline_ignores_a_malformed_scheduler_variable(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_END_TIME", "not-a-number")
    assert mg.job_deadline(None, now=0) is None
    assert mg.job_deadline(500, now=0) == 500


def test_deadline_rejects_a_nonsense_budget():
    with pytest.raises(ValueError, match="must be positive"):
        mg.job_deadline(0, now=0)


def test_expensive_dataset_is_not_started_without_time_for_it():
    # rel-amazon is 3348 s on potato; with safety 2 it needs ~6700 s.
    ok, why = mg.can_start("rel-amazon", deadline=1800, now=0, safety=2.0, reserve=0)
    assert not ok and "only" in why


def test_cheap_dataset_still_starts_in_the_same_window():
    ok, _ = mg.can_start("rel-f1", deadline=1800, now=0, safety=2.0, reserve=0)
    assert ok


def test_reserve_is_subtracted_from_the_remaining_time():
    # rel-hm needs 253*2 = 506 s. 600 s of wall minus a 200 s reserve is 400.
    assert mg.can_start("rel-hm", deadline=600, now=0, safety=2.0, reserve=0)[0]
    assert not mg.can_start("rel-hm", deadline=600, now=0, safety=2.0, reserve=200)[0]


def test_no_deadline_means_no_refusal():
    assert mg.can_start("rel-amazon", None, now=0, safety=2.0, reserve=0)[0]


def test_unknown_dataset_is_attempted_rather_than_refused():
    ok, why = mg.can_start("rel-unknown", deadline=60, now=0, safety=2.0, reserve=0)
    assert ok and "no estimate" in why


def test_can_start_rejects_a_nonsense_safety_factor():
    with pytest.raises(ValueError, match="must be positive"):
        mg.can_start("rel-f1", deadline=100, now=0, safety=0, reserve=0)


def test_disk_check_refuses_a_cache_that_cannot_fit(tmp_path, monkeypatch):
    import shutil

    monkeypatch.setattr(
        shutil, "disk_usage", lambda _p: shutil._ntuple_diskusage(0, 0, 1024)
    )
    ok, why = mg.can_fit_on_disk("rel-amazon", tmp_path)
    assert not ok and "free" in why


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_render_table_rejects_ragged_rows():
    with pytest.raises(ValueError, match="cells"):
        mg.render_table(["a", "b"], [["1"]])


def test_merge_results_keeps_the_requested_order(tmp_path):
    cache_dir, relbench_cache = make_tree(tmp_path, "rel-x", ["a"], ["a"])
    done = mg.inspect("rel-x", cache_dir, relbench_cache)
    todo = mg.DatasetState("rel-y", True, ["a"], [], [], False, 0, status="missing")
    merged = mg.merge_results([done, todo], [{"dataset": "rel-y", "ok": True}])
    assert [r["dataset"] for r in merged] == ["rel-x", "rel-y"]
    assert merged[0]["download_action"] == "cached"


def test_a_skipped_dataset_still_reports_what_it_cost(tmp_path):
    r"""The summary is a cost record, so a skipped dataset must not be blank.

    Its wall time and peak RSS come from the receipt written when it was
    actually built; without them, rerunning the job would quietly erase the
    only measurements anyone has of the expensive datasets.
    """
    cache_dir, relbench_cache = make_tree(tmp_path, "rel-x", ["a"], ["a"])
    receipt = mg.materialized_dir(cache_dir, "rel-x") / mg.RECEIPT_NAME
    receipt.write_text(json.dumps({"wall_s": 3348.0, "peak_rss_bytes": 150 * 1024**3}))
    state = mg.inspect("rel-x", cache_dir, relbench_cache)
    row = mg.summary_rows(mg.merge_results([state], []))[0]
    assert row[3] == "0:55:48"      # wall
    assert row[4] == "150.0 GiB"    # peak RSS


def test_a_skipped_dataset_without_a_receipt_reports_nothing_rather_than_zero(
    tmp_path,
):
    # Claiming 0:00:00 for a cache someone copied in would be a fabricated
    # measurement, which is worse than an empty cell.
    cache_dir, relbench_cache = make_tree(tmp_path, "rel-x", ["a"], ["a"])
    state = mg.inspect("rel-x", cache_dir, relbench_cache)
    row = mg.summary_rows(mg.merge_results([state], []))[0]
    assert row[3] == "-" and row[4] == "-"


def test_merge_results_refuses_to_drop_a_stray_record():
    state = mg.DatasetState("rel-x", True, ["a"], ["a"], [], True, 0, status="complete")
    with pytest.raises(ValueError, match="never requested"):
        mg.merge_results([state], [{"dataset": "rel-z"}])


def test_a_lower_bound_rss_is_marked_as_one():
    rows = mg.summary_rows(
        [{"dataset": "rel-x", "peak_rss_bytes": 2 * 1024**3, "rss_is_lower_bound": True}]
    )
    assert rows[0][4].startswith(">=")


def test_failure_reason_spells_out_a_kill_signal():
    reason = mg.failure_reason(-9, ["something"])
    assert "signal 9" in reason and "memory" in reason


def test_failure_reason_of_an_ordinary_error_is_the_last_line():
    assert mg.failure_reason(1, ["boom: bad thing"]) == "boom: bad thing"


# --------------------------------------------------------------------------
# Worker plumbing
# --------------------------------------------------------------------------


class _Args:
    def __init__(self, tmp_path):
        self.cache_dir = tmp_path / "cache"
        self.relbench_cache = tmp_path / "relbench"
        self.embedder = "glove"
        self.torch_threads = 2


def test_worker_result_is_parsed_and_the_rest_is_streamed(tmp_path, capsys,
                                                          monkeypatch):
    payload = json.dumps({"dataset": "rel-x", "ok": True, "wall_s": 1.0})
    monkeypatch.setattr(
        mg,
        "worker_command",
        lambda name, args: [
            sys.executable,
            "-c",
            f"print('progress line'); print({mg.RESULT_PREFIX + payload!r})",
        ],
    )
    result, code = mg.run_worker("rel-x", _Args(tmp_path))
    assert code == 0 and result["ok"] is True
    assert "progress line" in capsys.readouterr().out


def test_a_killed_worker_becomes_a_failed_row(tmp_path, monkeypatch):
    monkeypatch.setattr(
        mg,
        "worker_command",
        lambda name, args: [
            sys.executable,
            "-c",
            "import os, signal; print('started', flush=True); "
            "os.kill(os.getpid(), signal.SIGKILL)",
        ],
    )
    result, code = mg.run_worker("rel-x", _Args(tmp_path))
    assert code != 0
    assert result["ok"] is False and result["materialised"] == "failed"
    assert "signal 9" in result["error"]


def test_worker_command_carries_the_parents_cache_paths(tmp_path):
    cmd = mg.worker_command("rel-x", _Args(tmp_path))
    assert str(tmp_path / "cache") in cmd
    assert str(tmp_path / "relbench") in cmd
    assert cmd[cmd.index("--worker") + 1] == "rel-x"


# --------------------------------------------------------------------------
# --check
# --------------------------------------------------------------------------


def test_check_reports_state_without_importing_torch_or_relbench(tmp_path):
    r"""``--check`` has to be cheap enough to run on a login node.

    Importing torch costs seconds and hundreds of megabytes; importing relbench
    drags in pooch and the whole dataset registry. Neither is needed to list two
    directories, and the moment one gets hoisted to module scope the "cheap"
    claim in the job script stops being true.
    """
    make_tree(tmp_path, "rel-f1", ["a", "b"], ["a"])
    probe = (
        "import sys; sys.path.insert(0, %r);"
        "import scripts.materialize_graphs as m;"
        "code = m.main(['--check', 'rel-f1', '--cache-dir', %r,"
        " '--relbench-cache', %r]);"
        "assert 'torch' not in sys.modules, 'torch was imported';"
        "assert 'relbench' not in sys.modules, 'relbench was imported';"
        "print('EXIT', code)"
    ) % (str(REPO_ROOT), str(tmp_path / "cache"), str(tmp_path / "relbench"))
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert proc.returncode == 0, proc.stderr
    assert "partial" in proc.stdout
    assert "EXIT 1" in proc.stdout  # work remains


def test_check_exits_zero_when_everything_is_ready(tmp_path, capsys):
    make_tree(tmp_path, "rel-f1", ["a"], ["a"])
    code = mg.main(
        [
            "--check",
            "rel-f1",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--relbench-cache",
            str(tmp_path / "relbench"),
        ]
    )
    assert code == 0
    assert "nothing to do" in capsys.readouterr().out


def test_dry_run_plans_without_building(tmp_path, capsys):
    make_tree(tmp_path, "rel-f1", ["a", "b"], [])
    code = mg.main(
        [
            "--dry-run",
            "rel-f1",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--relbench-cache",
            str(tmp_path / "relbench"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0 and "=== plan ===" in out
    assert mg.materialised_tables(mg.materialized_dir(tmp_path / "cache", "rel-f1")) == []


def test_main_refuses_an_unknown_dataset(tmp_path):
    with pytest.raises(ValueError, match="no download policy"):
        mg.main(["--check", "rel-nonesuch", "--cache-dir", str(tmp_path)])
