r"""Refuse to launch a grid that is already doomed.

A continual-learning grid is a week of GPU time. Every one of the failures below
actually happened on this project, every one was silent at launch, and every one
was cheap to detect *before* the first trial started:

* a grid wrote NaN weights into every checkpoint because the encoder had no
  ``na_strategy``, while the logged metric still looked healthy;
* a chain stopped after episode 1 and exited 0, so the runner wrote a "done"
  marker for a cell that would never be retried;
* ``--resume`` restarted from episode 1 because a hardcoded quorum of 5 did not
  match the 3 seeds actually run;
* the MLflow experiment prefix collided with the published grid, whose
  checkpoints live on a machine the runner cannot reach;
* a run silently used the CPU torch build, because someone typed ``uv run``
  (which re-syncs the default ``cpu`` dependency group);
* ``torch.set_num_threads(1)`` sat inside a ``torch.cuda.is_available()``
  branch, so CPU runs used 64 threads per trial -- the worst possible setting.

This module runs a battery of cheap checks that would have caught each of them,
prints one line per check, and exits non-zero if any check FAILs. It reads the
filesystem, queries MLflow read-only, and takes a few optimiser steps on CPU.
It never writes, never downloads, and never launches anything.

Usage:
    .venv/bin/python scripts/preflight.py --tier A --expect-gpu
    .venv/bin/python scripts/preflight.py --pairs rel-f1:driver-dnf --expect-cpu
    .venv/bin/python scripts/preflight.py --tier A B --json > preflight.json

Exit codes: 0 = every check PASS/WARN/SKIP, 1 = at least one FAIL, 2 = bad usage.
"""

import argparse
import contextlib
import itertools
import json
import os
import shutil
import socket
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence
from urllib.parse import urlparse

# `scripts/` is not a package and this file is normally run as a script, so
# sys.path[0] is `scripts/`, not the repo root. Without this, `import
# scripts.run_grid` below fails when preflight is invoked the documented way.
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The grid definition is imported rather than copied. A preflight that disagreed
# with the launcher about which cells the grid contains would be worse than no
# preflight at all.
import scripts.run_grid as run_grid  # noqa: E402

VENV_PYTHON = REPO / ".venv" / "bin" / "python"

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"
STATUSES = (PASS, WARN, FAIL, SKIP)

# Measured episode counts, from analysis/dataset-episodes-measured.md. These are
# counts observed by running `ContinuousWrapper.get_splits()` on the real data,
# not ceilings derived from dataset metadata -- the ceilings were wrong by up to
# 2.5x, so cost projected from them is meaningless.
MEASURED_EPISODES = {
    ("rel-hm", "user-churn"): 52,
    ("rel-hm", "item-sales"): 52,
    ("rel-stack", "user-engagement"): 18,
    ("rel-stack", "post-votes"): 18,
    ("rel-stack", "user-badge"): 17,
    ("rel-amazon", "user-churn"): 15,
    ("rel-amazon", "user-ltv"): 15,
    ("rel-amazon", "item-churn"): 15,
    ("rel-amazon", "item-ltv"): 16,
    ("rel-ratebeer", "beer-churn"): 12,
    ("rel-ratebeer", "user-churn"): 12,
    ("rel-ratebeer", "user-count"): 12,
    ("rel-ratebeer", "brewer-dormant"): 9,
    ("rel-f1", "driver-position"): 11,
    ("rel-f1", "driver-dnf"): 11,
    ("rel-f1", "driver-top3"): 2,
    ("rel-trial", "study-outcome"): 7,
    ("rel-trial", "study-adverse"): 7,
    ("rel-trial", "site-success"): 7,
    ("rel-event", "user-attendance"): 10,
    ("rel-event", "user-repeat"): 9,
}

# Measured A100 GPU-hours per dataset for 7 modes x 5 seeds over that dataset's
# full task set, after the validation fix. Used only to project a headline
# number; scaled linearly in modes, seeds and episodes covered.
MEASURED_GPU_HOURS = {
    "rel-f1": 20.0,
    "rel-trial": 41.0,
    "rel-stack": 186.0,
    "rel-hm": 389.0,
    "rel-ratebeer": 150.0,
    "rel-amazon": 224.0,
}
GPU_HOURS_BASELINE_MODES = 7
GPU_HOURS_BASELINE_SEEDS = 5

# Measured on this repo's own output (logs/smoke-v2, rel-f1/driver-top3):
# best_model.pt is 20,554,925 B and one is written per trial per episode.
# The planning figure carried in the project notes is 17 MB; the larger measured
# value is used because under-projecting disk is the expensive direction.
CHECKPOINT_BYTES = 20_554_925

# Measured cl_state.pt per mode, same source. EWC dominates: its state is the
# parameter anchor plus the Fisher diagonal, i.e. two full copies of the model.
CL_STATE_BYTES = {
    "ewc": 40_904_221,
    "der_pp": 50_355,
    "er": 46_835,
    "freeze_extend": 1_395,
}
CL_STATE_BYTES_DEFAULT = 1_267

# A chain this short cannot show forgetting, and it is also the shape that hid
# the "stopped after episode 1, exited 0" bug: with 2 episodes a truncated chain
# looks almost like a complete one.
MIN_USEFUL_EPISODES = 3


@dataclass
class CheckResult:
    r"""One check's verdict, in a form both the table and ``--json`` can render.

    Args:
        name: Short slug, also the value accepted by ``--only``/``--skip``.
        status: One of ``PASS``, ``WARN``, ``FAIL``, ``SKIP``.
        reason: A single line explaining the verdict. Must not contain newlines,
            because the table gives it exactly one row.
        details: Structured payload for ``--json``. Must be JSON-serialisable.

    Raises:
        ValueError: If ``status`` is not a known status or ``reason`` spans
            more than one line.
    """

    name: str
    status: str
    reason: str
    details: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(
                f"unknown status {self.status!r}; expected one of {STATUSES}"
            )
        if "\n" in self.reason:
            raise ValueError(f"`reason` must be a single line, got {self.reason!r}")

    @property
    def failed(self) -> bool:
        return self.status == FAIL

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "details": self.details,
        }


def human_bytes(n: float) -> str:
    r"""Format a byte count with a binary unit suffix.

    Args:
        n: Byte count. May be negative (free-space deficits are reported that way).

    Returns:
        A short string such as ``"1.4 GiB"``.
    """
    sign = "-" if n < 0 else ""
    n = abs(float(n))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{sign}{n:.1f} {unit}" if unit != "B" else f"{sign}{n:.0f} B"
        n /= 1024
    raise AssertionError("unreachable")


def _dir_bytes(path: Path) -> int:
    """Total size of everything under `path`, following no symlinks."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                # A file that vanished mid-walk is not a preflight failure.
                continue
    return total


# ---------------------------------------------------------------------------
# 7. Code state
# ---------------------------------------------------------------------------


def check_git(repo: Path = REPO, runner: Optional[Callable] = None) -> CheckResult:
    r"""Report the commit the grid would run, and whether the tree is clean.

    A week-long grid whose results cannot be tied to a code version is a real
    problem: the numbers become unreproducible the moment someone edits a file.
    Tracked modifications FAIL; untracked files only WARN, because scratch files
    do not change what the grid executes.

    Args:
        repo: Repository root.
        runner: Injection point for tests; called as ``runner(args)`` and must
            return an object with ``returncode`` and ``stdout``. Defaults to
            :func:`subprocess.run`.

    Returns:
        A :class:`CheckResult` whose details carry ``commit``, ``branch``,
        ``dirty_tracked`` and ``untracked``.
    """
    run = runner or (
        lambda args: subprocess.run(
            args, cwd=str(repo), capture_output=True, text=True, timeout=30
        )
    )
    try:
        head = run(["git", "rev-parse", "HEAD"])
        branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        status = run(["git", "status", "--porcelain"])
    except (OSError, subprocess.SubprocessError) as exc:
        return CheckResult("git", FAIL, f"git unavailable: {type(exc).__name__}: {exc}")

    if head.returncode != 0:
        return CheckResult("git", FAIL, f"not a git repository: {repo}")

    commit = head.stdout.strip()
    branch_name = branch.stdout.strip() if branch.returncode == 0 else "?"
    lines = [ln for ln in status.stdout.splitlines() if ln.strip()]
    untracked = [ln[3:] for ln in lines if ln.startswith("??")]
    tracked = [ln[3:] for ln in lines if not ln.startswith("??")]

    details = {
        "commit": commit,
        "short_commit": commit[:8],
        "branch": branch_name,
        "dirty_tracked": tracked,
        "untracked": untracked,
    }
    if tracked:
        return CheckResult(
            "git", FAIL,
            f"{commit[:8]} ({branch_name}) has {len(tracked)} uncommitted tracked "
            f"change(s), e.g. {tracked[0]}; results could not be tied to a commit",
            details,
        )
    if untracked:
        return CheckResult(
            "git", WARN,
            f"{commit[:8]} ({branch_name}) clean, but {len(untracked)} untracked "
            f"file(s) present",
            details,
        )
    return CheckResult("git", PASS, f"{commit[:8]} ({branch_name}), tree clean", details)


# ---------------------------------------------------------------------------
# 1. Interpreter and build
# ---------------------------------------------------------------------------


def check_interpreter(
    executable: Optional[str] = None, venv_python: Path = VENV_PYTHON
) -> CheckResult:
    r"""Confirm the running interpreter is the repo's own ``.venv``.

    Args:
        executable: Interpreter path to judge. Defaults to :data:`sys.executable`.
        venv_python: The interpreter every run must use.

    Returns:
        A :class:`CheckResult` carrying ``executable``, ``expected`` and
        ``prefix``.
    """
    actual = Path(executable if executable is not None else sys.executable)
    details = {
        "executable": str(actual),
        "expected": str(venv_python),
        "prefix": sys.prefix,
    }
    try:
        same = actual.resolve() == Path(venv_python).resolve()
    except OSError:
        same = str(actual) == str(venv_python)
    if not same:
        return CheckResult(
            "interpreter", FAIL,
            f"running {actual}, expected {venv_python}", details,
        )
    return CheckResult("interpreter", PASS, f"{actual}", details)


def check_torch_build(
    expect_gpu: Optional[bool] = None, torch_module: Any = None
) -> CheckResult:
    r"""Confirm torch is a CUDA build and that CUDA availability matches intent.

    Two distinct failures live here. A torch whose ``version.cuda`` is ``None``
    is the *CPU wheel*: the venv was re-synced with the default ``cpu``
    dependency group, which is what a stray ``uv run`` does. A CUDA build whose
    ``cuda.is_available()`` is False is a *runtime* problem instead -- on this
    host, a container denied the nvidia device nodes -- and a GPU grid launched
    into it silently runs on CPU at roughly a hundredth of the speed.

    Args:
        expect_gpu: ``True`` to require usable GPUs, ``False`` to require their
            absence, ``None`` to report without judging.
        torch_module: Injection point for tests. Defaults to the real ``torch``.

    Returns:
        A :class:`CheckResult` carrying ``torch_version``, ``cuda_version``,
        ``is_available``, ``device_count`` and ``devices``.
    """
    if torch_module is None:
        import torch as torch_module  # noqa: PLC0415

    cuda_version = getattr(torch_module.version, "cuda", None)
    try:
        available = bool(torch_module.cuda.is_available())
    except Exception:  # a broken driver raises rather than returning False
        available = False
    try:
        count = int(torch_module.cuda.device_count()) if available else 0
    except Exception:
        count = 0
    devices = []
    for i in range(count):
        try:
            devices.append(str(torch_module.cuda.get_device_name(i)))
        except Exception:
            devices.append(f"cuda:{i} (name unavailable)")

    details = {
        "torch_version": str(getattr(torch_module, "__version__", "?")),
        "cuda_version": cuda_version,
        "is_available": available,
        "device_count": count,
        "devices": devices,
        "expect_gpu": expect_gpu,
    }

    if cuda_version is None:
        return CheckResult(
            "torch", FAIL if expect_gpu else WARN,
            f"torch {details['torch_version']} is a CPU-only build; the venv was "
            f"re-synced with the cpu group (a stray `uv run` does this)",
            details,
        )
    if expect_gpu is True and not available:
        return CheckResult(
            "torch", FAIL,
            f"torch {details['torch_version']} is a CUDA {cuda_version} build but "
            f"cuda.is_available() is False; --expect-gpu cannot be satisfied",
            details,
        )
    if expect_gpu is False and available:
        return CheckResult(
            "torch", WARN,
            f"--expect-cpu, yet {count} GPU(s) are visible; the run will leave "
            f"them idle",
            details,
        )
    where = f"{count} GPU(s): {', '.join(devices)}" if available else "no GPU visible"
    return CheckResult(
        "torch", PASS,
        f"torch {details['torch_version']} (CUDA {cuda_version}), {where}",
        details,
    )


# ---------------------------------------------------------------------------
# 2. Thread settings
# ---------------------------------------------------------------------------


def check_threads(
    torch_threads: int,
    cpus_per_trial: int,
    cpus_per_job: int,
    jobs: int,
    cpu_count: int,
    omp_num_threads: Optional[str],
    observed_torch_threads: int,
) -> CheckResult:
    r"""Confirm the intended per-trial parallelism does not oversubscribe the box.

    Concurrent trials multiply: ``jobs`` chains each run
    ``cpus_per_job // cpus_per_trial`` trials at once, and each trial asks torch
    for ``torch_threads`` intra-op threads. The historical bug set
    ``torch.set_num_threads(1)`` only on the CUDA branch, so CPU trials inherited
    the interpreter default -- one thread per core, per trial, all fighting.

    ``OMP_NUM_THREADS`` is checked separately because torch's thread count does
    not bound OpenMP inside BLAS: leaving it unset lets every worker process
    spawn ``cpu_count`` OpenMP threads regardless of ``torch_threads``.

    Args:
        torch_threads: Value that will be passed as ``--torch_threads``.
        cpus_per_trial: Value that will be passed as ``--cpus_per_trial``.
        cpus_per_job: CPU cores Ray is given per chain.
        jobs: Number of chains running concurrently.
        cpu_count: Cores actually available.
        omp_num_threads: Raw ``OMP_NUM_THREADS`` value, or ``None`` when unset.
        observed_torch_threads: ``torch.get_num_threads()`` in this process.

    Returns:
        A :class:`CheckResult` carrying the derived ``concurrent_trials`` and
        ``demanded_threads``.

    Raises:
        ValueError: If any count is not a positive integer.
    """
    for label, value in (
        ("torch_threads", torch_threads), ("cpus_per_trial", cpus_per_trial),
        ("cpus_per_job", cpus_per_job), ("jobs", jobs), ("cpu_count", cpu_count),
    ):
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"`{label}` must be a positive int, got {value!r}")

    trials_per_job = max(1, cpus_per_job // cpus_per_trial)
    concurrent_trials = jobs * trials_per_job
    demanded_threads = concurrent_trials * torch_threads
    demanded_cpus = jobs * cpus_per_job

    details = {
        "torch_threads": torch_threads,
        "cpus_per_trial": cpus_per_trial,
        "cpus_per_job": cpus_per_job,
        "jobs": jobs,
        "cpu_count": cpu_count,
        "omp_num_threads": omp_num_threads,
        "observed_torch_threads": observed_torch_threads,
        "trials_per_job": trials_per_job,
        "concurrent_trials": concurrent_trials,
        "demanded_threads": demanded_threads,
        "demanded_cpus": demanded_cpus,
    }

    if demanded_cpus > cpu_count:
        return CheckResult(
            "threads", FAIL,
            f"{jobs} jobs x {cpus_per_job} cpus = {demanded_cpus} cores requested, "
            f"only {cpu_count} present",
            details,
        )
    if demanded_threads > cpu_count:
        return CheckResult(
            "threads", FAIL,
            f"{concurrent_trials} concurrent trials x {torch_threads} torch threads "
            f"= {demanded_threads} > {cpu_count} cores; trials will fight",
            details,
        )
    if omp_num_threads is None:
        return CheckResult(
            "threads", WARN,
            f"OMP_NUM_THREADS unset: BLAS may spawn {cpu_count} threads per trial "
            f"regardless of --torch_threads={torch_threads}",
            details,
        )
    try:
        omp = int(omp_num_threads)
    except ValueError:
        return CheckResult(
            "threads", WARN,
            f"OMP_NUM_THREADS={omp_num_threads!r} is not an integer", details,
        )
    if omp * concurrent_trials > cpu_count:
        return CheckResult(
            "threads", WARN,
            f"OMP_NUM_THREADS={omp} x {concurrent_trials} trials > {cpu_count} cores",
            details,
        )
    return CheckResult(
        "threads", PASS,
        f"{concurrent_trials} trials x {torch_threads} torch / {omp} omp threads "
        f"<= {cpu_count} cores (this process: {observed_torch_threads})",
        details,
    )


# ---------------------------------------------------------------------------
# 3. Datasets
# ---------------------------------------------------------------------------


def relbench_cache_dir() -> Path:
    """Where relbench keeps raw tables, honouring RELBENCH_CACHE_DIR."""
    override = os.getenv("RELBENCH_CACHE_DIR")
    if override:
        return Path(override)
    import pooch  # noqa: PLC0415

    return Path(pooch.os_cache("relbench"))


def inspect_dataset(
    dataset: str,
    tasks: Sequence[str],
    relbench_cache: Path,
    graph_cache: Path,
) -> dict:
    r"""Describe one dataset's caches without loading or downloading anything.

    The materialised graph cache holds one ``<table>.pt`` per table in the raw
    database, so comparing the two counts detects a *partial* materialisation --
    the case that otherwise surfaces as the first trial silently paying a
    multi-hour rebuild, or reaching for the 480 MB glove embedder.

    Args:
        dataset: Dataset name, e.g. ``"rel-f1"``.
        tasks: Task names that will be run against it.
        relbench_cache: Root of the relbench cache.
        graph_cache: Root of the materialised graph cache (``.cache`` by default).

    Returns:
        A dict with ``problems`` (list of one-line strings, empty when fine),
        table and task counts, and byte sizes.
    """
    db_dir = relbench_cache / dataset / "db"
    mat_dir = graph_cache / dataset / "materialized"
    schema_file = graph_cache / dataset / "attribute-schema.json"

    parquet = sorted(p.name for p in db_dir.glob("*.parquet")) if db_dir.is_dir() else []
    tensors = sorted(p.stem for p in mat_dir.glob("*.pt")) if mat_dir.is_dir() else []
    missing_tasks = [
        t for t in tasks if not (relbench_cache / dataset / "tasks" / t).is_dir()
    ]

    problems = []
    if not parquet:
        problems.append(f"no relbench db cache at {db_dir}")
    if missing_tasks:
        problems.append(
            f"task cache missing for {', '.join(missing_tasks)} under "
            f"{relbench_cache / dataset / 'tasks'}"
        )
    if not tensors:
        problems.append(f"no materialised graph cache at {mat_dir}")
    elif parquet and len(tensors) != len(parquet):
        problems.append(
            f"materialised cache has {len(tensors)} of {len(parquet)} tables"
        )

    return {
        "dataset": dataset,
        "tasks": list(tasks),
        "db_dir": str(db_dir),
        "graph_dir": str(mat_dir),
        "n_tables": len(parquet),
        "n_materialized": len(tensors),
        "missing_tasks": missing_tasks,
        "has_schema": schema_file.is_file(),
        "relbench_bytes": _dir_bytes(relbench_cache / dataset)
        if (relbench_cache / dataset).is_dir() else 0,
        "graph_bytes": _dir_bytes(graph_cache / dataset)
        if (graph_cache / dataset).is_dir() else 0,
        "problems": problems,
    }


def check_datasets(
    pairs: Sequence[tuple], relbench_cache: Path, graph_cache: Path
) -> CheckResult:
    r"""Confirm every dataset and task the grid needs is already on disk.

    Args:
        pairs: ``(dataset, task)`` pairs the grid will run.
        relbench_cache: Root of the relbench cache.
        graph_cache: Root of the materialised graph cache.

    Returns:
        A :class:`CheckResult` whose details carry one entry per dataset.
    """
    if not pairs:
        return CheckResult("datasets", SKIP, "no dataset:task pairs requested")

    by_dataset: dict = {}
    for dataset, task in pairs:
        by_dataset.setdefault(dataset, []).append(task)

    reports = [
        inspect_dataset(ds, tasks, relbench_cache, graph_cache)
        for ds, tasks in sorted(by_dataset.items())
    ]
    total = sum(r["relbench_bytes"] + r["graph_bytes"] for r in reports)
    details = {
        "relbench_cache": str(relbench_cache),
        "graph_cache": str(graph_cache),
        "total_bytes": total,
        "datasets": reports,
    }

    broken = [r for r in reports if r["problems"]]
    if broken:
        first = broken[0]
        return CheckResult(
            "datasets", FAIL,
            f"{len(broken)}/{len(reports)} dataset(s) not ready: "
            f"{first['dataset']}: {first['problems'][0]}",
            details,
        )
    no_schema = [r["dataset"] for r in reports if not r["has_schema"]]
    if no_schema:
        return CheckResult(
            "datasets", WARN,
            f"{len(reports)} dataset(s) cached ({human_bytes(total)}), but "
            f"attribute-schema.json missing for {', '.join(no_schema)}",
            details,
        )
    return CheckResult(
        "datasets", PASS,
        f"{len(reports)} dataset(s) cached, {human_bytes(total)} "
        f"({', '.join(r['dataset'] for r in reports)})",
        details,
    )


# ---------------------------------------------------------------------------
# 4. Disk, and the shape of the plan
# ---------------------------------------------------------------------------


def project_grid(
    pairs: Sequence[tuple],
    modes: Sequence[str],
    num_samples: int,
    episodes: Optional[dict] = None,
    checkpoint_bytes: int = CHECKPOINT_BYTES,
) -> dict:
    r"""Project trial-episodes, bytes and GPU-hours for a grid.

    One checkpoint and one ``cl_state.pt`` are written per trial per episode and
    none are deleted, so volume is ``episodes x modes x seeds`` per pair. EWC
    dominates the state term: its anchor plus Fisher diagonal is two full copies
    of the model, 40 MB against the 20 MB checkpoint.

    Args:
        pairs: ``(dataset, task)`` pairs.
        modes: Learning modes.
        num_samples: Trials (seeds) per episode.
        episodes: Episode counts keyed by pair. Defaults to
            :data:`MEASURED_EPISODES`.
        checkpoint_bytes: Size of one ``best_model.pt``.

    Returns:
        A dict with ``total_bytes``, ``trial_episodes``, ``gpu_hours``,
        ``unknown_pairs`` and ``short_pairs``.

    Raises:
        ValueError: If ``num_samples`` is not a positive int or ``modes`` is empty.
    """
    if not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError(f"`num_samples` must be a positive int, got {num_samples!r}")
    if not modes:
        raise ValueError("`modes` must not be empty")

    table = MEASURED_EPISODES if episodes is None else episodes
    per_mode_state = [CL_STATE_BYTES.get(m, CL_STATE_BYTES_DEFAULT) for m in modes]
    bytes_per_trial_episode_all_modes = sum(
        checkpoint_bytes + state for state in per_mode_state
    )

    # GPU-hours are quoted per dataset for its whole task set; scale by the
    # fraction of that dataset's episodes this grid actually covers.
    dataset_total_episodes: dict = {}
    for (ds, _task), n in table.items():
        dataset_total_episodes[ds] = dataset_total_episodes.get(ds, 0) + n

    total_bytes = 0
    trial_episodes = 0
    gpu_hours = 0.0
    unknown_pairs = []
    short_pairs = []
    per_pair = []
    for dataset, task in pairs:
        n_ep = table.get((dataset, task))
        if n_ep is None:
            unknown_pairs.append(f"{dataset}:{task}")
            continue
        if n_ep < MIN_USEFUL_EPISODES:
            short_pairs.append(f"{dataset}:{task} ({n_ep})")
        pair_bytes = n_ep * num_samples * bytes_per_trial_episode_all_modes
        total_bytes += pair_bytes
        trial_episodes += n_ep * num_samples * len(modes)
        base = MEASURED_GPU_HOURS.get(dataset)
        pair_hours = 0.0
        if base is not None and dataset_total_episodes.get(dataset):
            pair_hours = (
                base
                * (n_ep / dataset_total_episodes[dataset])
                * (len(modes) / GPU_HOURS_BASELINE_MODES)
                * (num_samples / GPU_HOURS_BASELINE_SEEDS)
            )
            gpu_hours += pair_hours
        per_pair.append(
            {"dataset": dataset, "task": task, "episodes": n_ep,
             "bytes": pair_bytes, "gpu_hours": round(pair_hours, 1)}
        )

    return {
        "total_bytes": total_bytes,
        "trial_episodes": trial_episodes,
        "chains": len(pairs) * len(modes),
        "gpu_hours": round(gpu_hours, 1),
        "unknown_pairs": unknown_pairs,
        "short_pairs": short_pairs,
        "per_pair": per_pair,
        "checkpoint_bytes": checkpoint_bytes,
        "num_samples": num_samples,
        "modes": list(modes),
    }


def check_plan(projection: dict) -> CheckResult:
    r"""Judge the shape of the grid itself, before its resource cost.

    Args:
        projection: Output of :func:`project_grid`.

    Returns:
        A :class:`CheckResult`. Unknown pairs FAIL, because a pair with no
        measured episode count cannot be costed and may not even produce
        episodes. Chains shorter than :data:`MIN_USEFUL_EPISODES` WARN.
    """
    if not projection["per_pair"] and not projection["unknown_pairs"]:
        return CheckResult("plan", SKIP, "no dataset:task pairs requested")

    summary = (
        f"{projection['chains']} chains, {projection['trial_episodes']} "
        f"trial-episodes, ~{projection['gpu_hours']} A100-h, "
        f"{human_bytes(projection['total_bytes'])}"
    )
    if projection["unknown_pairs"]:
        return CheckResult(
            "plan", FAIL,
            f"no measured episode count for "
            f"{', '.join(projection['unknown_pairs'])}; cost cannot be projected",
            projection,
        )
    if projection["short_pairs"]:
        return CheckResult(
            "plan", WARN,
            f"{summary}; too few episodes to show forgetting in "
            f"{', '.join(projection['short_pairs'])}",
            projection,
        )
    return CheckResult("plan", PASS, summary, projection)


def check_disk(
    projection: dict,
    out_dir: Path,
    headroom: float = 1.25,
    free_bytes: Optional[int] = None,
) -> CheckResult:
    r"""Compare projected checkpoint volume against free space.

    Args:
        projection: Output of :func:`project_grid`.
        out_dir: Where checkpoints will be written. It need not exist yet; the
            nearest existing ancestor is measured, and nothing is created.
        headroom: Multiplier applied to the projection before comparing. Grids
            that fill a filesystem to the brim fail late and messily.
        free_bytes: Injection point for tests. Defaults to a real ``statvfs``.

    Returns:
        A :class:`CheckResult` carrying ``needed_bytes`` and ``free_bytes``.

    Raises:
        ValueError: If ``headroom`` is not at least 1.
    """
    if headroom < 1:
        raise ValueError(f"`headroom` must be >= 1, got {headroom}")

    probe = Path(out_dir).resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent

    if free_bytes is None:
        try:
            free_bytes = shutil.disk_usage(probe).free
        except OSError as exc:
            return CheckResult("disk", FAIL, f"cannot stat {probe}: {exc}")

    needed = int(projection["total_bytes"] * headroom)
    details = {
        "out_dir": str(out_dir),
        "measured_at": str(probe),
        "projected_bytes": projection["total_bytes"],
        "needed_bytes": needed,
        "free_bytes": int(free_bytes),
        "headroom": headroom,
    }
    if needed > free_bytes:
        return CheckResult(
            "disk", FAIL,
            f"needs {human_bytes(needed)} (incl. {headroom:g}x headroom), "
            f"{human_bytes(free_bytes)} free on {probe}",
            details,
        )
    if needed > free_bytes * 0.5:
        return CheckResult(
            "disk", WARN,
            f"needs {human_bytes(needed)} of {human_bytes(free_bytes)} free on "
            f"{probe} -- over half the filesystem",
            details,
        )
    return CheckResult(
        "disk", PASS,
        f"needs {human_bytes(needed)}, {human_bytes(free_bytes)} free on {probe}",
        details,
    )


# ---------------------------------------------------------------------------
# 5. MLflow
# ---------------------------------------------------------------------------


def experiment_names(prefix: str, modes: Sequence[str]) -> list:
    r"""Experiment names ``run_grid.py`` would use for these modes.

    Args:
        prefix: The ``--mlflow-experiment-prefix`` value.
        modes: Learning modes.

    Returns:
        One name per mode, in the order given.

    Raises:
        ValueError: If ``prefix`` is empty.
    """
    if not prefix:
        raise ValueError("`prefix` must not be empty")
    return [f"{prefix}_{mode}" for mode in modes]


def local_store_path(uri: str) -> Optional[Path]:
    r"""The filesystem path behind a non-HTTP tracking URI, if there is one.

    Args:
        uri: Tracking URI. ``sqlite:///rel.db`` is relative and
            ``sqlite:////abs.db`` absolute, per SQLAlchemy; ``file:`` and a bare
            path are taken literally.

    Returns:
        The path, or ``None`` for ``http``/``https`` and other remote schemes.
    """
    parsed = urlparse(uri)
    if parsed.scheme in ("http", "https"):
        return None
    if parsed.scheme == "sqlite":
        # urlparse turns sqlite:///rel.db into path "/rel.db"; SQLAlchemy reads
        # the same string as relative. Drop exactly one leading slash.
        return Path(parsed.path[1:]) if parsed.path.startswith("/") else Path(parsed.path)
    if parsed.scheme == "file":
        return Path(parsed.path)
    if not parsed.scheme:
        return Path(uri)
    return None


def probe_tracking_store(uri: str, timeout: float = 3.0) -> Optional[str]:
    r"""Check a tracking URI is usable without touching it.

    Two shapes. A remote server gets a TCP probe first, so an unreachable host
    costs three seconds rather than however long MLflow's own retry policy takes.
    A local store gets an existence check, because building an MLflow client
    against a missing sqlite file *creates* it -- a preflight that conjures the
    store it was asked to inspect always reports a healthy store.

    Args:
        uri: Tracking URI, e.g. ``http://host:2222`` or ``sqlite:///mlruns.db``.
        timeout: Connect timeout in seconds, for remote stores.

    Returns:
        ``None`` when the store is reachable, else a one-line description.
    """
    local = local_store_path(uri)
    if local is not None:
        if local.exists():
            return None
        return (
            f"tracking store {local} does not exist; preflight will not create it"
        )

    parsed = urlparse(uri)
    host = parsed.hostname
    if not host:
        return f"cannot parse a host out of {uri!r}"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return None
    except OSError as exc:
        return f"{host}:{port} unreachable: {exc}"


def check_mlflow(
    uri: str,
    names: Sequence[str],
    resume: bool = False,
    client: Any = None,
    timeout: float = 3.0,
) -> CheckResult:
    r"""Confirm the tracking server answers, and that no target name is occupied.

    Writing a fresh grid into an experiment that already holds runs is the trap
    that cost this project a launch: ``pelesjak_cl`` resolves to
    ``pelesjak_cl_from_scratch``, MLflow experiment 92, which holds 684 published
    runs whose ``model_save_dir`` values point at a cluster this host cannot
    reach. ``--resume`` *wants* to find existing runs, so the same observation is
    a pass there and a refusal otherwise.

    Args:
        uri: Tracking URI.
        names: Experiment names the grid would write to.
        resume: Whether the grid will be launched with ``--resume``.
        client: Injection point for tests; anything with
            ``get_experiment_by_name`` and ``search_runs``.
        timeout: Socket probe timeout in seconds.

    Returns:
        A :class:`CheckResult` carrying one entry per experiment name.
    """
    if not names:
        return CheckResult("mlflow", SKIP, "no experiment names to check")

    if client is None:
        problem = probe_tracking_store(uri, timeout=timeout)
        if problem is not None:
            return CheckResult("mlflow", FAIL, problem, {"uri": uri})
        try:
            from mlflow.tracking import MlflowClient  # noqa: PLC0415

            client = MlflowClient(tracking_uri=uri)
        except Exception as exc:
            return CheckResult(
                "mlflow", FAIL,
                f"cannot build an MLflow client for {uri}: "
                f"{type(exc).__name__}: {exc}",
                {"uri": uri},
            )

    found = []
    for name in names:
        try:
            experiment = client.get_experiment_by_name(name)
        except Exception as exc:
            return CheckResult(
                "mlflow", FAIL,
                f"query for {name!r} failed: {type(exc).__name__}: {exc}",
                {"uri": uri},
            )
        if experiment is None:
            found.append({"name": name, "exists": False, "runs": 0})
            continue
        try:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id], max_results=1000
            )
            n_runs = len(runs)
            more = getattr(runs, "token", None) is not None
        except Exception as exc:
            return CheckResult(
                "mlflow", FAIL,
                f"run search for {name!r} failed: {type(exc).__name__}: {exc}",
                {"uri": uri},
            )
        found.append({
            "name": name, "exists": True,
            "experiment_id": str(experiment.experiment_id),
            "runs": n_runs, "runs_truncated": more,
        })

    details = {"uri": uri, "resume": resume, "experiments": found}
    occupied = [f for f in found if f["exists"] and f["runs"] > 0]
    if occupied and not resume:
        worst = max(occupied, key=lambda f: f["runs"])
        return CheckResult(
            "mlflow", FAIL,
            f"{len(occupied)} target experiment(s) already hold runs, e.g. "
            f"{worst['name']} (id {worst['experiment_id']}, {worst['runs']}"
            f"{'+' if worst['runs_truncated'] else ''} runs); pick a fresh "
            f"prefix or pass --resume",
            details,
        )
    if occupied:
        return CheckResult(
            "mlflow", PASS,
            f"{uri} reachable; --resume will continue into "
            f"{len(occupied)}/{len(found)} existing experiment(s)",
            details,
        )
    if resume:
        return CheckResult(
            "mlflow", WARN,
            f"--resume requested but none of the {len(found)} target "
            f"experiment(s) hold any runs; every chain will start at episode 1",
            details,
        )
    return CheckResult(
        "mlflow", PASS,
        f"{uri} reachable; all {len(found)} target experiment(s) free",
        details,
    )


def chain_id_for(dataset: str, task: str, mode: str, out_root: Path) -> str:
    r"""The ``chain_id`` ``continuous_learning.py`` would log for this cell.

    Mirrors ``chain_id = f"{dataset}/{task}/{mode}/{model_save_dir}"`` and
    ``run_grid.py``'s directory layout, ``{out}/{dataset}_{task}_{mode}/models``.
    The path is used *as written on the command line*, so it stays relative.

    Args:
        dataset: Dataset name.
        task: Task name.
        mode: Learning mode.
        out_root: The launcher's ``--out`` value.

    Returns:
        The chain identifier.
    """
    save_dir = f"{Path(out_root)}/{dataset}_{task}_{mode}/models"
    return f"{dataset}/{task}/{mode}/{save_dir}"


def check_resume_quorum(
    pairs: Sequence[tuple],
    modes: Sequence[str],
    prefix: str,
    num_samples: int,
    out_root: Path,
    client: Any = None,
    uri: str = "",
) -> CheckResult:
    r"""Confirm ``--resume`` can actually meet its own quorum.

    An increment counts as complete only when it has ``len(seeds)`` finished
    runs. Relaunching a 3-seed chain with ``--num-samples 5`` makes every past
    increment look incomplete, and the chain silently restarts from episode 1 --
    the exact failure that a hardcoded quorum of 5 caused here. Relaunching with
    a *smaller* count is the mirror image: a partially finished increment then
    satisfies the quorum and the chain resumes off the best of a subset.

    Args:
        pairs: ``(dataset, task)`` pairs.
        modes: Learning modes.
        prefix: MLflow experiment prefix.
        num_samples: Seeds the relaunch will use.
        out_root: The launcher's ``--out`` value, which enters the chain id.
        client: Injection point for tests.
        uri: Tracking URI, used only when ``client`` is None.

    Returns:
        A :class:`CheckResult` listing every chain whose recorded trial count
        disagrees with ``num_samples``.
    """
    if not pairs:
        return CheckResult("resume", SKIP, "no dataset:task pairs requested")
    if client is None:
        problem = probe_tracking_store(uri)
        if problem is not None:
            return CheckResult("resume", FAIL, problem, {"uri": uri})
        try:
            from mlflow.tracking import MlflowClient  # noqa: PLC0415

            client = MlflowClient(tracking_uri=uri)
        except Exception as exc:
            return CheckResult(
                "resume", FAIL,
                f"cannot build an MLflow client for {uri}: "
                f"{type(exc).__name__}: {exc}", {"uri": uri},
            )

    mismatches = []
    resumable = 0
    inspected = 0
    for (dataset, task), mode in itertools.product(pairs, modes):
        name = f"{prefix}_{mode}"
        try:
            experiment = client.get_experiment_by_name(name)
        except Exception as exc:
            return CheckResult(
                "resume", FAIL,
                f"query for {name!r} failed: {type(exc).__name__}: {exc}",
                {"uri": uri},
            )
        if experiment is None:
            continue
        chain = chain_id_for(dataset, task, mode, out_root)
        filters = " and ".join([
            f"params.dataset_name = '{dataset}'",
            f"params.task_name = '{task}'",
            f"params.chain_id = '{chain}'",
            "attributes.status = 'FINISHED'",
        ])
        try:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filters, max_results=1000,
            )
        except Exception as exc:
            return CheckResult(
                "resume", FAIL,
                f"run search for {name!r} failed: {type(exc).__name__}: {exc}",
                {"uri": uri},
            )
        if not runs:
            continue
        inspected += 1
        counts: dict = {}
        for run in runs:
            increment = run.data.params.get("increment")
            if increment is None:
                continue
            counts[int(increment)] = counts.get(int(increment), 0) + 1
        if not counts:
            continue
        newest = max(counts)
        if counts[newest] == num_samples:
            resumable += 1
        else:
            mismatches.append(
                f"{dataset}:{task}:{mode} increment {newest} has "
                f"{counts[newest]} trial(s), --num-samples is {num_samples}"
            )

    details = {
        "num_samples": num_samples,
        "chains_with_runs": inspected,
        "chains_resumable": resumable,
        "mismatches": mismatches,
    }
    if mismatches:
        return CheckResult(
            "resume", FAIL,
            f"{len(mismatches)} chain(s) cannot meet the resume quorum: "
            f"{mismatches[0]}",
            details,
        )
    if inspected == 0:
        return CheckResult(
            "resume", WARN,
            "no finished runs match any chain id; --resume will start every "
            "chain at episode 1", details,
        )
    return CheckResult(
        "resume", PASS,
        f"{resumable}/{inspected} chain(s) with history can resume at "
        f"{num_samples} trials/increment", details,
    )


# ---------------------------------------------------------------------------
# 6. Numerical smoke test
# ---------------------------------------------------------------------------


class _RefusingTextEmbedder:
    """Raises rather than downloading. A cache miss must be loud, not slow."""

    def __call__(self, sentences):
        raise RuntimeError(
            "text embedder invoked during preflight: the materialised cache is "
            "incomplete, and preflight must not download the glove embedder"
        )


def _synthetic_graph():
    r"""A two-table graph with missing cells in every numerical column.

    Used when no real dataset is cached, and deliberately harsher than any real
    one: the na_strategy regression only shows up once a batch contains a missing
    numerical cell, and here every batch does.

    Returns:
        ``(data, col_stats_dict, entity_table)``.
    """
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    from relbench.base import Database, Table  # noqa: PLC0415
    from torch_frame import stype  # noqa: PLC0415

    from redelex.data import make_pkey_fkey_graph  # noqa: PLC0415

    n_users, n_visits = 64, 256
    rng = np.random.default_rng(0)

    age = rng.normal(size=n_users)
    age[::3] = np.nan  # every batch of 16 will contain several
    category = pd.Series((["a", "b", "c"] * (n_users // 3 + 1))[:n_users], dtype=object)
    users = pd.DataFrame({
        "__PK__": np.arange(n_users),
        "reg": pd.date_range("2020-01-01", periods=n_users, freq="D"),
        "age": age,
        "cat": category.values,
    })
    duration = rng.normal(size=n_visits)
    duration[::3] = np.nan
    visits = pd.DataFrame({
        "__PK__": np.arange(n_visits),
        "FK_users": np.arange(n_visits) % n_users,
        "vt": pd.date_range("2020-01-01", periods=n_visits, freq="6h"),
        "dur": duration,
    })

    db = Database({
        "users": Table(df=users, fkey_col_to_pkey_table={}, pkey_col="__PK__",
                       time_col="reg"),
        "visits": Table(df=visits, fkey_col_to_pkey_table={"FK_users": "users"},
                        pkey_col="__PK__", time_col="vt"),
    })
    schema = {
        "users": {"age": stype.numerical, "cat": stype.categorical},
        "visits": {"dur": stype.numerical},
    }
    # cache_dir=None: materialise in memory so preflight writes nothing.
    # The embedder is never reached (no text columns) but must not be left to
    # default, which would construct the 480 MB glove embedder.
    data, col_stats = make_pkey_fkey_graph(
        db, schema, text_embedder=_RefusingTextEmbedder(), cache_dir=None
    )
    return data, col_stats, "users"


def _cached_graph(dataset: str, graph_cache: Path):
    r"""Load a real dataset's materialised graph, reading only, never writing.

    Args:
        dataset: Dataset name.
        graph_cache: Root of the materialised graph cache.

    Returns:
        ``(data, col_stats_dict, entity_table)``, where ``entity_table`` is a
        node type that carries a time column, preferring one with missing
        numerical cells so the smoke actually exercises ``na_strategy``.
    """
    import torch  # noqa: PLC0415
    import torch_frame  # noqa: PLC0415
    from relbench.datasets import get_dataset  # noqa: PLC0415

    from experiments.continuous_learning.utils import (  # noqa: PLC0415
        get_attribute_schema,
    )
    from redelex.data import make_pkey_fkey_graph  # noqa: PLC0415

    # download=False everywhere: rel-stack's upstream db.zip was republished
    # without refreshing relbench's pinned hash, and preflight must not fetch.
    db = get_dataset(dataset, download=False).get_db(upto_test_timestamp=False)
    cache = graph_cache / dataset
    schema = get_attribute_schema(str(cache / "attribute-schema.json"), db)
    data, col_stats = make_pkey_fkey_graph(
        db, schema, text_embedder=_RefusingTextEmbedder(),
        cache_dir=str(cache / "materialized"),
    )

    timed = [nt for nt in data.node_types if hasattr(data[nt], "time")]
    if not timed:
        raise ValueError(f"{dataset} has no node type with a time column")
    with_nan = [
        nt for nt in timed
        if torch_frame.numerical in data[nt].tf.feat_dict
        and bool(torch.isnan(data[nt].tf.feat_dict[torch_frame.numerical]).any())
    ]
    return data, col_stats, (with_nan or timed)[0]


def smoke_verdict(
    source: str,
    entity: str,
    steps: int,
    channels: int,
    non_finite: Sequence[str],
    losses: Sequence[float],
    batches_with_missing: int,
) -> CheckResult:
    r"""Turn the smoke run's observations into a verdict.

    Split out from :func:`check_smoke` so each branch can be driven directly: a
    non-finite loss with finite parameters is a real state but an awkward one to
    provoke from a model, and a branch no test can reach is a branch that can
    silently stop working.

    Args:
        source: ``"synthetic"`` or the dataset name the smoke ran on.
        entity: Node type used as the seed table.
        steps: Optimiser steps taken.
        channels: GNN width used.
        non_finite: Names of parameter tensors holding a non-finite value.
        losses: Per-step loss values.
        batches_with_missing: How many sampled batches held a missing numerical
            cell. Zero means the run could not have detected the na_strategy
            regression, whatever else it showed.

    Returns:
        A :class:`CheckResult` named ``smoke``.

    Raises:
        ValueError: If ``losses`` is empty; a smoke run that took no step has
            nothing to report and must not be mistaken for a pass.
    """
    import numpy as np  # noqa: PLC0415

    if len(losses) == 0:
        raise ValueError("`losses` must hold at least one step")

    details = {
        "source": source,
        "entity_table": entity,
        "steps": steps,
        "channels": channels,
        "batches_with_missing_numericals": batches_with_missing,
        "non_finite_parameters": list(non_finite),
        "losses": [round(float(x), 6) for x in losses],
    }
    if non_finite:
        return CheckResult(
            "smoke", FAIL,
            f"{len(non_finite)} parameter tensor(s) went non-finite in {steps} "
            f"steps on {source}, e.g. {non_finite[0]}",
            details,
        )
    if not bool(np.all(np.isfinite(np.asarray(losses, dtype=float)))):
        return CheckResult(
            "smoke", FAIL,
            f"loss went non-finite in {steps} steps on {source}: "
            f"{details['losses']}", details,
        )
    if batches_with_missing == 0:
        return CheckResult(
            "smoke", WARN,
            f"{steps} steps on {source} stayed finite, but no sampled batch held "
            f"a missing numerical cell, so na_strategy was never exercised",
            details,
        )
    return CheckResult(
        "smoke", PASS,
        f"{steps} steps on {source} ({entity}) stayed finite; "
        f"{batches_with_missing}/{steps} batches held missing numerical cells",
        details,
    )


def pick_smoke_dataset(
    pairs: Sequence[tuple], relbench_cache: Path, graph_cache: Path
) -> Optional[str]:
    r"""The smallest fully-cached dataset among ``pairs``, or ``None``.

    Smallest, because the smoke must stay in the seconds range; fully cached,
    because a partial cache would make ``make_pkey_fkey_graph`` rebuild and write.

    Args:
        pairs: ``(dataset, task)`` pairs.
        relbench_cache: Root of the relbench cache.
        graph_cache: Root of the materialised graph cache.

    Returns:
        A dataset name, or ``None`` when none is usable.
    """
    candidates = []
    for dataset in sorted({ds for ds, _ in pairs}):
        report = inspect_dataset(dataset, [], relbench_cache, graph_cache)
        # Ignore task-cache problems: the smoke needs the graph, not the labels.
        blocking = [p for p in report["problems"] if not p.startswith("task cache")]
        if blocking or not report["has_schema"]:
            continue
        candidates.append((report["relbench_bytes"] + report["graph_bytes"], dataset))
    return min(candidates)[1] if candidates else None


def check_smoke(
    source: str = "synthetic",
    steps: int = 5,
    graph_cache: Path = REPO / ".cache",
    model_factory: Optional[Callable] = None,
    channels: int = 16,
    batch_size: int = 16,
) -> CheckResult:
    r"""Train the real model for a few steps on CPU and demand finite parameters.

    This is the check that would have caught the NaN grid. With the library's
    default numerical encoder, a column holding any missing cell yields a NaN
    gradient; Adam turns that into a NaN parameter on the first step and the
    column's output is dead for the rest of the chain. The forward pass hides it
    and the metric stays plausible, so nothing downstream complains.

    The check reports how many of the sampled batches actually contained a
    missing numerical cell. If none did, it cannot have tested the failure it
    exists for, and says so rather than claiming a pass.

    Args:
        source: ``"synthetic"``, or a dataset name whose materialised cache is
            complete.
        steps: Optimiser steps to take.
        graph_cache: Root of the materialised graph cache.
        model_factory: Injection point for tests; called as
            ``model_factory(data, col_stats, channels)``. Defaults to the real
            :class:`HeterogeneousSAGE`.
        channels: GNN width. Small by default: the failure is width-independent.
        batch_size: Seed nodes per batch.

    Returns:
        A :class:`CheckResult` naming the first non-finite parameter, if any.

    Raises:
        ValueError: If ``steps`` is not a positive int.
    """
    if not isinstance(steps, int) or steps < 1:
        raise ValueError(f"`steps` must be a positive int, got {steps!r}")

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torch_frame  # noqa: PLC0415
    from relbench.modeling.graph import AttachTargetTransform  # noqa: PLC0415
    from torch_geometric.loader import NeighborLoader  # noqa: PLC0415

    if model_factory is None:
        from experiments.continuous_learning.models import (  # noqa: PLC0415
            HeterogeneousSAGE,
        )

        def model_factory(data, col_stats, width):
            return HeterogeneousSAGE(
                data, col_stats, gnn_channels=width, gnn_layers=2, out_channels=1
            )

    try:
        # torch_frame's cache loader emits a UserWarning per table, and a
        # preflight table that scrolls off the screen is one nobody reads.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if source == "synthetic":
                data, col_stats, entity = _synthetic_graph()
            else:
                data, col_stats, entity = _cached_graph(source, Path(graph_cache))
    except Exception as exc:
        return CheckResult(
            "smoke", FAIL,
            f"could not build the {source} graph: {type(exc).__name__}: {exc}",
            {"source": source},
        )

    torch.manual_seed(0)
    n_nodes = int(data[entity].num_nodes)
    rng = np.random.default_rng(0)
    target = torch.from_numpy(rng.normal(size=n_nodes))
    loader = NeighborLoader(
        data,
        num_neighbors=[8, 4],
        time_attr="time",
        input_nodes=(entity, torch.arange(n_nodes)),
        input_time=data[entity].time,
        transform=AttachTargetTransform(entity, target),
        batch_size=batch_size,
        temporal_strategy="uniform",
        shuffle=True,
    )
    model = model_factory(data, col_stats, channels)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    batches_with_missing = 0
    losses = []
    try:
        for _step, batch in zip(range(steps), itertools.cycle(loader)):
            for node_type in batch.node_types:
                feats = getattr(batch[node_type], "tf", None)
                if feats is None:
                    continue
                numeric = feats.feat_dict.get(torch_frame.numerical)
                if numeric is not None and bool(torch.isnan(numeric).any()):
                    batches_with_missing += 1
                    break
            optimiser.zero_grad()
            out = model(batch, entity).squeeze(-1)
            loss = torch.nn.functional.mse_loss(out, batch[entity].y.float())
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))
    except Exception as exc:
        return CheckResult(
            "smoke", FAIL,
            f"training step raised {type(exc).__name__}: {exc}",
            {"source": source, "entity_table": entity},
        )

    non_finite = [
        name for name, param in model.named_parameters()
        if not bool(torch.isfinite(param).all())
    ]
    return smoke_verdict(
        source=source, entity=entity, steps=steps, channels=channels,
        non_finite=non_finite, losses=losses,
        batches_with_missing=batches_with_missing,
    )


# ---------------------------------------------------------------------------
# Rendering and the runner
# ---------------------------------------------------------------------------

CHECK_NAMES = (
    "git", "interpreter", "torch", "threads", "datasets", "plan", "disk",
    "mlflow", "resume", "smoke",
)


def render_table(results: Sequence[CheckResult]) -> str:
    r"""Render check results as a fixed-width table.

    Args:
        results: Results in display order.

    Returns:
        The table as a string, without a trailing newline.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("`results` must not be empty")
    width = max(len(r.name) for r in results)
    lines = [f"{'CHECK'.ljust(width)}  STATUS  REASON",
             f"{'-' * width}  ------  {'-' * 60}"]
    lines += [f"{r.name.ljust(width)}  {r.status.ljust(6)}  {r.reason}" for r in results]
    return "\n".join(lines)


def exit_code(results: Iterable[CheckResult], strict: bool = False) -> int:
    r"""Turn results into a process exit code.

    Args:
        results: The check results.
        strict: When True, WARN also fails the gate.

    Returns:
        ``0`` when the gate passes, ``1`` otherwise.
    """
    bad = {FAIL, WARN} if strict else {FAIL}
    return 1 if any(r.status in bad for r in results) else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--tier", nargs="+", choices=sorted(run_grid.TIERS),
                        help="Dataset tiers to be launched, as in run_grid.py.")
    source.add_argument("--pairs", nargs="+", metavar="DATASET:TASK",
                        help="Explicit dataset:task pairs.")
    parser.add_argument("--modes", nargs="+", default=run_grid.DEFAULT_MODES)
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Seeds per episode, as passed to run_grid.py.")
    gpu = parser.add_mutually_exclusive_group()
    gpu.add_argument("--expect-gpu", dest="expect_gpu", action="store_true",
                     default=None, help="Require usable CUDA devices.")
    gpu.add_argument("--expect-cpu", dest="expect_gpu", action="store_false",
                     help="Require that no CUDA device is used.")
    parser.add_argument("--jobs", type=int, default=None,
                        help="Concurrent chains. Default: one per visible GPU, "
                             "or 1 when there is none.")
    parser.add_argument("--cpus-per-job", type=int, default=4)
    parser.add_argument("--cpus-per-trial", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--mlflow-uri", default="http://potato.felk.cvut.cz:2222")
    parser.add_argument("--mlflow-experiment-prefix", default="pelesjak_cl_v2")
    parser.add_argument("--out", default="logs/grid",
                        help="Launcher --out; where checkpoints will land.")
    parser.add_argument("--resume", action="store_true",
                        help="Check as if the grid will be launched with --resume.")
    parser.add_argument("--graph-cache", default=str(REPO / ".cache"))
    parser.add_argument("--relbench-cache", default=None,
                        help="Default: RELBENCH_CACHE_DIR, else the pooch cache.")
    parser.add_argument("--checkpoint-bytes", type=int, default=CHECKPOINT_BYTES)
    parser.add_argument("--disk-headroom", type=float, default=1.25)
    parser.add_argument("--smoke-dataset", default="auto",
                        help="'auto' (smallest fully cached, else synthetic), "
                             "'synthetic', or a dataset name.")
    parser.add_argument("--smoke-steps", type=int, default=5)
    parser.add_argument("--only", nargs="+", choices=CHECK_NAMES, default=None)
    parser.add_argument("--skip", nargs="+", choices=CHECK_NAMES, default=[])
    parser.add_argument("--strict", action="store_true",
                        help="Treat WARN as failure.")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def resolve_pairs(args) -> list:
    r"""Resolve ``--tier``/``--pairs`` into concrete pairs.

    Args:
        args: Parsed arguments.

    Returns:
        A list of ``(dataset, task)`` tuples, possibly empty when neither flag
        was given.

    Raises:
        ValueError: If a ``--pairs`` entry is not ``dataset:task``.
    """
    if args.pairs:
        pairs = []
        for raw in args.pairs:
            if ":" not in raw:
                raise ValueError(f"--pairs entry {raw!r} must look like dataset:task")
            dataset, task = raw.split(":", 1)
            pairs.append((dataset, task))
        return pairs
    if args.tier:
        return [pair for tier in args.tier for pair in run_grid.TIERS[tier]]
    return []


def _visible_gpu_count() -> int:
    try:
        import torch  # noqa: PLC0415

        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


def run_checks(args) -> list:
    r"""Run every requested check, in table order.

    Args:
        args: Parsed arguments.

    Returns:
        The list of :class:`CheckResult`, one per check that was not skipped.
    """
    wanted = set(args.only) if args.only else set(CHECK_NAMES)
    wanted -= set(args.skip)

    pairs = resolve_pairs(args)
    graph_cache = Path(args.graph_cache)
    rb_cache = Path(args.relbench_cache) if args.relbench_cache else relbench_cache_dir()
    names = experiment_names(args.mlflow_experiment_prefix, args.modes)
    projection = project_grid(
        pairs, args.modes, args.num_samples, checkpoint_bytes=args.checkpoint_bytes
    )

    results = []
    if "git" in wanted:
        results.append(check_git())
    if "interpreter" in wanted:
        results.append(check_interpreter())
    if "torch" in wanted:
        results.append(check_torch_build(args.expect_gpu))
    if "threads" in wanted:
        import torch  # noqa: PLC0415

        jobs = args.jobs if args.jobs is not None else max(1, _visible_gpu_count())
        results.append(check_threads(
            torch_threads=args.torch_threads,
            cpus_per_trial=args.cpus_per_trial,
            cpus_per_job=args.cpus_per_job,
            jobs=jobs,
            cpu_count=os.cpu_count() or 1,
            omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
            observed_torch_threads=torch.get_num_threads(),
        ))
    if "datasets" in wanted:
        results.append(check_datasets(pairs, rb_cache, graph_cache))
    if "plan" in wanted:
        results.append(check_plan(projection))
    if "disk" in wanted:
        results.append(check_disk(projection, Path(args.out),
                                  headroom=args.disk_headroom))
    if "mlflow" in wanted:
        results.append(check_mlflow(args.mlflow_uri, names, resume=args.resume))
    if "resume" in wanted:
        if not args.resume:
            results.append(CheckResult("resume", SKIP, "--resume not requested"))
        else:
            results.append(check_resume_quorum(
                pairs, args.modes, args.mlflow_experiment_prefix,
                args.num_samples, Path(args.out), uri=args.mlflow_uri,
            ))
    if "smoke" in wanted:
        source = args.smoke_dataset
        if source == "auto":
            source = pick_smoke_dataset(pairs, rb_cache, graph_cache) or "synthetic"
        results.append(check_smoke(source, steps=args.smoke_steps,
                                   graph_cache=graph_cache))
    return results


def main(argv=None) -> int:
    try:
        args = parse_args(argv)
        # relbench, torch_frame and MLflow all print to stdout while the checks
        # run. With --json that chatter lands ahead of the document and makes it
        # unparseable, so the whole check phase is redirected to stderr and
        # stdout carries nothing but this tool's own output.
        with contextlib.redirect_stdout(sys.stderr):
            results = run_checks(args)
    except ValueError as exc:
        print(f"preflight: {exc}", file=sys.stderr)
        return 2

    code = exit_code(results, strict=args.strict)
    if args.json:
        commit = next((r.details.get("commit") for r in results if r.name == "git"), None)
        print(json.dumps({
            "ok": code == 0,
            "exit_code": code,
            "commit": commit,
            "strict": args.strict,
            "checks": [r.to_dict() for r in results],
        }, indent=2, default=str))
        return code

    git = next((r for r in results if r.name == "git"), None)
    if git is not None and git.details.get("commit"):
        print(f"commit {git.details['commit']}  "
              f"branch {git.details.get('branch', '?')}\n")
    print(render_table(results))
    counts = {s: sum(1 for r in results if r.status == s) for s in STATUSES}
    print(f"\n{counts[PASS]} pass, {counts[WARN]} warn, {counts[FAIL]} fail, "
          f"{counts[SKIP]} skip -> "
          f"{'LAUNCH' if code == 0 else 'DO NOT LAUNCH'}")
    return code


if __name__ == "__main__":
    sys.exit(main())
