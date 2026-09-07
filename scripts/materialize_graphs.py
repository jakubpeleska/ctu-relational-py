r"""Prepare the RelBench download cache and the materialised graph cache.

Two costs stand between a fresh machine and the first continual-learning trial:
the RelBench database has to be on disk, and the heterogeneous graph has to be
materialised into per-table tensor frames. Neither belongs on a login node and
neither belongs in the grid's critical path -- materialising is a one-off charge
that would otherwise be paid by whichever trial happens to touch the dataset
first, while every other trial on that GPU waits.

Measured on potato (CPU-only work, glove embedder, 8 torch threads, full builds
with an empty cache)::

    dataset       raw db      materialised   wall     peak RSS
    rel-f1           2 MB         11.5 MiB      4 s      1.9 GiB
    rel-hm         181 MB          2.8 GiB    253 s      8.2 GiB
    rel-trial      650 MB         13.8 GiB    493 s     20.3 GiB
    rel-stack     1005 MB         10.5 GiB    521 s     17.0 GiB
    rel-ratebeer   2.6 GB         29.5 GiB   1618 s     52.5 GiB
    rel-amazon     7.0 GB         51.7 GiB   3348 s    139.9 GiB

That is 1h44m of compute and 108 GiB of cache for the six-dataset grid, and
rel-amazon alone accounts for half of both. The peak RSS column is what sizes
the job's memory request; the wall column is what decides, per dataset, whether
there is still time to start one.

Usage::

    # cheap: report what is present and what is missing, no work, no torch
    .venv/bin/python scripts/materialize_graphs.py --check

    # print the execution plan (order, download decisions, time, disk) and stop
    .venv/bin/python scripts/materialize_graphs.py --dry-run

    # do the work; already-complete datasets are skipped
    .venv/bin/python scripts/materialize_graphs.py rel-f1 rel-trial

Each dataset is built in its own child process. That is not incidental: peak RSS
is a process high-water mark, so a shared process would report rel-amazon's
139.9 GB for whatever ran after it, and an OOM kill on one dataset would take
the whole job down instead of one entry in the summary table.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------
# Per-dataset download policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DownloadPolicy:
    r"""How a dataset's raw database may legitimately be obtained.

    Args:
        prepared: Whether ``get_dataset(name, download=True)`` -- the
            RelBench-prepared ``db.zip`` -- may be used when the extracted
            database is absent.
        reason: Why this dataset gets this policy. Kept next to the flag so a
            future reader does not have to rediscover the trap.
    """

    prepared: bool
    reason: str


# One rule per dataset, because the datasets genuinely differ. Assuming a single
# rule is how you either delete a good file (rel-stack) or 404 (rel-amazon).
DOWNLOAD_POLICY: dict[str, DownloadPolicy] = {
    "rel-f1": DownloadPolicy(
        prepared=True,
        reason="prepared db.zip is 1 MB and hashes correctly; make_db would "
        "re-fetch the raw archive for no benefit",
    ),
    "rel-trial": DownloadPolicy(
        prepared=True,
        reason="prepared db.zip hashes correctly",
    ),
    "rel-hm": DownloadPolicy(
        prepared=True,
        reason="make_db() cannot run unattended -- it requires a manually "
        "accepted Kaggle competition download",
    ),
    "rel-ratebeer": DownloadPolicy(
        prepared=True,
        reason="prepared db.zip hashes correctly; make_db() pulls from a "
        "Dropbox share link that is not a stable dependency",
    ),
    "rel-amazon": DownloadPolicy(
        prepared=True,
        reason="make_db() 404s -- UCSD removed the mcauley_group raw file it "
        "fetches, so the RelBench-prepared path is the only one that works",
    ),
    "rel-stack": DownloadPolicy(
        prepared=False,
        reason="relbench 2.1.1 pins a SHA256 for rel-stack/db.zip that upstream "
        "no longer serves; download=True DELETES the good local file and "
        "raises. Never fetch it. See notes/DECISIONS.md.",
    ),
    "rel-avito": DownloadPolicy(
        prepared=True,
        reason="prepared db.zip hashes correctly (not in the CL grid; listed so "
        "--check can report the shared cache honestly)",
    ),
}

# Order the job works through by default. rel-f1 and rel-trial come first
# because they are the first chains the grid launches (20 and 41 GPU-h, the two
# cheapest); rel-hm is third only because it materialises in four minutes. The
# two datasets that cost an hour and tens of gigabytes each go last, so a job
# killed at the wall clock loses the least.
DEFAULT_DATASETS: tuple[str, ...] = (
    "rel-f1",
    "rel-trial",
    "rel-hm",
    "rel-stack",
    "rel-ratebeer",
    "rel-amazon",
)


@dataclass(frozen=True)
class Estimate:
    r"""Expected cost of materialising one dataset.

    Args:
        seconds: Wall time on potato with 8 torch threads.
        cache_bytes: Size of the resulting ``materialized/`` directory.
        peak_rss_bytes: Process high-water mark observed during the build.
        measured: True when ``seconds`` and ``peak_rss_bytes`` come from a real
            run, False when they are extrapolated from cache size.
    """

    seconds: float
    cache_bytes: int
    peak_rss_bytes: int
    measured: bool


_GB = 1024**3
_MB = 1024**2

# Every entry for a grid dataset is a real full build on potato: rel-hm,
# rel-ratebeer and rel-amazon from logs/materialize.log (2026-09-06), rel-f1,
# rel-trial and rel-stack timed through this script on the same day. rel-avito
# is a guess -- it is not in the grid, and nothing here relies on it being
# right beyond deciding whether to start it.
ESTIMATES: dict[str, Estimate] = {
    "rel-f1": Estimate(5, 12 * _MB, int(1.9 * _GB), measured=True),
    "rel-trial": Estimate(493, 14133 * _MB, int(20.3 * _GB), measured=True),
    "rel-hm": Estimate(253, 2868 * _MB, int(8.2 * _GB), measured=True),
    "rel-stack": Estimate(521, 10708 * _MB, int(17.0 * _GB), measured=True),
    "rel-ratebeer": Estimate(1618, 30234 * _MB, int(52.5 * _GB), measured=True),
    "rel-amazon": Estimate(3348, 52975 * _MB, int(139.9 * _GB), measured=True),
    "rel-avito": Estimate(600, 8 * _GB, 16 * _GB, measured=False),
}

# Marker written next to the tensor frames once a dataset finishes cleanly. It
# is a convenience record (wall time, RSS, table list), never the skip
# predicate: the predicate is the tensor frames themselves, so a cache built by
# an older script or copied in by hand is still recognised as complete.
RECEIPT_NAME = "materialize-receipt.json"


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def default_relbench_cache() -> Path:
    r"""Locate the RelBench download cache the same way relbench itself does.

    Returns:
        Path: ``$RELBENCH_CACHE_DIR`` when set, else pooch's OS cache location.
    """
    env = os.environ.get("RELBENCH_CACHE_DIR")
    if env:
        return Path(env)
    try:
        import pooch  # light import; avoided entirely when the env var is set

        return Path(pooch.os_cache("relbench"))
    except ImportError:  # pragma: no cover - pooch is a relbench dependency
        return Path.home() / ".cache" / "relbench"


def db_dir(relbench_cache: Path, name: str) -> Path:
    r"""Directory of extracted parquet tables for `name`."""
    return relbench_cache / name / "db"


def materialized_dir(cache_dir: Path, name: str) -> Path:
    r"""Directory of materialised tensor frames for `name`."""
    return cache_dir / name / "materialized"


def schema_path(cache_dir: Path, name: str) -> Path:
    r"""Path of the cached attribute schema for `name`."""
    return cache_dir / name / "attribute-schema.json"


# --------------------------------------------------------------------------
# Inspection -- cheap, no torch, no database load
# --------------------------------------------------------------------------


def expected_tables(db_path: Path) -> Optional[list[str]]:
    r"""Table names a materialised cache for this database must contain.

    ``Database.load`` builds ``table_dict`` from ``*.parquet`` stems, and
    ``make_pkey_fkey_graph`` writes one ``<table>.pt`` per entry of that dict,
    so the parquet stems are exactly the expected tensor-frame stems. Reading
    the directory listing costs nothing, which is what makes ``--check`` cheap.

    Args:
        db_path: The dataset's extracted ``db/`` directory.

    Returns:
        Optional[list[str]]: Sorted table names, or None when the directory does
        not exist or holds no parquet files -- in which case the expected set is
        genuinely unknown rather than empty.
    """
    if not db_path.is_dir():
        return None
    names = sorted(p.stem for p in db_path.glob("*.parquet"))
    return names or None


def materialised_tables(mat_path: Path) -> list[str]:
    r"""Sorted stems of the tensor frames present in `mat_path`."""
    if not mat_path.is_dir():
        return []
    return sorted(p.stem for p in mat_path.glob("*.pt"))


def corrupt_tables(mat_path: Path) -> list[str]:
    r"""Tensor frames that cannot be opened, i.e. truncated writes.

    ``torch_frame.utils.io.save`` is a bare ``torch.save`` with no temp file and
    no rename, so a job killed at the wall clock mid-write leaves a truncated
    ``.pt`` behind. On the next attempt ``Dataset.materialize`` sees
    ``osp.isfile(path)`` and tries to load it, and the whole dataset dies on a
    file the job itself wrote. A torch checkpoint is a zip whose central
    directory is written last, so opening it confirms the write completed --
    and it reads only the directory, which takes microseconds even on the 48 GiB
    rel-amazon review frame.

    This catches truncation, which is the failure mode a kill produces. It does
    not catch silent bit corruption; nothing cheap does.

    Args:
        mat_path: The dataset's ``materialized/`` directory.

    Returns:
        list[str]: Sorted stems of unreadable tensor frames.
    """
    bad = []
    for path in sorted(Path(mat_path).glob("*.pt")) if Path(mat_path).is_dir() else []:
        try:
            with zipfile.ZipFile(path) as archive:
                archive.namelist()
        except Exception:
            bad.append(path.stem)
    return bad


def dir_bytes(path: Path) -> int:
    r"""Total size of the regular files directly inside `path` (0 if absent)."""
    if not Path(path).is_dir():
        return 0
    return sum(p.stat().st_size for p in Path(path).iterdir() if p.is_file())


@dataclass
class DatasetState:
    r"""What is on disk for one dataset, and what that implies."""

    name: str
    db_present: bool
    expected: Optional[list[str]]
    materialised: list[str]
    corrupt: list[str]
    schema_present: bool
    cache_bytes: int
    status: str = ""
    receipt: dict = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        r"""True when no further work is needed for this dataset."""
        return self.status == "complete"


def classify(
    db_present: bool,
    expected: Optional[list[str]],
    materialised: list[str],
    corrupt: list[str],
    schema_present: bool,
) -> str:
    r"""Reduce the on-disk facts to a single status word.

    Args:
        db_present: Whether the extracted parquet database exists.
        expected: Table names the cache must hold, or None if unknowable.
        materialised: Table names the cache does hold.
        corrupt: Subset of `materialised` that failed the integrity probe.
        schema_present: Whether ``attribute-schema.json`` exists.

    Returns:
        str: One of ``corrupt``, ``needs-download``, ``unverifiable``,
        ``missing``, ``partial``, ``stale``, ``needs-schema``, ``complete``.

    Raises:
        ValueError: If `corrupt` names a table that is not in `materialised`,
            which would mean the two were read from different directories.
    """
    unknown = set(corrupt) - set(materialised)
    if unknown:
        raise ValueError(
            f"corrupt tables {sorted(unknown)} are not among the materialised "
            f"tables {materialised}; the two listings disagree"
        )

    # A corrupt frame outranks everything: it is the one state where doing
    # nothing is worse than doing work, because the next run dies on it.
    if corrupt:
        return "corrupt"
    if expected is None:
        # No parquet database, so the expected table set is unknown. If frames
        # exist anyway the cache may well be fine, but it cannot be confirmed --
        # and the experiment loads the database on every run regardless.
        return "unverifiable" if materialised else "needs-download"
    if not db_present:  # pragma: no cover - expected is None whenever this holds
        return "needs-download"
    if not materialised:
        return "missing"
    extra = set(materialised) - set(expected)
    if extra:
        return "stale"
    if set(materialised) != set(expected):
        return "partial"
    return "complete" if schema_present else "needs-schema"


def inspect(name: str, cache_dir: Path, relbench_cache: Path) -> DatasetState:
    r"""Read the on-disk state of one dataset without loading anything heavy.

    Args:
        name: RelBench dataset name.
        cache_dir: Root of the materialised cache (the repo's ``.cache``).
        relbench_cache: Root of the RelBench download cache.

    Returns:
        DatasetState: Presence, completeness and size, with `status` filled in.
    """
    db_path = db_dir(relbench_cache, name)
    mat_path = materialized_dir(cache_dir, name)
    expected = expected_tables(db_path)
    materialised = materialised_tables(mat_path)
    corrupt = corrupt_tables(mat_path)
    schema_present = schema_path(cache_dir, name).is_file()

    receipt = {}
    receipt_path = mat_path / RECEIPT_NAME
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text())
        except (OSError, json.JSONDecodeError):
            receipt = {}

    return DatasetState(
        name=name,
        db_present=db_path.is_dir(),
        expected=expected,
        materialised=materialised,
        corrupt=corrupt,
        schema_present=schema_present,
        cache_bytes=dir_bytes(mat_path),
        status=classify(
            db_path.is_dir(), expected, materialised, corrupt, schema_present
        ),
        receipt=receipt,
    )


# --------------------------------------------------------------------------
# Download decision
# --------------------------------------------------------------------------


def download_decision(name: str, db_present: bool) -> tuple[bool, str]:
    r"""Decide whether ``get_dataset`` may be called with ``download=True``.

    Args:
        name: RelBench dataset name.
        db_present: Whether the extracted parquet database already exists.

    Returns:
        tuple[bool, str]: The ``download`` argument, and a short label for the
        summary table (``fetch``, ``cached``, ``never`` or ``from-raw``).

    Raises:
        ValueError: If `name` has no recorded policy. Guessing a policy is how
            the rel-stack and rel-amazon traps get re-triggered.
    """
    if name not in DOWNLOAD_POLICY:
        raise ValueError(
            f"no download policy recorded for {name!r}; add one to "
            f"DOWNLOAD_POLICY rather than assuming the common case. "
            f"Known: {sorted(DOWNLOAD_POLICY)}"
        )
    policy = DOWNLOAD_POLICY[name]
    if db_present:
        # The database is already extracted, so get_db loads it straight from
        # parquet and a download would at best re-verify a hash -- at worst
        # re-transfer 6 GB, or trip the rel-stack pin.
        return False, "cached"
    if not policy.prepared:
        # rel-stack with no local database: make_db rebuilds from the raw
        # Stanford archive. Slow, but it is the only path that is not a trap.
        return False, "from-raw"
    return True, "fetch"


# --------------------------------------------------------------------------
# Budget arithmetic
# --------------------------------------------------------------------------


def job_deadline(budget_seconds: Optional[float], now: float) -> Optional[float]:
    r"""Epoch time this job must be finished by, or None if unbounded.

    Slurm exports ``SLURM_JOB_END_TIME`` as an epoch second, which is
    authoritative -- it accounts for queue time and for any ``scontrol update``.
    An explicit ``--time-budget`` is honoured too, and the earlier of the two
    wins so a deliberately conservative budget is never widened by the scheduler.

    Args:
        budget_seconds: Explicit budget from the CLI, or None.
        now: Current epoch time.

    Returns:
        Optional[float]: The deadline, or None when neither source gives one.

    Raises:
        ValueError: If `budget_seconds` is not positive.
    """
    deadlines = []
    if budget_seconds is not None:
        if budget_seconds <= 0:
            raise ValueError(f"--time-budget must be positive, got {budget_seconds}")
        deadlines.append(now + budget_seconds)
    slurm_end = os.environ.get("SLURM_JOB_END_TIME")
    if slurm_end:
        try:
            deadlines.append(float(slurm_end))
        except ValueError:
            pass  # a malformed scheduler variable must not stop the job
    return min(deadlines) if deadlines else None


def can_start(
    name: str,
    deadline: Optional[float],
    now: float,
    safety: float,
    reserve: float,
) -> tuple[bool, str]:
    r"""Decide whether starting `name` now can plausibly finish in time.

    The named failure mode is being killed at the wall clock after five hours of
    embedding, having written a truncated frame and produced nothing. Refusing
    to start a dataset that cannot finish converts that into an honest line in
    the summary and a job that can simply be resubmitted.

    Args:
        name: RelBench dataset name.
        deadline: Epoch time the job ends, or None for no limit.
        now: Current epoch time.
        safety: Multiplier applied to the estimate, covering the fact that the
            estimates were taken on potato and this may be a slower node.
        reserve: Seconds held back for writing the summary and exiting cleanly.

    Returns:
        tuple[bool, str]: Whether to start, and the reason when not.

    Raises:
        ValueError: If `safety` is not positive.
    """
    if safety <= 0:
        raise ValueError(f"--time-safety must be positive, got {safety}")
    if deadline is None:
        return True, "no deadline"
    remaining = deadline - now - reserve
    if remaining <= 0:
        return False, "no time left"
    estimate = ESTIMATES.get(name)
    if estimate is None:
        # Unknown cost: attempting is better than a blanket refusal, but say so.
        return True, "no estimate on record"
    needed = estimate.seconds * safety
    if needed > remaining:
        return False, (
            f"needs ~{fmt_duration(needed)} (est {fmt_duration(estimate.seconds)} "
            f"x{safety:g}), only {fmt_duration(remaining)} left"
        )
    return True, f"needs ~{fmt_duration(needed)}, {fmt_duration(remaining)} left"


def can_fit_on_disk(
    name: str, cache_dir: Path, headroom: float = 1.15
) -> tuple[bool, str]:
    r"""Check that the filesystem holding `cache_dir` can take the new cache.

    Args:
        name: RelBench dataset name.
        cache_dir: Root of the materialised cache.
        headroom: Multiplier over the estimated cache size.

    Returns:
        tuple[bool, str]: Whether to proceed, and a human-readable reason.

    Note:
        ``shutil.disk_usage`` reports filesystem free space, not a per-user
        quota. On a cluster home directory the quota usually binds first, so
        this catches the coarse failure only.
    """
    estimate = ESTIMATES.get(name)
    if estimate is None:
        return True, "no size estimate on record"
    root = Path(cache_dir)
    while not root.exists() and root != root.parent:
        root = root.parent
    free = shutil.disk_usage(root).free
    needed = int(estimate.cache_bytes * headroom)
    if free < needed:
        return False, (
            f"needs ~{fmt_bytes(needed)} free, filesystem has {fmt_bytes(free)}"
        )
    return True, f"needs ~{fmt_bytes(needed)}, {fmt_bytes(free)} free"


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------


def fmt_bytes(n: Optional[float]) -> str:
    r"""Human-readable byte count, or ``-`` for None/zero."""
    if not n:
        return "-"
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{value:.0f} B"
        value /= 1024
    return f"{value:.1f} TiB"  # pragma: no cover - unreachable, loop returns


def fmt_duration(seconds: Optional[float]) -> str:
    r"""Human-readable duration as ``H:MM:SS``, or ``-`` for None."""
    if seconds is None:
        return "-"
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    r"""Left-aligned fixed-width table with a rule under the header.

    Args:
        headers: Column titles.
        rows: Row cells; every row must have as many cells as there are headers.

    Returns:
        str: The rendered table, without a trailing newline.

    Raises:
        ValueError: If any row's width differs from the header's.
    """
    for i, row in enumerate(rows):
        if len(row) != len(headers):
            raise ValueError(
                f"row {i} has {len(row)} cells but there are "
                f"{len(headers)} headers: {row}"
            )
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip()]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())
    return "\n".join(lines)


def check_rows(states: list[DatasetState]) -> list[list[str]]:
    r"""Rows for the ``--check`` table, one per dataset."""
    rows = []
    for st in states:
        expected = len(st.expected) if st.expected is not None else "?"
        tables = f"{len(st.materialised)}/{expected}"
        if st.corrupt:
            tables += f" ({len(st.corrupt)} corrupt)"
        rows.append(
            [
                st.name,
                "yes" if st.db_present else "no",
                tables,
                "yes" if st.schema_present else "no",
                fmt_bytes(st.cache_bytes),
                st.status,
            ]
        )
    return rows


CHECK_HEADERS = ["dataset", "db", "tables", "schema", "cache", "status"]
SUMMARY_HEADERS = [
    "dataset",
    "downloaded",
    "materialised",
    "wall",
    "peak RSS",
    "cache",
]


def summary_rows(results: list[dict]) -> list[list[str]]:
    r"""Rows for the final summary table, one per dataset the job considered.

    Args:
        results: Worker result dicts, in the order the datasets were handled.

    Returns:
        list[list[str]]: Cells matching :data:`SUMMARY_HEADERS`.
    """
    rows = []
    for res in results:
        rss = res.get("peak_rss_bytes")
        rss_text = fmt_bytes(rss)
        if rss and res.get("rss_is_lower_bound"):
            rss_text = f">={rss_text}"
        rows.append(
            [
                res["dataset"],
                res.get("download_action", "-"),
                res.get("materialised", "-"),
                fmt_duration(res.get("wall_s")),
                rss_text,
                fmt_bytes(res.get("cache_bytes")),
            ]
        )
    return rows


# --------------------------------------------------------------------------
# The heavy part -- runs in a child process, one dataset per child
# --------------------------------------------------------------------------


def build_dataset(
    name: str,
    cache_dir: Path,
    relbench_cache: Path,
    embedder_name: str,
    torch_threads: int,
) -> dict:
    r"""Download if needed and materialise the graph cache for one dataset.

    Args:
        name: RelBench dataset name.
        cache_dir: Root of the materialised cache.
        relbench_cache: Root of the RelBench download cache.
        embedder_name: Text embedder passed to ``get_text_embedder``.
        torch_threads: Threads for the one-off build.

    Returns:
        dict: A result record for the summary table.

    Raises:
        ValueError: If the download policy for `name` is unknown, or if the
            decision function somehow asked to download a dataset whose policy
            forbids it.
    """
    # tqdm's env var is read when tqdm is imported, not when a bar is created,
    # so this must precede every heavy import. Without it torch_frame's embedder
    # bars turn a batch log into three megabytes of carriage returns.
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ["RELBENCH_CACHE_DIR"] = str(relbench_cache)

    import warnings

    warnings.filterwarnings("ignore")

    import resource

    import torch

    torch.set_num_threads(torch_threads)

    from relbench.datasets import get_dataset

    from experiments.continuous_learning.utils import (
        get_attribute_schema,
        get_text_embedder,
    )
    from redelex.data import make_pkey_fkey_graph

    mat_path = materialized_dir(cache_dir, name)
    mat_path.mkdir(parents=True, exist_ok=True)

    # Self-heal before deciding anything: a truncated frame left by a previous
    # wall-clock kill would otherwise be loaded and blow up the whole dataset.
    quarantined = []
    for stem in corrupt_tables(mat_path):
        bad = mat_path / f"{stem}.pt"
        print(
            f"  quarantining truncated frame {bad.name} "
            f"({fmt_bytes(bad.stat().st_size)})",
            flush=True,
        )
        bad.unlink()
        quarantined.append(stem)

    before = set(materialised_tables(mat_path))
    db_present = db_dir(relbench_cache, name).is_dir()
    download, action = download_decision(name, db_present)
    if download and not DOWNLOAD_POLICY[name].prepared:  # pragma: no cover
        raise ValueError(
            f"refusing to call download=True for {name}: {DOWNLOAD_POLICY[name].reason}"
        )
    print(
        f"  download={download} ({action}): {DOWNLOAD_POLICY[name].reason}",
        flush=True,
    )

    started = time.perf_counter()
    dataset = get_dataset(name, download=download)
    db = dataset.get_db(upto_test_timestamp=False)
    print(f"  db loaded: {len(db.table_dict)} tables, "
          f"span {db.min_timestamp} -> {db.max_timestamp}", flush=True)

    schema = get_attribute_schema(str(schema_path(cache_dir, name)), db)
    embedder = get_text_embedder(embedder_name, device=torch.device("cpu"))
    data, _ = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=schema,
        text_embedder=embedder,
        cache_dir=str(mat_path),
    )
    wall = time.perf_counter() - started
    after = set(materialised_tables(mat_path))

    result = {
        "dataset": name,
        "ok": True,
        "download_action": action,
        "materialised": f"{len(after)}/{len(db.table_dict)}",
        "tables_built": sorted(after - before),
        "tables_reused": sorted(before),
        "tables_quarantined": quarantined,
        "node_types": len(data.node_types),
        "edge_types": len(data.edge_types),
        "wall_s": wall,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "cache_bytes": dir_bytes(mat_path),
        "error": None,
    }
    del data, db

    receipt = dict(result)
    receipt["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    receipt["host"] = os.uname().nodename
    (mat_path / RECEIPT_NAME).write_text(json.dumps(receipt, indent=2))
    return result


# --------------------------------------------------------------------------
# Parent process
# --------------------------------------------------------------------------

RESULT_PREFIX = "__RESULT__ "


def worker_command(name: str, args: argparse.Namespace) -> list[str]:
    r"""Argument vector that builds exactly one dataset in a child process.

    Args:
        name: RelBench dataset name.
        args: Parsed CLI namespace, whose cache and embedder settings the child
            must inherit -- a child that guessed its own paths could write the
            tensor frames somewhere the parent never looks.

    Returns:
        list[str]: The command to run.
    """
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        name,
        "--cache-dir",
        str(args.cache_dir),
        "--relbench-cache",
        str(args.relbench_cache),
        "--embedder",
        args.embedder,
        "--torch-threads",
        str(args.torch_threads),
    ]


def failure_reason(returncode: int, tail: list[str]) -> str:
    r"""Describe why a worker produced no result.

    Args:
        returncode: The child's exit status; negative means it died on a signal.
        tail: The last lines it managed to print.

    Returns:
        str: A one-line explanation. A negative return code is spelled out
        because that is the OOM killer's signature (SIGKILL = -9) and it is the
        difference between "the code is wrong" and "the job needs more memory".
    """
    last = tail[-1].strip() if tail else ""
    if returncode < 0:
        signal_number = -returncode
        hint = " (out of memory? raise --mem)" if signal_number == 9 else ""
        return f"killed by signal {signal_number}{hint}: {last}" if last else (
            f"killed by signal {signal_number}{hint}"
        )
    return last or f"exit status {returncode}"


def run_worker(name: str, args: argparse.Namespace) -> tuple[dict, int]:
    r"""Run one dataset in a child process and collect its result.

    The child's output is streamed line by line rather than captured and dumped
    at the end. A dataset can take an hour; buffering would mean an hour of
    silence in the job log, and if the scheduler kills the job at the wall clock
    the buffer dies with it -- the exact situation where the log matters most.

    Args:
        name: RelBench dataset name.
        args: Parsed CLI namespace, forwarded to the child.

    Returns:
        tuple[dict, int]: The result record and the child's exit status. When
        the child died without reporting, the record is synthesised from the
        parent's own timing and from ``RUSAGE_CHILDREN``, whose ``ru_maxrss`` is
        the high-water mark over every child reaped so far -- a lower bound on
        the dead child's peak, and the only figure available for a job the OOM
        killer removed.
    """
    import collections
    import resource  # cheap, and only the parent path needs it here

    rss_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    started = time.perf_counter()
    proc = subprocess.Popen(
        worker_command(name, args),
        cwd=str(Path(__file__).resolve().parent.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # one interleaved stream, so the traceback
        text=True,                 # lands next to the log line that preceded it
        bufsize=1,
    )
    result = None
    tail: collections.deque[str] = collections.deque(maxlen=20)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        tail.append(line)
        if line.startswith(RESULT_PREFIX):
            # The machine-readable record; the parent renders it as a summary
            # row instead of repeating it verbatim.
            try:
                result = json.loads(line[len(RESULT_PREFIX) :])
            except json.JSONDecodeError:
                print(line, flush=True)
        else:
            print(line, flush=True)
    returncode = proc.wait()
    wall = time.perf_counter() - started
    rss_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

    if result is not None:
        return result, returncode

    return (
        {
            "dataset": name,
            "ok": False,
            "download_action": "?",
            "materialised": "failed",
            "wall_s": wall,
            "peak_rss_bytes": rss_after * 1024 if rss_after > rss_before else None,
            "rss_is_lower_bound": True,
            "cache_bytes": dir_bytes(materialized_dir(args.cache_dir, name)),
            "error": failure_reason(returncode, list(tail)),
        },
        returncode or 1,
    )


def print_check(states: list[DatasetState]) -> int:
    r"""Print the ``--check`` report.

    Args:
        states: Inspected datasets, in the order requested.

    Returns:
        int: Number of datasets that still need work.
    """
    print(render_table(CHECK_HEADERS, check_rows(states)))
    outstanding = [s for s in states if not s.complete]
    total_needed = sum(
        ESTIMATES[s.name].cache_bytes
        for s in outstanding
        if s.name in ESTIMATES and s.status != "unverifiable"
    )
    total_time = sum(
        ESTIMATES[s.name].seconds
        for s in outstanding
        if s.name in ESTIMATES and s.status != "unverifiable"
    )
    print()
    if not outstanding:
        print(f"all {len(states)} dataset(s) complete; nothing to do")
    else:
        print(
            f"WORK REMAINING: {len(outstanding)}/{len(states)} dataset(s): "
            + ", ".join(f"{s.name} [{s.status}]" for s in outstanding)
        )
        print(
            f"estimated cost: {fmt_duration(total_time)} of compute and "
            f"{fmt_bytes(total_needed)} of disk (potato rates; a slower node "
            f"takes proportionally longer)"
        )
    return len(outstanding)


def cached_result(state: DatasetState) -> dict:
    r"""Summary record for a dataset that was already complete before the job.

    The wall time and peak RSS come from the receipt left by whichever run
    actually built it, so the summary reports what the cache cost rather than a
    row of dashes -- and reports nothing at all when no receipt exists, instead
    of implying this job did the work.

    Args:
        state: The inspected dataset.

    Returns:
        dict: A record in the same shape a worker would have returned.
    """
    count = len(state.materialised)
    return {
        "dataset": state.name,
        "ok": True,
        "download_action": "cached",
        "materialised": f"{count}/{count} cached",
        "wall_s": state.receipt.get("wall_s"),
        "peak_rss_bytes": state.receipt.get("peak_rss_bytes"),
        "cache_bytes": state.cache_bytes,
        "error": None,
    }


def merge_results(states: list[DatasetState], results: list[dict]) -> list[dict]:
    r"""Put the worker results back into the requested dataset order.

    Args:
        states: Datasets in the order the user asked for them.
        results: Records for the datasets this job actually handled, in any
            order and possibly a subset.

    Returns:
        list[dict]: One record per entry of `states`, worker records where the
        job did work and cached records where it skipped.

    Raises:
        ValueError: If `results` names a dataset absent from `states`, which
            would mean a record was about to be silently dropped from the
            summary table.
    """
    by_name = {r["dataset"]: r for r in results}
    stray = sorted(set(by_name) - {s.name for s in states})
    if stray:
        raise ValueError(
            f"results contain datasets that were never requested: {stray}"
        )
    return [by_name.get(s.name, cached_result(s)) for s in states]


def build_argparser() -> argparse.ArgumentParser:
    r"""CLI for both the parent job and its per-dataset workers."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help=f"datasets to prepare (default: {' '.join(DEFAULT_DATASETS)})",
    )
    parser.add_argument("--cache-dir", default=".cache", type=Path)
    parser.add_argument(
        "--relbench-cache",
        default=None,
        type=Path,
        help="RelBench download cache (default: $RELBENCH_CACHE_DIR or the "
        "pooch OS cache, i.e. what relbench itself would use)",
    )
    parser.add_argument("--embedder", default="glove", choices=["glove", "potion"])
    parser.add_argument("--torch-threads", default=8, type=int)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what is present and what is missing, then exit; does no "
        "work and imports neither torch nor relbench",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check, then print the execution plan (order, download decision, "
        "time and disk) without doing any of it",
    )
    parser.add_argument(
        "--json",
        default=None,
        type=Path,
        help="also write the report or the summary to this path as JSON",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild datasets that are already complete",
    )
    parser.add_argument(
        "--time-budget",
        default=None,
        type=float,
        help="seconds this job may use; combined with SLURM_JOB_END_TIME, the "
        "earlier deadline wins",
    )
    parser.add_argument(
        "--time-safety",
        default=2.0,
        type=float,
        help="multiplier on the potato estimates when deciding whether a "
        "dataset can still finish (default: 2.0)",
    )
    parser.add_argument(
        "--time-reserve",
        default=600.0,
        type=float,
        help="seconds held back for the summary and a clean exit",
    )
    parser.add_argument(
        "--worker",
        default=None,
        help=argparse.SUPPRESS,  # internal: build exactly this one dataset
    )
    return parser


def main(argv=None) -> int:
    r"""Entry point for the parent job and, via ``--worker``, for the children.

    Args:
        argv: Argument list, or None to read ``sys.argv``.

    Returns:
        int: 0 when every requested dataset ended complete, 1 otherwise.
    """
    args = build_argparser().parse_args(argv)
    if args.relbench_cache is None:
        args.relbench_cache = default_relbench_cache()
    args.cache_dir = Path(args.cache_dir)
    args.relbench_cache = Path(args.relbench_cache)

    if args.worker:
        result = build_dataset(
            args.worker,
            args.cache_dir,
            args.relbench_cache,
            args.embedder,
            args.torch_threads,
        )
        print(RESULT_PREFIX + json.dumps(result), flush=True)
        return 0

    unknown = [d for d in args.datasets if d not in DOWNLOAD_POLICY]
    if unknown:
        raise ValueError(
            f"no download policy recorded for {unknown}; add one to "
            f"DOWNLOAD_POLICY rather than assuming the common case. "
            f"Known: {sorted(DOWNLOAD_POLICY)}"
        )

    print(f"materialised cache : {args.cache_dir.resolve()}")
    print(f"relbench cache     : {args.relbench_cache}")
    print(f"embedder           : {args.embedder}")
    print()
    states = [inspect(d, args.cache_dir, args.relbench_cache) for d in args.datasets]
    outstanding = print_check(states)

    if args.check:
        if args.json:
            args.json.write_text(json.dumps([vars(s) for s in states], indent=2))
        return 0 if outstanding == 0 else 1

    now = time.time()
    deadline = job_deadline(args.time_budget, now)
    if deadline is not None:
        print(f"deadline           : {time.strftime('%H:%M:%S', time.localtime(deadline))} "
              f"({fmt_duration(deadline - now)} from now)")

    todo = [s for s in states if args.force or not s.complete]
    if args.dry_run:
        print("\n=== plan ===")
        plan_rows = []
        for st in todo:
            _, action = download_decision(st.name, st.db_present)
            est = ESTIMATES.get(st.name)
            _, disk_reason = can_fit_on_disk(st.name, args.cache_dir)
            plan_rows.append(
                [
                    st.name,
                    st.status,
                    action,
                    fmt_duration(est.seconds) if est else "?",
                    "measured" if est and est.measured else "estimated",
                    fmt_bytes(est.cache_bytes) if est else "?",
                    fmt_bytes(est.peak_rss_bytes) if est else "?",
                ]
            )
        if plan_rows:
            print(
                render_table(
                    ["dataset", "status", "download", "wall", "source",
                     "cache", "peak RSS"],
                    plan_rows,
                )
            )
        else:
            print("nothing to do")
        return 0

    results: list[dict] = []
    for st in todo:
        print(f"\n=== {st.name} === (status: {st.status})", flush=True)
        ok, why = can_start(st.name, deadline, time.time(), args.time_safety,
                            args.time_reserve)
        if not ok:
            print(f"  SKIPPED: {why}; resubmit the job to continue", flush=True)
            results.append(
                {
                    "dataset": st.name,
                    "ok": False,
                    "download_action": "-",
                    "materialised": "skipped (time)",
                    "cache_bytes": st.cache_bytes,
                    "error": why,
                }
            )
            continue
        fits, disk_why = can_fit_on_disk(st.name, args.cache_dir)
        if not fits:
            print(f"  SKIPPED: {disk_why}", flush=True)
            results.append(
                {
                    "dataset": st.name,
                    "ok": False,
                    "download_action": "-",
                    "materialised": "skipped (disk)",
                    "cache_bytes": st.cache_bytes,
                    "error": disk_why,
                }
            )
            continue
        print(f"  budget: {why} | disk: {disk_why}", flush=True)
        result, _ = run_worker(st.name, args)
        results.append(result)

    print("\n=== summary ===")
    print(render_table(SUMMARY_HEADERS, summary_rows(merge_results(states, results))))
    failed = [r for r in results if not r.get("ok")]
    if failed:
        print()
        for res in failed:
            print(f"  {res['dataset']}: {res.get('error')}")

    final = [inspect(d, args.cache_dir, args.relbench_cache) for d in args.datasets]
    incomplete = [s for s in final if not s.complete]
    print()
    if incomplete:
        print(
            f"INCOMPLETE: {len(incomplete)} dataset(s) still need work: "
            + ", ".join(f"{s.name} [{s.status}]" for s in incomplete)
        )
        print("resubmit this job -- finished datasets are skipped on the next run")
    else:
        print(f"COMPLETE: all {len(final)} dataset(s) ready")

    if args.json:
        args.json.write_text(
            json.dumps({"results": results, "final": [vars(s) for s in final]}, indent=2)
        )
    return 1 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
