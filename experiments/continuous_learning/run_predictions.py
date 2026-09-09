r"""Score every saved checkpoint of a task over its whole timeline.

Produces the wide table the continual-learning metrics reduce over: one row per
target row of the task (train, val and test windows concatenated), one column
per finished MLflow run, named ``{increment}_{run_id}``. Grouping those rows by
episode turns the table into the evaluation matrix R[i, j] consumed by
:mod:`redelex.continual.metrics`.

Two properties this pass has to have, neither of which is free:

**Every cell of R must be measured the same way.** ``NeighborLoader`` draws its
neighbour sample from the global torch RNG, so iterating one loader twice hands
out two different samples -- verified, not assumed. Left alone, every cell of R
carries its own independent sampling perturbation, and backward transfer is a
*difference* of two such cells, so the noise does not cancel. The pass therefore
reseeds immediately before each checkpoint's forward pass, and records the seed
in a sidecar JSON next to the CSV.

**Only one chain may enter R.** Filtering on (dataset, task, FINISHED) alone
also selects any other chain logged to the same experiment -- a two-increment
smoke run silently becomes rows 0-1 of R. Pass ``--chain_id`` to pin one chain;
the protocol table printed at startup is what makes a mismatch visible when you
cannot.

The CSV is written incrementally and re-read on startup, so an interrupted pass
resumes rather than recomputing.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence

import json
import random
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

import pandas as pd

import numpy as np

import torch

from torch_geometric.loader import NeighborLoader

from relbench.datasets import get_dataset
from relbench.tasks import get_task

from redelex.data.graph import make_pkey_fkey_graph
from redelex.continual.adapters import AdapterStack

from experiments.continuous_learning.utils import (
    get_attribute_schema,
    get_experiment_runs_df,
    get_potato_client,
    get_text_embedder,
    get_table_input,
)
from experiments.continuous_learning.models import HeterogeneousSAGE
from experiments.continuous_learning.continuous_task import ContinuousWrapper


__all__ = [
    "build_run_filter",
    "order_runs_by_start_time",
    "increment_label",
    "protocol_summary",
    "protocol_conflicts",
    "format_protocol_report",
    "format_conflict_message",
    "seed_all",
    "sidecar_path",
    "new_sidecar",
    "load_sidecar",
    "write_sidecar",
    "seed_conflict_message",
    "prediction_columns",
    "unrecorded_columns",
    "generate_all_predictions_df",
]


DEFAULT_SEED = 42

# Hyperparameters that change the model or the sampling, and so require
# rebuilding one or both before a checkpoint can be scored.
# "n_adapters" and "adapter_rank" are part of the ARCHITECTURE, not of the
# training config: a freeze_extend checkpoint from episode k carries k-1 adapter
# modules, and rebuilding a bare backbone to receive it fails with "Unexpected
# key(s) in state_dict: adapters.adapters.0.down.weight". Every mode except
# freeze_extend logs n_adapters=0 and is unaffected.
ARCH_PARAMS = ("gnn_channels", "gnn_layers", "gnn_aggr", "num_neighbors",
               "batch_size", "n_adapters", "adapter_rank")

# Params that fix what a "score" means for a run. They must be identical across
# every run that contributes a row to R: `best_val_*` is a max over
# `max_training_steps / val_check_interval` validations of a `val_max_rows`
# subsample, so two runs with different values are not measuring the same thing
# even when they train identically.
PROTOCOL_PARAMS = (
    "max_training_steps",
    "val_check_interval",
    "val_max_rows",
    "val_delta_days",
)

# Displayed in place of a param a run never logged. It is deliberately a value
# like any other in the conflict check: a chain half of whose runs predate a
# param is a chain whose protocol cannot be shown to match.
MISSING = "<missing>"

# Column-name increment for a run whose `increment` param is absent. Such a
# column never parses as a checkpoint downstream, which is the intent -- it is
# visible in the CSV but cannot silently become a row of R.
UNKNOWN_INCREMENT = "unknown"


def _run_arch(run: pd.Series, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Architecture used by a run, from its logged params, falling back to defaults.

    Every trial logs its config (see `get_hyperparams_logging`), so a checkpoint
    is normally rebuilt with the exact architecture that produced it. Older runs
    that predate a parameter fall back to the CLI default.
    """
    arch = {}
    for key in ARCH_PARAMS:
        value = run.get(key, None)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            arch[key] = defaults[key]
        elif key == "gnn_aggr":
            arch[key] = str(value)
        else:
            arch[key] = int(float(value))
    return arch


# --- run selection ----------------------------------------------------------


def adapter_stack_for(arch: Dict[str, Any]) -> Optional[AdapterStack]:
    r"""The adapter stack a checkpoint with this architecture was saved with.

    Returns ``None`` for every mode but freeze_extend, which is what those
    checkpoints expect: a model with an ``adapters`` child it never had would
    report the adapter keys as *missing*, the mirror image of the bug this
    exists to prevent.

    The stack is grown with repeated :meth:`AdapterStack.add_adapter` rather
    than constructed at size, because that is what fixes the submodule names
    (``adapters.adapters.<i>.*``) the state_dict keys refer to. The weights are
    overwritten by ``load_state_dict`` immediately afterwards, so only shape and
    naming matter here -- and :meth:`AdapterStack._load_from_state_dict` will
    resize the stack anyway, so the count only has to be non-zero to be correct.
    """
    if int(arch.get("n_adapters", 0)) <= 0:
        return None
    stack = AdapterStack(
        channels=int(arch["gnn_channels"]), rank=int(arch.get("adapter_rank", 16))
    )
    for _ in range(int(arch["n_adapters"])):
        stack.add_adapter()
    return stack


def build_scoring_model(data, col_stats_dict, arch: Dict[str, Any], device):
    r"""The model to load one checkpoint into.

    Single construction site on purpose: scoring rebuilds only when the
    architecture changes, so a checkpoint quietly loaded into the previous
    checkpoint's model is the failure mode here, and it is silent.
    """
    return HeterogeneousSAGE(
        data=data,
        col_stats_dict=col_stats_dict,
        gnn_channels=arch["gnn_channels"],
        gnn_layers=arch["gnn_layers"],
        gnn_aggr=arch["gnn_aggr"],
        adapters=adapter_stack_for(arch),
    ).to(device)


def build_run_filter(
    dataset_name: str,
    task_name: str,
    chain_id: Optional[str] = None,
    status: Optional[str] = "FINISHED",
) -> str:
    r"""MLflow filter string selecting the runs whose checkpoints to score.

    ``chain_id`` is the only clause that separates one chain from another.
    Without it the filter also matches every other chain logged to the same
    experiment -- including a short smoke chain, whose increments 1 and 2 would
    be picked up as rows 0-1 of R and scored as if they were the real thing.

    Args:
        dataset_name: Value of the run's ``dataset_name`` param.
        task_name: Value of the run's ``task_name`` param.
        chain_id: Value of the run's ``chain_id`` param. ``None`` selects every
            chain, which is only safe in an experiment holding exactly one.
        status: Run status to require, or ``None`` for any status.

    Returns:
        A filter string for ``MlflowClient.search_runs``.

    Raises:
        ValueError: If a value is empty or contains a single quote, which would
            terminate the quoted literal and silently change the query.
    """
    values = {
        "dataset_name": dataset_name,
        "task_name": task_name,
        "chain_id": chain_id,
        "status": status,
    }
    for name, value in values.items():
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"`{name}` must be a non-empty string, got {value!r}")
        if "'" in value:
            raise ValueError(f"`{name}` must not contain a quote, got {value!r}")
    if dataset_name is None or task_name is None:
        raise ValueError("`dataset_name` and `task_name` are required")

    clauses = [
        f"params.dataset_name = '{dataset_name}'",
        f"params.task_name = '{task_name}'",
    ]
    if status is not None:
        clauses.append(f"attributes.status = '{status}'")
    if chain_id is not None:
        clauses.append(f"params.chain_id = '{chain_id}'")
    return " and ".join(clauses)


def order_runs_by_start_time(
    runs_df: pd.DataFrame, column: str = "_start_time"
) -> pd.DataFrame:
    r"""The selected runs, oldest first.

    ``search_runs`` defaults to ``start_time DESC``, and the shared helper this
    module calls takes no ``order_by``, so the ordering is restored here. It
    decides which columns survive an interrupted pass: newest-first leaves a
    *head* gap -- the earliest increments missing -- and the matrix builder
    truncates the tail, so R would be assembled from the wrong episodes without
    anything looking wrong.

    Args:
        runs_df: Frame from ``get_experiment_runs_df``.
        column: Start-time column; MLflow's ``RunInfo`` exposes ``_start_time``.

    Returns:
        A new frame ordered oldest first, with runs missing a start time last
        so a malformed row cannot displace well-formed ones. The input is
        returned unchanged when it is empty or carries no such column.
    """
    if runs_df.empty or column not in runs_df.columns:
        return runs_df
    ordered = runs_df.sort_values(column, kind="stable", na_position="last")
    return ordered.reset_index(drop=True)


def increment_label(value: Any) -> str:
    r"""Increment as it appears in a prediction column name.

    Normalising to a bare integer matters downstream: the matrix builder parses
    checkpoint columns as ``^(\d+)_(.+)$``, so an increment that arrives as
    ``3.0`` -- which is what a float-typed MLflow param column yields -- would
    produce ``"3.0_<run>"``, fail that regex, and be dropped from R silently.

    Args:
        value: Raw ``increment`` param of a run.

    Returns:
        The integer as a string, or ``"unknown"`` when the value is missing.
        Values that are neither are passed through as-is.
    """
    if value is None:
        return UNKNOWN_INCREMENT
    if isinstance(value, float) and pd.isna(value):
        return UNKNOWN_INCREMENT
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or UNKNOWN_INCREMENT


def _param_label(value: Any) -> str:
    """One protocol value as displayed and compared.

    MLflow returns params as strings, but a frame assembled from runs that
    disagree on which params exist promotes the column to float, turning
    ``"2000"`` into ``2000.0``. Both must read as the same value or the conflict
    check fires on a difference that is not there.
    """
    if value is None:
        return MISSING
    if isinstance(value, float):
        if pd.isna(value):
            return MISSING
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none"):
        return MISSING
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else str(number)


def _increment_sort_key(label: str):
    """Numeric increments first and in order, then anything else alphabetically."""
    try:
        return (0, int(label), "")
    except (TypeError, ValueError):
        return (1, 0, str(label))


def protocol_summary(
    runs_df: pd.DataFrame, params: Sequence[str] = PROTOCOL_PARAMS
) -> Dict[str, Dict[str, Any]]:
    r"""Run count and distinct protocol values, per increment.

    This is what makes a wrong selection visible. Two chains merged into one
    selection show up either as an increment with twice the expected run count,
    or as an increment reporting two values for a protocol param.

    Args:
        runs_df: Frame from ``get_experiment_runs_df``.
        params: Param names to report. Missing columns report ``"<missing>"``
            rather than being skipped, so a param no run logged is still visible.

    Returns:
        Mapping of increment label to ``{"n_runs": int, <param>: [values]}``,
        ordered by increment with non-numeric labels last. Empty for an empty
        frame.
    """
    if runs_df is None or len(runs_df) == 0:
        return {}

    params = tuple(params)
    if "increment" in runs_df.columns:
        labels = [increment_label(value) for value in runs_df["increment"]]
    else:
        labels = [UNKNOWN_INCREMENT] * len(runs_df)

    summary: Dict[str, Dict[str, Any]] = {}
    for position, label in enumerate(labels):
        entry = summary.setdefault(label, {"n_runs": 0, **{param: [] for param in params}})
        entry["n_runs"] += 1
        for param in params:
            # A param no run logged is reported as missing rather than omitted:
            # a blank in the table would read as "fine", and it is not.
            if param in runs_df.columns:
                value = _param_label(runs_df[param].iloc[position])
            else:
                value = MISSING
            if value not in entry[param]:
                entry[param].append(value)

    for entry in summary.values():
        for param in params:
            entry[param] = sorted(entry[param])
    return dict(sorted(summary.items(), key=lambda item: _increment_sort_key(item[0])))


def protocol_conflicts(
    summary: Mapping[str, Mapping[str, Any]], params: Sequence[str] = PROTOCOL_PARAMS
) -> Dict[str, List[str]]:
    r"""Protocol params that are not constant across every selected run.

    Values are pooled over increments, not compared within one: a chain must use
    one protocol from end to end, so a smoke chain contributing increments 1-2
    at ``max_training_steps=50`` conflicts with the real chain's 2000 even
    though no single increment mixes the two.

    Args:
        summary: Output of :func:`protocol_summary`.
        params: Param names to check.

    Returns:
        Mapping of param name to its sorted distinct values, holding only the
        params with more than one. Empty when the selection is coherent.
    """
    pooled: Dict[str, set] = {param: set() for param in params}
    for entry in summary.values():
        for param in params:
            pooled[param].update(entry.get(param, []))
    return {
        param: sorted(values) for param, values in pooled.items() if len(values) > 1
    }


def format_protocol_report(
    summary: Mapping[str, Mapping[str, Any]], params: Sequence[str] = PROTOCOL_PARAMS
) -> List[str]:
    r"""The per-increment protocol table, as printable lines.

    Args:
        summary: Output of :func:`protocol_summary`.
        params: Param names to tabulate, in column order.

    Returns:
        Header line, then one line per increment. A single explanatory line when
        nothing was selected.
    """
    if not summary:
        return ["(no runs selected)"]

    header = ["increment", "runs", *params]
    rows = [
        [
            str(label),
            str(entry.get("n_runs", 0)),
            *("|".join(entry.get(param, [MISSING])) for param in params),
        ]
        for label, entry in summary.items()
    ]
    widths = [
        max(len(header[i]), *(len(row[i]) for row in rows)) for i in range(len(header))
    ]
    return [
        "  ".join(cell.rjust(width) for cell, width in zip(row, widths))
        for row in [header, *rows]
    ]


def format_conflict_message(conflicts: Mapping[str, Sequence[str]]) -> str:
    r"""One sentence naming every protocol param the selection disagrees on.

    Args:
        conflicts: Output of :func:`protocol_conflicts`.

    Returns:
        A message ending in the remedy, or ``""`` when there is no conflict.
    """
    if not conflicts:
        return ""
    parts = ", ".join(
        f"{param} in {{{', '.join(values)}}}" for param, values in sorted(conflicts.items())
    )
    return (
        f"the selected runs mix evaluation protocols: {parts}. "
        "Rows of R would come from runs that do not measure the same thing; "
        "pass --chain_id to pin a single chain."
    )


# --- reproducibility --------------------------------------------------------


def seed_all(seed: int) -> int:
    r"""Seed every RNG this pass draws from.

    ``NeighborLoader`` samples neighbours from the global torch RNG, so this is
    what makes one checkpoint's evaluation batches identical to the next one's.
    ``random`` and numpy are seeded too: neither drives the sampler today, but a
    pass that grows a dependency on one of them would otherwise become
    irreproducible without anything changing here.

    Args:
        seed: Seed to install, in ``[0, 2**32)`` -- numpy's accepted range.

    Returns:
        The seed as an ``int``, for recording alongside the output.

    Raises:
        ValueError: If ``seed`` is not an integer or is out of range.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"`seed` must be an integer, got {seed!r}")
    seed = int(seed)
    if not 0 <= seed < 2**32:
        raise ValueError(f"`seed` must be in [0, 2**32), got {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def sidecar_path(csv_path) -> Path:
    r"""Path of the seed record belonging to a predictions CSV.

    The seed lives beside the CSV rather than inside it because the matrix
    builder treats every non-data column of that file as a candidate checkpoint;
    a column of seeds is one rename away from being scored as a model.

    Args:
        csv_path: Path of the predictions CSV.

    Returns:
        The sidecar path, ``<name>.seeds.json``.
    """
    return Path(csv_path).with_suffix(".seeds.json")


def new_sidecar(
    seed: int,
    dataset_name: str,
    task_name: str,
    mlflow_experiment: Optional[str] = None,
    chain_id: Optional[str] = None,
    protocol: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    r"""An empty seed record for a predictions CSV.

    Args:
        seed: Seed the pass will use.
        dataset_name: Dataset the CSV scores.
        task_name: Task the CSV scores.
        mlflow_experiment: Experiment the checkpoints came from.
        chain_id: Chain the runs were restricted to, if any.
        protocol: Output of :func:`protocol_summary`, kept so the CSV carries
            the protocol its columns were produced under.

    Returns:
        A JSON-serialisable dict with an empty ``columns`` map.
    """
    return {
        "seed": int(seed),
        "dataset_name": dataset_name,
        "task_name": task_name,
        "mlflow_experiment": mlflow_experiment,
        "chain_id": chain_id,
        "protocol": dict(protocol) if protocol else {},
        # Per column, not just per file: a resumed pass may legitimately carry
        # columns from an earlier seed, and mixing is only defensible if which
        # column got which seed is written down.
        "columns": {},
    }


def load_sidecar(path) -> Optional[Dict[str, Any]]:
    r"""Read a seed record, or ``None`` when there is none to read.

    Args:
        path: Sidecar path.

    Returns:
        The parsed record, or ``None`` if the file is absent or unreadable. A
        corrupt file is reported and treated as absent rather than raising: it
        must not block a pass that can still produce correct columns.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError) as error:
        print(f"WARNING: could not read seed record {path}: {error}")
        return None
    if not isinstance(payload, dict):
        print(f"WARNING: seed record {path} is not an object; ignoring it.")
        return None
    payload.setdefault("columns", {})
    return payload


def write_sidecar(path, payload: Mapping[str, Any]) -> None:
    r"""Write a seed record next to its CSV.

    Args:
        path: Sidecar path.
        payload: Record to serialise.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(dict(payload), handle, indent=2, default=str)


def seed_conflict_message(
    existing: Optional[Mapping[str, Any]], seed: int
) -> Optional[str]:
    r"""Why an existing predictions CSV must not be extended with this seed.

    Args:
        existing: Record returned by :func:`load_sidecar`, or ``None``.
        seed: Seed this pass would use.

    Returns:
        An explanatory message, or ``None`` when the seeds agree, when there is
        no record, or when the record predates seed tracking.
    """
    if not existing:
        return None
    previous = existing.get("seed")
    if previous is None:
        return None
    try:
        previous = int(previous)
    except (TypeError, ValueError):
        return None
    if previous == int(seed):
        return None
    return (
        f"the existing columns were scored with seed {previous} but this pass "
        f"would use seed {seed}. Columns scored under different neighbour "
        "samples are not comparable, and backward transfer is a difference of "
        "two such columns."
    )


def prediction_columns(csv_columns: Sequence, table_columns: Sequence) -> List[str]:
    r"""Columns of a predictions CSV that hold a checkpoint's scores.

    Args:
        csv_columns: Every column of the predictions CSV.
        table_columns: The task table's own columns, which this script copies
            into the CSV verbatim.

    Returns:
        The remaining column names, in CSV order, as strings.
    """
    known = {str(column) for column in table_columns}
    return [str(column) for column in csv_columns if str(column) not in known]


def unrecorded_columns(
    csv_columns: Sequence,
    table_columns: Sequence,
    existing: Optional[Mapping[str, Any]],
) -> List[str]:
    r"""Prediction columns whose seed is not written down anywhere.

    These are columns from a pass that ran before the seed was recorded, or
    under a sidecar that has since been lost. Their neighbour sample is unknown,
    so they cannot be compared with columns this pass writes.

    Args:
        csv_columns: Every column of the predictions CSV.
        table_columns: The task table's own columns.
        existing: Record returned by :func:`load_sidecar`, or ``None``.

    Returns:
        The unrecorded prediction column names.
    """
    recorded = set((existing or {}).get("columns", {}))
    return [
        column
        for column in prediction_columns(csv_columns, table_columns)
        if column not in recorded
    ]


# --- the pass ---------------------------------------------------------------


def generate_all_predictions_df(
    dataset_name: str,
    task_name: str,
    mlflow_experiment: str,
    cache_dir: str = ".cache",
    batch_size: int = 128,
    num_neighbors: int = 32,
    gnn_channels: int = 128,
    gnn_layers: int = 2,
    gnn_aggr: str = "sum",
    mlflow_uri: Optional[str] = None,
    out_dir: Optional[str] = None,
    seed: int = DEFAULT_SEED,
    chain_id: Optional[str] = None,
    strict_protocol: bool = False,
    allow_seed_change: bool = False,
):
    r"""Score every finished checkpoint of one chain over the whole task table.

    Args:
        dataset_name: RelBench dataset name.
        task_name: RelBench task name.
        mlflow_experiment: Experiment holding the runs to score.
        cache_dir: Root of the materialised-graph cache.
        batch_size: Fallback batch size for runs that predate the param.
        num_neighbors: Fallback fan-out for runs that predate the param.
        gnn_channels: Fallback width for runs that predate the param.
        gnn_layers: Fallback depth for runs that predate the param.
        gnn_aggr: Fallback aggregation for runs that predate the param.
        mlflow_uri: Tracking URI, or ``None`` for the group server.
        out_dir: Where the CSV and its seed record go.
        seed: Seed installed once up front and again before every checkpoint's
            forward pass, so all checkpoints see the identical neighbour sample.
        chain_id: Restrict to one chain's runs. Strongly recommended -- see
            :func:`build_run_filter`.
        strict_protocol: Abort instead of warning when the selected runs
            disagree on a protocol param.
        allow_seed_change: Permit extending a CSV whose existing columns were
            scored with a different seed.

    Returns:
        The wide predictions frame, or ``None`` when nothing matched.

    Raises:
        ValueError: If the seed is invalid, if ``strict_protocol`` is set and
            the selection mixes protocols, or if the CSV was scored with a
            different seed and ``allow_seed_change`` is not set.
    """
    # Seeded once here and again per checkpoint below. The pass-level seeding is
    # what makes the graph materialisation and model construction reproducible;
    # the per-checkpoint reseeding is what makes the cells of R comparable.
    seed = seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | seed: {seed}")

    defaults = {
        "gnn_channels": gnn_channels,
        "gnn_layers": gnn_layers,
        "gnn_aggr": gnn_aggr,
        "num_neighbors": num_neighbors,
        "batch_size": batch_size,
        # A run that never logged these is a run from a mode that has no
        # adapters, so an absent value means "bare backbone", not "unknown".
        "n_adapters": 0,
        "adapter_rank": 16,
    }

    mlflow_client = get_potato_client(mlflow_uri)
    runs_df = get_experiment_runs_df(
        mlflow_client,
        mlflow_experiment,
        filter_string=build_run_filter(dataset_name, task_name, chain_id=chain_id),
    )

    if runs_df.empty:
        print("No finished runs found matching the criteria.")
        return None

    # Oldest first: an interrupted pass must leave the tail of R missing, which
    # the matrix builder truncates correctly, rather than the head.
    runs_df = order_runs_by_start_time(runs_df)

    summary = protocol_summary(runs_df)
    scope = f"chain_id={chain_id!r}" if chain_id else "ALL chains (no --chain_id given)"
    print(f"Found {len(runs_df)} finished runs in {mlflow_experiment!r} for {scope}.")
    for line in format_protocol_report(summary):
        print(line)

    conflicts = protocol_conflicts(summary)
    if conflicts:
        message = format_conflict_message(conflicts)
        if strict_protocol:
            raise ValueError(message)
        print(f"WARNING: {message}")

    # Setup Data and Task
    cache_path = Path(cache_dir).absolute() / dataset_name
    dataset = get_dataset(dataset_name, download=False)
    db = dataset.get_db(upto_test_timestamp=False)

    task = get_task(dataset_name, task_name)
    wrapped_task = ContinuousWrapper(task)

    text_embedder = get_text_embedder("glove", device=torch.device("cpu"))
    attribute_schema = get_attribute_schema(f"{cache_path}/attribute-schema.json", db)

    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=attribute_schema,
        text_embedder=text_embedder,
        cache_dir=f"{cache_path}/materialized",
    )

    # A table covering the entire dataset duration, so every checkpoint is scored
    # on every episode -- including the ones it was never trained on.
    full_input = get_table_input(wrapped_task.full_table, task)

    data_dir = Path(out_dir or f"data/{mlflow_experiment}").absolute()
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"{dataset_name}_{task_name}_predictions.csv"
    seeds_path = sidecar_path(csv_path)

    existing = load_sidecar(seeds_path)
    conflict = seed_conflict_message(existing, seed)
    if conflict:
        if not allow_seed_change:
            raise ValueError(
                f"{conflict} Re-run with --seed {existing['seed']}, write to a "
                "fresh --out_dir, or pass --allow_seed_change to mix seeds "
                "deliberately (each column's seed is recorded either way)."
            )
        print(f"WARNING: {conflict}")

    # Resume rather than restart: the base table is only written once, so
    # prediction columns from earlier passes survive.
    if csv_path.exists():
        results_df = pd.read_csv(csv_path)
        print(f"Resuming from {csv_path} with {len(results_df.columns)} columns.")
        stale = unrecorded_columns(
            results_df.columns, wrapped_task.full_table.df.columns, existing
        )
        if stale:
            print(
                f"WARNING: {len(stale)} existing prediction column(s) have no "
                f"recorded seed (e.g. {stale[0]}); they were written before the "
                "seed was tracked and their neighbour sample is unknown."
            )
    else:
        results_df = wrapped_task.full_table.df.copy()
        results_df.to_csv(csv_path, index=False)

    payload = existing or new_sidecar(
        seed, dataset_name, task_name, mlflow_experiment, chain_id, summary
    )
    payload["seed"] = seed
    payload["chain_id"] = chain_id
    payload["protocol"] = summary
    payload.setdefault("columns", {})
    # Written before the loop so an interrupted pass still records its seed.
    write_sidecar(seeds_path, payload)

    # Built lazily and reused while consecutive runs share an architecture.
    current_arch: Optional[Dict[str, Any]] = None
    model = None
    full_loader = None

    for _, run in tqdm(runs_df.iterrows(), total=len(runs_df), desc="Evaluating Runs"):
        run_id = run["_run_id"]
        increment = increment_label(run.get("increment", None))

        weights_path = None
        if "model_save_dir" in run and pd.notna(run["model_save_dir"]):
            weights_path = Path(run["model_save_dir"]) / "best_model.pt"

        if weights_path is None or not weights_path.exists():
            print(f"\nSkipping run_id {run_id} - Model weights not found.")
            continue

        col_name = f"{increment}_{run_id}"
        if col_name in results_df.columns:
            continue

        arch = _run_arch(run, defaults)
        if arch != current_arch:
            model = build_scoring_model(data, col_stats_dict, arch, device)
            full_loader = NeighborLoader(
                data,
                num_neighbors=[
                    int(arch["num_neighbors"] / 2**i) for i in range(arch["gnn_layers"])
                ],
                time_attr="time",
                input_nodes=full_input.nodes,
                input_time=full_input.time,
                transform=full_input.transform,
                batch_size=arch["batch_size"],
                temporal_strategy="uniform",
                shuffle=False,
            )
            current_arch = arch

        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.to(device)
        model.eval()

        # Reseed immediately before the pass, not once at the top of the run.
        # The loader is shared by every checkpoint with this architecture and
        # draws its neighbour sample from the global RNG, so without this each
        # checkpoint would be scored on a different subgraph and every cell of R
        # would carry its own sampling noise -- noise that does not cancel in
        # backward transfer, which is a difference of two cells.
        seed_all(seed)

        all_preds = []
        with torch.no_grad():
            for batch in full_loader:
                batch = batch.to(device)
                preds = model(batch, task.entity_table)
                all_preds.append(preds.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)

        # Flatten array if the task returns shape (N, 1)
        if all_preds.ndim > 1 and all_preds.shape[1] == 1:
            all_preds = all_preds.flatten()

        if len(all_preds) != len(results_df):
            print(
                f"\nSkipping run_id {run_id} - produced {len(all_preds)} predictions "
                f"for {len(results_df)} rows."
            )
            continue

        results_df[col_name] = all_preds
        results_df.to_csv(csv_path, index=False)
        payload["columns"][col_name] = seed
        write_sidecar(seeds_path, payload)

    print(f"Wrote {csv_path}")
    print(f"Wrote {seeds_path}")
    return results_df


def parse_args(argv: Optional[Sequence[str]] = None):
    """CLI for the prediction pass."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--mlflow_experiment", type=str, default=None)
    parser.add_argument("--mlflow_uri", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed installed before every checkpoint's forward pass, so all "
        "checkpoints see the identical neighbour sample. Recorded in the "
        "<name>.seeds.json written next to the CSV.",
    )
    parser.add_argument(
        "--chain-id",
        "--chain_id",
        dest="chain_id",
        type=str,
        default=None,
        help="Score only this chain's runs. Without it, every chain logged to "
        "the experiment for this dataset/task is scored, so a smoke chain "
        "can become the first rows of R.",
    )
    parser.add_argument(
        "--strict-protocol",
        "--strict_protocol",
        dest="strict_protocol",
        action="store_true",
        help="Abort instead of warning when the selected runs disagree on "
        "max_training_steps, val_check_interval, val_max_rows or val_delta_days.",
    )
    parser.add_argument(
        "--allow-seed-change",
        "--allow_seed_change",
        dest="allow_seed_change",
        action="store_true",
        help="Extend a predictions CSV whose existing columns were scored with "
        "a different seed. Off by default: the mixed columns are not comparable.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(args)

    generate_all_predictions_df(
        dataset_name=args.dataset,
        task_name=args.task,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_uri=args.mlflow_uri,
        out_dir=args.out_dir,
        seed=args.seed,
        chain_id=args.chain_id,
        strict_protocol=args.strict_protocol,
        allow_seed_change=args.allow_seed_change,
    )
