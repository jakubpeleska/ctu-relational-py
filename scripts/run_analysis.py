r"""Drive the whole continual-learning analysis for one (dataset, task).

Once every learning mode of a ``(dataset, task)`` pair has finished, the numbers
that go into the paper come out of a three-stage pipeline that was previously
driven by hand, once per mode:

1. ``experiments/continuous_learning/run_predictions.py`` scores every saved
   checkpoint of one chain over the whole task timeline into
   ``data/{experiment}/{dataset}_{task}_predictions.csv``.
2. ``scripts/build_evaluation_matrix.py`` reduces that wide CSV to the
   evaluation matrix ``R`` and the continual-learning metrics.
3. ``redelex.continual.drift`` describes how the target marginal moved over the
   same episode grid.

This module is the one command that runs all three for every mode and emits a
single **tidy long-format** table -- one row per
``(mode, seed, episode, metric)`` -- which is the shape that makes plotting and
significance testing downstream trivial. Nothing here re-implements a metric:
every number comes from :mod:`scripts.build_evaluation_matrix`,
:mod:`redelex.continual.metrics` or :mod:`redelex.continual.drift`.

What it refuses to do
---------------------
A mode-by-mode table is only worth printing if every row of it was measured the
same way, so a mode is **excluded, loudly**, rather than averaged in, when

* its predictions CSV is missing or holds no checkpoint columns;
* its increments are not the contiguous prefix ``1..k`` (episode ``j`` is
  aligned to increment ``j + 1``, so a hole slides rows against columns);
* its own runs disagree on ``max_training_steps``, ``val_check_interval``,
  ``val_max_rows`` or ``val_delta_days`` -- ``best_val_*`` is a maximum over
  ``max_training_steps / val_check_interval`` validations of a ``val_max_rows``
  subsample, so two runs with different values are not measuring the same thing.

And the whole table is refused outright, with nothing written, when the
*surviving* modes disagree with each other on those protocol params or on how
many episodes they cover: comparing ``er`` over 11 episodes against ``naive``
over 7 is not a comparison. ``--truncate-to-common`` is the explicit opt-in that
cuts every mode down to the shortest one instead.

Exit codes are meant to be read by a script: ``0`` every requested mode was
included, ``1`` a table was written but at least one mode was excluded, ``2``
nothing was written because the surviving modes are not comparable.

The ``seed`` column, and what it does *not* mean
------------------------------------------------
The protocol runs ``--num_samples`` trials per episode and every trial of
episode ``i + 1`` warm-starts from the single best checkpoint of episode ``i``.
There is therefore **no per-seed chain** to follow: the trials of two adjacent
episodes are not paired by anything. ``--replicates`` builds one ``R`` per
"trial slot" -- replicate ``r`` takes the ``r``-th run (by run id) of every
increment -- which is a resampling of the trial-level noise and is what the
mode-to-mode spread in the printed summary is computed over. It is not an
independent chain, and the docstring of :func:`replicate_column_sets` says so
again where someone is most likely to read it.

Rows built by collapsing an increment's trials instead carry
``seed = "agg:first"`` or ``"agg:mean"``. Note that ``"mean"`` averages the
*predictions*, i.e. it scores an **ensemble** of the trials, which is
systematically better than any single trial; the two are never pooled.

Usage:
    # everything, from existing prediction CSVs, no MLflow and no GPU
    .venv/bin/python scripts/run_analysis.py \
        --dataset rel-f1 --task driver-position --from-csv

    # run the missing prediction passes first (needs MLflow + checkpoints)
    .venv/bin/python scripts/run_analysis.py \
        --dataset rel-f1 --task driver-position \
        --experiment-prefix pelesjak_cl_v2 --grid-root logs/grid

    # freeze the task metadata on a host that has the relbench cache, so the
    # analysis can then run anywhere with --task-spec and --from-csv
    .venv/bin/python scripts/run_analysis.py \
        --dataset rel-f1 --task driver-position --write-task-spec spec.json

Never invoke via a plain ``uv run``: that re-syncs the default ``cpu``
dependency group and silently swaps torch for the CPU build.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from redelex.continual.drift import per_episode_target_stats, target_drift
from scripts.build_evaluation_matrix import (
    apply_checkpoint_aggregation,
    assign_episodes,
    build_matrix,
    checkpoint_column_candidates,
    checkpoint_columns_by_increment,
    episode_boundaries_from_splits,
    final_model_decay_avg,
    metric_for_task_type,
    roc_auc,
    summarise,
)

__all__ = [
    "REFERENCE_MODES",
    "DEFAULT_MODES",
    "DRIFT_MODE",
    "TIDY_COLUMNS",
    "SUMMARY_METRICS",
    "order_modes",
    "experiment_name",
    "predictions_csv_path",
    "chain_id_for",
    "prediction_command",
    "load_task_spec",
    "task_spec_from_relbench",
    "parse_splits",
    "score_metric_name",
    "increment_gap_message",
    "pooled_protocol",
    "protocol_verdict",
    "replicate_column_sets",
    "ModeResult",
    "analyse_mode",
    "drift_rows",
    "tidy_frame",
    "cross_mode_refusal",
    "scalar_summary",
    "format_summary_table",
    "format_exclusions",
]

REPO = Path(__file__).resolve().parent.parent
RUN_PREDICTIONS = REPO / "experiments" / "continuous_learning" / "run_predictions.py"

# The roster from scripts/run_grid.py. Kept as a literal rather than imported so
# that this module stays importable without the experiment package (which pulls
# in torch, ray and relbench at module scope); `run_grid.DEFAULT_MODES` and this
# list are asserted equal by tests/test_run_analysis.py.
DEFAULT_MODES = (
    "from_scratch",
    "joint",
    "naive",
    "er",
    "der_pp",
    "ewc",
    "lwf",
    "freeze_extend",
)

# Printed first, in this order. These three are the frame every CL method is read
# against: `from_scratch` retrains per episode (no transfer, no forgetting),
# `joint` trains on all data seen so far (the retention ceiling) and `naive`
# fine-tunes on the new increment only (the forgetting floor). A CL method that
# does not sit between `naive` and `joint` has not done anything.
REFERENCE_MODES = ("from_scratch", "joint", "naive")

# The `mode` value of rows that describe the *task*, not a learning mode: the
# per-episode target statistics and the drift curve are properties of the data
# and identical for every mode sharing the episode grid. Parenthesised so it can
# never collide with a real mode name, which is always an identifier.
DRIFT_MODE = "(task)"

# Prefix of a `seed` value produced by collapsing an increment's trials rather
# than by picking one of them; see the module docstring.
AGG_SEED_PREFIX = "agg:"

TIDY_COLUMNS = (
    "dataset",
    "task",
    "mode",
    "seed",
    "episode",
    "train_episode",
    "metric_name",
    "value",
    "score_metric",
)

# (metric name in the tidy table, column header in the printed summary).
SUMMARY_METRICS = (
    ("average_accuracy", "ACC"),
    ("backward_transfer", "BWT"),
    ("forward_transfer", "FWT"),
    ("final_model_decay_avg", "decay_final"),
    ("first_model_episode_decay_avg", "decay_first"),
)

# Params that must agree within a mode and across modes. Mirrors
# `run_predictions.PROTOCOL_PARAMS`; that module is imported lazily (see
# `protocol_verdict`) because importing it costs ~13 s of torch.
PROTOCOL_PARAMS = (
    "max_training_steps",
    "val_check_interval",
    "val_max_rows",
    "val_delta_days",
)


# --- naming and paths -------------------------------------------------------


def order_modes(modes: Iterable[str]) -> List[str]:
    r"""Reference modes first, in :data:`REFERENCE_MODES` order, then the rest.

    Args:
        modes: Mode names in any order.

    Returns:
        The same names, deduplicated, with ``from_scratch``, ``joint`` and
        ``naive`` leading so a reader meets the bounds before the methods that
        have to be read against them. Remaining modes keep alphabetical order.
    """
    seen = list(dict.fromkeys(modes))
    reference = [mode for mode in REFERENCE_MODES if mode in seen]
    return reference + sorted(mode for mode in seen if mode not in reference)


def experiment_name(prefix: str, mode: str) -> str:
    r"""MLflow experiment holding one mode's chains.

    ``run_grid.py`` launches every chain with
    ``--mlflow_experiment={prefix}_{mode}``, and ``run_predictions.py`` writes
    its CSV under ``data/{experiment}``, so this one join determines both.

    Args:
        prefix: ``--mlflow-experiment-prefix`` as given to ``run_grid.py``.
        mode: Learning mode.

    Returns:
        The experiment name.
    """
    return f"{prefix}_{mode}"


def predictions_csv_path(data_root, experiment: str, dataset: str, task: str) -> Path:
    r"""Where ``run_predictions.py`` puts one mode's predictions CSV.

    Args:
        data_root: Root the prediction passes write under, ``data`` by default.
        experiment: MLflow experiment name, from :func:`experiment_name`.
        dataset: RelBench dataset name.
        task: RelBench task name.

    Returns:
        ``{data_root}/{experiment}/{dataset}_{task}_predictions.csv``.
    """
    return Path(data_root) / experiment / f"{dataset}_{task}_predictions.csv"


def chain_id_for(dataset: str, task: str, mode: str, model_save_dir: str) -> str:
    r"""The ``chain_id`` param ``continuous_learning.py`` logs for one chain.

    Must be byte-identical to the string that run built, because it is what
    ``run_predictions.py --chain_id`` filters MLflow on. That string is
    ``f"{dataset}/{task}/{mode}/{model_save_dir}"`` using the **raw**
    ``--model_save_dir`` argument, before it is made absolute -- so a chain
    launched by ``run_grid.py`` with a relative ``logs/grid/...`` path has a
    relative path inside its chain id.

    Args:
        dataset: RelBench dataset name.
        task: RelBench task name.
        mode: Learning mode.
        model_save_dir: The ``--model_save_dir`` string the chain was launched
            with, verbatim.

    Returns:
        The chain id.
    """
    return f"{dataset}/{task}/{mode}/{model_save_dir}"


def prediction_command(
    dataset: str,
    task: str,
    experiment: str,
    out_dir,
    seed: int,
    chain_id: Optional[str] = None,
    mlflow_uri: Optional[str] = None,
    python: Optional[str] = None,
    strict_protocol: bool = True,
) -> List[str]:
    r"""The ``run_predictions.py`` invocation for one mode.

    Args:
        dataset: RelBench dataset name.
        task: RelBench task name.
        experiment: MLflow experiment to score.
        out_dir: Directory the CSV and its seed sidecar go in.
        seed: Neighbour-sampling seed; must be the same for every mode, or the
            cells of two modes' matrices are drawn from different subgraphs.
        chain_id: Chain to pin. Omitting it lets every chain logged for this
            dataset/task join the same CSV, which is how a two-increment smoke
            run becomes the first rows of ``R``.
        mlflow_uri: Tracking server, or ``None`` for the module default.
        python: Interpreter to use. Defaults to the repo venv, falling back to
            the running interpreter.
        strict_protocol: Pass ``--strict-protocol`` so the pass aborts, rather
            than warns, when the selected runs mix evaluation protocols. On by
            default here because this driver's whole contract is that it will
            not print a table it cannot stand behind.

    Returns:
        The argv list, ready for :func:`subprocess.call`.
    """
    interpreter = python or _default_python()
    command = [
        str(interpreter),
        "-u",
        str(RUN_PREDICTIONS),
        f"--dataset={dataset}",
        f"--task={task}",
        f"--mlflow_experiment={experiment}",
        f"--out_dir={out_dir}",
        f"--seed={int(seed)}",
    ]
    if mlflow_uri:
        command.append(f"--mlflow_uri={mlflow_uri}")
    if chain_id:
        command.append(f"--chain_id={chain_id}")
    if strict_protocol:
        command.append("--strict-protocol")
    return command


def _default_python() -> str:
    """The repo venv interpreter when it exists, else the running one."""
    venv = REPO / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


# --- task metadata ----------------------------------------------------------

_SPEC_KEYS = ("task_type", "target_col", "time_col", "splits", "table_columns")


def parse_splits(raw: Sequence) -> List:
    r"""Episode boundaries from a JSON spec, as timestamps or plain numbers.

    Args:
        raw: The ``splits`` entry of a task spec: ISO-8601 strings, or numbers
            for a task whose time axis is numeric.

    Returns:
        A list of ``pd.Timestamp`` (or floats), strictly increasing.

    Raises:
        ValueError: If fewer than three boundaries are given -- which leaves no
            evaluation episode once the first training window is dropped -- or
            if they are not strictly increasing.
    """
    values = list(raw)
    if len(values) < 3:
        raise ValueError(
            f"`splits` needs at least 3 boundaries to define one evaluation "
            f"episode, got {len(values)}"
        )
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        parsed: List = [float(value) for value in values]
        ordinals = np.asarray(parsed, dtype=float)
    else:
        stamps = pd.to_datetime(pd.Index(values))
        parsed = list(stamps)
        ordinals = stamps.asi8.astype(float)
    if np.any(np.diff(ordinals) <= 0):
        raise ValueError("`splits` must be strictly increasing")
    return parsed


def load_task_spec(path) -> Dict[str, Any]:
    r"""Read the frozen task metadata the analysis needs, without relbench.

    The episode grid is not recoverable from a predictions CSV: it depends on
    the dataset's ``val_timestamp``/``test_timestamp`` and on the width the run
    was swept with. Freezing it into a small JSON file is what lets the whole
    analysis run on a host with no relbench cache -- and what lets the tests run
    with no network at all.

    Args:
        path: JSON file written by :func:`task_spec_from_relbench`, holding
            ``task_type``, ``target_col``, ``time_col``, ``table_columns`` and
            the raw ``splits`` list from ``ContinuousWrapper.get_splits()``.

    Returns:
        The spec, with ``splits`` parsed by :func:`parse_splits`.

    Raises:
        ValueError: If a required key is missing. Guessing any of them would
            silently score the wrong column or cut the wrong episodes.
    """
    with open(path, "r") as handle:
        spec = json.load(handle)
    missing = [key for key in _SPEC_KEYS if key not in spec]
    if missing:
        raise ValueError(f"task spec {path} is missing key(s) {missing}")
    spec = dict(spec)
    spec["splits"] = parse_splits(spec["splits"])
    spec["table_columns"] = [str(column) for column in spec["table_columns"]]
    return spec


def task_spec_from_relbench(
    dataset: str, task: str, val_delta_days: Optional[float] = None
) -> Dict[str, Any]:
    r"""Build a task spec by asking relbench, for later offline use.

    Delegates to ``build_evaluation_matrix._load_task_context`` so there is one
    definition of "the run's own episode grid" rather than two.

    Args:
        dataset: RelBench dataset name.
        task: RelBench task name.
        val_delta_days: Episode width the chains were run with. Must match
            ``continuous_learning.py --val_delta_days``: a grid of the right
            *length* but the wrong *edges* passes every downstream check and
            scores each checkpoint against somebody else's episodes.

    Returns:
        A spec dict, ``splits`` already parsed.
    """
    # Imported here, not at module scope: it pulls in relbench and the experiment
    # package, and the whole point of `--task-spec` is to run without them.
    from scripts.build_evaluation_matrix import _load_task_context

    relbench_task, splits, table_columns = _load_task_context(
        dataset, task, val_delta_days=val_delta_days
    )
    return {
        "dataset": dataset,
        "task": task,
        "task_type": str(getattr(relbench_task.task_type, "value", relbench_task.task_type)),
        "target_col": relbench_task.target_col,
        "time_col": relbench_task.time_col,
        "table_columns": [str(column) for column in table_columns],
        "splits": list(splits),
        "val_delta_days": val_delta_days,
    }


def score_metric_name(task_type) -> str:
    r"""Short name of the higher-is-better scorer used for a task type.

    Args:
        task_type: A ``relbench`` ``TaskType`` or its string value.

    Returns:
        ``"roc_auc"`` or ``"neg_mae"``. Carried in every row of the tidy table
        so that a file concatenated across tasks cannot have AUCs averaged
        together with negative MAEs.
    """
    return "roc_auc" if metric_for_task_type(task_type) is roc_auc else "neg_mae"


# --- per-mode validation ----------------------------------------------------


def increment_gap_message(increments: Sequence[int]) -> Optional[str]:
    r"""Why a set of increments cannot be matched to episodes, or ``None``.

    Episode ``j`` of ``R`` is aligned to increment ``j + 1``, so only a *tail*
    truncation is recoverable. A head or interior gap is not: ``run_predictions``
    queries MLflow with no ``order_by`` and MLflow defaults to ``start_time
    DESC``, so an interrupted pass leaves the *newest* increments in the CSV.
    Matching those against episodes ``0..k`` would score increment 3 against
    episode 0 and print a drift curve that improves over time, from models that
    never saw the early episodes.

    Args:
        increments: Increment numbers present in the predictions CSV.

    Returns:
        An explanation naming the missing increments, or ``None`` when the
        increments are exactly the prefix ``1..k``.
    """
    present = sorted(set(int(value) for value in increments))
    if not present:
        return "no checkpoint columns in the predictions CSV"
    if present == list(range(1, len(present) + 1)):
        return None
    missing = sorted(set(range(1, max(present) + 1)) - set(present))
    return (
        f"increments {present} are not the prefix 1..{len(present)}: "
        f"increment(s) {missing} are missing, so episodes cannot be matched to "
        "checkpoints. Re-run predictions for the missing increments, or drop "
        "the later columns to leave a prefix."
    )


def pooled_protocol(
    summary: Optional[Dict[str, Any]], params: Sequence[str] = PROTOCOL_PARAMS
) -> Dict[str, List[str]]:
    r"""Distinct value of each protocol param, pooled over every increment.

    Pooled rather than compared within an increment because a chain must use one
    protocol from end to end: a smoke chain contributing increments 1-2 at
    ``max_training_steps=50`` conflicts with the real chain's 2000 even though no
    single increment mixes the two.

    Args:
        summary: The ``protocol`` block of a predictions sidecar, i.e. the output
            of ``run_predictions.protocol_summary``. ``None`` or empty yields an
            empty result.
        params: Param names to pool.

    Returns:
        Mapping of param name to its sorted distinct values.
    """
    pooled: Dict[str, set] = {param: set() for param in params}
    if summary:
        for entry in summary.values():
            for param in params:
                pooled[param].update(entry.get(param, []) or [])
    return {param: sorted(values) for param, values in pooled.items() if values}


def protocol_verdict(
    sidecar: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, List[str]], Optional[str], bool]:
    r"""Pooled protocol of one mode, and why it is unusable if it is.

    Args:
        sidecar: Parsed ``<name>.seeds.json`` written next to the predictions
            CSV, or ``None`` when there is none.

    Returns:
        ``(pooled, conflict_message, verified)``. ``conflict_message`` is
        ``None`` when the mode's runs agree. ``verified`` is ``False`` when no
        sidecar recorded a protocol at all -- the CSV may still be perfectly
        good, it simply cannot be *shown* to be, so the caller warns instead of
        excluding unless ``--require-protocol`` says otherwise.
    """
    summary = (sidecar or {}).get("protocol") or {}
    pooled = pooled_protocol(summary)
    if not pooled:
        return {}, None, False

    # Reuse run_predictions' own conflict wording so the message a reader sees
    # here is the message they would have seen from the prediction pass. Imported
    # lazily: that module imports torch, relbench and torch_geometric at module
    # scope, ~13 s, and the from-csv path must not pay for it.
    from experiments.continuous_learning.run_predictions import (
        format_conflict_message,
        protocol_conflicts,
    )

    conflicts = protocol_conflicts(summary, PROTOCOL_PARAMS)
    message = format_conflict_message(conflicts) if conflicts else None
    return pooled, message, True


def _recorded_seed(sidecar: Optional[Dict[str, Any]]) -> Optional[int]:
    r"""The neighbour-sampling seed a predictions CSV was scored under.

    Args:
        sidecar: Parsed ``<name>.seeds.json``, or ``None``.

    Returns:
        The seed, or ``None`` when there is no record or it is unreadable. An
        unreadable seed is reported as unknown rather than as a value, so it
        cannot silently match another mode's.
    """
    value = (sidecar or {}).get("seed")
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def replicate_column_sets(
    groups: Dict[int, List[str]], replicates: int
) -> List[Tuple[str, List[str]]]:
    r"""One checkpoint column per increment, repeated over the trial slots.

    **A replicate is not a chain.** Every trial of increment ``i + 1``
    warm-starts from the single best checkpoint of increment ``i``, so the
    ``r``-th trial of increment 3 has no relationship to the ``r``-th trial of
    increment 2 beyond both being the ``r``-th name in a sorted list of random
    MLflow run ids. What varying ``r`` does measure is the trial-level noise of
    the protocol, which is the right spread to put an error bar on -- and it is
    honest in a way that ``--aggregate mean`` is not, since that averages
    *predictions* and therefore scores an ensemble.

    Args:
        groups: Mapping of increment to its checkpoint columns, sorted, as
            ``build_evaluation_matrix.checkpoint_columns_by_increment`` returns.
        replicates: How many slots to emit. ``0`` emits none, a negative value
            emits as many as the thinnest increment supports. Capped at that
            same number either way: a slot that existed for some increments but
            not others would make ``R`` non-square.

    Returns:
        ``[(seed_label, columns_ordered_by_increment), ...]``, empty when
        ``replicates`` is 0 or no increment is present.
    """
    if not groups or replicates == 0:
        return []
    available = min(len(columns) for columns in groups.values())
    wanted = available if replicates < 0 else min(replicates, available)
    increments = sorted(groups)
    return [
        (f"trial{slot}", [groups[increment][slot] for increment in increments])
        for slot in range(wanted)
    ]


# --- the per-mode analysis --------------------------------------------------


@dataclass
class ModeResult:
    r"""Everything the driver learned about one learning mode.

    Attributes:
        mode: Learning mode name.
        included: Whether its rows may enter the table.
        reason: Why it was excluded; ``""`` when included.
        rows: Tidy rows, empty when excluded.
        n_episodes: Episodes its matrices cover, ``0`` when excluded.
        n_replicates: Trial slots that produced a matrix.
        protocol: Pooled protocol values, for the cross-mode check.
        protocol_verified: Whether a sidecar recorded a protocol at all.
        predict_seed: Neighbour-sampling seed the CSV's columns were scored
            with, from the sidecar, or ``None`` when unrecorded.
        predictions_csv: Path the numbers came from, for provenance.
    """

    mode: str
    included: bool
    reason: str = ""
    rows: List[Dict[str, Any]] = field(default_factory=list)
    n_episodes: int = 0
    n_replicates: int = 0
    protocol: Dict[str, List[str]] = field(default_factory=dict)
    protocol_verified: bool = False
    predict_seed: Optional[int] = None
    predictions_csv: Optional[str] = None


def _row(
    dataset: str,
    task: str,
    mode: str,
    seed: str,
    metric_name: str,
    value: float,
    score_metric: str,
    episode: Optional[int] = None,
    train_episode: Optional[int] = None,
) -> Dict[str, Any]:
    """One tidy row. Keys are exactly :data:`TIDY_COLUMNS`, in that order."""
    return {
        "dataset": dataset,
        "task": task,
        "mode": mode,
        "seed": seed,
        "episode": episode,
        "train_episode": train_episode,
        "metric_name": metric_name,
        "value": float(value),
        "score_metric": score_metric,
    }


def _summary_rows(
    summary: Dict[str, Any],
    matrix: np.ndarray,
    dataset: str,
    task: str,
    mode: str,
    seed: str,
    score_metric: str,
) -> List[Dict[str, Any]]:
    r"""Expand one :func:`~scripts.build_evaluation_matrix.summarise` result into tidy rows.

    Args:
        summary: The dict ``summarise`` returned.
        matrix: The same ``R`` it was computed from, needed for the cells and
            the final model's per-episode scores, which ``summarise`` does not
            return.
        dataset: Dataset name, copied into every row.
        task: Task name, copied into every row.
        mode: Learning mode, copied into every row.
        seed: Seed label -- a ``trial{r}`` slot or an ``agg:`` label.
        score_metric: ``"roc_auc"`` or ``"neg_mae"``.

    Returns:
        Scalar metrics (``episode`` null), per-episode series (``episode`` set),
        and every cell of ``R`` (``train_episode`` and ``episode`` both set).
    """
    rows: List[Dict[str, Any]] = []

    def add(metric_name, value, episode=None, train_episode=None):
        rows.append(
            _row(
                dataset, task, mode, seed, metric_name, value, score_metric,
                episode=episode, train_episode=train_episode,
            )
        )

    for key in (
        "n_episodes",
        "average_accuracy",
        "backward_transfer",
        "forward_transfer",
        "first_model_episode_decay_avg",
        "final_model_decay_avg",
    ):
        add(key, summary[key])

    for episode, value in enumerate(summary["drift_curve"]):
        add("drift_curve", value, episode=episode)
    for episode, value in enumerate(summary["forward_transfer_baseline"]):
        add("forward_transfer_baseline", value, episode=episode)
    # One entry shorter than the others: forgetting is undefined for the last
    # episode, which has no later model to be forgotten by.
    for episode, value in enumerate(summary["per_episode_forgetting"]):
        add("per_episode_forgetting", value, episode=episode)
    # The last row of R -- the deployed model per episode. `average_accuracy` is
    # its mean, and keeping the components is what lets a plot show *where* a
    # mode loses rather than only that it does.
    for episode, value in enumerate(np.asarray(matrix)[-1, :]):
        add("final_model_score", value, episode=episode)

    for i, row in enumerate(np.asarray(matrix)):
        for j, value in enumerate(row):
            add("R", value, episode=j, train_episode=i)
    return rows


def analyse_mode(
    mode: str,
    predictions_df: pd.DataFrame,
    spec: Dict[str, Any],
    dataset: str,
    task: str,
    sidecar: Optional[Dict[str, Any]] = None,
    aggregate: str = "first",
    decay: float = 0.0,
    replicates: int = -1,
    require_protocol: bool = False,
    max_episodes: Optional[int] = None,
    predictions_csv: Optional[str] = None,
) -> ModeResult:
    r"""Reduce one mode's predictions CSV to tidy rows, or refuse to.

    Args:
        mode: Learning mode the CSV belongs to.
        predictions_df: The wide table ``run_predictions.py`` wrote.
        spec: Task spec from :func:`load_task_spec` or
            :func:`task_spec_from_relbench`.
        dataset: Dataset name, copied into every row.
        task: Task name, copied into every row.
        sidecar: Parsed seed record next to the CSV, used for the protocol check.
        aggregate: How to collapse an increment's trials for the ``agg:`` rows;
            ``"first"`` picks one, ``"mean"`` averages the predictions and so
            scores an ensemble.
        decay: Forwarded to both decay averages; ``0`` is a plain mean.
        replicates: Trial slots to build separate matrices for; see
            :func:`replicate_column_sets`.
        require_protocol: Exclude the mode when no sidecar recorded a protocol,
            instead of warning.
        max_episodes: Keep only the first ``k`` increments and the first ``k``
            episodes, so that modes whose chains stopped at different points can
            be compared over a span they all cover. Every scalar is then computed
            over that span, which is why this rebuilds the matrix rather than
            slicing a finished one: ``average_accuracy`` and backward transfer
            are averages over the episodes their matrix held, so a scalar
            computed at full length still describes the full chain no matter how
            many rows are dropped afterwards.
        predictions_csv: Path the frame came from, recorded for provenance.

    Returns:
        A :class:`ModeResult`. Excluded modes carry an empty ``rows`` and a
        ``reason``: a mode that cannot be shown to be comparable is never merged
        into the table, because a silently wrong row is worse than a missing one.
    """
    target_col = spec["target_col"]
    time_col = spec["time_col"]
    task_type = spec["task_type"]
    score_metric = score_metric_name(task_type)

    for column in (target_col, time_col):
        if column not in predictions_df.columns:
            return ModeResult(
                mode, False,
                f"column {column!r} is not in the predictions CSV",
                predict_seed=_recorded_seed(sidecar), predictions_csv=predictions_csv,
            )

    protocol, conflict, verified = protocol_verdict(sidecar)
    # The neighbour-sampling seed of the prediction pass. Two modes scored under
    # different seeds saw different subgraphs, so their cells differ by sampling
    # noise before any learning-mode effect -- and backward transfer is a
    # difference of two such cells, where that noise does not cancel.
    predict_seed = _recorded_seed(sidecar)
    if conflict is not None:
        return ModeResult(
            mode, False, conflict, protocol=protocol, protocol_verified=True,
            predict_seed=predict_seed, predictions_csv=predictions_csv,
        )
    if require_protocol and not verified:
        return ModeResult(
            mode, False,
            "no protocol recorded next to the predictions CSV, and "
            "--require-protocol was given",
            protocol=protocol, predict_seed=predict_seed,
            predictions_csv=predictions_csv,
        )

    data_cols = set(map(str, spec["table_columns"])) | {str(target_col), str(time_col)}
    candidates = checkpoint_column_candidates(predictions_df.columns, data_cols)
    groups = checkpoint_columns_by_increment(candidates)

    gap = increment_gap_message(sorted(groups))
    if gap is not None:
        return ModeResult(
            mode, False, gap, protocol=protocol, protocol_verified=verified,
            predict_seed=predict_seed, predictions_csv=predictions_csv,
        )

    boundaries = episode_boundaries_from_splits(spec["splits"])
    if len(groups) > len(boundaries) - 1:
        return ModeResult(
            mode, False,
            f"{len(groups)} checkpoint increment(s) but the episode grid only "
            f"defines {len(boundaries) - 1}; the grid is not the one this chain "
            "ran on (check --val-delta-days)",
            protocol=protocol, protocol_verified=verified,
            predict_seed=predict_seed, predictions_csv=predictions_csv,
        )
    # A chain that stopped early leaves a *prefix* of the episodes -- the gap
    # check above has already established that -- so keeping the episodes its
    # checkpoints line up with is sound. The count is reported, and the
    # cross-mode check refuses to compare two different counts.
    keep_increments = sorted(groups)
    if max_episodes is not None:
        if max_episodes < 1:
            raise ValueError(f"`max_episodes` must be at least 1, got {max_episodes}")
        keep_increments = keep_increments[:max_episodes]
    groups = {increment: groups[increment] for increment in keep_increments}
    boundaries = boundaries[: len(groups) + 1]

    frame = _ensure_time_axis(predictions_df, time_col, spec["splits"])
    if max_episodes is not None:
        # Drop the discarded increments' columns as well as their episodes: the
        # aggregation helpers select one column per increment *present*, so a
        # leftover column would make R taller than the boundary list.
        kept_columns = {column for increment in keep_increments for column in groups[increment]}
        frame = frame[
            [c for c in frame.columns if str(c) in data_cols or str(c) in kept_columns]
        ]

    column_sets: List[Tuple[str, Optional[List[str]]]] = [
        (f"{AGG_SEED_PREFIX}{aggregate}", None)
    ]
    column_sets.extend(replicate_column_sets(groups, replicates))

    rows: List[Dict[str, Any]] = []
    n_episodes = 0
    n_replicates = 0
    for seed_label, columns in column_sets:
        if columns is None:
            source, how = frame, aggregate
        else:
            # One column per increment plus the data columns, so that reusing
            # `build_matrix(..., aggregate="first")` selects exactly this slot.
            keep = [c for c in frame.columns if str(c) in data_cols] + list(columns)
            source, how = frame[keep], "first"
        try:
            matrix, _ = build_matrix(
                source,
                boundaries,
                target_col=target_col,
                time_col=time_col,
                task_type=task_type,
                aggregate=how,
                data_cols=data_cols,
            )
            scored, scored_columns = apply_checkpoint_aggregation(
                source[checkpoint_column_candidates(source.columns, data_cols)],
                aggregate=how,
            )
            summary = summarise(
                matrix,
                decay=decay,
                # The deployed model over its own window and everything after it,
                # including the RelBench test split that R has no column for.
                final_model_decay=final_model_decay_avg(
                    source[time_col],
                    source[target_col],
                    scored[scored_columns[-1]],
                    metric_for_task_type(task_type),
                    start=boundaries[-2],
                    decay=decay,
                ),
            )
        except ValueError as error:
            return ModeResult(
                mode, False, f"{seed_label}: {error}", protocol=protocol,
                protocol_verified=verified, predict_seed=predict_seed,
                predictions_csv=predictions_csv,
            )
        rows.extend(
            _summary_rows(summary, matrix, dataset, task, mode, seed_label, score_metric)
        )
        n_episodes = int(matrix.shape[0])
        if columns is not None:
            n_replicates += 1

    return ModeResult(
        mode, True, "", rows=rows, n_episodes=n_episodes, n_replicates=n_replicates,
        protocol=protocol, protocol_verified=verified, predict_seed=predict_seed,
        predictions_csv=predictions_csv,
    )


def _coerce_time(values: pd.Series, splits: Sequence) -> pd.Series:
    """Put a CSV time column on the same axis as the splits.

    ``pd.read_csv`` hands back timestamps as strings, and comparing a string to a
    ``Timestamp`` boundary raises rather than silently misclassifying -- but only
    sometimes, so the conversion is done up front instead of relying on that.
    """
    if len(splits) and isinstance(splits[0], (int, float)):
        return pd.to_numeric(values)
    return pd.to_datetime(values)


def _ensure_time_axis(frame: pd.DataFrame, time_col: str, splits: Sequence) -> pd.DataFrame:
    r"""``frame`` with its time column comparable to ``splits``.

    Returns the frame **unchanged** when the dtype already matches, so a caller
    that converted once does not pay for a second full copy of a predictions
    table that can run to hundreds of megabytes.

    Args:
        frame: A predictions table.
        time_col: Its timestamp column.
        splits: Episode boundaries, whose type fixes the axis.

    Returns:
        The same frame, or a copy whose ``time_col`` has been converted.
    """
    numeric_axis = bool(len(splits)) and isinstance(splits[0], (int, float))
    column = frame[time_col]
    aligned = (
        pd.api.types.is_numeric_dtype(column)
        if numeric_axis
        else pd.api.types.is_datetime64_any_dtype(column)
    )
    if aligned:
        return frame
    converted = frame.copy()
    converted[time_col] = _coerce_time(column, splits)
    return converted


def drift_rows(
    predictions_df: pd.DataFrame,
    spec: Dict[str, Any],
    dataset: str,
    task: str,
) -> List[Dict[str, Any]]:
    r"""Per-episode target statistics and the drift curve, as tidy rows.

    These describe the *data*, not any model, so they are emitted once under
    ``mode = "(task)"`` rather than duplicated per mode. They are computed on the
    **full** episode grid, not any mode's truncated one, so the curve does not
    change shape because one chain stopped early.

    Note what is being measured: ``P(y)``, the target marginal. That is prior
    probability shift, which is neither necessary nor sufficient for the concept
    drift ``P(y | x)`` that actually breaks a fitted model -- see
    :func:`redelex.continual.drift.target_drift`.

    Args:
        predictions_df: A predictions CSV; only its target and time columns are
            read, so any mode's file gives the same answer.
        spec: Task spec.
        dataset: Dataset name, copied into every row.
        task: Task name, copied into every row.

    Returns:
        Rows for ``episode_n_rows``, one row per remaining statistic column
        (``target_mean``, ``target_positive_rate``, ...) and ``target_drift``.
    """
    target_col = spec["target_col"]
    time_col = spec["time_col"]
    task_type = spec["task_type"]
    score_metric = score_metric_name(task_type)

    boundaries = episode_boundaries_from_splits(spec["splits"])
    frame = _ensure_time_axis(predictions_df, time_col, spec["splits"])
    episodes = assign_episodes(frame[time_col], boundaries)
    stats = per_episode_target_stats(frame[target_col], episodes, task_type)

    rows: List[Dict[str, Any]] = []
    for episode, record in stats.iterrows():
        for column, value in record.items():
            name = "episode_n_rows" if column == "n" else f"target_{column}"
            rows.append(
                _row(
                    dataset, task, DRIFT_MODE, "", name, value, score_metric,
                    episode=int(episode),
                )
            )
    for episode, value in target_drift(stats, task_type).items():
        rows.append(
            _row(
                dataset, task, DRIFT_MODE, "", "target_drift", value, score_metric,
                episode=int(episode),
            )
        )
    return rows


def tidy_frame(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    r"""Collect tidy rows into the long-format results table.

    Args:
        rows: Dicts produced by :func:`analyse_mode` and :func:`drift_rows`.

    Returns:
        A frame with exactly :data:`TIDY_COLUMNS`, in that order. ``episode`` and
        ``train_episode`` are nullable integers so a scalar metric writes an
        empty cell rather than ``-1`` or ``0.0``, either of which would read as a
        real episode index.
    """
    frame = pd.DataFrame(list(rows), columns=list(TIDY_COLUMNS))
    for column in ("episode", "train_episode"):
        frame[column] = frame[column].astype("Int64")
    frame["value"] = frame["value"].astype(float)
    return frame


# --- cross-mode validation --------------------------------------------------


def cross_mode_refusal(results: Sequence[ModeResult]) -> Optional[str]:
    r"""Why the included modes must not be tabulated together, or ``None``.

    Three things make a mode-by-mode table meaningless even when every mode is
    individually sound: modes measured under different protocols, modes scored
    under different neighbour-sampling seeds, and modes covering different
    numbers of episodes. ``average_accuracy`` over 11 episodes and over 7 are not
    the same statistic, and putting them in adjacent rows invites exactly the
    comparison they do not support.

    Args:
        results: The included :class:`ModeResult` objects.

    Returns:
        A message naming the disagreement and the flag that overrides it, or
        ``None`` when the modes are comparable.
    """
    included = [result for result in results if result.included]
    if len(included) < 2:
        return None

    # `protocol_conflicts` pools values across the entries of a summary-shaped
    # mapping, which is exactly the cross-mode question with one entry per mode.
    from experiments.continuous_learning.run_predictions import protocol_conflicts

    by_mode = {result.mode: result.protocol for result in included if result.protocol}
    conflicts = protocol_conflicts(by_mode, PROTOCOL_PARAMS) if by_mode else {}
    if conflicts:
        parts = ", ".join(
            f"{param} in {{{', '.join(values)}}}" for param, values in sorted(conflicts.items())
        )
        owners = ", ".join(
            f"{mode}={{{'; '.join(f'{k}={v}' for k, v in sorted(protocol.items()))}}}"
            for mode, protocol in sorted(by_mode.items())
        )
        return (
            f"the included modes were run under different evaluation protocols: "
            f"{parts}. Per mode: {owners}. Their scores are not comparable, so no "
            "table is written. Re-run the odd mode out, or restrict --modes."
        )

    seeds = {
        result.mode: result.predict_seed
        for result in included
        if result.predict_seed is not None
    }
    if len(set(seeds.values())) > 1:
        listing = ", ".join(f"{mode}={seed}" for mode, seed in sorted(seeds.items()))
        return (
            f"the included modes were scored with different prediction seeds "
            f"({listing}). NeighborLoader draws its neighbour sample from the "
            "global torch RNG, so their cells sit on different subgraphs and "
            "differ before any learning mode does. Re-run the odd mode out with "
            "--predict-seed matching the rest."
        )

    counts = {result.mode: result.n_episodes for result in included}
    if len(set(counts.values())) > 1:
        listing = ", ".join(f"{mode}={n}" for mode, n in sorted(counts.items()))
        return (
            f"the included modes cover different numbers of episodes ({listing}). "
            "average_accuracy and backward transfer are averages over those "
            "episodes, so the rows would not be comparable. Finish the short "
            "chains, restrict --modes, or pass --truncate-to-common to cut every "
            "mode down to the shortest."
        )
    return None


# --- the printed summary ----------------------------------------------------


def scalar_summary(tidy: pd.DataFrame) -> pd.DataFrame:
    r"""Per-mode point estimate and spread for each scalar CL metric.

    The trial replicates are preferred whenever a mode has at least two, because
    the mean of per-trial *scores* is the quantity to report and to test. The
    ``agg:mean`` row is not a substitute: it averages the trials' *predictions*
    and therefore scores an ensemble, which beats the average trial by
    construction. The two are never pooled, and ``source`` records which was used.

    Args:
        tidy: The long results table.

    Returns:
        Frame with ``mode``, ``metric_name``, ``mean``, ``sd``, ``n`` and
        ``source`` (``"trials"`` or ``"aggregate"``). ``n`` counts the
        non-``nan`` observations that were averaged, ``sd`` is the sample
        standard deviation and is ``nan`` below two of them.
    """
    wanted = [name for name, _ in SUMMARY_METRICS] + ["n_episodes"]
    frame = tidy[
        (tidy["mode"] != DRIFT_MODE)
        & tidy["metric_name"].isin(wanted)
        & tidy["episode"].isna()
        & tidy["train_episode"].isna()
    ]
    records: List[Dict[str, Any]] = []
    for mode, block in frame.groupby("mode", sort=False):
        trials = block[block["seed"].str.startswith("trial")]
        # Two is the minimum for a spread to exist; one replicate is no better
        # evidenced than the aggregate and would print a bare "nan" error bar.
        use, source = (
            (trials, "trials")
            if trials["seed"].nunique() >= 2
            else (block[block["seed"].str.startswith(AGG_SEED_PREFIX)], "aggregate")
        )
        for metric_name, values in use.groupby("metric_name", sort=False)["value"]:
            # `nan` observations are dropped rather than skipped silently, so
            # `n` counts what was really averaged: forward transfer is undefined
            # below three episodes and ROC-AUC is undefined on a single-class
            # episode, and a mean over one usable replicate must not be reported
            # as a mean over five.
            usable = values.dropna()
            records.append(
                {
                    "mode": mode,
                    "metric_name": metric_name,
                    "mean": float(usable.mean()) if len(usable) else float("nan"),
                    "sd": float(usable.std(ddof=1)) if len(usable) > 1 else float("nan"),
                    "n": int(len(usable)),
                    "source": source,
                }
            )
    return pd.DataFrame(
        records, columns=["mode", "metric_name", "mean", "sd", "n", "source"]
    )


def _cell(mean: float, sd: float) -> str:
    """One summary cell: the mean, with a spread only when one was measured."""
    if np.isnan(mean):
        return "nan"
    if np.isnan(sd):
        return f"{mean:.4f}"
    return f"{mean:.4f}+-{sd:.4f}"


def format_summary_table(
    tidy: pd.DataFrame, dataset: str, task: str, score_metric: str
) -> List[str]:
    r"""The compact mode-by-metric table, as printable lines.

    Reference modes come first, separated by a rule, so a reader meets the bounds
    (``from_scratch``, ``joint``, ``naive``) before the methods that only mean
    something relative to them.

    Args:
        tidy: The long results table.
        dataset: Dataset name, for the header.
        task: Task name, for the header.
        score_metric: ``"roc_auc"`` or ``"neg_mae"``, for the header.

    Returns:
        Header lines followed by one line per mode, plus a legend.
    """
    summary = scalar_summary(tidy)
    if summary.empty:
        return [f"{dataset}/{task}: no mode produced a usable matrix."]

    lookup = {
        (record["mode"], record["metric_name"]): record
        for record in summary.to_dict("records")
    }
    modes = order_modes(summary["mode"].unique())

    header = ["mode", "n_ep", "n_obs"] + [label for _, label in SUMMARY_METRICS]
    rows = []
    for mode in modes:
        episodes = lookup.get((mode, "n_episodes"))
        rows.append(
            [
                mode,
                "-" if episodes is None else f"{int(round(episodes['mean']))}",
                "-" if episodes is None else f"{episodes['n']}",
                *(
                    _cell(lookup[(mode, name)]["mean"], lookup[(mode, name)]["sd"])
                    if (mode, name) in lookup
                    else "-"
                    for name, _ in SUMMARY_METRICS
                ),
            ]
        )

    widths = [max(len(header[i]), *(len(row[i]) for row in rows)) for i in range(len(header))]

    def line(cells):
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths)).rstrip()

    sources = set(summary["source"])
    out = [
        f"{dataset}/{task}   metric={score_metric} (higher is better)",
        "",
        line(header),
        line(["-" * width for width in widths]),
    ]
    n_reference = sum(1 for mode in modes if mode in REFERENCE_MODES)
    for position, row in enumerate(rows):
        # A rule between the reference frame and the CL methods, so the eye does
        # not read `er` as just another baseline.
        if position == n_reference and n_reference:
            out.append(line(["-" * width for width in widths]))
        out.append(line(row))
    out.append("")
    if sources == {"trials"}:
        legend = (
            "values: mean over trial replicates +- sample sd; n_obs is how many "
            "replicates were averaged"
        )
    elif sources == {"aggregate"}:
        legend = (
            "values: the single aggregate column per mode (n_obs=1); no trial "
            "replicates available, so no spread is measured"
        )
    else:
        legend = (
            "values: mean and sample sd over trial replicates for the modes that "
            "have at least two (n_obs > 1), the single aggregate column for the "
            "rest (n_obs = 1)"
        )
    out.append(legend)
    out.append(
        "ACC=average accuracy  BWT=backward transfer (negative = forgetting)  "
        "FWT=forward transfer"
    )
    return out


def format_exclusions(results: Sequence[ModeResult], requested: Sequence[str]) -> List[str]:
    r"""The loud block naming every mode that did not make it into the table.

    Args:
        results: All :class:`ModeResult` objects, included and excluded.
        requested: Modes the user asked for, so a mode with no result at all
            (its CSV never existed) is still named.

    Returns:
        Printable lines, empty when every requested mode was included.
    """
    by_mode = {result.mode: result for result in results}
    lines: List[str] = []
    for mode in order_modes(requested):
        result = by_mode.get(mode)
        if result is None:
            lines.append(f"  {mode}: no result (mode was never analysed)")
        elif not result.included:
            lines.append(f"  {mode}: {result.reason}")
    if not lines:
        return []
    return [
        "",
        f"EXCLUDED {len(lines)} of {len(set(requested))} requested mode(s) -- "
        "their rows are NOT in the table:",
        *lines,
    ]


# --- CLI --------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--modes", nargs="+", default=list(DEFAULT_MODES),
        help="Modes to include. A requested mode that cannot be analysed is "
             "excluded loudly and makes the exit code 1.",
    )
    parser.add_argument(
        "--from-csv", "--from_csv", dest="from_csv", action="store_true",
        help="Never run a prediction pass. Builds everything from the CSVs that "
             "already exist, so it needs no MLflow, no checkpoints and no GPU.",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Run run_predictions.py even when the CSV exists. It is resumable, "
             "so this only rescores checkpoints the CSV is missing.",
    )
    parser.add_argument("--data-root", "--data_root", dest="data_root", default="data")
    parser.add_argument(
        "--experiment-prefix", "--experiment_prefix", dest="experiment_prefix",
        default="pelesjak_cl_v2",
        help="Experiment names are '{prefix}_{mode}', matching run_grid.py.",
    )
    parser.add_argument("--mlflow-uri", "--mlflow_uri", dest="mlflow_uri", default=None)
    parser.add_argument(
        "--grid-root", "--grid_root", dest="grid_root", default="logs/grid",
        help="--out of the run_grid.py invocation that produced the chains. Used "
             "to rebuild each chain's chain_id.",
    )
    parser.add_argument(
        "--model-save-dir-template", "--model_save_dir_template",
        dest="model_save_dir_template",
        default="{grid_root}/{dataset}_{task}_{mode}/models",
        help="The --model_save_dir each chain was launched with, verbatim: it is "
             "half of the chain_id and must match byte for byte.",
    )
    parser.add_argument(
        "--no-chain-id", "--no_chain_id", dest="use_chain_id",
        action="store_false", default=True,
        help="Do not pin a chain when running predictions. Off by default "
             "because an unpinned pass lets every chain in the experiment -- "
             "smoke runs included -- become rows of R.",
    )
    parser.add_argument(
        "--predict-seed", "--predict_seed", dest="predict_seed", type=int, default=42,
        help="Neighbour-sampling seed for run_predictions.py. Must be the same "
             "for every mode.",
    )
    parser.add_argument(
        "--task-spec", "--task_spec", dest="task_spec", default=None,
        help="JSON holding task_type/target_col/time_col/table_columns/splits. "
             "Skips relbench entirely; write one with --write-task-spec.",
    )
    parser.add_argument(
        "--write-task-spec", "--write_task_spec", dest="write_task_spec", default=None,
        help="Ask relbench for the task metadata, write it here, and exit.",
    )
    parser.add_argument(
        "--val-delta-days", "--val_delta_days", dest="val_delta_days",
        type=float, default=None,
        help="Episode width the chains were run with. Must match the runs' own "
             "--val_delta_days; only used when the spec comes from relbench.",
    )
    parser.add_argument(
        "--aggregate", default="first", choices=("first", "mean"),
        help="How the 'agg:' rows collapse an increment's trials. 'mean' "
             "averages predictions and so scores an ensemble, not a trial.",
    )
    parser.add_argument("--decay", type=float, default=0.0)
    parser.add_argument(
        "--replicates", type=int, default=-1,
        help="Trial slots to build a separate R for. -1 uses as many as the "
             "thinnest increment supports, 0 disables them.",
    )
    parser.add_argument(
        "--truncate-to-common", "--truncate_to_common", dest="truncate_to_common",
        action="store_true",
        help="Rebuild every mode over the first k episodes, k being the shortest "
             "included chain, instead of refusing to tabulate unequal chains.",
    )
    parser.add_argument(
        "--require-protocol", "--require_protocol", dest="require_protocol",
        action="store_true",
        help="Exclude a mode whose predictions CSV has no recorded protocol, "
             "rather than warning about it.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output prefix. Defaults to analysis/results/{dataset}_{task}. "
             "Writes <prefix>_results.csv and <prefix>_meta.json.",
    )
    parser.add_argument(
        "--dry-run", "--dry_run", dest="dry_run", action="store_true",
        help="Print the prediction commands that would run, then stop.",
    )
    return parser.parse_args(argv)


def _load_sidecar(csv_path: Path) -> Optional[Dict[str, Any]]:
    """The seed record beside a predictions CSV, without importing run_predictions."""
    path = Path(csv_path).with_suffix(".seeds.json")
    if not path.exists():
        return None
    try:
        with open(path, "r") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError) as error:
        print(f"WARNING: could not read {path}: {error}", file=sys.stderr)
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_spec(args) -> Dict[str, Any]:
    """The task spec, from --task-spec when given and from relbench otherwise."""
    if args.task_spec:
        return load_task_spec(args.task_spec)
    spec = task_spec_from_relbench(args.dataset, args.task, args.val_delta_days)
    spec["splits"] = parse_splits(spec["splits"])
    return spec


def _analyse_all(
    modes: Sequence[str],
    args,
    spec: Dict[str, Any],
    prediction_failures: Dict[str, str],
    max_episodes: Optional[int] = None,
) -> Tuple[List[ModeResult], List[Dict[str, Any]]]:
    r"""Analyse every requested mode, and the task's drift curve once.

    Args:
        modes: Requested modes, already ordered.
        args: Parsed CLI namespace.
        spec: Task spec.
        prediction_failures: Modes whose prediction pass exited non-zero, mapped
            to the reason, so they are excluded rather than analysed from a
            half-written CSV.
        max_episodes: Forwarded to :func:`analyse_mode`.

    Returns:
        ``(results, drift_rows)``. The drift rows are computed on the **full**
        episode grid even under truncation: they describe the data, not a chain,
        and shortening the curve because one mode stopped early would make a
        property of the task look like a property of the run.
    """
    results: List[ModeResult] = []
    drift: List[Dict[str, Any]] = []
    for mode in modes:
        experiment = experiment_name(args.experiment_prefix, mode)
        csv_path = predictions_csv_path(args.data_root, experiment, args.dataset, args.task)
        if mode in prediction_failures:
            results.append(
                ModeResult(
                    mode, False, prediction_failures[mode], predictions_csv=str(csv_path)
                )
            )
            continue
        if not csv_path.exists():
            results.append(
                ModeResult(
                    mode, False,
                    f"no predictions CSV at {csv_path}"
                    + ("" if args.from_csv else "; the prediction pass produced none"),
                    predictions_csv=str(csv_path),
                )
            )
            continue
        try:
            predictions = _ensure_time_axis(
                pd.read_csv(csv_path), spec["time_col"], spec["splits"]
            )
            result = analyse_mode(
                mode,
                predictions,
                spec,
                dataset=args.dataset,
                task=args.task,
                sidecar=_load_sidecar(csv_path),
                aggregate=args.aggregate,
                decay=args.decay,
                replicates=args.replicates,
                require_protocol=args.require_protocol,
                max_episodes=max_episodes,
                predictions_csv=str(csv_path),
            )
        except Exception as exc:  # noqa: BLE001 -- see below
            # One unreadable CSV must not cost the other modes. A chain that was
            # killed mid-write (preemption, OOM, a full filesystem) leaves a
            # truncated file, and parsing it raises anywhere from pd.read_csv to
            # the metric code. Letting that propagate loses a whole batch of
            # modes -- 8 chains of GPU time -- to one bad file, and the analysis
            # is per-mode by construction, so the honest response is to exclude
            # this mode with its reason and keep going. The exclusion is
            # reported in the table exactly like a missing CSV, so a mode that
            # vanished this way can never be mistaken for one that was analysed.
            results.append(
                ModeResult(
                    mode, False,
                    f"could not analyse {csv_path}: {type(exc).__name__}: {exc}",
                    predictions_csv=str(csv_path),
                )
            )
            print(
                f"WARNING: excluding {mode}; {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        results.append(result)
        if result.included and not result.protocol_verified:
            print(
                f"WARNING: {mode} has no recorded protocol next to {csv_path}; its "
                "runs cannot be shown to share max_training_steps, "
                "val_check_interval, val_max_rows and val_delta_days.",
                file=sys.stderr,
            )
        if result.included and not drift:
            # Once per (dataset, task): the target statistics are a property of
            # the data, so any included mode's CSV gives the same answer.
            try:
                drift = drift_rows(predictions, spec, args.dataset, args.task)
            except Exception as exc:  # noqa: BLE001
                # The drift curve is a side panel, not the result. Losing it must
                # not cost the mode table it sits next to; the next included mode
                # gets a turn at computing it.
                print(
                    f"WARNING: drift curve failed on {mode}; "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
    return results, drift


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.write_task_spec:
        spec = task_spec_from_relbench(args.dataset, args.task, args.val_delta_days)
        target = Path(args.write_task_spec)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as handle:
            json.dump(spec, handle, indent=2, default=str)
        print(f"Wrote {target}")
        return 0

    modes = order_modes(args.modes)
    prefix = Path(args.out or f"analysis/results/{args.dataset}_{args.task}")

    # 1. Run or reuse the prediction CSV per mode.
    commands: List[Tuple[str, List[str]]] = []
    for mode in modes:
        experiment = experiment_name(args.experiment_prefix, mode)
        csv_path = predictions_csv_path(args.data_root, experiment, args.dataset, args.task)
        if args.from_csv or (csv_path.exists() and not args.refresh):
            continue
        model_save_dir = args.model_save_dir_template.format(
            grid_root=args.grid_root, dataset=args.dataset, task=args.task, mode=mode
        )
        commands.append(
            (
                mode,
                prediction_command(
                    args.dataset,
                    args.task,
                    experiment,
                    out_dir=csv_path.parent,
                    seed=args.predict_seed,
                    chain_id=(
                        chain_id_for(args.dataset, args.task, mode, model_save_dir)
                        if args.use_chain_id
                        else None
                    ),
                    mlflow_uri=args.mlflow_uri,
                ),
            )
        )

    if args.dry_run:
        if not commands:
            print("No prediction pass needed; every CSV is already present.")
        for mode, command in commands:
            print(f"[{mode}] {' '.join(command)}")
        return 0

    prediction_failures: Dict[str, str] = {}
    for mode, command in commands:
        print(f"[{mode}] {' '.join(command)}", flush=True)
        code = subprocess.call(command, cwd=REPO)
        if code != 0:
            prediction_failures[mode] = f"run_predictions.py exited {code}"

    # 2. Build R per mode.
    spec = _resolve_spec(args)
    score_metric = score_metric_name(spec["task_type"])

    results, drift = _analyse_all(modes, args, spec, prediction_failures)
    included = [result for result in results if result.included]

    if not included:
        for line in format_exclusions(results, modes):
            print(line, file=sys.stderr)
        print(
            f"\nREFUSING to write a table: no requested mode of "
            f"{args.dataset}/{args.task} produced a usable evaluation matrix.",
            file=sys.stderr,
        )
        return 2

    if args.truncate_to_common:
        common = min(result.n_episodes for result in included)
        if any(result.n_episodes != common for result in included):
            # Rebuilt, not sliced. ACC, BWT and both decay averages are averages
            # over the episodes the matrix held, so cutting rows off a finished
            # table would leave scalars that still describe the full chain while
            # sitting next to a shorter one. Re-reading the CSVs is the price of
            # every row in the table meaning the same thing.
            print(
                f"--truncate-to-common: rebuilding every mode over the first "
                f"{common} episode(s), the most any included mode covers in "
                "common.",
                file=sys.stderr,
            )
            results, drift = _analyse_all(
                modes, args, spec, prediction_failures, max_episodes=common
            )
            included = [result for result in results if result.included]
            if not included:
                for line in format_exclusions(results, modes):
                    print(line, file=sys.stderr)
                print(
                    "\nREFUSING to write a table: truncating to "
                    f"{common} episode(s) left no usable mode.",
                    file=sys.stderr,
                )
                return 2

    exclusions = format_exclusions(results, modes)
    refusal = cross_mode_refusal(included)
    if refusal:
        for line in exclusions:
            print(line, file=sys.stderr)
        print(f"\nREFUSING to write a table: {refusal}", file=sys.stderr)
        return 2

    # 3. One tidy long-format table.
    tidy = tidy_frame([row for result in included for row in result.rows] + drift)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    results_path = prefix.with_name(prefix.name + "_results.csv")
    meta_path = prefix.with_name(prefix.name + "_meta.json")
    tidy.to_csv(results_path, index=False)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv[1:] if argv is None else argv),
        "dataset": args.dataset,
        "task": args.task,
        "task_type": spec["task_type"],
        "target_col": spec["target_col"],
        "time_col": spec["time_col"],
        "score_metric": score_metric,
        "aggregate": args.aggregate,
        "decay": args.decay,
        "replicates_requested": args.replicates,
        "episode_grid": [str(edge) for edge in episode_boundaries_from_splits(spec["splits"])],
        "modes": {
            result.mode: {
                "included": result.included,
                "reason": result.reason,
                "n_episodes": result.n_episodes,
                "n_replicates": result.n_replicates,
                "protocol": result.protocol,
                "protocol_verified": result.protocol_verified,
                "predict_seed": result.predict_seed,
                "predictions_csv": result.predictions_csv,
            }
            for result in results
        },
    }
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=2, default=str)

    # 4. The compact human-readable summary.
    for line in format_summary_table(tidy, args.dataset, args.task, score_metric):
        print(line)
    print(f"\nWrote {results_path} ({len(tidy)} rows) and {meta_path}")

    if exclusions:
        for line in exclusions:
            print(line, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
