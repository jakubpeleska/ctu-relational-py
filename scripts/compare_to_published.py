r"""Compare a re-run against the published MLflow runs, episode by episode.

The port-fidelity gate: the experiment code that produced the paper lived on
`origin/rci`, and the version now on this branch is a port of it. This script
asks whether the port selects the same quality of model per episode.

What it compares is the distribution of `best_val_{metric}` across seeds. That
statistic is a **maximum over the validations a run performed**, so it is
optimistically biased, and the size of the bias grows with the number of
validations and shrinks with the size of the validation set. Comparing it across
two different validation protocols therefore measures the protocols as much as
the code: the published runs validated up to 500 times on the full window, while
current runs validate 20 times on a 25k-row subsample. Under that mismatch a
faithful port can read as a failure and an unfaithful one as a pass, so this
script **refuses to compare** until the protocol params agree.

Even with the protocols matched, agreement here is evidence and not proof: it
says the two sides select equally good models on the same evaluation set, which
an unfaithful port could also do. Disagreement on episode 1 is the stronger
signal -- it trains from scratch on the same window with the same seeds. Later
episodes inherit weights from the previous episode's best trial and drift, which
is expected and not a failure.

Usage:
    .venv/bin/python scripts/compare_to_published.py \
        --dataset rel-f1 --task driver-position \
        --published pelesjak_cl_from_scratch --candidate pelesjak_cl_verify_port
"""

import argparse
import math
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from experiments.continuous_learning.utils import (
    DEFAULT_MLFLOW_URI,
    get_experiment_runs_df,
    get_potato_client,
)


__all__ = [
    "distinct_params",
    "protocol_comparison",
    "protocol_blockers",
    "format_protocol_comparison",
    "validation_budget",
    "format_selection_bias_note",
    "na_strategy_note",
    "by_episode",
    "detect_metric",
]


# Params that define what `best_val_*` means. `max_training_steps` and
# `val_check_interval` set how many validations a run maximises over;
# `val_max_rows` sets what it maximises over. All three must agree before two
# `best_val_*` distributions can be read as measuring the same thing.
PROTOCOL_PARAMS = ("max_training_steps", "val_check_interval", "val_max_rows")

MISSING = "<missing>"

# `na_strategy` is not logged anywhere, but the `chain_id` param and the
# `na_strategy=MEAN` pin landed in the same commit (b3bf3f7). Presence of
# `chain_id` on one side only therefore means the two sides straddle that
# commit. See `na_strategy_note`.
NA_STRATEGY_PROXY_PARAM = "chain_id"

# Statuses that stop the comparison. "unverified" is included on purpose:
# neither side logging a param is not evidence that the two agree on it.
BLOCKING_STATUSES = ("differs", "ambiguous", "unverified")


def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--published", required=True, help="MLflow experiment with the paper's runs.")
    p.add_argument("--candidate", required=True, help="MLflow experiment with the re-run.")
    p.add_argument("--metric", default=None,
                   help="Metric column, e.g. best_val_mae. Auto-detected if omitted.")
    p.add_argument("--mlflow-uri", default=DEFAULT_MLFLOW_URI)
    p.add_argument("--tolerance", type=float, default=0.02,
                   help="Max relative gap on episode 1 before the episode reads as inconsistent.")
    p.add_argument("--allow-protocol-mismatch", "--allow_protocol_mismatch",
                   dest="allow_protocol_mismatch", action="store_true",
                   help="Compare anyway when the validation protocols differ or "
                        "cannot be verified. The numbers are then not comparable; "
                        "the report says so on every line it prints.")
    return p.parse_args(argv)


def load(client, experiment, dataset, task):
    df = get_experiment_runs_df(
        client, experiment,
        filter_string=(
            f"params.dataset_name = '{dataset}' and params.task_name = '{task}' "
            f"and attributes.status = 'FINISHED'"
        ),
    )
    return df


def detect_metric(df):
    candidates = [c for c in df.columns if c.startswith("best_val_")]
    if not candidates:
        raise SystemExit("No best_val_* column found; pass --metric explicitly.")
    # prefer the tune metric names the experiment actually selects on
    for preferred in ("best_val_roc_auc", "best_val_mae", "best_val_macro_roc_auc"):
        if preferred in candidates:
            return preferred
    return candidates[0]


# --- protocol comparison ------------------------------------------------------


def _value_label(value) -> str:
    """One param value as displayed and compared.

    MLflow returns params as strings, but a frame built from runs that disagree
    on which params exist promotes the column to float, turning ``"2000"`` into
    ``2000.0``. Both must read alike or the gate fires on a difference that is
    an artifact of the frame rather than of the runs.
    """
    if value is None:
        return MISSING
    if isinstance(value, float):
        if pd.isna(value):
            return MISSING
        return str(int(value)) if float(value).is_integer() else str(value)
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none"):
        return MISSING
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else str(number)


def distinct_params(
    df: pd.DataFrame, params: Sequence[str] = PROTOCOL_PARAMS
) -> Dict[str, List[str]]:
    r"""The distinct values each param takes across a set of runs.

    Args:
        df: Runs frame from ``get_experiment_runs_df``.
        params: Param names to read.

    Returns:
        Mapping of param name to its sorted distinct values. A param no run
        logged, or an empty frame, yields ``["<missing>"]`` -- never an empty
        list, so callers never have to special-case absence.
    """
    out: Dict[str, List[str]] = {}
    for param in params:
        if df is None or len(df) == 0 or param not in df.columns:
            out[param] = [MISSING]
            continue
        values = {_value_label(value) for value in df[param]}
        out[param] = sorted(values) if values else [MISSING]
    return out


def protocol_comparison(
    published: pd.DataFrame,
    candidate: pd.DataFrame,
    params: Sequence[str] = PROTOCOL_PARAMS,
) -> List[Dict[str, object]]:
    r"""Per-param verdict on whether the two sides ran the same protocol.

    Args:
        published: Runs frame for the published experiment.
        candidate: Runs frame for the re-run.
        params: Param names to compare.

    Returns:
        One dict per param, in the order given, with keys ``param``,
        ``published``, ``candidate`` and ``status``. ``status`` is:

        - ``"match"``: one value on each side and the two are equal.
        - ``"differs"``: the sides disagree.
        - ``"ambiguous"``: one side is internally inconsistent, so there is no
          single protocol to compare against.
        - ``"unverified"``: neither side logged the param. Silence is not
          agreement, so this blocks like a mismatch does.
    """
    pub_values = distinct_params(published, params)
    cand_values = distinct_params(candidate, params)

    comparison: List[Dict[str, object]] = []
    for param in params:
        pub, cand = pub_values[param], cand_values[param]
        if len(pub) > 1 or len(cand) > 1:
            status = "ambiguous"
        elif pub != cand:
            status = "differs"
        elif pub == [MISSING]:
            status = "unverified"
        else:
            status = "match"
        comparison.append(
            {"param": param, "published": pub, "candidate": cand, "status": status}
        )
    return comparison


def protocol_blockers(comparison: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    r"""The entries of a protocol comparison that make the numbers incomparable.

    Args:
        comparison: Output of :func:`protocol_comparison`.

    Returns:
        The entries whose status is not ``"match"``, in input order.
    """
    return [entry for entry in comparison if entry["status"] in BLOCKING_STATUSES]


def format_protocol_comparison(comparison: Sequence[Dict[str, object]]) -> List[str]:
    r"""The protocol comparison as printable lines.

    Args:
        comparison: Output of :func:`protocol_comparison`.

    Returns:
        A header line followed by one line per param.
    """
    header = ["param", "published", "candidate", "status"]
    rows = [
        [
            str(entry["param"]),
            "|".join(entry["published"]),
            "|".join(entry["candidate"]),
            str(entry["status"]),
        ]
        for entry in comparison
    ]
    if not rows:
        return ["  ".join(header)]
    widths = [
        max(len(header[i]), *(len(row[i]) for row in rows)) for i in range(len(header))
    ]
    return [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths))
        for row in [header, *rows]
    ]


def validation_budget(values: Dict[str, List[str]]) -> Optional[int]:
    r"""How many validations a run maximised ``best_val_*`` over.

    This is the size of the best-of-N optimism: with N validations of a
    stochastic training curve, the maximum drifts upward in N even when nothing
    about the model differs.

    Args:
        values: Output of :func:`distinct_params` for one side.

    Returns:
        ``ceil(max_training_steps / val_check_interval)``, or ``None`` when
        either param is missing, ambiguous or unusable.
    """
    steps = values.get("max_training_steps", [MISSING])
    interval = values.get("val_check_interval", [MISSING])
    if len(steps) != 1 or len(interval) != 1:
        return None
    try:
        n_steps = float(steps[0])
        every = float(interval[0])
    except ValueError:
        return None
    if every <= 0 or n_steps <= 0:
        return None
    return int(math.ceil(n_steps / every))


def format_selection_bias_note(
    published: pd.DataFrame, candidate: pd.DataFrame
) -> Optional[str]:
    r"""A line quantifying the best-of-N asymmetry, when it can be computed.

    Args:
        published: Runs frame for the published experiment.
        candidate: Runs frame for the re-run.

    Returns:
        A message naming both validation budgets, or ``None`` when either is
        unknown or the two agree.
    """
    pub_n = validation_budget(distinct_params(published))
    cand_n = validation_budget(distinct_params(candidate))
    if pub_n is None or cand_n is None or pub_n == cand_n:
        return None
    return (
        f"best_val_* is a maximum over {pub_n} validation(s) on the published "
        f"side and {cand_n} on the candidate side. The larger budget is "
        "optimistically biased upward, by an amount this script cannot estimate."
    )


def _param_presence(df: pd.DataFrame, param: str) -> str:
    """``"all"``, ``"none"`` or ``"partial"`` -- how many runs logged ``param``."""
    if df is None or len(df) == 0 or param not in df.columns:
        return "none"
    labels = [_value_label(value) for value in df[param]]
    present = sum(1 for label in labels if label != MISSING)
    if present == 0:
        return "none"
    return "all" if present == len(labels) else "partial"


def na_strategy_note(
    published: pd.DataFrame, candidate: pd.DataFrame
) -> Optional[str]:
    r"""Whether the two sides can be shown to differ on the encoder's NA strategy.

    Commit b3bf3f7 pins ``na_strategy=MEAN`` for numerical columns, a deliberate
    departure from the RelBench default the published numbers were produced
    under, so runs straddling it are not comparable no matter how the validation
    protocol is set. The strategy itself is not a logged param; the ``chain_id``
    param landed in the same commit, so its presence on one side only is the
    detectable trace of the difference.

    Args:
        published: Runs frame for the published experiment.
        candidate: Runs frame for the re-run.

    Returns:
        A message describing the difference, or ``None`` when nothing detectable
        separates the two sides.
    """
    direct = distinct_params(published, ("na_strategy",))["na_strategy"]
    cand_direct = distinct_params(candidate, ("na_strategy",))["na_strategy"]
    if direct != [MISSING] or cand_direct != [MISSING]:
        if direct == cand_direct:
            return None
        return (
            f"na_strategy differs: published {'|'.join(direct)}, candidate "
            f"{'|'.join(cand_direct)}. The encoder's handling of missing "
            "numerical cells changes the model, so the two sides are not "
            "comparable."
        )

    pub_proxy = _param_presence(published, NA_STRATEGY_PROXY_PARAM)
    cand_proxy = _param_presence(candidate, NA_STRATEGY_PROXY_PARAM)
    if pub_proxy == cand_proxy and pub_proxy != "partial":
        return None
    words = {"all": "all", "none": "none", "partial": "some"}
    return (
        "na_strategy is not logged, but these runs straddle commit b3bf3f7: the "
        f"{NA_STRATEGY_PROXY_PARAM} param, added in that commit, is present in "
        f"{words[pub_proxy]} of the published runs and {words[cand_proxy]} of "
        "the candidate runs. b3bf3f7 pins na_strategy=MEAN for numerical "
        "columns, departing from the RelBench default the published numbers "
        "used, so those numbers are not comparable to current ones without "
        "re-running them."
    )


# --- per-episode comparison ---------------------------------------------------


def by_episode(df, metric):
    r"""Metric values grouped by the episode the run trained through.

    Args:
        df: Runs frame from ``get_experiment_runs_df``.
        metric: Metric column to read, e.g. ``"best_val_mae"``.

    Returns:
        Mapping of integer episode to the metric values of its runs. Rows whose
        episode or metric is missing or unparseable are dropped rather than
        entering a mean as ``nan``, which would poison the whole episode.
    """
    out = defaultdict(list)
    for _, row in df.iterrows():
        inc = row.get("increment")
        value = row.get(metric)
        if inc is None or value is None:
            continue
        try:
            if pd.isna(inc) or pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        try:
            out[int(float(inc))].append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def main(argv=None):
    args = parse_args(argv)
    client = get_potato_client(args.mlflow_uri)

    published = load(client, args.published, args.dataset, args.task)
    candidate = load(client, args.candidate, args.dataset, args.task)

    if published.empty:
        raise SystemExit(f"No published runs in {args.published!r} for {args.dataset}/{args.task}")
    if candidate.empty:
        raise SystemExit(f"No candidate runs in {args.candidate!r} for {args.dataset}/{args.task}")

    metric = args.metric or detect_metric(published)
    print(f"Metric: {metric}\n")

    # The protocol gate runs before anything is compared: `best_val_*` is a
    # best-of-N statistic, so two sides with different N or different validation
    # sets are not measuring the same quantity and no gap between them can be
    # attributed to the port.
    comparison = protocol_comparison(published, candidate)
    print("Validation protocol:")
    for line in format_protocol_comparison(comparison):
        print(f"  {line}")
    bias_note = format_selection_bias_note(published, candidate)
    if bias_note:
        print(f"\n  {bias_note}")

    na_note = na_strategy_note(published, candidate)
    if na_note:
        print(f"\n  CAVEAT: {na_note}")

    blockers = protocol_blockers(comparison)
    if blockers and not args.allow_protocol_mismatch:
        print()
        for entry in blockers:
            print(
                f"REFUSING TO COMPARE: {entry['param']} is {entry['status']} "
                f"(published {'|'.join(entry['published'])}, candidate "
                f"{'|'.join(entry['candidate'])})."
            )
        print(
            "\nbest_val_* is a maximum over the validations a run performed, on "
            "whatever rows it validated. Across two protocols that statistic "
            "differs even when the code is identical, so a faithful port can "
            "read as a failure and an unfaithful one as a pass. Re-run one side "
            "with the other's --max_training_steps/--val_check_interval/"
            "--val_max_rows, or pass --allow-protocol-mismatch to see the "
            "numbers anyway (they will not establish port fidelity)."
        )
        return 3
    if blockers:
        print(
            "\nWARNING: comparing across protocols that differ or cannot be "
            "verified. Every number below reflects the protocol as much as the "
            "code; it cannot establish port fidelity."
        )

    pub, cand = by_episode(published, metric), by_episode(candidate, metric)

    print()
    header = f"{'ep':>3}  {'published (n, mean+/-sd)':>28}  {'candidate (n, mean+/-sd)':>28}  {'rel gap':>8}"
    print(header)
    print("-" * len(header))

    episode1_gap = None
    for ep in sorted(set(pub) | set(cand)):
        p, c = pub.get(ep, []), cand.get(ep, [])
        p_txt = f"{len(p):>2}  {np.mean(p):.4f}+/-{np.std(p):.4f}" if p else " -"
        c_txt = f"{len(c):>2}  {np.mean(c):.4f}+/-{np.std(c):.4f}" if c else " -"
        if p and c and np.mean(p) != 0:
            gap = abs(np.mean(c) - np.mean(p)) / abs(np.mean(p))
            gap_txt = f"{gap:7.2%}"
            if ep == 1:
                episode1_gap = gap
        else:
            gap_txt = "      -"
        print(f"{ep:>3}  {p_txt:>28}  {c_txt:>28}  {gap_txt}")

    print()
    if episode1_gap is None:
        print("INCONCLUSIVE: episode 1 missing from one side; cannot judge port fidelity.")
        return 2
    if episode1_gap <= args.tolerance:
        print(f"CONSISTENT: episode 1 agrees within {episode1_gap:.2%} "
              f"(tolerance {args.tolerance:.2%}).")
        print("This says the two sides select models of the same quality under the "
              "same protocol. It is evidence for the port, not proof: a port that "
              "changed semantics but still selects as well would look identical here.")
        print("Later episodes may drift: they inherit weights from the previous "
              "episode's best trial, so small differences compound.")
        return 0
    print(f"INCONSISTENT: episode 1 differs by {episode1_gap:.2%}, above the "
          f"{args.tolerance:.2%} tolerance.")
    print("Episode 1 trains from scratch on the same window with the same seeds, so "
          "a gap this size is more than seed noise should produce -- but check the "
          "per-episode spread and the run counts above before concluding the port "
          "changed semantics; best_val_* is a best-of-N statistic and a thin sample "
          "of seeds is noisy on its own.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
