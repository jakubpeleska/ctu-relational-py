"""Compare a re-run against the published MLflow runs, episode by episode.

The port-fidelity gate: the experiment code that produced the paper lived on
`origin/rci`, and the version now on this branch is a port of it. This script
checks that the port reproduces the published per-episode numbers rather than
merely running.

Compares the distribution of `best_val_{metric}` across seeds for each episode.
Episode 1 should agree closely -- it is trained from scratch on the same window
with the same seeds -- while later episodes inherit weights from the best trial
of the previous episode and so drift, which is expected and not a failure.

Usage:
    .venv/bin/python scripts/compare_to_published.py \
        --dataset rel-f1 --task driver-position \
        --published pelesjak_cl_from_scratch --candidate pelesjak_cl_verify_port
"""

import argparse
import sys
from collections import defaultdict

import numpy as np

from experiments.continuous_learning.utils import (
    DEFAULT_MLFLOW_URI,
    get_experiment_runs_df,
    get_potato_client,
)


def parse_args(argv=None):
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
                   help="Max relative gap on episode 1 before the gate fails.")
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


def by_episode(df, metric):
    out = defaultdict(list)
    for _, row in df.iterrows():
        inc = row.get("increment")
        value = row.get(metric)
        if inc is None or value is None:
            continue
        try:
            out[int(inc)].append(float(value))
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

    pub, cand = by_episode(published, metric), by_episode(candidate, metric)

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
        print(f"PASS: episode 1 within {episode1_gap:.2%} (tolerance {args.tolerance:.2%}).")
        print("Later episodes may drift: they inherit weights from the previous "
              "episode's best trial, so small differences compound.")
        return 0
    print(f"FAIL: episode 1 differs by {episode1_gap:.2%}, above the {args.tolerance:.2%} tolerance.")
    print("Episode 1 trains from scratch on the same window with the same seeds, "
          "so a gap here means the port changed semantics.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
