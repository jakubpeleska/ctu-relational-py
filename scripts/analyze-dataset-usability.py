"""Assess which RelBench datasets support the multi-episode incremental protocol.

For every candidate dataset, downloads the data, then for each non-recommendation,
non-autocomplete EntityTask reports the ACTUAL number of training episodes that
survive ContinuousWrapper.get_splits()'s 10%-of-val-window filter, along with
per-episode target-row counts.

Results stream to JSON/CSV as each task completes, so a partial run is still useful.

Usage:
    uv run --no-group cpu --group cu128 python scripts/analyze-dataset-usability.py
    uv run ... python scripts/analyze-dataset-usability.py --datasets rel-f1 rel-trial
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import pandas as pd

from relbench.base import EntityTask, RecommendationTask
from relbench.datasets import get_dataset
from relbench.tasks import get_task, get_task_names

from experiments.continuous_learning.continuous_task import ContinuousWrapper

try:
    from relbench.base.task_autocomplete import AutoCompleteTask
except ImportError:  # pragma: no cover - defensive, API may move
    AutoCompleteTask = ()

# Cheapest first, so useful rows land early; rel-amazon/rel-stack are the big downloads.
DEFAULT_DATASETS = [
    "rel-f1",
    "rel-avito",
    "rel-event",
    "rel-trial",
    "rel-hm",
    "rel-arxiv",
    "rel-ratebeer",
    "rel-stack",
    "rel-amazon",
]

# Ruled out structurally - see plan. dbinfer-* are static, tgb* use a different task
# base class, rel-mimic needs PhysioNet credentials + BigQuery, rel-salt is all autocomplete.
SKIP_REASON = {
    "rel-mimic": "PhysioNet credentialing + BigQuery client required",
    "rel-salt": "~5 episodes and 100% autocomplete tasks",
}


def analyse_task(dataset_name: str, task_name: str) -> dict:
    task = get_task(dataset_name, task_name, download=True)

    if isinstance(task, RecommendationTask):
        return {"status": "skipped", "reason": "recommendation/link-prediction"}
    if AutoCompleteTask and isinstance(task, AutoCompleteTask):
        return {"status": "skipped", "reason": "autocomplete"}
    if not isinstance(task, EntityTask):
        return {"status": "skipped", "reason": f"not an EntityTask ({type(task).__name__})"}

    wrapper = ContinuousWrapper(task)
    splits = wrapper.get_splits()

    # run_ray_tuner loops `for i in range(start_inc=1, len(splits) - 1)`
    n_episodes = max(len(splits) - 2, 0)

    episode_rows = []
    for i in range(1, len(splits) - 1):
        episode_rows.append(len(wrapper.get_table(start=splits[i], end=splits[i + 1])))

    delta_w = task.timedelta
    delta_i = task.dataset.test_timestamp - task.dataset.val_timestamp
    rows = pd.Series(episode_rows, dtype="int64")

    return {
        "status": "ok",
        "task_type": str(task.task_type.value),
        "entity_table": getattr(task, "entity_table", None),
        "delta_w_days": delta_w / pd.Timedelta(days=1),
        "delta_i_days": delta_i / pd.Timedelta(days=1),
        "delta_i_over_w": (delta_i / delta_w) if delta_w else None,
        "n_episodes": n_episodes,
        "n_splits": len(splits),
        "first_split": str(splits[0]),
        "val_timestamp": str(task.dataset.val_timestamp),
        "test_timestamp": str(task.dataset.test_timestamp),
        "rows_total": int(rows.sum()) if len(rows) else 0,
        "rows_min": int(rows.min()) if len(rows) else 0,
        "rows_median": float(rows.median()) if len(rows) else 0.0,
        "rows_max": int(rows.max()) if len(rows) else 0,
        "episode_rows": episode_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--out-dir", default="analysis")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "dataset-usability.json"
    csv_path = out_dir / "dataset-usability.csv"

    results = []

    for dataset_name in args.datasets:
        if dataset_name in SKIP_REASON:
            print(f"[skip] {dataset_name}: {SKIP_REASON[dataset_name]}", flush=True)
            continue

        print(f"\n=== {dataset_name} ===", flush=True)
        try:
            dataset = get_dataset(dataset_name, download=True)
            db = dataset.get_db(upto_test_timestamp=False)
            db_min, db_max = str(db.min_timestamp), str(db.max_timestamp)
            print(f"  db span: {db_min} -> {db_max}", flush=True)
            del db
        except Exception as exc:
            print(f"  [FAIL] could not load dataset: {exc}", flush=True)
            traceback.print_exc()
            results.append({"dataset": dataset_name, "task": None,
                            "status": "dataset_error", "error": str(exc)})
            continue

        for task_name in get_task_names(dataset_name):
            print(f"  - {task_name} ... ", end="", flush=True)
            row = {"dataset": dataset_name, "task": task_name,
                   "db_min_timestamp": db_min, "db_max_timestamp": db_max}
            try:
                row.update(analyse_task(dataset_name, task_name))
                if row["status"] == "ok":
                    print(f"{row['n_episodes']} episodes, "
                          f"median {row['rows_median']:.0f} rows/episode", flush=True)
                else:
                    print(f"skipped ({row['reason']})", flush=True)
            except Exception as exc:
                row.update({"status": "error", "error": str(exc)})
                print(f"ERROR: {exc}", flush=True)
                traceback.print_exc()

            results.append(row)
            json_path.write_text(json.dumps(results, indent=2, default=str))
            pd.DataFrame([{k: v for k, v in r.items() if k != "episode_rows"}
                          for r in results]).to_csv(csv_path, index=False)

    ok = [r for r in results if r.get("status") == "ok"]
    print(f"\n\n=== SUMMARY ({len(ok)} usable tasks) ===", flush=True)
    if ok:
        summary = pd.DataFrame(ok)[
            ["dataset", "task", "task_type", "n_episodes",
             "rows_median", "rows_total", "delta_i_over_w"]
        ].sort_values(["dataset", "task"])
        print(summary.to_string(index=False), flush=True)
    print(f"\nWrote {json_path} and {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
