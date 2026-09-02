"""Score every saved checkpoint of a task over its whole timeline.

Produces the wide table the continual-learning metrics reduce over: one row per
target row of the task (train, val and test windows concatenated), one column
per finished MLflow run, named ``{increment}_{run_id}``. Grouping those rows by
episode turns the table into the evaluation matrix R[i, j] consumed by
:mod:`redelex.continual.metrics`.

The CSV is written incrementally and re-read on startup, so an interrupted pass
resumes rather than recomputing.
"""

from typing import Any, Dict, Optional

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

from experiments.continuous_learning.utils import (
    get_attribute_schema,
    get_experiment_runs_df,
    get_potato_client,
    get_text_embedder,
    get_table_input,
)
from experiments.continuous_learning.models import HeterogeneousSAGE
from experiments.continuous_learning.continuous_task import ContinuousWrapper


# Hyperparameters that change the model or the sampling, and so require
# rebuilding one or both before a checkpoint can be scored.
ARCH_PARAMS = ("gnn_channels", "gnn_layers", "gnn_aggr", "num_neighbors", "batch_size")


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
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    defaults = {
        "gnn_channels": gnn_channels,
        "gnn_layers": gnn_layers,
        "gnn_aggr": gnn_aggr,
        "num_neighbors": num_neighbors,
        "batch_size": batch_size,
    }

    mlflow_client = get_potato_client(mlflow_uri)
    runs_df = get_experiment_runs_df(
        mlflow_client,
        mlflow_experiment,
        filter_string=(
            f"params.dataset_name = '{dataset_name}' and "
            f"params.task_name = '{task_name}' and attributes.status = 'FINISHED'"
        ),
    )

    if runs_df.empty:
        print("No finished runs found matching the criteria.")
        return None

    print(f"Found {len(runs_df)} finished runs.")

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

    # Resume rather than restart: the base table is only written once, so
    # prediction columns from earlier passes survive.
    if csv_path.exists():
        results_df = pd.read_csv(csv_path)
        print(f"Resuming from {csv_path} with {len(results_df.columns)} columns.")
    else:
        results_df = wrapped_task.full_table.df.copy()
        results_df.to_csv(csv_path, index=False)

    # Built lazily and reused while consecutive runs share an architecture.
    current_arch: Optional[Dict[str, Any]] = None
    model = None
    full_loader = None

    for _, run in tqdm(runs_df.iterrows(), total=len(runs_df), desc="Evaluating Runs"):
        run_id = run["_run_id"]
        increment = run.get("increment", "unknown")

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
            model = HeterogeneousSAGE(
                data=data,
                col_stats_dict=col_stats_dict,
                gnn_channels=arch["gnn_channels"],
                gnn_layers=arch["gnn_layers"],
                gnn_aggr=arch["gnn_aggr"],
            ).to(device)
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

    print(f"Wrote {csv_path}")
    return results_df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--mlflow_experiment", type=str, default=None)
    parser.add_argument("--mlflow_uri", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)

    args = parser.parse_args()
    print(args)

    generate_all_predictions_df(
        dataset_name=args.dataset,
        task_name=args.task,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_uri=args.mlflow_uri,
        out_dir=args.out_dir,
    )
