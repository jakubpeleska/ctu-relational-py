"""Run a grid of continual-learning jobs across the local GPUs.

Replaces the Slurm job array (`slurm/continuous_learning/*.sh`) on hosts with no
scheduler. One job = one (dataset, task, learning_mode) episode chain, pinned to
a single GPU. Several jobs run concurrently, one per GPU slot.

Why one GPU per job rather than one job across all GPUs: a chain is sequential
across episodes, and each episode runs `num_samples` trials. Giving a chain four
GPUs only parallelises the trials inside one episode; running four chains of one
GPU each keeps every GPU busy and advances four tasks at once.

Restartable: a job whose marker file exists in `--out/markers` is skipped, so
re-running the same command resumes where it stopped.

Usage:
    .venv/bin/python scripts/run_grid.py --tier A --modes from_scratch ft_full
    .venv/bin/python scripts/run_grid.py --pairs "rel-f1:driver-dnf" --dry-run

Never invoke via a plain `uv run`: that re-syncs the default `cpu` dependency
group and silently swaps torch for the CPU build.
"""

import argparse
import itertools
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = REPO / ".venv" / "bin" / "python"
ENTRY = REPO / "experiments" / "continuous_learning" / "continuous_learning.py"

# Measured episode counts, not metadata ceilings. See analysis/dataset-usability.json.
TIERS = {
    "A": [("rel-f1", "driver-position"), ("rel-f1", "driver-dnf"), ("rel-f1", "driver-top3"),
          ("rel-trial", "study-outcome"), ("rel-trial", "study-adverse"), ("rel-trial", "site-success")],
    "B": [("rel-hm", "user-churn"), ("rel-hm", "item-sales")],
    "C": [("rel-stack", "user-engagement"), ("rel-stack", "post-votes"), ("rel-stack", "user-badge")],
    "D": [("rel-ratebeer", "beer-churn"), ("rel-ratebeer", "user-churn"),
          ("rel-ratebeer", "user-count"), ("rel-ratebeer", "brewer-dormant")],
}

# The roster: three reference points plus one mode per CL family. See
# experiments/continuous_learning/cl_modes.py for why each earns a slot.
DEFAULT_MODES = ["from_scratch", "joint", "naive", "er", "der_pp", "ewc", "lwf",
                 "freeze_extend"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tier", nargs="+", choices=sorted(TIERS), help="Dataset tiers to run.")
    src.add_argument("--pairs", nargs="+", metavar="DATASET:TASK",
                     help="Explicit dataset:task pairs.")
    p.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    p.add_argument("--gpus", type=int, nargs="+", default=None,
                   help="Physical GPU ids to use. Default: all visible.")
    p.add_argument("--num-samples", type=int, default=3, help="Seeds per episode.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpus-per-job", type=int, default=4)
    p.add_argument("--mlflow-uri", default="http://potato.felk.cvut.cz:2222")
    p.add_argument(
        "--mlflow-experiment-prefix", default="pelesjak_cl_v2",
        help="Must NOT collide with the published grid: 'pelesjak_cl' resolves to "
             "pelesjak_cl_from_scratch, which is MLflow experiment 92 holding 684 "
             "runs whose checkpoints live on a cluster this host cannot reach.",
    )
    p.add_argument("--out", default="logs/grid", help="Root for logs, models and markers.")
    p.add_argument("--resume-chain", action="store_true",
                   help="Pass --resume so each chain resumes from MLflow.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def resolve_pairs(args):
    if args.pairs:
        out = []
        for raw in args.pairs:
            if ":" not in raw:
                raise SystemExit(f"--pairs entry {raw!r} must look like dataset:task")
            ds, task = raw.split(":", 1)
            out.append((ds, task))
        return out
    return [pair for tier in args.tier for pair in TIERS[tier]]


def detect_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
        return [int(line) for line in out.split() if line.strip().isdigit()]
    except (OSError, subprocess.CalledProcessError):
        return []


def build_command(job, gpu, args, run_dir):
    dataset, task, mode = job
    cmd = [
        str(PYTHON), "-u", str(ENTRY),
        f"--dataset={dataset}", f"--task={task}", f"--learning_mode={mode}",
        "--ray_address=local",
        f"--ray_storage={run_dir}", f"--model_save_dir={run_dir}/models",
        f"--run_name={dataset}_{task}_{mode}",
        f"--mlflow_experiment={args.mlflow_experiment_prefix}_{mode}",
        f"--num_cpus={args.cpus_per_job}", "--num_gpus=1", f"--gpu_ids={gpu}",
        f"--num_samples={args.num_samples}", f"--seed={args.seed}",
    ]
    if args.mlflow_uri:
        cmd.append(f"--mlflow_uri={args.mlflow_uri}")
    if args.resume_chain:
        cmd.append("--resume")
    return cmd


def main(argv=None):
    args = parse_args(argv)
    pairs = resolve_pairs(args)
    jobs = [(ds, task, mode) for (ds, task), mode in itertools.product(pairs, args.modes)]

    gpus = args.gpus if args.gpus is not None else detect_gpus()
    if not gpus:
        raise SystemExit("No GPUs detected; pass --gpus explicitly to override.")

    out_root = Path(args.out)
    marker_dir = out_root / "markers"
    marker_dir.mkdir(parents=True, exist_ok=True)

    pending, skipped = [], []
    for job in jobs:
        marker = marker_dir / ("_".join(job) + ".done")
        (skipped if marker.exists() else pending).append(job)

    print(f"{len(jobs)} jobs ({len(pairs)} pairs x {len(args.modes)} modes); "
          f"{len(skipped)} already done, {len(pending)} to run on GPUs {gpus}")
    if args.dry_run:
        for job in pending:
            print("  would run:", " ".join(build_command(job, gpus[0], args, out_root / "_dry")))
        return 0
    if not pending:
        print("Nothing to do.")
        return 0

    work = queue.Queue()
    for job in pending:
        work.put(job)

    failures, lock = [], threading.Lock()

    def worker(gpu):
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            dataset, task, mode = job
            name = "_".join(job)
            run_dir = out_root / name
            (run_dir / "models").mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "run.log"
            cmd = build_command(job, gpu, args, run_dir)

            started = datetime.now(timezone.utc)
            with lock:
                print(f"[gpu{gpu}] START {name} -> {log_path}", flush=True)
            with open(log_path, "ab") as log:
                log.write(f"\n=== {started.isoformat()} :: {' '.join(cmd)}\n".encode())
                log.flush()
                rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, cwd=REPO)
            mins = (datetime.now(timezone.utc) - started).total_seconds() / 60

            with lock:
                status = "OK" if rc == 0 else f"FAILED rc={rc}"
                print(f"[gpu{gpu}] {status} {name} ({mins:.1f} min)", flush=True)
                with open(out_root / "RUNLOG.tsv", "a") as f:
                    f.write(f"{started.isoformat()}\t{name}\tgpu{gpu}\t{mins:.1f}\t{status}\n")
                if rc != 0:
                    failures.append(name)
            if rc == 0:
                (marker_dir / f"{name}.done").write_text(started.isoformat())
            work.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\nDone in {(time.time()-t0)/60:.1f} min. "
          f"{len(pending)-len(failures)}/{len(pending)} succeeded.")
    if failures:
        print("Failed:", ", ".join(sorted(failures)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
