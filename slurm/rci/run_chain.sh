#!/bin/bash
#SBATCH --job-name=cl_chunk
#SBATCH --partition=amdgpufast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/rci/run_chain_%j.log
#
# One CHUNK of one continual-learning chain on RCI.
#
# A chain is one (dataset, task, learning_mode) run over episodes 1..N, each
# episode warm-starting from the previous episode's best checkpoint. rel-hm has
# 52 episodes; as a single job that is a multi-day `gpuextralong` allocation,
# which is exactly what the fair-share budget punishes. So a chain is cut into
# chunks of K episodes: every job runs `--max_increments K` and `--resume`, and
# the next job picks the chain up where this one left it.
#
# Chunking is only safe because `--resume` recovers the CL *state* (replay
# buffer, EWC anchor) alongside the weights, and refuses to continue a stateful
# chain whose state file is missing. Three preconditions come with that, and
# this script exists to hold all three:
#
#   1. `chain_id` is `dataset/task/mode/model_save_dir` built from the string
#      passed on the command line (continuous_learning.py:946, computed *before*
#      the path is made absolute). Every chunk of a chain must therefore pass a
#      byte-identical `--model_save_dir`, so it is derived here from the chain
#      key alone and never from the chunk index or the job id.
#   2. `expected_trials` is `len(seeds)`. Resuming a chain with MORE seeds than
#      the earlier chunks ran makes every finished increment look incomplete and
#      silently restarts the chain at episode 1. `chain-params.txt` pins the
#      value on the first chunk and this script refuses to run if a later chunk
#      disagrees.
#   3. The chain must never have two chunks in flight at once -- they would both
#      resume from the same increment. scripts/submit_rci_grid.py guarantees
#      that by putting every chunk of a chain in one serial dependency lane.
#
# cwd is the repo, so `.cache/<dataset>/materialized` is shared by every chunk:
# re-materialising a graph per job would cost more than the training it feeds.
#
# Submitted by scripts/submit_rci_grid.py. The #SBATCH directives above are
# defaults for a hand-run `sbatch slurm/rci/run_chain.sh ...`; the submitter
# overrides partition, time, resources, name, output and dependency on the
# command line, where they take precedence.

set -euo pipefail

REPO="${CL_REPO:-$HOME/git/claude-redelex}"
PYTHON="$REPO/.venv/bin/python"
ENTRY="$REPO/experiments/continuous_learning/continuous_learning.py"

# The exact line run_ray_tuner prints when the chain has no episodes left. Used
# to mark the chain complete, so the remaining queued chunks of a chain that
# finished early exit in seconds instead of re-loading a dataset to discover
# there is nothing to do.
CHAIN_DONE_SENTINEL="All increments are already completed"

DATASET=""
TASK=""
MODE=""
CHUNK=""
EPISODES=""
NUM_SAMPLES=5
SEED=42
OUT=""
MLFLOW_URI="http://potato.felk.cvut.cz:2222"
MLFLOW_EXPERIMENT=""
CPUS_PER_TRIAL=4
TORCH_THREADS=1
EXTRA=""

usage() {
    cat <<'USAGE'
usage: run_chain.sh --dataset D --task T --mode M --chunk N --episodes K
                    --out DIR --mlflow-experiment NAME
                    [--num-samples 5] [--seed 42] [--mlflow-uri URI]
                    [--cpus-per-trial 4] [--torch-threads 1] [--extra "ARGS"]

  --chunk     index of this chunk within the chain; labels the marker only.
  --episodes  K, passed to the experiment as --max_increments.
  --extra     verbatim extra flags for continuous_learning.py, e.g.
              --extra "--val_delta_days=30 --buffer_size=20000"
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)           DATASET="$2"; shift 2 ;;
        --task)              TASK="$2"; shift 2 ;;
        --mode)              MODE="$2"; shift 2 ;;
        --chunk)             CHUNK="$2"; shift 2 ;;
        --episodes)          EPISODES="$2"; shift 2 ;;
        --num-samples)       NUM_SAMPLES="$2"; shift 2 ;;
        --seed)              SEED="$2"; shift 2 ;;
        --out)               OUT="$2"; shift 2 ;;
        --mlflow-uri)        MLFLOW_URI="$2"; shift 2 ;;
        --mlflow-experiment) MLFLOW_EXPERIMENT="$2"; shift 2 ;;
        --cpus-per-trial)    CPUS_PER_TRIAL="$2"; shift 2 ;;
        --torch-threads)     TORCH_THREADS="$2"; shift 2 ;;
        --extra)             EXTRA="$2"; shift 2 ;;
        -h|--help)           usage; exit 0 ;;
        *) echo "run_chain.sh: unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
done

for required in DATASET TASK MODE CHUNK EPISODES OUT MLFLOW_EXPERIMENT; do
    if [[ -z "${!required}" ]]; then
        flag="--${required,,}"
        echo "run_chain.sh: ${flag//_/-} is required" >&2
        exit 2
    fi
done

cd "$REPO"

# `__` separates the three fields because mode names contain single underscores
# (from_scratch, der_pp, freeze_extend) and dataset/task names contain hyphens.
CHAIN_KEY="${DATASET}__${TASK}__${MODE}"
CHUNK_KEY="$(printf '%s__c%02d' "$CHAIN_KEY" "$CHUNK")"
CHAIN_DIR="$OUT/$CHAIN_KEY"
MARKER_DIR="$OUT/markers"
MODEL_DIR="$CHAIN_DIR/models"
RAY_DIR="$CHAIN_DIR/ray"
LOG="$CHAIN_DIR/${CHUNK_KEY}.log"

mkdir -p "$MARKER_DIR" "$MODEL_DIR" "$RAY_DIR"

# Two cheap exits before anything heavy. Both make re-submission idempotent: the
# submitter also filters on these markers, but a job already queued when the
# chain finished can only be stopped here.
if [[ -e "$MARKER_DIR/${CHAIN_KEY}.chain-complete" ]]; then
    echo "chain $CHAIN_KEY already complete; nothing to do."
    exit 0
fi
if [[ -e "$MARKER_DIR/${CHUNK_KEY}.done" ]]; then
    echo "chunk $CHUNK_KEY already done; nothing to do."
    exit 0
fi

# Precondition 2: the resume quorum is len(seeds). Changing --num-samples or
# --seed mid-chain either restarts it from episode 1 (more seeds than before) or
# accepts a partially finished increment as complete (fewer). Both are silent,
# so fail here instead.
PARAMS_FILE="$CHAIN_DIR/chain-params.txt"
SIGNATURE="num_samples=$NUM_SAMPLES seed=$SEED model_save_dir=$MODEL_DIR"
if [[ -f "$PARAMS_FILE" ]]; then
    PREVIOUS="$(cat "$PARAMS_FILE")"
    if [[ "$PREVIOUS" != "$SIGNATURE" ]]; then
        echo "run_chain.sh: chain $CHAIN_KEY was started with [$PREVIOUS] but this" >&2
        echo "chunk asks for [$SIGNATURE]. Resuming across that change would" >&2
        echo "silently restart or mis-resume the chain. Refusing." >&2
        exit 2
    fi
else
    printf '%s\n' "$SIGNATURE" > "$PARAMS_FILE"
fi

# Slurm has already restricted this job to its allocated device; take that
# device rather than letting the experiment's free-memory auto-selection
# renumber CUDA_VISIBLE_DEVICES relative to the visible set and land on the
# wrong physical GPU (see continuous_learning.py:882-890).
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
GPU_ID="${GPU_ID:-0}"
NUM_CPUS="${SLURM_CPUS_PER_TASK:-4}"

echo "=== $(date -Is) node $(hostname) job ${SLURM_JOB_ID:-none} ==="
echo "chain   : $CHAIN_KEY"
echo "chunk   : $CHUNK_KEY (up to $EPISODES episode(s) from wherever the chain stands)"
echo "gpu     : id $GPU_ID of CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-unset}'"
echo "models  : $MODEL_DIR"
echo "log     : $LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086  # EXTRA is a deliberately word-split flag string.
rc=0
"$PYTHON" -u "$ENTRY" \
    --dataset="$DATASET" \
    --task="$TASK" \
    --learning_mode="$MODE" \
    --ray_address=local \
    --ray_storage="$RAY_DIR" \
    --model_save_dir="$MODEL_DIR" \
    --run_name="$CHUNK_KEY" \
    --mlflow_uri="$MLFLOW_URI" \
    --mlflow_experiment="$MLFLOW_EXPERIMENT" \
    --num_samples="$NUM_SAMPLES" \
    --seed="$SEED" \
    --max_increments="$EPISODES" \
    --num_gpus=1 \
    --gpu_ids="$GPU_ID" \
    --num_cpus="$NUM_CPUS" \
    --cpus_per_trial="$CPUS_PER_TRIAL" \
    --torch_threads="$TORCH_THREADS" \
    --resume \
    $EXTRA \
    >>"$LOG" 2>&1 || rc=$?

# The chain ran out of episodes: mark it so the queued tail of this lane exits
# immediately instead of paying a dataset load each to learn the same thing.
if grep -q "$CHAIN_DONE_SENTINEL" "$LOG"; then
    touch "$MARKER_DIR/${CHAIN_KEY}.chain-complete"
    echo "chain $CHAIN_KEY reports no episodes left; marked complete."
fi

if [[ $rc -eq 0 ]]; then
    date -Is > "$MARKER_DIR/${CHUNK_KEY}.done"
    echo "=== OK $CHUNK_KEY ==="
else
    # No marker on failure, and no marker on a Slurm timeout kill (which never
    # reaches this line): the next chunk in the lane resumes from the last
    # increment MLflow recorded as complete, so the work is retried, not lost.
    echo "=== FAILED $CHUNK_KEY rc=$rc -- see $LOG ===" >&2
    tail -n 40 "$LOG" >&2 || true
fi
exit $rc
