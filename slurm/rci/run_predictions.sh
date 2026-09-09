#!/bin/bash
#SBATCH --job-name=cl_predict
#SBATCH --partition=amdgpufast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/rci/run_predictions_%j.log
#
# The PREDICTION pass for one (dataset, task, mode) chain on RCI.
#
# Training produces one checkpoint per episode; this scores every one of them
# over a table covering the WHOLE dataset timeline, so each checkpoint is
# evaluated on episodes it was never trained on. That full-timeline column is
# what `scripts/run_analysis.py` slices into the evaluation matrix R[i,j], and
# R is what BWT, forward transfer and forgetting are computed from. Without
# this pass the grid yields only final-episode scores -- which cannot show
# forgetting at all, and forgetting is the paper's subject.
#
# One job per chain, because the chains are independent and a per-chain job is
# restartable: run_predictions.py skips any column already in the CSV, so a
# re-run costs only the checkpoints it has not scored yet.
#
# Resource sizing matches run_chain.sh and is deliberate; see the note there.
# amdgpufast bills CPU and RAM only (TRESBillingWeights=CPU=1.0,Mem=0.25G), so
# 4 CPU + 32G is billing=12 rather than 24 for the same GPU.

set -uo pipefail

REPO="${REPO:-$HOME/git/claude-redelex}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
ENTRY="${ENTRY:-experiments/continuous_learning/run_predictions.py}"

DATASET=""; TASK=""; MODE=""; OUT=""
MLFLOW_URI="http://potato.felk.cvut.cz:2222"
MLFLOW_EXPERIMENT=""
DATA_ROOT=""
SEED=42
EXTRA=""

usage() {
    cat <<'USAGE'
run_predictions.sh --dataset D --task T --mode M --out DIR --data-root DIR
                   --mlflow-experiment E [--mlflow-uri U] [--seed N] [--extra "..."]

Scores every checkpoint of one chain over the full dataset timeline and writes
<data-root>/<mlflow-experiment>/<dataset>_<task>_predictions.csv.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)           DATASET="$2"; shift 2 ;;
        --task)              TASK="$2"; shift 2 ;;
        --mode)              MODE="$2"; shift 2 ;;
        --out)               OUT="$2"; shift 2 ;;
        --data-root)         DATA_ROOT="$2"; shift 2 ;;
        --mlflow-uri)        MLFLOW_URI="$2"; shift 2 ;;
        --mlflow-experiment) MLFLOW_EXPERIMENT="$2"; shift 2 ;;
        --seed)              SEED="$2"; shift 2 ;;
        --extra)             EXTRA="$2"; shift 2 ;;
        -h|--help)           usage; exit 0 ;;
        *) echo "run_predictions.sh: unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
done

for required in DATASET TASK MODE OUT DATA_ROOT MLFLOW_EXPERIMENT; do
    if [[ -z "${!required}" ]]; then
        flag="--${required,,}"
        echo "run_predictions.sh: ${flag//_/-} is required" >&2
        exit 2
    fi
done

cd "$REPO" || exit 2

# Same BLAS pinning rationale as run_chain.sh: these are 128-core nodes and the
# libraries size themselves from the machine, not from the cgroup.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

CHAIN_KEY="${DATASET}__${TASK}__${MODE}"
CHAIN_DIR="$OUT/$CHAIN_KEY"
MODEL_DIR="$CHAIN_DIR/models"
MARKER_DIR="$OUT/markers"
LOG="$CHAIN_DIR/${CHAIN_KEY}__predict.log"
mkdir -p "$MARKER_DIR" "$CHAIN_DIR"

if [[ -e "$MARKER_DIR/${CHAIN_KEY}.predicted" ]]; then
    echo "predictions for $CHAIN_KEY already done; nothing to do."
    exit 0
fi

# Refuse to score a chain that has not finished training. A partial chain still
# produces a CSV, and that CSV looks exactly like a complete one to the analysis
# driver -- it just has fewer columns, which reads as a shorter experiment
# rather than an unfinished one.
if [[ ! -e "$MARKER_DIR/${CHAIN_KEY}__c00.done" ]] \
   && [[ ! -e "$MARKER_DIR/${CHAIN_KEY}.chain-complete" ]]; then
    echo "run_predictions.sh: chain $CHAIN_KEY has no completion marker;" >&2
    echo "scoring it now would produce a short R that looks complete. Refusing." >&2
    exit 2
fi

# Must be byte-identical to the string continuous_learning.py logged, built from
# the RAW --model_save_dir. A mismatch does not error: it selects zero runs and
# writes an empty CSV.
CHAIN_ID="${DATASET}/${TASK}/${MODE}/${MODEL_DIR}"
DATA_DIR="$DATA_ROOT/$MLFLOW_EXPERIMENT"
mkdir -p "$DATA_DIR"

echo "=== $(date -Is) node $(hostname) job ${SLURM_JOB_ID:-none} ==="
echo "chain    : $CHAIN_KEY"
echo "chain_id : $CHAIN_ID"
echo "out      : $DATA_DIR/${DATASET}_${TASK}_predictions.csv"
echo "seed     : $SEED (identical across modes, or the matrices are incomparable)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086  # EXTRA is a deliberately word-split flag string.
rc=0
"$PYTHON" -u "$ENTRY" \
    --dataset="$DATASET" \
    --task="$TASK" \
    --mlflow_uri="$MLFLOW_URI" \
    --mlflow_experiment="$MLFLOW_EXPERIMENT" \
    --out_dir="$DATA_DIR" \
    --seed="$SEED" \
    --chain_id="$CHAIN_ID" \
    --strict_protocol \
    $EXTRA \
    >>"$LOG" 2>&1 || rc=$?

CSV="$DATA_DIR/${DATASET}_${TASK}_predictions.csv"
if [[ $rc -eq 0 && -s "$CSV" ]]; then
    date -Is > "$MARKER_DIR/${CHAIN_KEY}.predicted"
    echo "=== OK $CHAIN_KEY -> $(wc -l < "$CSV") rows ==="
else
    # No marker on failure, and none on an empty CSV either: an empty file is
    # the signature of a chain_id that matched nothing, which is the one failure
    # mode here that does not raise.
    echo "=== FAILED $CHAIN_KEY (rc=$rc, csv=$( [[ -s "$CSV" ]] && echo nonempty || echo EMPTY)) ===" >&2
    tail -20 "$LOG" >&2
fi
exit $rc
