#!/bin/bash
#SBATCH --job-name=cl_prepdata
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rci/prepare_datasets_%j.log
#
# Put the six continual-learning datasets on RCI: download what is missing,
# materialise the graph cache for all of them.
#
#   mkdir -p logs/rci                                 # once, see below
#   sbatch slurm/rci/prepare_datasets.sh              # everything still missing
#   sbatch slurm/rci/prepare_datasets.sh rel-f1 rel-trial   # just these
#   bash   slurm/rci/prepare_datasets.sh --check      # safe on the login node
#
# Submit from the repository root: --output is resolved relative to the
# submission directory, and `mkdir -p logs/rci` is not optional -- the one
# inside this script runs too late, because Slurm opens the --output file
# before the script does anything, and a missing directory fails the
# submission itself with "Unable to open file".
#
# The compute node needs outbound network: for the RelBench db.zip files, and
# for the 480 MB glove embedder that sentence-transformers pulls from
# HuggingFace into ~/.cache/huggingface on first use. rel-f1 runs first and
# exercises both in about four seconds, so a network problem surfaces
# immediately rather than an hour in.
#
# The job is idempotent: a dataset whose tensor frames are all present is
# skipped, so a job that hits the wall clock is fixed by resubmitting it. It is
# also safe to run two of them on disjoint dataset lists, but not on the same
# dataset -- torch_frame's per-table cache has no locking.
#
# ---------------------------------------------------------------------------
# Why these resources
# ---------------------------------------------------------------------------
#
# --mem=240G
#   Every one of the six was measured on potato, from an empty cache. Peak
#   resident set: rel-amazon 139.9 GB, rel-ratebeer 52.5 GB, rel-trial 20.3 GB,
#   rel-stack 17.0 GB, rel-hm 8.2 GB, rel-f1 1.9 GB. 240 GB is ~1.7x the
#   largest of them and still fits a 384 GB `cpu` node, so the job stays
#   schedulable. Do not size this for the typical dataset: the OOM killer sends
#   SIGKILL, and torch_frame writes tensor frames with a bare torch.save, so a
#   kill mid-write leaves a truncated .pt behind. The script detects and
#   rebuilds those, but the hour spent producing one is gone.
#
# --partition=cpu --time=12:00:00
#   Measured full-build wall time, same runs: rel-f1 4 s, rel-hm 253 s,
#   rel-trial 493 s, rel-stack 521 s, rel-ratebeer 1618 s, rel-amazon 3348 s
#   -- 1h44m of compute in total, plus ~5 GB of downloads for rel-f1, rel-hm
#   and rel-ratebeer. 12 h is ~7x that, which covers an RCI node being slower
#   than potato and a cold HuggingFace cache for the glove embedder.
#   `cpufast` (4 h limit) does NOT have that margin: rel-amazon alone is an
#   hour, and being killed at the wall clock after an hour of single-threaded
#   embedding is the failure this sizing exists to avoid. `cpu` has a 24 h
#   limit, so 12 h is a request the partition can actually honour, and the
#   script refuses to start a dataset that cannot finish inside what is left.
#
# NOT a GPU job. Materialisation is pandas, pyarrow and a CPU glove embedder;
# asking for a V100 would burn the fair-share budget on an idle card.
#
# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------
#
# The six materialised caches total ~108 GiB (rel-amazon 52, rel-ratebeer 30,
# rel-trial 14, rel-stack 11, rel-hm 2.8, rel-f1 0.012), on top of ~5 GB of new
# downloads in the RelBench cache. Check the quota before submitting -- the
# script refuses a dataset whose estimated cache does not fit in the free space
# it can see, but it cannot see a per-user quota.

set -uo pipefail

# RCI's rule is that all work happens inside ~/git/claude-redelex. CL_REPO
# overrides it only so the script can be exercised from a checkout
# elsewhere; on RCI leave it unset.
REPO="${CL_REPO:-$HOME/git/claude-redelex}"
cd "$REPO" || { echo "no repo at $REPO"; exit 1; }
mkdir -p logs/rci

# tqdm reads this when it is imported, not when a bar is created, so it has to
# be in the environment before python starts. Without it the embedder's
# progress bars turn this log into megabytes of carriage returns -- the
# rel-hm/ratebeer/amazon run on potato produced a 3 MB log for 80 lines of
# actual content.
export TQDM_DISABLE=1

# Keep every thread pool at the cpus this job was actually allocated. Left
# unset, OpenMP and MKL size themselves from the physical core count of a
# 128-core node and thrash against the cgroup.
THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

# Share the RelBench download cache with everything else on this account, so
# the rel-amazon/rel-stack/rel-trial/rel-avito databases already there are
# reused rather than re-fetched.
export RELBENCH_CACHE_DIR="${RELBENCH_CACHE_DIR:-$HOME/.cache/relbench}"

# ARGS is written this way because `"${DATASETS[@]}"` on an empty array is
# an unbound-variable error under `set -u` on bash < 4.4, and RCI's bash
# version is not something this script should depend on. With no datasets
# named it expands to nothing at all, letting the python script fall back to
# its own default order.
DATASETS=("$@")
ARGS=(${DATASETS[@]+"${DATASETS[@]}"})

echo "=== node       : $(hostname) ==="
echo "=== job        : ${SLURM_JOB_ID:-<no slurm>} on ${SLURM_JOB_PARTITION:-<none>}"
echo "=== cpus       : $THREADS"
echo "=== mem        : ${SLURM_MEM_PER_NODE:-?} MB"
echo "=== relbench   : $RELBENCH_CACHE_DIR"
echo "=== free disk  :"
df -h "$REPO" "$RELBENCH_CACHE_DIR" 2>/dev/null | sed 's/^/    /'
quota -s 2>/dev/null | sed 's/^/    /' || true
echo

# --check does no work and imports neither torch nor relbench, so this branch is
# the one thing here that is safe to run on a login node.
if [ "${1:-}" = "--check" ]; then
    .venv/bin/python scripts/materialize_graphs.py --check "${@:2}"
    exit $?     # 0 when everything is ready, 1 when work remains
fi

# Refuse to do the heavy part outside Slurm. Downloading 5 GB and materialising
# 108 GB on a shared login node is exactly what the cluster rules forbid, and
# `bash slurm/rci/prepare_datasets.sh` is an easy thing to type by accident.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    cat >&2 <<'MSG'
ERROR: this job downloads several GB and materialises ~108 GB of tensor frames.
       Run it with sbatch, never on a login node:

           sbatch slurm/rci/prepare_datasets.sh

       For a cheap read-only report of what is present, use:

           bash slurm/rci/prepare_datasets.sh --check
MSG
    exit 2
fi

# --time-budget is the fallback for a Slurm that does not export
# SLURM_JOB_END_TIME; the script takes whichever deadline is earlier, so the
# scheduler's own value wins when it is available. Keep this in sync with the
# #SBATCH --time above -- sbatch directives cannot read shell variables, so the
# two really do have to be written twice.
WALL_SECONDS=$(( 12 * 3600 ))

.venv/bin/python scripts/materialize_graphs.py \
    ${ARGS[@]+"${ARGS[@]}"} \
    --cache-dir "$REPO/.cache" \
    --relbench-cache "$RELBENCH_CACHE_DIR" \
    --embedder glove \
    --torch-threads "$THREADS" \
    --time-budget "$WALL_SECONDS" \
    --json "logs/rci/prepare_datasets_${SLURM_JOB_ID:-local}.json"
STATUS=$?

echo
echo "=== disk after ==="
df -h "$REPO" "$RELBENCH_CACHE_DIR" 2>/dev/null | sed 's/^/    /'
du -sh "$REPO/.cache"/*/materialized 2>/dev/null | sed 's/^/    /'

if [ "$STATUS" -ne 0 ]; then
    echo
    echo "=== NOT FINISHED (exit $STATUS) -- resubmit to continue: ==="
    echo "    sbatch slurm/rci/prepare_datasets.sh ${DATASETS[*]:-}"
fi

echo "=== DONE (exit $STATUS) ==="
exit "$STATUS"
