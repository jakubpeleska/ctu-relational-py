#!/bin/bash
#SBATCH --job-name=cl_setup
#SBATCH --partition=cpufast
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/rci/setup_env_%j.log
#
# Build the project environment on RCI. Runs as a batch job because creating the
# venv downloads several GB of wheels, which does not belong on a login node.
#
#   sbatch slurm/rci/setup_env.sh
#
# Not a GPU job on purpose: `uv sync` needs no GPU, and cpufast queues faster.
# The CUDA build is verified separately by slurm/rci/verify_gpu.sh.

set -euo pipefail
cd "$HOME/git/claude-redelex"
mkdir -p logs/rci
export PATH="$HOME/.local/bin:$PATH"

echo "=== node: $(hostname) ==="
uv --version

# cu128, never the default `cpu` group: a plain `uv sync` silently installs the
# CPU build of torch and every later run is ~7x slower with no error.
echo "=== uv sync (cu128) ==="
uv sync --no-group cpu --group cu128

echo "=== interpreter ==="
.venv/bin/python -V
.venv/bin/python -c "
import torch, relbench, torch_geometric
print(f'  torch {torch.__version__}')
print(f'  relbench {__import__(\"importlib.metadata\", fromlist=[\"x\"]).version(\"relbench\")}')
print(f'  torch_geometric {torch_geometric.__version__}')
print(f'  cuda compiled: {torch.version.cuda}')
"

echo "=== import the experiment modules ==="
.venv/bin/python -c "
from experiments.continuous_learning.cl_modes import DEFAULT_ROSTER
from redelex.continual import ReservoirBuffer, ParameterAnchor
print('  roster:', ', '.join(DEFAULT_ROSTER))
"

echo "=== unit tests ==="
.venv/bin/python -m pytest -q 2>&1 | tail -3

echo "=== DONE ==="
