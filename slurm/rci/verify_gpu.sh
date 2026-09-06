#!/bin/bash
#SBATCH --job-name=cl_gpucheck
#SBATCH --partition=gpufast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --output=logs/rci/verify_gpu_%j.log
#
# Confirm the CUDA build actually works on an RCI GPU node, and record which GPU
# it is. The cluster has V100s (gpu*) and A100s (amdgpu*); per-batch cost differs
# between them, so the compute model must know which one produced a number.

set -euo pipefail
cd "$HOME/git/claude-redelex"
mkdir -p logs/rci

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

.venv/bin/python -c "
import torch, time
print(f'  torch {torch.__version__} | cuda available: {torch.cuda.is_available()}')
assert torch.cuda.is_available(), 'no CUDA on a GPU node -- environment is wrong'
d = torch.device('cuda')
print(f'  device: {torch.cuda.get_device_name(0)} | capability {torch.cuda.get_device_capability(0)}')
a = torch.randn(4096, 4096, device=d); torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(20): a = a @ a.clamp(-1, 1)
torch.cuda.synchronize()
print(f'  20 x 4096^2 matmul: {time.perf_counter()-t:.2f}s | finite: {bool(torch.isfinite(a).all())}')
"
echo "=== DONE ==="
