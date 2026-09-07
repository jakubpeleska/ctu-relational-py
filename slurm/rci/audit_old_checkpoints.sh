#!/bin/bash
#SBATCH --job-name=cl_ckptaudit
#SBATCH --partition=cpufast
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/rci/audit_ckpt_%j.log
#
# Are the published checkpoints reusable, or are they evidence?
#
# ~/git/ctu-relational-py/logs holds 103 job directories from the runs behind the
# submitted paper, and their paths match the model_save_dir values recorded in
# MLflow. If those weights are sound we can compute forgetting metrics for the
# original regimes by re-scoring them, instead of re-running the whole thing.
#
# But they predate the na_strategy fix (commit b3bf3f7), under which every
# numerical column holding a missing cell acquired a NaN gradient and died. This
# job settles which it is. CPU-only and read-only.

set -euo pipefail
cd "$HOME/git/claude-redelex"
mkdir -p logs/rci

.venv/bin/python - <<'PY'
import glob, os, random, torch, collections

ROOT = os.path.expanduser("~/git/ctu-relational-py/logs")
jobs = sorted(d for d in glob.glob(f"{ROOT}/cl_*") if os.path.isdir(d))
print(f"job directories: {len(jobs)}")

# Sample rather than walk 103 trees: enough to tell "all poisoned" from "none".
random.seed(0)
found = []
for job in jobs:
    hits = glob.glob(f"{job}/*/models/increment_*/*/best_model.pt")
    if hits:
        found.extend(random.sample(hits, min(3, len(hits))))
print(f"checkpoints sampled: {len(found)}")

bad, total, by_tensor = 0, 0, collections.Counter()
sizes = set()
for path in found[:200]:
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"  unreadable {path}: {type(exc).__name__}")
        continue
    total += 1
    sizes.add(sum(v.numel() for v in sd.values() if torch.is_tensor(v)))
    nan = [k for k, v in sd.items() if torch.is_tensor(v) and not torch.isfinite(v).all()]
    if nan:
        bad += 1
        for k in nan:
            by_tensor[k] += 1

print(f"\nRESULT: {bad}/{total} sampled checkpoints contain non-finite weights")
print(f"distinct parameter counts seen: {sorted(sizes)[:5]}")
if by_tensor:
    print("most frequently poisoned tensors:")
    for name, n in by_tensor.most_common(5):
        print(f"  {n:4d}x  {name}")
print("\nVERDICT:", "poisoned -- evidence, not reusable models"
      if bad == total and total else
      ("clean -- reusable for re-scoring" if bad == 0 else "mixed -- inspect per run"))
PY
echo "=== DONE ==="
