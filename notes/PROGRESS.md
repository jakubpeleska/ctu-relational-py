# PROGRESS — CL Benchmark Reframe (ICLR 2027)

**Entry point for any new session.** Read this, then `notes/DECISIONS.md`.
Plan: `~/.claude/plans/happy-twirling-lantern.md`

Target: ICLR 2027. HARD DEADLINE: results by ~2026-09-16 (12 days from 2026-09-04).
Priority: (1) integrate real CL techniques, (2) add datasets or rigorously argue exclusions.

---

## Current phase: post-review fixes done, BLOCKED on GPUs (2026-09-06)

**The code review returned NO-GO and all six defects are fixed and re-validated.**
See `notes/DECISIONS.md` and commit b3bf3f7.

| step | status |
|---|---|
| Five-lens code review with adversarial refutation | DONE - returned NO-GO |
| M1-M6 fixes | DONE, committed b3bf3f7 |
| Post-fix smoke, 8 modes x 2 seeds on rel-f1 | **8/8 succeeded** |
| **Main grid launch** | **BLOCKED - no GPUs** |

### Verified on real artefacts after the fixes

- **NaN weights: 21/21 checkpoints before -> 0/32 after.**
- **Checkpoint size 8,272,841 -> 5,068,690 params** (the duplicated encoder registration is gone).
- Mode mechanics all correct: `from_scratch`/`joint` train the full window at both episodes;
  `naive`/`er`/`der_pp`/`ewc`/`lwf`/`freeze_extend` switch to the increment at episode 2;
  `er`/`der_pp` replay a 830-exemplar buffer; `freeze_extend` adds exactly 1 adapter.

## BLOCKER: GPU access revoked at the CONTAINER level (2026-09-06)

`nvidia-smi` -> `Failed to initialize NVML: Unknown Error`; `torch.cuda.is_available()` is False
under every `CUDA_VISIBLE_DEVICES` setting tried.

**Diagnosis (this is NOT a driver crash):** the device nodes exist and are world read/write
(`crw-rw-rw-` on `/dev/nvidia1`-`4`, `/dev/nvidiactl`, `/dev/nvidia-uvm`), yet `os.open()` on any of
them returns **EPERM**. World-writable + EPERM means the **container's device cgroup is denying
access**, not the driver failing and not file permissions. Note also that `/dev/nvidia0` is absent
while `nvidia1`-`nvidia4` are present.

So `nvidia-smi -r` or a driver reload will NOT fix this. What is needed is for the container to be
granted GPU device access again (e.g. `--gpus all` / the correct `device_cgroup_rules`), by whoever
runs the sandbox. Driver 560.35.03 is still loaded on the host.

## Machine facts that constrain the CPU fallback

- **64 PHYSICAL cores** (128 logical, 2 threads/core) on an AMD EPYC 7742. Parallelism projections
  must use 64, not 128.
- **`/dev/shm` is 32 GB**, not RAM-sized -- this hard-caps Ray's plasma object store, which is where
  a shared graph would live.
- 503 GB RAM, ~339 GB available (another tenant is using ~164 GB).
- **Load average ~8.8: the box is NOT idle.** Roughly 55 physical cores are actually free.

## Key context a new session needs

- The `rci` port is DONE and smoke-tested but was uncommitted as of 2026-09-02. Working tree =
  rci + import-path adaptation + `num_classes` + mlflow-name fallback.
- MLflow `http://potato.felk.cvut.cz:2222` is LIVE. Experiments: 92=cl_from_scratch, 93=cl_ft_full,
  90=cl_ft_newonly, 91=cl_ft_upsample. 2,371 published runs with per-episode metrics AND timings.
  This is the ground truth for verifying the port.
- Published CHECKPOINTS are NOT reachable (they point at `/home/pelesjak/git/ctu-relational-py/...`
  on the old cluster). Metrics reusable, weights not => anchors must be re-run for forgetting metrics.
- No Slurm on this host (`dev-sandbox`). The 46 `slurm/` scripts target the remote server.
- ALWAYS use `.venv/bin/python` or `uv run --no-group cpu --group cu128`. A plain `uv run` re-syncs
  the default `cpu` group and silently reverts torch to the CPU build (`pyproject.toml:91`).
- Pass `--num_gpus=1`; it defaults to 0 and will silently run on CPU at ~7x the cost.

## Measured episode counts (NOT metadata ceilings — those ran ~2.5x high)

ALL MEASURED - see `analysis/dataset-episodes-measured.md` for the full table.
rel-hm 52/52 (104) | rel-stack 18/18/17 (53) | rel-amazon 15/15/15/16 (61) |
rel-ratebeer 12/12/12/9 (45) | rel-f1 11/11/2 (24) | rel-trial 7/7/7 (21) |
rel-event 10/9 (corrupt DB span, caveat) | rel-arxiv 3 (drop) | rel-avito 2 (drop)
rel-stack and rel-f1/rel-trial counts independently confirmed against the published MLflow grid.

## Blocked / needs the user

- (none currently) - the rel-stack hash question is settled, see DECISIONS.md.

## Port fidelity result (the gate)

`scripts/compare_to_published.py` vs MLflow experiment 92, rel-f1 driver-position, best_val_mae:

| ep | published | candidate | gap |
|---|---|---|---|
| 1 | 5.8383 +/- 0.0137 | 5.8838 +/- 0.0468 | 0.78% |
| 2 | 5.9045 +/- 0.0450 | 5.9401 +/- 0.0375 | 0.60% |
| 3 | 4.2773 +/- 0.0209 | 4.2593 +/- 0.0071 | 0.42% |
| 4 | 4.5775 +/- 0.0571 | 4.5989 +/- 0.0368 | 0.47% |
| 5 | 5.2796 +/- 0.0398 | 5.2001 +/- 0.0692 | 1.50% |
| 6 | 4.7581 +/- 0.0691 | 4.7273 +/- 0.0307 | 0.65% |
| 7 | 4.7839 +/- 0.0472 | 4.7293 +/- 0.0338 | 1.14% |
| 8 | 5.4637 +/- 0.0232 | 5.4196 +/- 0.0419 | 0.81% |
| 9 | 3.4286 +/- 0.0459 | 3.4572 +/- 0.0559 | 0.83% |
| 10 | 3.2665 +/- 0.0438 | 3.2127 +/- 0.0414 | 1.65% |
| 11 | 3.1107 +/- 0.0550 | 3.0975 +/- 0.0191 | 0.42% |

**PASSED across the whole chain.** Every episode within 1.65%, most under 1%, all well inside
seed spread. Episode 1 is the strict gate (same window, same seeds, trained from scratch): 0.78%.
Published episode 11 is 3.1107 vs the paper's Table 2 Scratch MAE of 3.106 -- the chain reproduces.
Re-run cost: 55 runs, ~3.2 min each on one A100.

## Next action

1. Wire the three CL families into `continuous_learning.py` as new `--learning_mode` values
   (replay_reservoir, replay_herding, ewc), threading buffer/anchor state episode-to-episode
   the same way `weights_path` already is (`:497` -> `:474` -> `:200-201`).
2. Then parameter isolation + LwF, then launch Tier A (45 episodes, ~71 GPU-h, <1 day).
