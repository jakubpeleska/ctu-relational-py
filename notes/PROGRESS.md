# PROGRESS — CL Benchmark Reframe (ICLR 2027)

**Entry point for any new session.** Read this, then `notes/DECISIONS.md`.
Plan: `~/.claude/plans/happy-twirling-lantern.md`

Target: ICLR 2027, ~3 weeks from 2026-09-02. Anytime-shaped: every phase leaves a submittable paper.
Priority: (1) integrate real CL techniques, (2) add datasets or rigorously argue exclusions.

---

## Current phase: Phase 0 — finish the foundation

| Step | Status |
|---|---|
| Cross-session persistence (`notes/`) | DONE 2026-09-02 |
| 0a. Commit the working tree | DONE - 6 commits, tree clean |
| 0d. Robustness fixes | DONE - seeds/mlflow_uri/trial-tolerance |
| 0f. rel-stack download + probe | DONE - 53 episodes, use download=False |
| 0f. rel-amazon download + probe | DONE - 61 episodes, span 2008-2018 |
| 0b. Port verification vs MLflow (GATE) | PASSING - ep1 gap 0.78%, ep 8-11 still running |
| 0c. Local multi-GPU runner | DONE - scripts/run_grid.py |
| Phase 1: CL metrics (ACC/BWT/FWT/forgetting + decay) | DONE - redelex/continual/metrics.py |
| Phase 1: Replay family (reservoir + herding) | DONE - redelex/continual/replay.py |
| Phase 1: Regularisation family (EWC) | DONE - redelex/continual/regularization.py |
| Phase 1: Wire CL families into the experiment script | TODO - next |
| Phase 1: Parameter isolation family | TODO |
| Phase 1: LwF distillation | TODO |
| Phase 2: HeteroGAT backbone seam, dI ablation | TODO |

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

Episode 1 is the gate (same window, same seeds, from scratch). PASS.
Published episode 11 is 3.1107, essentially the paper's Table 2 Scratch MAE of 3.106.

## Next action

1. Wire the three CL families into `continuous_learning.py` as new `--learning_mode` values
   (replay_reservoir, replay_herding, ewc), threading buffer/anchor state episode-to-episode
   the same way `weights_path` already is (`:497` -> `:474` -> `:200-201`).
2. Then parameter isolation + LwF, then launch Tier A (45 episodes, ~71 GPU-h, <1 day).
