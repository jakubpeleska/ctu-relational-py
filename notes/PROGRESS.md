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
| 0f. rel-stack + rel-amazon downloads | RUNNING (bg) |
| 0a. Commit the working tree | IN PROGRESS |
| 0d. Robustness fixes | TODO |
| 0c. Local multi-GPU runner | TODO |
| 0b. Port verification vs MLflow (GATE) | TODO |
| Phase 1: CL methods | TODO |

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

rel-hm 52/52 | rel-stack 18/18/17 | rel-ratebeer 12/12/12/9 | rel-f1 11/11/2 | rel-trial 7/7/7
rel-event 10/9 (corrupt DB span 1912->2222, usable w/ caveat) | rel-arxiv 3 (drop) | rel-avito 2 (drop)
rel-amazon: unknown, gated on download.

## Blocked / needs the user

- (none currently)

## Next action

Finish 0a (commit), then 0d robustness fixes, then 0c runner, then the 0b verification gate.
