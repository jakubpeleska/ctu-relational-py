# Settled decisions

Append-only. One line per decision + the evidence that settled it.
Do not re-litigate anything here without new evidence.

## Verified corrections (2026-09-02)

- **Seeds ARE reproducible.** `random`/`np.random`/`torch` are seeded from `--seed` at
  `continuous_learning.py:365-367`, *before* `param_space` is built, so `tune.randint(0,1000)`
  draws deterministically. `--seed=42` always yields `[102, 435, 860, 270, 106]`.
  Real defect is only that effective seeds are opaque and unlogged. (Earlier "not reproducible"
  claim was WRONG.)

- **There is NO test-window leak.** `val_table = get_table(start=train_timestamp, end=val_timestamp)`
  (`continuous_learning.py:245`) is a half-open window `[splits[i], splits[i+1])`; `get_table` masks
  with `< end` (`continuous_task.py:30`). On the last episode that window is
  `[val_timestamp, test_timestamp)` = the NATIVE VALIDATION WINDOW. `val_timestamp` is the exclusive
  END bound, not the start. Data at/after `test_timestamp` is never used in training or selection.
  (Earlier "test leak" claim was WRONG — withdrawn.)

- **`develop` did NOT break `ComposedLoader`.** `origin/rci`'s `rnd_uni` branch is byte-identical to
  today's. Commit `6ae3489 "Simplify composed loader"` did not regress it.

- **Upsampled was NEVER 50/50.** `rnd_uni` truncates the epoch to `min(len)*n_loaders` but builds the
  draw order by `repeat_interleave` over the FULL lengths, so each loader's share is proportional to
  its own size. Measured new-share: 50% @100:100, 10.5% @100:900, 5.2% @50:1000.
  Decisive: `origin/rci` shipped an unused `rnd_weighted` mode (`torch.multinomial`, replacement=True)
  that WOULD have given 50/50 — but `continuous_learning.py:242` calls `rnd_uni`. Intent existed,
  never wired up. See `analysis/upsampling-ratio-finding.md`.

- **GPUs are healthy** (2026-09-02): 4x A100-SXM4-40GB, torch 2.9.1+cu128,
  `torch.cuda.is_available() == True`, `device_count() == 4`. The prior session's NVML failure has
  cleared. Earlier smoke run's "0 GPUs" was just `--num_gpus` defaulting to 0, not a broken build.

## Design decisions

- **Model selection: METRIC-based** (`val_{tune_metric}`), not loss-based. Evidence: MLflow logs
  `val_roc_auc`/`best_val_roc_auc` for the published runs, and `val_loss_epoch` is NEVER logged
  because `validation_step` computes no loss (`entity_wrapper.py:101-113`). HEAD's loss-based
  monitor could never have resolved. Do not mix.

- **Upsample mixing: keep `rnd_uni` published runs as the documented baseline, add `weighted` as the
  swept rehearsal family in Phase 1.** Satisfies reviewer CMu9's ratio-sweep commitment without
  invalidating existing numbers, and folds the fix into the CL-methods work instead of a separate re-run.

- **Datasets committed:** Tier A (rel-f1, rel-trial), B (rel-hm), C (rel-stack), D (rel-ratebeer).
  Dropped: rel-avito (2 episodes), rel-arxiv (3 episodes). Gated on download: rel-amazon.

- **Compute:** local 4x A100 + remote Slurm server with ~4 more => plan on ~8 GPUs.
