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

- **rel-stack: always call with `download=False`. Do NOT override the SHA256 pin.**
  relbench 2.1.1 pins `f1374fda...` for `rel-stack/db.zip` but the server now serves a file
  hashing to `5a97bf65...`. Verified NOT corruption: pooch's download and an independent curl
  produced the identical `5a97bf65...` digest, i.e. upstream republished without refreshing the
  pin (same class of bug as `rel-event/user-ignore`). The db.zip was extracted manually into
  `~/.cache/relbench/rel-stack/db/` and loads fine (span 2009-02-02 -> 2023-09-03).
  A registry-override helper was written and then DELIBERATELY REMOVED: disabling an integrity
  check is not worth the convenience. Task archives under `rel-stack/tasks/` hash correctly and
  download normally; only the dataset call needs `download=False`.

## Evaluation protocol change (2026-09-03)

- **Validation is now step-based, not epoch-based.** `val_check_interval=100` with
  `check_val_every_n_epoch=None` gives exactly 20 validations per run everywhere.
  **Why:** `limit_train_batches=100` resolves to `min(100, len(train_loader))`, so short early
  episodes had short epochs and validated far more often. Measured from published MLflow runs:
  rel-f1 driver-position validated **500** times at episode 1 and **33** at episode 11.
  Validation drives SaveModelCallback AND ReduceLROnPlateau(patience=3, interval="epoch"), so the
  LR collapsed on small episodes and barely decayed on large ones - a confound scaling with episode
  size, i.e. along the exact temporal axis the paper studies.

- **LR scheduler moved to `interval="step", frequency=100`** so patience=3 means 300 optimiser
  steps in every run.

- **Validation window capped at 25,000 rows** by uniform subsample, seeded on
  `(dataset, task, increment)` and NEVER on the trial seed, so every method and seed selects
  against an identical evaluation set. Helper: `subsample_val_table` in
  `experiments/continuous_learning/utils.py`.
  **Why not `limit_val_batches`:** the val loader is `shuffle=False`, so capping batches would keep
  only the temporally earliest rows - a biased subsample.
  **Why this is safe:** reported metrics never come from this set. They come from
  `run_predictions.py` re-scoring every checkpoint over the full timeline; the notebook computes the
  paper's tables from `full_table` predictions. The subsample drives model selection and LR only.
  **Cost:** validation was 15.1x training on rel-stack (30,100 vs 2,000 batches), 13.2x on
  rel-amazon, 8.2x on rel-hm. Capping cuts validation work ~7.7x on rel-stack.

- **All three are CLI flags** (`--val_check_interval`, `--val_max_rows`, `--max_training_steps`);
  passing 0 restores the old behaviour so it stays reachable as a control.

- **Comparability:** switched wholesale. The port gate passed under the OLD protocol, so it must be
  re-earned on rel-f1 under the new one. Expect a real shift - the fix deliberately removes the LR
  collapse on small episodes.

## Hyperparameters (2026-09-03)

- **They were never chosen.** Every model constant was written in commit `a15a5f8` (2026-04-22) and
  never revisited; no rationale documented anywhere. Copied from
  `experiments/universal_encoder/universal_encoder_supervised.py` (identical values AND key naming).
  Against the repo's own sweep (`experiments/original/dbgnn_hyperparams.py:296-315`):
  `gnn_channels=128` is **outside** its range (that sweep fixed 64, commented space topped at 64);
  `batch_size=128` vs its 512; **`lr` was never searched anywhere** - always a literal, with
  `tune.choice([0.001,0.005])` commented out. Only `num_neighbors` and `gnn_layers` have any tuning
  precedent, and that was for the non-continual setting.
- **`head_norm` was dead config** - in param_space and logged to MLflow, never passed to the model.
  Now passed. Same value, so no results change.
- **`out_channels` was never passed** either; always 1. Correct for every task in the grid (all
  binary/regression) but would have silently broken multiclass. Now derived from task type.
- **Learning rate is now logged** (`LearningRateMonitor`). It was logged in NO published run, which
  is why the LR collapse went unnoticed.
- Row encoder still runs at relbench defaults (`{"channels":128,"num_layers":4}`) - a 4-layer ResNet
  feeding a 2-layer GNN, never chosen. Relevant to CMu9's encoder-vs-GNN attribution.

## Method roster (2026-09-03) - 7 modes

`from_scratch`, `ft_full`, `ft_newonly`, `er_reservoir`, `der_pp`, `ewc`, `lwf`.

- **Dropped `ft_upsample`**: a defective Experience Replay with an uncontrolled mixing ratio,
  subsumed by ER with an explicit ratio.
- **Dropped `replay_herding`**: iCaRL herding approximates *class* means; every task here is binary
  or regression in a domain-incremental setting, so the motivation does not transfer. Code retained.
- **Added DER++**: literature recommends DER/DER++ as the starting baseline specifically for
  Domain-IL, which is what this setting is.
- **Rejected A-GEM/GEM**: gradient-projection methods are reported as less effective than replay in
  Domain-IL, and ER already outperforms A-GEM.
- **Parameter isolation excluded on principle**: it needs task identity at test time and allocates
  disjoint capacity per task, but this is domain-incremental (one task, drifting distribution).
  PackNet also leaves 0.02% of the network free by episode 12, the median chain length here.
  Write it up as a finding, not a gap.
