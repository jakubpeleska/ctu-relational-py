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

- **GPUs are healthy** (2026-09-02) **[NO LONGER TRUE as of 2026-09-06 - see below]**:
  4x A100-SXM4-40GB, torch 2.9.1+cu128,
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
  **[SUPERSEDED 2026-09-06: 15.1x is the *median episode*, not the chain. Per chain the honest
  figure is 5.9x - `analysis/evaluation-cost-finding.md`. Do not quote 15.1x in the paper.]**

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
  **[CORRECTED 2026-09-06: passing it changed nothing - it is still inert. See "Corrections to
  earlier verified lines" below. Do not read this bullet as "fixed".]**
- **`out_channels` was never passed** either; always 1. Correct for every task in the grid (all
  binary/regression) but would have silently broken multiclass. Now derived from task type.
- **Learning rate is now logged** (`LearningRateMonitor`). It was logged in NO published run, which
  is why the LR collapse went unnoticed.
- Row encoder still runs at relbench defaults (`{"channels":128,"num_layers":4}`) - a 4-layer ResNet
  feeding a 2-layer GNN, never chosen. Relevant to CMu9's encoder-vs-GNN attribution.

## Method roster (2026-09-03) - 7 modes

**[STALE 2026-09-06: the shipped roster is 8 modes under different names, and it INCLUDES parameter
isolation. Take the roster from `cl_modes.DEFAULT_ROSTER`, not from this section - details below.]**

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

## Corrections to earlier "verified" lines (2026-09-06)

Every entry below was re-checked by running the code, not by re-reading the note. The original
lines are left in place with a pointer rather than rewritten: a decisions file that quietly edits
its own history stops being evidence. What is corrected here is what a later session would
otherwise be told to trust.

- **`head_norm` is STILL INERT. The Hyperparameters bullet "Now passed" reads as fixed and is not.**
  The head is `MLP(in_channels=gnn_channels, out_channels=out_channels, norm=head_norm,
  num_layers=1)` (`experiments/continuous_learning/models.py:78-83`), and PyG's `MLP` places a norm
  after each *hidden* layer only. A 1-layer MLP has no hidden layer, so the argument is discarded
  whatever its value. Verified: `MLP(in_channels=128, out_channels=1, norm='batch_norm',
  num_layers=1)` gives `norms == []` and state keys `['lins.0.bias', 'lins.0.weight']`, identical to
  `norm=None` and to `norm='layer_norm'`, and the outputs are bit-equal for a fixed seed.
  So the parameter went from dead config to *live config wired to a no-op*: it is threaded to the
  model (`continuous_learning.py:476`) and logged to MLflow as a hyperparameter that cannot affect
  anything. "Same value, so no results change" was true, but for the wrong reason - it would still
  change nothing at a different value.
  **Not fixed here, because it is a behaviour change, not a doc fix.** The two honest options are
  to drop `head_norm` from `param_space` and the MLflow params, or to give the head
  `num_layers >= 2` with an explicit `hidden_channels` and re-earn the port gate.

- **`freeze_extend` does NOT give zero forgetting.** The phrase "zero forgetting by construction"
  had spread to `redelex/continual/adapters.py` and `cl_modes.py` and is now removed from both.
  `AdapterStack.forward` chains every adapter onto every input and this is domain-incremental, so
  there is no task identity to route an old input around later adapters: training episode t's
  adapter changes the function on episodes 1..t-1 while every one of their parameters stays
  bit-identical. Measured on the real model: episode-1 predictions moved 151% of their own
  magnitude after episode 3's adapter, backbone unchanged. Pinned by
  `tests/test_continual_adapters.py::test_later_episodes_still_move_the_function_on_earlier_ones`
  (toy reproduction, 0.857 relative drift; verified to fail if inputs are routed per episode).
  Every *mechanical* property claimed for the stack does hold: identity at insertion, optimiser
  scoping to the newest adapter, freeze invariant across save/load. The correct claim is
  "previously learned parameters are never overwritten; forgetting is bounded by adapter capacity
  rather than zero".

- **The fixed step budget is not the neutral control it was written up as.**
  `analysis/evaluation-cost-finding.md` used to argue *for* keeping it; that conclusion is
  withdrawn in place (the measurement it rests on is fine, the inference was not). A constant
  `max_training_steps x batch_size = 256,000` examples means a mode training on the increment gets
  up to ~52x more passes over its own training set than a mode training on all history, at rel-hm
  episode 52. See that file for the per-dataset table.

- **"GPUs are healthy (2026-09-02)" is a dated observation, not a standing fact.** As of 2026-09-06
  the container's device cgroup denies `/dev/nvidia*` (EPERM on world-writable nodes,
  `/dev/nvidia0` absent) and `torch.cuda.is_available()` is False. See `notes/PROGRESS.md`.

- **The roster section is stale in three ways.** The code ships 8 modes -
  `from_scratch, joint, naive, er, der_pp, ewc, lwf, freeze_extend` (`cl_modes.DEFAULT_ROSTER`).
  (1) `ft_full`/`ft_newonly` are now aliases of `joint`/`naive`, and `er_reservoir` is `er`.
  (2) "Parameter isolation excluded on principle" is contradicted by `freeze_extend` being in the
  default roster; it is in, and the finding to write up is the measured drift above rather than the
  exclusion argument. (3) `ft_upsample` was not dropped - it is retained as a reproduction-only
  mode (`legacy=True`) and is excluded from the roster rather than from the codebase.

- **DER++ here is a single-draw variant with `beta` implicitly 1.** Buzzega et al. draw `x'` and
  `x''` independently; the mixed loader emits one buffer batch at a time and the wrapper applies the
  task loss to every batch, so one batch carries both the alpha-weighted distillation and the replay
  task loss at weight 1. There is no `--der_beta`. Documented in `cl_modes.make_der_penalty`; the
  paper's method description must match it.

- **Line numbers in the 2026-09-02 and 2026-09-03 sections have drifted; the claims still hold.**
  Re-checked: the val window is built at `continuous_learning.py:614` (not `:245`) and is still
  `[splits[i], splits[i+1])` against a `< end` mask (`continuous_task.py:29-31`), so there is still
  no test-window leak; `validation_step` is at `entity_wrapper.py:120` (not `:101-113`) and still
  computes no loss, so metric-based selection still stands. The seed bullet's *mechanism* is stale -
  seeds are no longer drawn by `tune.randint` inside the episode loop but eagerly at
  `continuous_learning.py:850-860` - while its conclusion is unchanged and now stronger:
  `--seed=42` still yields `[102, 435, 860, 270, 106]`, and resuming can no longer shift the draw.

## Stay on relbench 2.1.1 (2026-09-07)

**Decision: do NOT upgrade to relbench 3.0.1.** Confirmed by Jakub.

3.0.1 is a breaking rewrite, not a point release. `relbench.datasets` and `relbench.tasks` no longer
exist; the package is now `hf.py` / `load.py` / `manifest.py` / `schema.py` / `submit.py` with a
single `load_dataset()` backed by HuggingFace. **Our repo has 36 call sites** using the 2.x API.

It very likely fixes the stale `rel-stack/db.zip` SHA256 (the pinned-hash problem disappears once
data moves to HF hosting) and may fix rel-amazon too. That is not worth taking 9 days from the
deadline, because it would require: rewriting the data layer, re-verifying that task definitions and
splits are unchanged (they gate every episode count in `analysis/dataset-episodes-measured.md`), and
re-earning the port-fidelity gate. RelArena also pins `relbench==2.1.2`, so a 3.0.1 upgrade would
fight the RelGNN/RelGT integration.

The two workarounds we carry instead are one line each and both are tested:
- **rel-stack: `download=False`.** Upstream republished `db.zip` without refreshing the pin.
- **rel-amazon: `download=True`.** It CANNOT be built from raw -- `make_db()` fetches
  `https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Books.json.gz`, which
  UCSD removed and now 404s. `download=True` fetches RelBench's prepared DB instead.

## The 4-GPU limit is enforced by Slurm, not by us (2026-09-07)

`scripts/submit_rci_grid.py` deals work into 4 lanes and reads `squeue` to see which are busy. An
earlier version REFUSED to submit when `squeue` was unreachable, on the grounds that two submissions
could then hold 8 GPUs. Jakub corrected this: **the Slurm account is capped at 4 concurrent GPUs
regardless**, so an over-submission queues rather than over-runs.

So an unreadable `squeue` costs latency -- work dealt into a lane that is actually busy waits behind
it -- not quota. It now warns and continues. Do not "fix" this back into a refusal.
