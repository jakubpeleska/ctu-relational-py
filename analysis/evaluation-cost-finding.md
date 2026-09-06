# Validation dominates runtime, and its frequency is an artifact

**Status:** measured on A100, 2026-09-03/04. Two calibration runs, 2 increments each,
`PhaseTimerCallback` on the *old* protocol.

**Correction, 2026-09-06.** Every measurement in this file stands. The *conclusion* drawn in
point 1 - that the constant training time is an argument for keeping the fixed step budget - is
withdrawn. See "What the constant training time actually buys" at the end.

## What was measured

| task | ep | train s | val s | **val share** | val passes | val batches |
|---|---|---|---|---|---|---|
| rel-trial site-success | 1 | 112.09 | 393.05 | **77.8%** | **111** | 6,771 |
| rel-trial site-success | 2 | 112.32 | 165.55 | 59.6% | **25** | 2,700 |
| rel-stack user-badge | 1 | 133.12 | 239.93 | 64.3% | 20 | 3,960 |
| rel-stack user-badge | 2 | 131.97 | 323.73 | **71.0%** | 20 | 5,720 |

Three things fall out:

1. **The fixed step budget holds wall-clock training time constant.** Training time is flat within a
   dataset (112.09 vs 112.32 s; 133.12 vs 131.97 s) regardless of how much history the episode
   holds - the budget does what it was built to do.
   ~~This is the argument for keeping it, and it should be stated in the paper.~~
   **WITHDRAWN 2026-09-06:** constant cost is not a controlled comparison. Last section.
2. **Validation is 60-78% of wall-clock**, and it was never bounded.
3. **The frequency artifact is real**: rel-trial validated **111** times at episode 1 and **25** at
   episode 2. Same task, same run, same step budget. Cause: `limit_train_batches=100` resolves to
   `min(100, len(train_loader))`, so a short early history means short epochs.

A validation batch costs about the same as a training batch (rel-trial 0.058 vs 0.056 s; rel-stack
0.060 vs 0.067), so batch counts translate to time almost directly.

## Projected effect of the fix, per chain

20 validation passes everywhere, validation window capped at 25k rows (196 batches). Computed from
real per-episode row counts, counting total batches over a whole episode chain:

| task | episodes | old batches | new batches | **speed-up** |
|---|---|---|---|---|
| rel-amazon user-churn | 15 | 819,180 | 88,800 | **9.2x** |
| rel-stack user-badge | 17 | 596,040 | 100,640 | **5.9x** |
| rel-hm item-sales | 52 | 962,000 | 307,840 | **3.1x** |
| rel-trial site-success | 7 | 41,253 | 36,160 | **1.1x** |

**The saving grows with episode index**, because the validation window grows with the episode while
the cap does not: rel-amazon episode 1 gains 2.1x, episode 15 gains 11.2x.

**Correction to an earlier estimate.** A previous note quoted "15x validation overhead" for
rel-stack from its *median* episode. Per-chain the honest figure is 5.9x, and rel-trial - whose
windows are already small - gains almost nothing. The fix is still worth making there: rel-trial's
gain is not speed but removing the 111-vs-25 validation-frequency confound, which is the
methodological half of the problem.

## Grid cost, recomputed from measured per-batch time

At ~0.065 s/batch, 7 modes x 5 seeds: rel-trial ~68 GPU-h, rel-stack ~191, rel-hm ~389,
rel-amazon ~224, rel-ratebeer ~150, rel-f1 ~15. **Total ~1,040 GPU-h**, about 10.8 days on the four
local GPUs or 5.4 days on eight. Down from the ~2,720 GPU-h estimated before the fix.

**[Stale mode count, 2026-09-06: `cl_modes.DEFAULT_ROSTER` ships 8 modes, not 7. Cost per mode is
uniform under the fixed step budget, so scale by 8/7: ~1,190 GPU-h. The per-dataset figures above
are per-7-modes.]**

## What the constant training time actually buys (correction, 2026-09-06)

The measurement is sound and is kept: `max_training_steps=2000` at `batch_size=128` fixes every run
at 256,000 training examples, so training wall-clock is flat across episodes and across modes. The
inference drawn from it was not. A constant number of *examples* is not a constant amount of
*learning* when the size of the training set is precisely what differs between the modes being
compared, and grows with the episode index.

Passes a mode gets over its own training set = `256,000 / rows in that mode's window`. Row counts
from `analysis/dataset-episodes-measured.md` (median rows/episode for the increment, total rows for
the history), so these are representative rather than exact for one episode:

| task | episodes | `naive` rows | its passes | `joint` / `from_scratch` rows | its passes | ratio |
|---|---|---|---|---|---|---|
| rel-hm item-sales | 52 | 105,542 | 2.43 | 5,488,184 | **0.05** | **52x** |
| rel-hm user-churn | 52 | 71,643 | 3.57 | 3,878,451 | **0.07** | 54x |
| rel-stack user-badge | 17 | 192,556 | 1.33 | 3,595,917 | **0.07** | 19x |
| rel-trial site-success | 7 | 25,826 | 9.91 | 168,903 | 1.52 | 6.5x |
| rel-f1 driver-position | 11 | ~690 | 371 | 7,507 | 34 | 11x |

Three consequences:

1. **The budget starves the data-hungry modes.** On rel-hm and rel-stack the full-window modes never
   complete a tenth of one pass: most of their training rows are never sampled even once, while
   `naive` sees its rows two or three times over. Any gap the paper reports between `joint` and
   `naive` at high episode index is therefore partly a budget artifact and not only a CL result.
2. **The starvation is confounded with the episode index** - the axis the paper is about. Passes for
   a full-window mode fall roughly as `1/i`, so the artifact grows monotonically along exactly the
   dimension being studied. Constant wall-clock guarantees this; it does not excuse it.
3. **It is already visible in the data.** Reported from the measurement that prompted this
   correction: on rel-stack, `from_scratch` degrades relative to the warm-started modes at the
   episodes where its budget drops below one pass over its own training set. Not re-derived in this
   file - it is the first thing to confirm when the grid runs.

**What to do about it.** The fixed budget has to be reported as a constraint of the study, not
defended as a control:

- *Cheapest, no re-run:* state the budget in the protocol section and carry a passes-over-data
  column alongside the results table, so a reader can see which cells are budget-limited.
- *Correct but unaffordable now:* scale steps with training-set size (a fixed number of epochs).
  That restores learning parity and destroys wall-clock parity, multiplying grid cost by roughly the
  mean history-to-increment ratio - 52x in the worst case on rel-hm. Not possible before the
  deadline.
- *The compromise worth doing:* run one dataset both ways as a budget-sensitivity arm. rel-trial is
  the candidate: 7 episodes, and `joint` already gets 1.5 passes there, so an epoch-based arm is
  cheap and directly measures how much of the `joint`-vs-`naive` gap the budget is producing.
