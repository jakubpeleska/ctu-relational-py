# Validation dominates runtime, and its frequency is an artifact

**Status:** measured on A100, 2026-09-03/04. Two calibration runs, 2 increments each,
`PhaseTimerCallback` on the *old* protocol.

## What was measured

| task | ep | train s | val s | **val share** | val passes | val batches |
|---|---|---|---|---|---|---|
| rel-trial site-success | 1 | 112.09 | 393.05 | **77.8%** | **111** | 6,771 |
| rel-trial site-success | 2 | 112.32 | 165.55 | 59.6% | **25** | 2,700 |
| rel-stack user-badge | 1 | 133.12 | 239.93 | 64.3% | 20 | 3,960 |
| rel-stack user-badge | 2 | 131.97 | 323.73 | **71.0%** | 20 | 5,720 |

Three things fall out:

1. **The fixed step budget works exactly as designed.** Training time is constant within a dataset
   (112.09 vs 112.32 s; 133.12 vs 131.97 s) regardless of how much history the episode holds. This
   is the argument for keeping it, and it should be stated in the paper.
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
