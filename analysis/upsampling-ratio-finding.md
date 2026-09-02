# The Upsampled regime's new-data share is not 50%

**Status:** confirmed by measurement, 2026-09-01. Affects §5.2 and Appendix A.1 of the
submitted paper, and Reviewer CMu9's Q4.

## What the paper claims

> "For the *Finetune (Upsampled)* regime, a composed data loader samples mini-batches
> equally (50% probability) from the newest data increment and the historical data."
> — §5.2, repeated in Appendix A.1

`experiments/continuous_learning/continuous_learning.py:242` carries the same intent in a
comment: *"old data and sample from them with 0.5 probability each"*.

## What the code does

That line constructs `ComposedLoader({"new": ..., "old": ...}, mode="rnd_uni")`.

In `rnd_uni` (`redelex/loaders/composed_loader.py`), the epoch length is truncated to
`min(len)*n_loaders`, but the draw order is built by `repeat_interleave` over the **full**
loader lengths. Each loader's share of an epoch is therefore proportional to its own size,
not uniform.

Measured over 20 epochs per configuration with length-only stand-in loaders:

| new : old batches | `rnd_uni` new-share | `minimum` new-share |
|---|---|---|
| 100 : 100 | 50.0% | 50.0% |
| 100 : 300 | 25.5% | 50.0% |
| 100 : 900 | 10.5% | 50.0% |
|  50 : 1000 |  5.2% | 50.0% |
| 200 : 1800 |  9.9% | 50.0% |

The stated 50% holds only when the increment and the history happen to be the same size —
i.e. roughly the first episode. Thereafter the history loader grows while the increment
loader does not, so the new-data share decays toward a few percent.

## Why it matters

1. **The described method was not the method run.** The regime's defining hyperparameter
   was never 0.5 in any episode after the first.
2. **It explains a published result.** As the new-data share decays, Upsampled converges
   behaviourally toward Cumulative fine-tuning — both end up training mostly on history.
   The paper's difficulty separating those two regimes is the expected consequence.
3. **The effect is time-varying, not a constant offset.** The mixing ratio drifts across
   episodes as a side effect of history growth, so it confounds exactly the temporal
   comparison the paper is about.
4. **It blocks a rebuttal commitment.** CMu9's Q4 asked whether the 50% was cross-validated
   and we promised a sweep. A sweep over a nominal parameter that the loader ignores would
   be meaningless.

## Why the tests did not catch it

`tests/test_loaders.py::test_composed_loader_rnd_uni` asserted only
`counts["a"] <= 5 and counts["b"] <= 8` — upper bounds satisfied by almost any split,
including 0/13.

## What has been done

- Added `mode="weighted"` with explicit per-loader `weights`, giving a controlled ratio and
  the longest epoch that does not exhaust any loader. `minimum` is its uniform special case.
- Added regression tests pinning the realized ratios of `rnd_uni` (proportional),
  `minimum` (uniform), and `weighted` (as configured).
- **Not changed:** `continuous_learning.py:242` still uses `rnd_uni`. Switching it alters
  experiment semantics and is a deliberate decision, not a bug fix to apply silently.

## Decision required

Either (a) switch the regime to `weighted`/`minimum` so it matches the paper's description
and re-run, reporting the ratio as a swept hyperparameter; or (b) keep `rnd_uni`, correct
the paper to describe proportional mixing, and note that the effective new-data share decays
across episodes. Option (a) is preferable: it honours the CMu9 commitment and makes
Upsampled a genuine rehearsal baseline rather than a slow drift into Cumulative.
