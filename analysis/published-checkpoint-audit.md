# The published checkpoints are ~47% NaN-poisoned

**Status:** measured on RCI, 2026-09-07, job 11513722. Read-only audit of
`~/git/ctu-relational-py/logs` — the artefacts behind the submitted paper.

## What was found

| | |
|---|---|
| job directories holding checkpoints | 33 (of 103 total dirs) |
| checkpoints sampled | 45 |
| **containing non-finite weights** | **21 / 45 (47%)** |
| distinct parameter counts | 5,602,681 / 6,838,541 / 8,272,841 / 16,549,933 |

The poisoned tensors are exactly the ones the M1 diagnosis predicts — the numerical column encoders,
and nothing else:

```
9x  row_encoder.encoders.users.encoder.encoder_dict.numerical.weight
9x  encoder.encoders.users.encoder.encoder_dict.numerical.weight
7x  row_encoder.encoders.results.encoder.encoder_dict.numerical.weight
7x  encoder.encoders.results.encoder.encoder_dict.numerical.weight
4x  row_encoder.encoders.studies.encoder.encoder_dict.numerical.weight
```

`users`, `results` and `studies` are the entity tables of rel-amazon/rel-hm, rel-f1 and rel-trial.
The doubled names (`encoder.*` and `row_encoder.*`) are the duplicate registration also fixed in
b3bf3f7 — the same tensor stored twice per checkpoint. `8,272,841` matches the pre-fix parameter
count measured on potato exactly, so those are the continual-learning runs.

## Why "mixed" rather than "all"

Whether a checkpoint is poisoned depends on whether that dataset's numerical columns happen to hold
a missing cell, and on whether the sampler drew one before training ended. So the defect fires
per (dataset, task) and, within a chain, from whichever episode first samples a missing value.

**That is worse than a uniform bug, not better.** A defect that hit everything equally would be a
constant offset. This one silently removes a different subset of features from a different subset of
runs, so it adds uncontrolled variance across exactly the axes the paper compares.

## Consequences

1. **Do not re-score the published checkpoints to obtain forgetting metrics for the paper.** Roughly
   half describe models with dead feature columns, and the split is not along any line we control.
   The hoped-for saving -- reusing them instead of re-running the original regimes -- is not safe.
2. **They are strong evidence.** This is the M1 defect confirmed on the real published artefacts,
   independently of anything reproduced on potato, with the poisoned tensors matching the mechanism
   exactly. Worth reporting.
3. Re-running under the `na_strategy=MEAN` fix is the right call, and the published numbers should
   be described as "the previous protocol" rather than compared cell-for-cell.
