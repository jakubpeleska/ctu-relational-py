# Measured episode counts per dataset

**Status:** measured, 2026-09-02. Supersedes every metadata-derived ceiling.

`n_episodes = len(get_splits()) - 2`, matching the loop
`for i in range(start_inc=1, len(splits) - 1)` in `continuous_learning.py`.
Counts come from running `ContinuousWrapper.get_splits()` on the real data, and
where the dataset was in the published grid they are **independently confirmed**
by the increment range of the 2,371 runs on MLflow (experiments 90-93).

| Dataset | Task | Episodes | Δw | rows/episode (median) | rows total |
|---|---|---|---|---|---|
| rel-hm | user-churn | **52** | 7 d | 71,643 | 3,878,451 |
| rel-hm | item-sales | **52** | 7 d | 105,542 | 5,488,184 |
| rel-stack | user-engagement | **18** ✓ | 91 d | 85,838 | 1,436,114 |
| rel-stack | post-votes | **18** ✓ | 91 d | 155,009 | 2,591,061 |
| rel-stack | user-badge | **17** ✓ | 91 d | 192,556 | 3,595,917 |
| rel-amazon | user-churn | **15** | 91 d | 169,156 | 5,049,718 |
| rel-amazon | user-ltv | **15** | 91 d | 169,156 | 5,049,718 |
| rel-amazon | item-churn | **15** | 91 d | 127,602 | 2,643,050 |
| rel-amazon | item-ltv | **16** | 91 d | 138,250 | 2,842,509 |
| rel-ratebeer | beer-churn | 12 | 90 d | 149,662 | 2,546,599 |
| rel-ratebeer | user-churn | 12 | 90 d | 20,082 | 391,115 |
| rel-ratebeer | user-count | 12 | 90 d | 20,082 | 391,115 |
| rel-ratebeer | brewer-dormant | 9 | 365 d | 9,655 | 112,885 |
| rel-f1 | driver-position | **11** ✓ | 60 d | ~690 | 7,507 |
| rel-f1 | driver-dnf | **11** ✓ | 30 d | ~990 | 11,204 |
| rel-f1 | driver-top3 | **2** ✓ | 30 d | ~555 | 1,111 |
| rel-trial | study-outcome | **7** ✓ | 365 d | 1,837 | 12,580 |
| rel-trial | study-adverse | **7** ✓ | 365 d | 7,034 | 45,842 |
| rel-trial | site-success | **7** ✓ | 365 d | 25,826 | 168,903 |
| rel-event | user-attendance | 10 | 7 d | 1,629 | 20,948 |
| rel-event | user-repeat | 9 | 7 d | 285 | 3,994 |
| rel-arxiv | all 3 tasks | 3 | 182 d | — | — |
| rel-avito | all 3 tasks | 2 | 4-6 d | — | — |

✓ = independently confirmed against the published MLflow increment range.

## Metadata ceilings were wrong by up to 2.5x

| Dataset | Ceiling from metadata | Measured | Error |
|---|---|---|---|
| rel-trial | ~19 | **7** | 2.7x high |
| rel-stack | ~45 | **17-18** | 2.5x high |
| rel-amazon | ~30 | **15-16** | 2x high |
| rel-hm | ~52 | **52** | correct |
| rel-f1 | ~10 | **11 / 11 / 2** | mostly correct |

The gap is `get_splits`'s 10%-of-val-window row filter
(`continuous_task.py:59-72`), which drops sparse early episodes. Ceilings assume
every `val_delta`-wide window survives; in practice early history is thin.

**Do not plan compute from ceilings.** Measure first.

## Notes

- **rel-f1 `driver-top3` really does yield 2 episodes**, despite sharing
  `timedelta=30d` with `driver-dnf` (11). Not a truncated run -- probed directly,
  the two episodes hold 523 and 588 rows. Different tasks have different target
  rows, hence different unique timestamps and a different filter outcome.
- **rel-event has a corrupt DB span** (`1912-01-01` -> `2222-02-02`, sentinel
  garbage) and `user-ignore` fails an upstream SHA256 check inside relbench.
  Splits still work because they derive from task-table anchors rather than the
  DB range, so it is usable with a caveat.
- **rel-stack requires `download=False`** -- upstream republished `db.zip`
  without refreshing relbench's pinned hash. See `notes/DECISIONS.md`.
- **`rel-stack/badges-class` yields 41 episodes** despite being an AutoComplete
  task with `timedelta = 1 second`, and it built without the feared per-second
  materialisation. Out of scope for v1, but worth revisiting: AutoComplete may
  suit a fine-grained protocol better than assumed.
