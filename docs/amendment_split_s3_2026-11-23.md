# Split amendment S3 (2026-11-23): ~70/30 split with test-side type-pair coverage

**Status:** proposed at the project owner's request on 2026-11-23. Not yet adopted. Nothing in `data/` is rewritten. The manifest `split` column, `configs/experiment.yaml`, the 2026-11-14 freeze (`results/prespec/`), the CV folds (`results/cv/folds.json`) and all V1/V2 caches and results are unchanged.

## 1. Why

The owner wants a larger evaluation set, about 70/30. Evaluation subanalyses need enough TEST pools in which two rule types occur together. Training only needs enough data to fit the probes. The 2026-11-14 split (15 test groups) has a minimum pair coverage of **0**. Of the 98 compatible type pairs, 5 never co-occur in a TEST pool and 28 co-occur in only one.

## 2. The new rule

The related-task units, the balance score and the per-type checks are exactly those of `tasks.split_groups` (§2 of the earlier explanation: units from `RELATED_CLUSTERS`; at least 1 TEST and at least 5 TRAIN groups per type; score = L1 family-share gap + L1 type-share gap against all 72 groups). One check is added:

- **Pair coverage:** every **compatible** pair of instruction types must co-occur in the six-rule pool of at least `min_pair_test_groups` TEST groups. A pair is compatible when `source.conflicts` allows it: 105 pairs minus the 7 inside the mutually exclusive groups gives 98.

## 3. Feasibility (pool metadata only)

Pools are fixed per group, so coverage can only come from choosing groups. The checks below were run with an exact CP-SAT solver on pool types and families from `data/manifest/prompts.parquet`. No generations, labels or features were read.

| Requirement | Result |
|---|---|
| Pair coverage ≥ 3 | **Infeasible at any size.** `change_case:capital_word_frequency` + `length_constraints:number_words` occurs in only 2 of the 72 pools. With that pair exempt, the best achievable minimum at 22 or 24 test groups is still 2. |
| Pair coverage ≥ 2, free choice | Feasible from **20** test groups upward. 15 or 18 groups reach only 1. |
| Pair coverage ≥ 2, keeping the current 15 test groups and adding more | Not reachable at 22 (best minimum is 1). It needs 24 or more groups, and then only with the rare pair exempt. |
| Most pairs at ≥ 3 with ≥ 2 enforced, 22 groups | At most 76 of 98 (exact optimum). The proposed split, which optimises balance instead, has 66. |

So the threshold defaults to **2**. It is configurable in `configs/split_s3.yaml`.

## 4. Search

Random unit fills, as in `tasks.split_groups`, almost never satisfy pair coverage 2: at 22 groups, 0 of 20,000 draws did. `split_v3.split_groups_pair_coverage` therefore runs 1,000 seeded restarts (seed 20261123). Each restart is a random unit fill, then best-improvement swaps of equal-size units minimising (hard shortfall, balance score). Among admissible results the lowest score wins, and ties keep the earliest restart. Re-running reproduces the same split exactly (`tests/test_split_s3.py`).

## 5. Proposed split (`results/split_s3/proposed_split.json`)

- **Test (22 groups, 66 tasks, 16,236 prompts):** g03, g04, g09, g11, g17, g19, g21, g24, g27, g30, g34, g36, g37, g47, g48, g54, g56, g58, g62, g63, g65, g68.
- **Train (50 groups, 150 tasks, 36,900 prompts).**
- **Coverage:** minimum pair coverage 2, and 66 of 98 pairs reach at least 3 (`results/split_s3/pair_coverage.csv`). Each type is in 6 to 11 TEST groups and at least 10 TRAIN groups.
- **Balance score:** 0.093, compared with 0.139 for the 2026-11-14 split. 386 of the 1,000 restarts were admissible.
- **Change from 2026-11-14:** 6 old test groups stay in test (g03, g09, g30, g48, g63, g68). 9 move to train (g02, g08, g10, g14, g35, g41, g52, g55, g67). 16 train groups move to test.

## 6. Consequences if adopted (not done in this change)

- **Manifest:** rewrite the `split` column in the manifest files, or add a split-version overlay, and update `manifest_summary.json`. Row order and prompt IDs do not change.
- **CV:** rebuild the folds over the 50 new TRAIN groups and update `tests/test_no_test_leakage.py`.
- **Previously trained models:** the 16 groups that move to test were used for TRAIN-side CV and selection in V1 and V2. The owner accepted this on 2026-11-23 ("leakage does not matter right now"). Report it with any result.
- **V2 caches:** these are TRAIN-only (42,066 rows). The 9 groups that move to train have no V2 max-attention features (A1max); extracting them needs authorisation. The 16 groups that move to test must be dropped from the TRAIN caches.
- **Rare pair:** the `capital_word_frequency` + `number_words` pair appears in TRAIN in 0 pools, because both of its pools are in TEST. Per-type heads are unaffected, since each type keeps at least 10 TRAIN groups.
- **Pre-specification:** the 2026-11-14 freeze describes 15 test groups and would need a dated revision before any unblinding.
