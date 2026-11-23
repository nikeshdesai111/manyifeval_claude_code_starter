# Split amendment S3 (2026-11-23): 70/30 split with type-pair coverage in test and train

**Status: adopted by the project owner on 2026-11-23.** The probing code now uses the S3 split, selected in `configs/active_split.yaml`. The frozen manifest files are not rewritten: their `split` column still records the 2026-11-14 split, and `mife.splits` overlays the S3 group assignment whenever `mife.probing.data` reads the manifest. `configs/experiment.yaml`, the 2026-11-14 pre-specification (`results/prespec/`), the V1 folds (`results/cv/folds.json`) and every V1/V2 cache and result are left as they are. They are historical artifacts of the 2026-11-14 split.

## 1. Why

The owner wants a larger evaluation set, about 70/30. Evaluation subanalyses need TEST pools in which two rule types occur together, and each pair should also be seen in training. Training only needs enough data to fit the probes. The 2026-11-14 split (15 test groups) has a minimum TEST pair coverage of **0**. Of the 98 compatible type pairs, 5 never co-occur in a TEST pool and 28 co-occur in only one.

## 2. Rules

Unchanged from `tasks.split_groups`:
- Related-task units (`RELATED_CLUSTERS`) stay on one side.
- Every type is in at least 1 TEST and at least 5 TRAIN groups.
- The balance score is the L1 family-share gap plus the L1 type-share gap against all 72 groups. Lower is better.

Added (`configs/split_s3.yaml`):
- **Test pairs:** every compatible pair of types (98 = 105 minus the 7 inside the mutually exclusive groups) co-occurs in the six-rule pool of at least `min_pair_test_groups = 2` TEST groups.
- **Train pairs:** every compatible pair co-occurs in at least `min_pair_train_groups = 1` TRAIN group.
- **Exception (owner's decision, 2026-11-23):** `change_case:capital_word_frequency` + `length_constraints:number_words` occurs in only 2 of the 72 pools, so it cannot meet both minima. With `exception_priority: test` it keeps 2 TEST groups and has 0 TRAIN groups. It is the only pair that needed an exception.

## 3. Feasibility (pool metadata only)

These checks used an exact CP-SAT solver on pool types and families from `data/manifest/prompts.parquet`. No generations, labels or features were read.

| Requirement | Result |
|---|---|
| Test pair coverage ≥ 3 | Infeasible at any size (the rare pair above). Even with it exempt, the best minimum at 22 or 24 test groups is 2. |
| Test pair coverage ≥ 2 | Feasible from 20 test groups upward. Not reachable at 22 if the 15 old test groups must stay in test. |
| Pairs at ≥ 3 in test, with ≥ 2 enforced, 22 groups | At most 76 of 98 (78 if related units are relaxed). |

## 4. Search

Plain random unit fills, the method of `tasks.split_groups`, never satisfy these constraints: at 22 groups, 0 of 20,000 draws did. `split_v3.split_groups_pair_coverage` therefore makes 20,000 seeded random draws (seed 20261123), each a random unit fill like the original. Each draw is then improved by best-improvement swaps of equal-size units, minimising (hard shortfall, balance score). Among the draws that end admissible, the lowest score wins, and ties keep the earliest draw. Re-running reproduces the split exactly (`tests/test_split_s3.py`).

## 5. Adopted split (`results/split_s3/proposed_split.json`, `pair_coverage.csv`)

- **Test, 22 groups (66 tasks, 16,236 prompts):** g03, g04, g10, g11, g14, g17, g19, g20, g21, g24, g27, g30, g33, g36, g48, g54, g56, g58, g62, g63, g64, g65.
- **Train, 50 groups (150 tasks, 36,900 prompts):** the other 50 groups.
- **Test pair coverage:** every compatible pair is in 2 to 8 TEST groups, and 63 of 98 pairs are in 3 or more.
- **Train pair coverage:** 97 of 98 pairs are in at least 1 TRAIN group. Two pairs (`english_capital` + `number_sentences`, `english_lowercase` + `number_words`) are in exactly 1. The rare pair is in 0.
- **Types:** each type is in 6 to 12 TEST groups and at least 10 TRAIN groups.
- **Balance:** score 0.0871, compared with 0.1389 for the 2026-11-14 split. 7,916 of the 20,000 draws were admissible.
- **Change from 2026-11-14:** 6 old test groups stay in test (g03, g10, g14, g30, g48, g63). 9 move to train, and 16 groups move from train to test. No related unit is split.

## 6. Variants kept for comparison

| | **Adopted S3** (`split_s3`) | Test minimum only (`split_s3_test_only`) | Test minimum only, units relaxed (`split_s3_no_units`) |
|---|---|---|---|
| Pairs in ≥ 2 test groups | 98 | 98 | 98 |
| Pairs in ≥ 3 test groups | 63 | 66 | 65 |
| Pairs in ≥ 1 train group | 97 (rare pair excepted) | 97 | 97 |
| Balance score | 0.0871 | 0.0930 | 0.0918 |
| Related units split | none | none | 4 |

The two earlier variants used 1,000 restarts. In both, the only pair absent from training is the rare pair, by chance rather than by rule. The adopted split enforces the train minimum for every other pair.

## 7. How it is applied (this change)

- **`src/mife/splits.py`, `configs/active_split.yaml`:** select the active split. `mife.probing.data.prompts()` and `prompt_rules()` take `split` from the S3 assignment. The test-set gate is unchanged: TEST rows stay locked until `results/prespec/FROZEN.json` approves unblinding.
- **CV folds:** `results/cv_s3/folds.json` holds 5 GroupKFold folds of 10 groups each (7,380 prompts per fold) over the 50 S3 TRAIN groups. They were made with the V1 procedure and fold seed 20261201; the same code reproduces the V1 `results/cv/folds.json` exactly. `probing.cv.CV_DIR` and the V2 CV read the active split's folds, and the V2 CV refuses folds made for another split.
- **Leakage tests:** `tests/test_no_test_leakage.py` checks the loaders and the new folds against the S3 TEST groups. It checks the V1 caches, predictions and folds against the 2026-11-14 split they were built under.

## 8. Still to do before evaluation (not in this change)

- **V2 caches:** these hold the old 42,066 TRAIN rows, and their key checks now refuse to load against the new split. They need rebuilding for the 36,900 S3 TRAIN prompts. This requires max-attention (A1max) extraction on the Mac for the 9 groups that move from the old test set to train (6,642 prompts). These are TRAIN prompts under S3, so the extraction falls under the existing TRAIN-only exception.
- **V2 retraining:** CV selection and the final refit need re-running on the S3 folds.
- **Contamination to report:** the 16 groups that move to test were used for TRAIN-side CV and selection in V1 and V2. The owner accepted this ("leakage does not matter right now"). Any result must report it, and the new models must be refit without those groups.
- **Pre-specification:** the 2026-11-14 freeze (15 test groups) needs a dated revision for S3 before the owner authorises unblinding.
