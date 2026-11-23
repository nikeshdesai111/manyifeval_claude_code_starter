# Expansion amendment E-S3 (2026-11-23): all rule subsets on the 22 S3 TEST groups — PROPOSED

Status: **proposed, awaiting the project owner's adoption.** This note was written before any TEST outcome was inspected. It does not change the 2026-11-14 pre-specification (`results/prespec/`), the V2 amendment or the S3 split amendment. It only adds held-out data.

## 1. What changes

The 22 S3 TEST groups (see `docs/amendment_split_s3_2026-11-23.md`) currently have 6 nested-chain subsets per k. Each group gains every other subset of its six exact rules at k = 2, 3 and 4, ordered by the same COA template per k. That adds 210 configurations per task, 13,860 prompts per generator, and 20 responses each from both generators. The design, counts and checks are in `docs/s3_22group_expansion_generation_plan.md`. The tasks, rule pools, exact rule instances, decoding settings and verifiers are unchanged. TRAIN groups are not expanded.

## 2. Matching and analysis (to be fixed before unblinding)

- **Original vs new prompts.** The 246 frozen configurations per task keep their IDs and responses (`main` run). New prompts carry `is_original = False` and come from run `expansion_s3_20261123`. Any analysis that compares original and expanded designs must report both, and must not silently pool them as one sampling procedure. The seeding mode differs (per-chunk engine seed in `main`, per-sample seed in the expansion), while the sampling distribution is the same.
- **Factor contrasts.** Strict-append (k → k+1), replacement and position contrasts use only ordered pairs present in the expanded design, matched within group and task. The specific estimands and their weights are to be written into the S3 evaluation pre-specification, not here.
- **Uncertainty.** The unit is the group (22 groups), as in the S3 amendment. Expanded subsets add prompts within groups. They do not add independent groups.

## 3. Verification protocol (locked until authorized)

New responses are verified with the same official ManyIFEval/IFEval checkers and the same per-rule binary labels as `main` (`mife.verify_stage`). The expansion manifest's `prompt_rules.parquet` lists every included rule with its exact parameters. Verification of TEST responses, original or new, runs only after the owner authorizes unblinding. Until then, the only check is `mife.expansion_s3 check-complete`, which reads provenance only.

## 4. Caveats carried forward

- The 16 groups that moved from TRAIN into TEST under S3 had their outcomes used in V1/V2 training and tuning. This is accepted and recorded in the S3 amendment, and results must report it.
- The type pair `capital_word_frequency` + `number_words` appears in 2 TEST pools and 0 TRAIN pools.
