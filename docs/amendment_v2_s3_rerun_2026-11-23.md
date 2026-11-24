# V2 rerun on split S3 (2026-11-23), training-only

**Status:** run at the project owner's request on 2026-11-23, after split S3 was adopted (PR #4, `docs/amendment_split_s3_2026-11-23.md`). This rerun is training-only. Nothing was extracted, read or scored for the 22 S3 TEST groups. The pre-specification still needs its S3 revision before any unblinding.

## 1. What was rerun, and what was not changed

- **Recipe:** the V2 recipe (`configs/v2_recipe.yaml`) and the 2026-11-22 decisions are unchanged. Primary comparisons are 1, 2, 3 and 4. Comparisons 5–7 freeze the baseline. Parsed parameters stay out of S. The recipe file is byte-identical, so its sha256 is unchanged. It was recorded for S3 in `results/v2_s3/recipe_recorded.json` (2026-11-22 23:08:46 local) before any S3 selection ran.
- **Folds:** 5 grouped folds of 10 groups over the 50 S3 TRAIN groups (`results/cv_s3/folds.json`, made by PR #4). V2 CV refuses folds made for another split.
- **Paths:** every split-dependent V2 artifact now lives in a separate tree, and its sidecars carry `split: split-s3-2026-11-23`. Loaders refuse an artifact built for another split. The 2026-11-14-split V2 run (`results/v2/`, `data/probing_features_v2/{shared,llama,qwen}`) is untouched.
  - Caches: `data/probing_features_v2/split_s3/{shared,llama,qwen}`
  - Results: `results/v2_s3/` (selections, out-of-fold predictions, refits, tables, `SUMMARY.md`)
- **Max-attention store** (`data/probing_features_v2/<model>/maxattn/`): split-independent, because each prompt's values do not depend on the split. The 9 groups newly in TRAIN (g02 g08 g09 g35 g41 g52 g55 g67 g68; 6,642 prompts) were extracted with the same frozen models, tokenization, spans and backend (HF eager, BF16, Apple MPS). This falls under the existing TRAIN-only exception. The store also holds prompts extracted while they were TRAIN under the old split, including the 16 groups that are now TEST. `maxattn.load` refuses every prompt that is TEST under the active split (tested).

## 2. Checks

| Check | Result |
|---|---|
| Max-attention post-run check, both models: 23 TRAIN prompts and 75 spans each, 4 prompts per model from the newly extracted groups | Span alignment OK; stored max equals the direct max for every layer and head; recomputed span sum reproduces the V1 sum (max difference 6e-8 for Llama, 1.2e-7 for Qwen; tolerance 1e-3) |
| BGE-M3 (frozen, pinned) | 30,258 prompts and 246 rules reused from the V1 cache; 6,642 prompts and 54 rules encoded now with the same encoder and settings. Re-encoding 64 cached prompts and all 246 cached rules reproduced the cache exactly (max difference 0.0). |
| Fallback on the S3 folds | Fires only for Qwen `detectable_format:title`, in every fold and in the final refit (2 TRAIN groups with a violation; 0 in fold 2). The smallest group coverage among the other heads is 5, which sits exactly at the threshold. |
| [P] defaults | Still appropriate. Grids, structure penalties, tie precision and the fallback threshold are unchanged. Boundary expansions follow the recorded rule and are listed in `results/v2_s3/SUMMARY.md`. |
| Tests | The full suite passes under S3 (139 tests). Two V2 tests now use the active split's TEST set instead of the manifest column. Two V1 tests in `tests/test_probing.py`, which check V1 folds and caches built for the 2026-11-14 split, failed after PR #4 activated S3; they now pin the manifest split explicitly. |

## 3. Contamination to report with any S3 result

The 16 groups that moved from TRAIN to TEST under S3 were used for TRAIN-side CV and selection in V1 and in the first V2 run. That earlier use includes choosing the recipe's [P] defaults, which were set before the first V2 run and are unchanged here. The S3 models are refit without those groups. The owner accepted this on 2026-11-23.

## 4. Recording log

- 2026-11-22 23:03 (local): PR #4 merged. Max-attention extraction for the 6,642 new TRAIN prompts started (Llama, then Qwen).
- 23:06: V2 paths made split-aware. S3 caches were built: S1/S2 and the 5 hidden rows per model from the frozen features, plus BGE-M3 as above.
- 23:08: recipe recorded for S3. Cache-filling CV pieces then ran in parallel; every selection, expansion and refit is made by `cv.run` from the cache.
- 23:38 to 00:16: the Mac was mostly asleep and the jobs paused. From 00:36 the session kept the Mac awake at the owner's request.
- 01:58: to save time, the Objective 1 H2 grid was split across three processes by rule-penalty value. This affects caching only; the selection is unchanged.

## 5. Results (training CV only)

The selected rows are unchanged from the first V2 run: Objective 1 uses Llama row 12 and Qwen row 18; Objective 2 uses Llama row 12 and Qwen row 27. Objective 1 needed no boundary expansion for either model. Objective 2 extended the prompt and rule penalty grids upwards by one step for both models. Every selection, out-of-fold prediction and refit is in `results/v2_s3/` and summarised in `results/v2_s3/SUMMARY.md`.
