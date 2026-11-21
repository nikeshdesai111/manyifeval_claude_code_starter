# V2 training-only milestone: summary

Training-only. The 15 TEST groups were not read: no TEST features, labels or scores. Every loss below is a **training-CV** value (5 grouped folds over the 57 TRAIN groups). These values were used to select settings, so they document the selection and are not held-out performance estimates. Amendment: `docs/amendment_v2_2026-11-20.md`; recipe: `configs/v2_recipe.yaml` (recorded in `results/v2/recipe_recorded.json`).

Recipe sha256 `d286c35b2e0c2c73…`, recorded 2026-11-20T16:46:47-0500, before any selection reported here.

## What changed from V1

- **Predictors:** 7 planned comparisons (3 primary, 4 secondary) plus references, instead of 23 configurations. B2 is now B1 plus a *frozen* exact-rule embedding (no fine-tuning). A1 is the per-layer×head **max** attention over the target-rule span (new training-only extraction). The rule-conditioned attention probe is A1 + exact-rule embedding + structure.
- **Architecture:** every Objective 1 predictor uses 15 per-type ridge-logistic heads with the same structure block. Every Objective 2 predictor uses one shared ridge fractional-logit head on the existing q_hat target. There are no MLP heads.
- **Fitting:** summed binomial NLL (Objective 1) or fractional logit (Objective 2), with penalty λ/N. There are three penalty slots (fixed structure, tuned prompt representation, tuned rule embedding) on identical grids, a deterministic tie rule, a one-time ×10 boundary expansion, a block-off option for the incremental comparisons, and a sparse-head fallback that also uses group coverage.
- **Layer search:** widened to 5 rows per generator, selected by H1 + structure, separately per objective.
- **Kept:** the 2026-11-14 freeze, every V1 file, the folds, labels and frozen features. Everything V2 is versioned (`v2-2026-11-20`) under `data/probing_features_v2/` and `results/v2/`.

## 1. Max-attention extraction (TRAIN prompts only)

| model | status | n_prompts | n_train_prompts | wall_h_recorded_sessions | wall_h_shard_span | storage_gb | backend | torch | transformers | pre_prompts/spans | pre_alignment_ok | pre_max_equals_direct | pre_sum_max_abs_diff_vs_v1 | sum_tolerance | straddling_spans_whitespace | post_prompts/spans | post_alignment_ok | post_max_equals_direct | post_stored_equals_direct | post_sum_max_abs_diff_vs_v1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama | complete | 42066 | 42066 | 4.97 | 5.01 | 0.438 | transformers+torch-mps, eager, bfloat16 | 2.15.0 | 5.22.2 | 23/74 | True | True | 5.960e-08 | 0.001 | 74 | 23/74 | True | True | True | 5.960e-08 |
| qwen | complete | 42066 | 42066 | 2.5 | 6.25 | 0.492 | transformers+torch-mps, eager, bfloat16 | 2.15.0 | 5.22.2 | 23/74 | True | True | 5.960e-08 | 0.001 | 74 | 23/74 | True | True | True | 5.960e-08 |

Checks: (1) each indexed span, rebuilt from character offsets, equals the frozen span and contains the exact rule text. The frozen tokenization flags no TRAIN span with a boundary token covering NON-whitespace text outside the rule; every sampled span has a boundary token that covers whitespace outside it. (2) For every layer and head, the extracted max equals the direct max over the span from the full attention row (exact equality); after the run, the stored value equals it too. (3) The recomputed span SUM reproduces the cached V1 sum within the documented absolute tolerance.

Wall time: `wall_h_recorded_sessions` sums the sessions written to the metadata. The Qwen run was restarted once, and the killed session wrote no record, so `wall_h_shard_span` (from the shard commit times) is the elapsed time. Both runs shared the Mac with CV jobs and other applications under memory pressure; the unloaded rate was about 0.25–0.30 s per prompt.

## 2. Hidden-layer selection (H1 + structure, training CV)

Row convention: 0 = embeddings; r < L = output of decoder layer r; L = layer L after the final norm; L+1 = layer L before the final norm (Llama L = 32, Qwen L = 36). Disclosure: hidden-state methods get this 5-row search; the text and attention methods do not.

| model | objective | which | row | meaning | depth | cv_loss_best_setting | selected |
|---|---|---|---|---|---|---|---|
| llama | 1 | own (H1 + structure) | 12 | output of decoder layer 12 | 0.375 | 0.393361 | True |
| llama | 1 | own (H1 + structure) | 16 | output of decoder layer 16 | 0.5 | 0.398848 | False |
| llama | 1 | own (H1 + structure) | 24 | output of decoder layer 24 | 0.75 | 0.428105 | False |
| llama | 1 | own (H1 + structure) | 33 | output of decoder layer 32 BEFORE the final norm | 1 | 0.425329 | False |
| llama | 1 | own (H1 + structure) | 8 | output of decoder layer 8 | 0.25 | 0.408508 | False |
| llama | 1 | other generator (reference) | 14 | output of decoder layer 14 | 0.3889 | 0.411345 | False |
| llama | 1 | other generator (reference) | 18 | output of decoder layer 18 | 0.5 | 0.384485 | True |
| llama | 1 | other generator (reference) | 27 | output of decoder layer 27 | 0.75 | 0.402554 | False |
| llama | 1 | other generator (reference) | 37 | output of decoder layer 36 BEFORE the final norm | 1 | 0.435446 | False |
| llama | 1 | other generator (reference) | 9 | output of decoder layer 9 | 0.25 | 0.425332 | False |
| llama | 2 | own (H1 + structure) | 12 | output of decoder layer 12 | 0.375 | 0.016343 | True |
| llama | 2 | own (H1 + structure) | 16 | output of decoder layer 16 | 0.5 | 0.0167519 | False |
| llama | 2 | own (H1 + structure) | 24 | output of decoder layer 24 | 0.75 | 0.0193834 | False |
| llama | 2 | own (H1 + structure) | 33 | output of decoder layer 32 BEFORE the final norm | 1 | 0.0194092 | False |
| llama | 2 | own (H1 + structure) | 8 | output of decoder layer 8 | 0.25 | 0.0198686 | False |
| llama | 2 | other generator (reference) | 14 | output of decoder layer 14 | 0.3889 | 0.0189299 | False |
| llama | 2 | other generator (reference) | 18 | output of decoder layer 18 | 0.5 | 0.0163524 | True |
| llama | 2 | other generator (reference) | 27 | output of decoder layer 27 | 0.75 | 0.0169374 | False |
| llama | 2 | other generator (reference) | 37 | output of decoder layer 36 BEFORE the final norm | 1 | 0.0195245 | False |
| llama | 2 | other generator (reference) | 9 | output of decoder layer 9 | 0.25 | 0.0206547 | False |
| qwen | 1 | own (H1 + structure) | 14 | output of decoder layer 14 | 0.3889 | 0.303978 | False |
| qwen | 1 | own (H1 + structure) | 18 | output of decoder layer 18 | 0.5 | 0.289243 | True |
| qwen | 1 | own (H1 + structure) | 27 | output of decoder layer 27 | 0.75 | 0.293633 | False |
| qwen | 1 | own (H1 + structure) | 37 | output of decoder layer 36 BEFORE the final norm | 1 | 0.315094 | False |
| qwen | 1 | own (H1 + structure) | 9 | output of decoder layer 9 | 0.25 | 0.312015 | False |
| qwen | 1 | other generator (reference) | 12 | output of decoder layer 12 | 0.375 | 0.298769 | False |
| qwen | 1 | other generator (reference) | 16 | output of decoder layer 16 | 0.5 | 0.296325 | True |
| qwen | 1 | other generator (reference) | 24 | output of decoder layer 24 | 0.75 | 0.313908 | False |
| qwen | 1 | other generator (reference) | 33 | output of decoder layer 32 BEFORE the final norm | 1 | 0.307676 | False |
| qwen | 1 | other generator (reference) | 8 | output of decoder layer 8 | 0.25 | 0.301793 | False |
| qwen | 2 | own (H1 + structure) | 14 | output of decoder layer 14 | 0.3889 | 0.0140353 | False |
| qwen | 2 | own (H1 + structure) | 18 | output of decoder layer 18 | 0.5 | 0.0129819 | False |
| qwen | 2 | own (H1 + structure) | 27 | output of decoder layer 27 | 0.75 | 0.0115264 | True |
| qwen | 2 | own (H1 + structure) | 37 | output of decoder layer 36 BEFORE the final norm | 1 | 0.0129809 | False |
| qwen | 2 | own (H1 + structure) | 9 | output of decoder layer 9 | 0.25 | 0.0143184 | False |
| qwen | 2 | other generator (reference) | 12 | output of decoder layer 12 | 0.375 | 0.011802 | True |
| qwen | 2 | other generator (reference) | 16 | output of decoder layer 16 | 0.5 | 0.0121182 | False |
| qwen | 2 | other generator (reference) | 24 | output of decoder layer 24 | 0.75 | 0.0129835 | False |
| qwen | 2 | other generator (reference) | 33 | output of decoder layer 32 BEFORE the final norm | 1 | 0.012867 | False |
| qwen | 2 | other generator (reference) | 8 | output of decoder layer 8 | 0.25 | 0.0137514 | False |

## 3. Grids, boundary expansions and edges

| model | objective | final_grids | expansions | base_round_edge_hits | edges_after_expansion |
|---|---|---|---|---|---|
| llama | 1 | {"P": [100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0], "R": [100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0]} | [] | [] | [] |
| llama | 2 | {"P": [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0], "R": [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]} | [{"direction": "high", "factor": 10, "slot": "P"}, {"direction": "high", "factor": 10, "slot": "R"}] | [{"direction": "high", "predictor": "B1", "selected": 100000.0, "slot": "P"}, {"direction": "high", "predictor": "B2", "selected": 100000.0, "slot": "P"}, {"direction": "high", "predictor": "H2", "selected": 100000.0, "slot": "R"}] | [{"direction": "high", "note": "best among the tested values", "predictor": "B2", "slot": "P"}, {"direction": "high", "note": "best among the tested values", "predictor": "H2", "slot": "R"}] |
| qwen | 1 | {"P": [100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0], "R": [100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0, 100000000.0]} | [{"direction": "high", "factor": 10, "slot": "R"}] | [{"direction": "high", "predictor": "RCAP", "selected": 10000000.0, "slot": "R"}] | [] |
| qwen | 2 | {"P": [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0], "R": [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]} | [{"direction": "high", "factor": 10, "slot": "P"}, {"direction": "high", "factor": 10, "slot": "R"}] | [{"direction": "high", "predictor": "B1", "selected": 100000.0, "slot": "P"}, {"direction": "high", "predictor": "B2", "selected": 100000.0, "slot": "P"}, {"direction": "high", "predictor": "H2", "selected": 100000.0, "slot": "R"}, {"direction": "high", "predictor": "RCAP", "selected": 100000.0, "slot": "R"}] | [] |

Penalties are on the summed-loss scale. An expansion adds one ×10 step, once. A setting still on an edge after expansion is reported as best among the tested values.

## 4. Selected settings per generator, objective and predictor

| model | objective | predictor | label | selected_setting | frozen_baseline | hidden_row | n_features | cv_loss | note |
|---|---|---|---|---|---|---|---|---|---|
| llama | 1 | B1 | B1 | P=1e+06 |  | nan | 1044 | 0.440064 |  |
| llama | 1 | B1+H1 | B1 + H1 | X=100000 | {"F_P": 1000000.0} | 12 | 5140 | 0.393873 |  |
| llama | 1 | B2 | B2 | P=1e+06,R=1e+06 |  | nan | 2068 | 0.382076 |  |
| llama | 1 | B2+A1 | B2 + A1 | X=10000 | {"F_P": 1000000.0, "F_R": 1000000.0} | nan | 3092 | 0.356137 |  |
| llama | 1 | B2+H1 | B2 + H1 | X=1e+06 | {"F_P": 1000000.0, "F_R": 1000000.0} | 12 | 6164 | 0.372671 |  |
| llama | 1 | H1 | H1 + structure | P=100000 |  | 12 | 4116 | 0.393361 |  |
| llama | 1 | H2 | H2 + structure | P=1e+06,R=1e+06 |  | 12 | 5140 | 0.371108 |  |
| llama | 1 | RCAP | Rule-conditioned attention probe | P=10000,R=1e+06 |  | nan | 2068 | 0.356952 |  |
| llama | 1 | REF_structure | Structure-only reference | none |  | nan | 20 | 0.448188 |  |
| llama | 1 | XH1 | Other generator's H1 + structure (reference) | P=100000 |  | 18 | 4116 | 0.384485 |  |
| llama | 2 | B1 | B1 | P=100000 |  | nan | 1116 | 0.0217161 |  |
| llama | 2 | B1+H1 | B1 + H1 | X=1000 | {"F_P": 100000.0} | 12 | 5212 | 0.0163707 |  |
| llama | 2 | B2 | B2 | P=1e+06,R=10000 |  | nan | 2140 | 0.0181024 |  |
| llama | 2 | B2+A1 | B2 + A1 | X=1000 | {"F_P": 1000000.0, "F_R": 10000.0} | nan | 3164 | 0.0161441 |  |
| llama | 2 | B2+H1 | B2 + H1 | X=1000 | {"F_P": 1000000.0, "F_R": 10000.0} | 12 | 6236 | 0.0165282 |  |
| llama | 2 | H1 | H1 + structure | P=1000 |  | 12 | 4188 | 0.016343 |  |
| llama | 2 | H2 | H2 + structure | P=1000,R=1e+06 |  | 12 | 5212 | 0.0163428 |  |
| llama | 2 | RCAP | Rule-conditioned attention probe | P=1000,R=10000 |  | nan | 2140 | 0.0161523 |  |
| llama | 2 | REF_structure | Structure-only reference | none |  | nan | 92 | 0.0219807 |  |
| llama | 2 | XH1 | Other generator's H1 + structure (reference) | P=1000 |  | 18 | 4188 | 0.0163524 |  |
| qwen | 1 | B1 | B1 | P=1e+06 |  | nan | 1044 | 0.318036 |  |
| qwen | 1 | B1+H1 | B1 + H1 | X=100000 | {"F_P": 1000000.0} | 18 | 5140 | 0.289044 |  |
| qwen | 1 | B2 | B2 | P=1e+06,R=1e+06 |  | nan | 2068 | 0.283032 |  |
| qwen | 1 | B2+A1 | B2 + A1 | X=10000 | {"F_P": 1000000.0, "F_R": 1000000.0} | nan | 3220 | 0.269658 |  |
| qwen | 1 | B2+H1 | B2 + H1 | X=1e+06 | {"F_P": 1000000.0, "F_R": 1000000.0} | 18 | 6164 | 0.27872 |  |
| qwen | 1 | H1 | H1 + structure | P=100000 |  | 18 | 4116 | 0.289243 |  |
| qwen | 1 | H2 | H2 + structure | P=1e+06,R=1e+06 |  | 18 | 5140 | 0.278812 |  |
| qwen | 1 | RCAP | Rule-conditioned attention probe | P=10000,R=1e+07 |  | nan | 2196 | 0.269491 |  |
| qwen | 1 | REF_structure | Structure-only reference | none |  | nan | 20 | 0.323091 |  |
| qwen | 1 | XH1 | Other generator's H1 + structure (reference) | P=100000 |  | 16 | 4116 | 0.296325 |  |
| qwen | 2 | B1 | B1 | P=100000 |  | nan | 1116 | 0.0146791 |  |
| qwen | 2 | B1+H1 | B1 + H1 | X=1000 | {"F_P": 100000.0} | 27 | 5212 | 0.0114992 |  |
| qwen | 2 | B2 | B2 | P=100000,R=10000 |  | nan | 2140 | 0.0137244 |  |
| qwen | 2 | B2+A1 | B2 + A1 | X=1000 | {"F_P": 100000.0, "F_R": 10000.0} | nan | 3292 | 0.0123576 |  |
| qwen | 2 | B2+H1 | B2 + H1 | X=1000 | {"F_P": 100000.0, "F_R": 10000.0} | 27 | 6236 | 0.0115196 |  |
| qwen | 2 | H1 | H1 + structure | P=1000 |  | 27 | 4188 | 0.0115264 |  |
| qwen | 2 | H2 | H2 + structure | P=1000,R=100000 |  | 27 | 5212 | 0.0114682 |  |
| qwen | 2 | RCAP | Rule-conditioned attention probe | P=1000,R=100000 |  | nan | 2268 | 0.0121365 |  |
| qwen | 2 | REF_structure | Structure-only reference | none |  | nan | 92 | 0.0147855 |  |
| qwen | 2 | XH1 | Other generator's H1 + structure (reference) | P=1000 |  | 12 | 4188 | 0.011802 |  |

Objective 1 loss is the pooled mean log loss per completion; Objective 2 loss is the pooled squared error of the predicted fraction. `P` = prompt-representation penalty, `R` = rule-embedding penalty, `X` = added block in comparisons 4–6 ("off" = CV did not select the block, so the gain is 0 by construction; this is not evidence that the block carries no information).

## 5. Sparse-head fallbacks that fired (Objective 1)

| model | fold | type_id | violated | compliant | groups_with_violation | reason |
|---|---|---|---|---|---|---|
| qwen | 0 | detectable_format:title | 116 | 276904 | 2 | groups_with_violation<5 |
| qwen | 1 | detectable_format:title | 116 | 225604 | 2 | groups_with_violation<5 |
| qwen | 2 | detectable_format:title | 115 | 276905 | 1 | groups_with_violation<5 |
| qwen | 3 | detectable_format:title | 1 | 246239 | 1 | violated<50;groups_with_violation<5 |
| qwen | 4 | detectable_format:title | 116 | 246124 | 2 | groups_with_violation<5 |
| qwen | final | detectable_format:title | 116 | 317944 | 2 | groups_with_violation<5 |

The decision depends only on the fitting rows' labels, so a fallback type contributes the same intercept-only prediction (v + 0.5) / (n + 1) to every method.

## 6. The seven comparisons on training CV (descriptive only)

Positive `right_minus_left` = lower training-CV loss for the left-hand predictor. These are the selection criteria themselves, with no interval and no held-out data. The planned test evaluation (Part 1 and 2 reporting plans) has not been run.

| model | objective | comparison | left | right | cv_left | cv_right | right_minus_left |
|---|---|---|---|---|---|---|---|
| llama | 1 | Primary 1 | H1 + structure | B1 | 0.393361 | 0.440064 | 0.0467021 |
| llama | 1 | Primary 2 | H2 + structure | B2 | 0.371108 | 0.382076 | 0.0109688 |
| llama | 1 | Primary 3 | Rule-conditioned attention probe | B2 | 0.356952 | 0.382076 | 0.0251247 |
| llama | 1 | Secondary 4 | B1 + H1 | B1 | 0.393873 | 0.440064 | 0.0461905 |
| llama | 1 | Secondary 5 | B2 + H1 | B2 | 0.372671 | 0.382076 | 0.00940574 |
| llama | 1 | Secondary 6 | B2 + A1 | B2 | 0.356137 | 0.382076 | 0.0259393 |
| llama | 1 | Secondary 7 | H2 + structure | Rule-conditioned attention probe | 0.371108 | 0.356952 | -0.0141559 |
| llama | 2 | Primary 1 | H1 + structure | B1 | 0.016343 | 0.0217161 | 0.00537314 |
| llama | 2 | Primary 2 | H2 + structure | B2 | 0.0163428 | 0.0181024 | 0.00175963 |
| llama | 2 | Primary 3 | Rule-conditioned attention probe | B2 | 0.0161523 | 0.0181024 | 0.00195012 |
| llama | 2 | Secondary 4 | B1 + H1 | B1 | 0.0163707 | 0.0217161 | 0.0053454 |
| llama | 2 | Secondary 5 | B2 + H1 | B2 | 0.0165282 | 0.0181024 | 0.00157417 |
| llama | 2 | Secondary 6 | B2 + A1 | B2 | 0.0161441 | 0.0181024 | 0.00195829 |
| llama | 2 | Secondary 7 | H2 + structure | Rule-conditioned attention probe | 0.0163428 | 0.0161523 | -1.905e-04 |
| qwen | 1 | Primary 1 | H1 + structure | B1 | 0.289243 | 0.318036 | 0.0287932 |
| qwen | 1 | Primary 2 | H2 + structure | B2 | 0.278812 | 0.283032 | 0.00422056 |
| qwen | 1 | Primary 3 | Rule-conditioned attention probe | B2 | 0.269491 | 0.283032 | 0.0135417 |
| qwen | 1 | Secondary 4 | B1 + H1 | B1 | 0.289044 | 0.318036 | 0.0289925 |
| qwen | 1 | Secondary 5 | B2 + H1 | B2 | 0.27872 | 0.283032 | 0.00431255 |
| qwen | 1 | Secondary 6 | B2 + A1 | B2 | 0.269658 | 0.283032 | 0.0133743 |
| qwen | 1 | Secondary 7 | H2 + structure | Rule-conditioned attention probe | 0.278812 | 0.269491 | -0.0093211 |
| qwen | 2 | Primary 1 | H1 + structure | B1 | 0.0115264 | 0.0146791 | 0.00315264 |
| qwen | 2 | Primary 2 | H2 + structure | B2 | 0.0114682 | 0.0137244 | 0.00225614 |
| qwen | 2 | Primary 3 | Rule-conditioned attention probe | B2 | 0.0121365 | 0.0137244 | 0.0015879 |
| qwen | 2 | Secondary 4 | B1 + H1 | B1 | 0.0114992 | 0.0146791 | 0.0031799 |
| qwen | 2 | Secondary 5 | B2 + H1 | B2 | 0.0115196 | 0.0137244 | 0.00220478 |
| qwen | 2 | Secondary 6 | B2 + A1 | B2 | 0.0123576 | 0.0137244 | 0.00136675 |
| qwen | 2 | Secondary 7 | H2 + structure | Rule-conditioned attention probe | 0.0114682 | 0.0121365 | 6.682e-04 |

## 7. Objective 2: ordered-slot sensitivity and averaged Objective 1 predictions

| model | analysis | predictor | setting | cv_mse | n_features | note |
|---|---|---|---|---|---|---|
| llama | ordered-slot sensitivity | B2o | P=1e+06,R=10000 | 0.0188505 | 6236 |  |
| llama | ordered-slot sensitivity | B2o+A1 | X=1000 | 0.0161712 | 11356 |  |
| llama | ordered-slot sensitivity | B2o+H1 | X=1000 | 0.0164392 | 10332 |  |
| llama | ordered-slot sensitivity | H2o | P=1000,R=100000 | 0.0163384 | 9308 |  |
| llama | ordered-slot sensitivity | RCAPo | P=1000,R=10000 | 0.0161773 | 10332 |  |
| llama | averaged Objective 1 OOF predictions | B1 | Objective 1 selection | 0.0222126 | nan |  |
| llama | averaged Objective 1 OOF predictions | B1+H1 | Objective 1 selection | 0.0165605 | nan |  |
| llama | averaged Objective 1 OOF predictions | B2 | Objective 1 selection | 0.0164545 | nan |  |
| llama | averaged Objective 1 OOF predictions | B2+A1 | Objective 1 selection | 0.0139741 | nan |  |
| llama | averaged Objective 1 OOF predictions | B2+H1 | Objective 1 selection | 0.015074 | nan |  |
| llama | averaged Objective 1 OOF predictions | H1 | Objective 1 selection | 0.0164467 | nan |  |
| llama | averaged Objective 1 OOF predictions | H2 | Objective 1 selection | 0.0148958 | nan |  |
| llama | averaged Objective 1 OOF predictions | RCAP | Objective 1 selection | 0.0140776 | nan |  |
| llama | averaged Objective 1 OOF predictions | REF_structure | Objective 1 selection | 0.0226704 | nan |  |
| llama | averaged Objective 1 OOF predictions | XH1 | Objective 1 selection | 0.0155616 | nan |  |
| qwen | ordered-slot sensitivity | B2o | P=100000,R=10000 | 0.0143249 | 6236 |  |
| qwen | ordered-slot sensitivity | B2o+A1 | X=1000 | 0.0125164 | 11996 |  |
| qwen | ordered-slot sensitivity | B2o+H1 | X=1000 | 0.0116433 | 10332 |  |
| qwen | ordered-slot sensitivity | H2o | P=1000,R=100000 | 0.0115107 | 9308 |  |
| qwen | ordered-slot sensitivity | RCAPo | P=1000,R=100000 | 0.0123917 | 10972 |  |
| qwen | averaged Objective 1 OOF predictions | B1 | Objective 1 selection | 0.0162942 | nan |  |
| qwen | averaged Objective 1 OOF predictions | B1+H1 | Objective 1 selection | 0.0132859 | nan |  |
| qwen | averaged Objective 1 OOF predictions | B2 | Objective 1 selection | 0.0132075 | nan |  |
| qwen | averaged Objective 1 OOF predictions | B2+A1 | Objective 1 selection | 0.0112847 | nan |  |
| qwen | averaged Objective 1 OOF predictions | B2+H1 | Objective 1 selection | 0.0127983 | nan |  |
| qwen | averaged Objective 1 OOF predictions | H1 | Objective 1 selection | 0.0132405 | nan |  |
| qwen | averaged Objective 1 OOF predictions | H2 | Objective 1 selection | 0.0127666 | nan |  |
| qwen | averaged Objective 1 OOF predictions | RCAP | Objective 1 selection | 0.0112188 | nan |  |
| qwen | averaged Objective 1 OOF predictions | REF_structure | Objective 1 selection | 0.0163817 | nan |  |
| qwen | averaged Objective 1 OOF predictions | XH1 | Objective 1 selection | 0.0141576 | nan |  |

## 8. Where the outputs are

- OOF predictions (every tested setting plus `selected`): `results/v2/predictions/<model>/obj<o>/<predictor>.parquet`; Objective 2 ordered-slot: `.../obj2/ordered/`; averaged Objective 1: `.../obj2/avg_obj1/`.
- Selections: `results/v2/cv/<model>/obj<o>/selection.json` (+ `selection_ordered.json`, `avg_obj1_reference.json`).
- Refitted heads (all 57 TRAIN groups): `results/v2/models/<model>/obj<o>/<predictor>.npz/.json`.
- Tables: `results/v2/tables/`. Caches: `data/probing_features_v2/`.
- Parquet, NumPy and npz files are git-ignored and stay on the Mac; JSON, CSV and Markdown are committed.
