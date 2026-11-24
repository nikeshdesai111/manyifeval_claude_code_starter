# V2 training-only milestone: summary

Split: `split-s3-2026-11-23`. Training-only. The 22 TEST groups were not read: no TEST features, labels or scores. Every loss below is a **training-CV** value (5 grouped folds over the 50 TRAIN groups). These values were used to select settings, so they document the selection and are not held-out performance estimates. Amendment: `docs/amendment_v2_2026-11-20.md`; recipe: `configs/v2_recipe.yaml` (recorded in `results/v2_s3/recipe_recorded.json`).

Recipe sha256 `d286c35b2e0c2c73…`, recorded 2026-11-22T23:08:46-0500, before any selection reported here.

## What changed from V1

- **Predictors:** 7 planned comparisons (primary: 1, 2, 3 and 4; secondary: 5, 6 and 7) plus references, instead of 23 configurations. B2 is now B1 plus a *frozen* exact-rule embedding (no fine-tuning). A1 is the per-layer×head **max** attention over the target-rule span (new training-only extraction). The rule-conditioned attention probe is A1 + exact-rule embedding + structure.
- **Architecture:** every Objective 1 predictor uses 15 per-type ridge-logistic heads with the same structure block. Every Objective 2 predictor uses one shared ridge fractional-logit head on the existing q_hat target. There are no MLP heads.
- **Fitting:** summed binomial NLL (Objective 1) or fractional logit (Objective 2), with penalty λ/N. There are three penalty slots (fixed structure, tuned prompt representation, tuned rule embedding) on identical grids, a deterministic tie rule, a one-time ×10 boundary expansion, a block-off option for the incremental comparisons, and a sparse-head fallback that also uses group coverage.
- **Layer search:** widened to 5 rows per generator, selected by H1 + structure, separately per objective.
- **Naming (B1 / B2):** in V2, B1 = S + P (structure + frozen BGE-M3 prompt embedding) and B2 = S + P + R (B1 + frozen BGE-M3 exact-rule embedding). In V1 (`results/RESULTS.md`, `results/tables/`, `results/cv/`, `docs/probing_schema.md`), **B2 means the LoRA-adapted BGE-M3 baseline**, which V2 retired. The two are different models and are never compared or mixed.
- **Kept:** the 2026-11-14 freeze, every V1 file, the folds, labels and frozen features. Everything V2 is versioned (`v2-2026-11-20`) under `data/probing_features_v2/` and `results/v2/`.

## 1. Max-attention extraction (TRAIN prompts only)

| model | status | n_prompts | n_train_prompts | wall_h_recorded_sessions | wall_h_shard_span | storage_gb | backend | torch | transformers | post_prompts/spans | post_alignment_ok | post_max_equals_direct | post_stored_equals_direct | post_sum_max_abs_diff_vs_v1 | sum_tolerance | straddling_spans_whitespace |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama | complete | 48708 | 36900 | 6.03 | 57.95 | 0.507 | transformers+torch-mps, eager, bfloat16 | 2.15.0 | 5.22.2 | 23/75 | True | True | True | 5.960e-08 | 0.001 | 75 |
| qwen | complete | 48708 | 36900 | 3.15 | 53.53 | 0.569 | transformers+torch-mps, eager, bfloat16 | 2.15.0 | 5.22.2 | 23/75 | True | True | True | 1.192e-07 | 0.001 | 75 |

Checks: (1) each indexed span, rebuilt from character offsets, equals the frozen span and contains the exact rule text. The frozen tokenization flags no TRAIN span with a boundary token covering NON-whitespace text outside the rule; every sampled span has a boundary token that covers whitespace outside it. (2) For every layer and head, the extracted max equals the direct max over the span from the full attention row (exact equality); after the run, the stored value equals it too. (3) The recomputed span SUM reproduces the cached V1 sum within the documented absolute tolerance.

Wall time: `wall_h_recorded_sessions` sums the sessions written to the metadata. The Qwen run was restarted once, and the killed session wrote no record, so `wall_h_shard_span` (from the shard commit times) is the elapsed time. Both runs shared the Mac with CV jobs and other applications under memory pressure; the unloaded rate was about 0.25–0.30 s per prompt.

## 2. Hidden-layer selection (H1 + structure, training CV)

Row convention: 0 = embeddings; r < L = output of decoder layer r; L = layer L after the final norm; L+1 = layer L before the final norm (Llama L = 32, Qwen L = 36). Disclosure: hidden-state methods get this 5-row search; the text and attention methods do not.

| model | objective | which | row | meaning | depth | cv_loss_best_setting | selected |
|---|---|---|---|---|---|---|---|
| llama | 1 | own (H1 + structure) | 12 | output of decoder layer 12 | 0.375 | 0.413778 | True |
| llama | 1 | own (H1 + structure) | 16 | output of decoder layer 16 | 0.5 | 0.444574 | False |
| llama | 1 | own (H1 + structure) | 24 | output of decoder layer 24 | 0.75 | 0.468623 | False |
| llama | 1 | own (H1 + structure) | 33 | output of decoder layer 32 BEFORE the final norm | 1 | 0.451841 | False |
| llama | 1 | own (H1 + structure) | 8 | output of decoder layer 8 | 0.25 | 0.448361 | False |
| llama | 1 | other generator (reference) | 14 | output of decoder layer 14 | 0.3889 | 0.428395 | False |
| llama | 1 | other generator (reference) | 18 | output of decoder layer 18 | 0.5 | 0.397407 | True |
| llama | 1 | other generator (reference) | 27 | output of decoder layer 27 | 0.75 | 0.451886 | False |
| llama | 1 | other generator (reference) | 37 | output of decoder layer 36 BEFORE the final norm | 1 | 0.483392 | False |
| llama | 1 | other generator (reference) | 9 | output of decoder layer 9 | 0.25 | 0.461187 | False |
| llama | 2 | own (H1 + structure) | 12 | output of decoder layer 12 | 0.375 | 0.0160586 | True |
| llama | 2 | own (H1 + structure) | 16 | output of decoder layer 16 | 0.5 | 0.0175785 | False |
| llama | 2 | own (H1 + structure) | 24 | output of decoder layer 24 | 0.75 | 0.0200556 | False |
| llama | 2 | own (H1 + structure) | 33 | output of decoder layer 32 BEFORE the final norm | 1 | 0.01859 | False |
| llama | 2 | own (H1 + structure) | 8 | output of decoder layer 8 | 0.25 | 0.0182123 | False |
| llama | 2 | other generator (reference) | 14 | output of decoder layer 14 | 0.3889 | 0.0177606 | False |
| llama | 2 | other generator (reference) | 18 | output of decoder layer 18 | 0.5 | 0.0168147 | True |
| llama | 2 | other generator (reference) | 27 | output of decoder layer 27 | 0.75 | 0.0173386 | False |
| llama | 2 | other generator (reference) | 37 | output of decoder layer 36 BEFORE the final norm | 1 | 0.0191352 | False |
| llama | 2 | other generator (reference) | 9 | output of decoder layer 9 | 0.25 | 0.0210728 | False |
| qwen | 1 | own (H1 + structure) | 14 | output of decoder layer 14 | 0.3889 | 0.302781 | False |
| qwen | 1 | own (H1 + structure) | 18 | output of decoder layer 18 | 0.5 | 0.282156 | True |
| qwen | 1 | own (H1 + structure) | 27 | output of decoder layer 27 | 0.75 | 0.303284 | False |
| qwen | 1 | own (H1 + structure) | 37 | output of decoder layer 36 BEFORE the final norm | 1 | 0.339581 | False |
| qwen | 1 | own (H1 + structure) | 9 | output of decoder layer 9 | 0.25 | 0.331936 | False |
| qwen | 1 | other generator (reference) | 12 | output of decoder layer 12 | 0.375 | 0.289374 | True |
| qwen | 1 | other generator (reference) | 16 | output of decoder layer 16 | 0.5 | 0.31 | False |
| qwen | 1 | other generator (reference) | 24 | output of decoder layer 24 | 0.75 | 0.325047 | False |
| qwen | 1 | other generator (reference) | 33 | output of decoder layer 32 BEFORE the final norm | 1 | 0.31868 | False |
| qwen | 1 | other generator (reference) | 8 | output of decoder layer 8 | 0.25 | 0.309171 | False |
| qwen | 2 | own (H1 + structure) | 14 | output of decoder layer 14 | 0.3889 | 0.01274 | False |
| qwen | 2 | own (H1 + structure) | 18 | output of decoder layer 18 | 0.5 | 0.0123939 | False |
| qwen | 2 | own (H1 + structure) | 27 | output of decoder layer 27 | 0.75 | 0.012192 | True |
| qwen | 2 | own (H1 + structure) | 37 | output of decoder layer 36 BEFORE the final norm | 1 | 0.0137358 | False |
| qwen | 2 | own (H1 + structure) | 9 | output of decoder layer 9 | 0.25 | 0.0133419 | False |
| qwen | 2 | other generator (reference) | 12 | output of decoder layer 12 | 0.375 | 0.0116078 | True |
| qwen | 2 | other generator (reference) | 16 | output of decoder layer 16 | 0.5 | 0.0120057 | False |
| qwen | 2 | other generator (reference) | 24 | output of decoder layer 24 | 0.75 | 0.0130405 | False |
| qwen | 2 | other generator (reference) | 33 | output of decoder layer 32 BEFORE the final norm | 1 | 0.0128615 | False |
| qwen | 2 | other generator (reference) | 8 | output of decoder layer 8 | 0.25 | 0.0129732 | False |

## 3. Grids, boundary expansions and edges

| model | objective | final_grids | expansions | base_round_edge_hits | edges_after_expansion |
|---|---|---|---|---|---|
| llama | 1 | {"P": [100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0], "R": [100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0]} | [] | [] | [] |
| llama | 2 | {"P": [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0], "R": [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]} | [{"direction": "high", "factor": 10, "slot": "P"}, {"direction": "high", "factor": 10, "slot": "R"}] | [{"direction": "high", "predictor": "B1", "selected": 100000.0, "slot": "P"}, {"direction": "high", "predictor": "B2", "selected": 100000.0, "slot": "P"}, {"direction": "high", "predictor": "H2", "selected": 100000.0, "slot": "R"}] | [] |
| qwen | 1 | {"P": [100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0], "R": [100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0]} | [] | [] | [] |
| qwen | 2 | {"P": [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0], "R": [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]} | [{"direction": "high", "factor": 10, "slot": "P"}, {"direction": "high", "factor": 10, "slot": "R"}] | [{"direction": "high", "predictor": "B1", "selected": 100000.0, "slot": "P"}, {"direction": "high", "predictor": "B2", "selected": 100000.0, "slot": "P"}, {"direction": "high", "predictor": "H2", "selected": 100000.0, "slot": "R"}, {"direction": "high", "predictor": "RCAP", "selected": 100000.0, "slot": "R"}] | [] |

Penalties are on the summed-loss scale. An expansion adds one ×10 step, once. A setting still on an edge after expansion is reported as best among the tested values.

## 4. Selected settings per generator, objective and predictor

| model | objective | predictor | label | selected_setting | frozen_baseline | hidden_row | n_features | cv_loss | note |
|---|---|---|---|---|---|---|---|---|---|
| llama | 1 | B1 | B1 | P=100000 |  | nan | 1044 | 0.48204 |  |
| llama | 1 | B1+H1 | B1 + H1 | X=10000 | {"F_P": 100000.0} | 12 | 5140 | 0.413074 |  |
| llama | 1 | B2 | B2 | P=100000,R=100000 |  | nan | 2068 | 0.401254 |  |
| llama | 1 | B2+A1 | B2 + A1 | X=10000 | {"F_P": 100000.0, "F_R": 100000.0} | nan | 3092 | 0.382092 |  |
| llama | 1 | B2+H1 | B2 + H1 | X=100000 | {"F_P": 100000.0, "F_R": 100000.0} | 12 | 6164 | 0.384945 |  |
| llama | 1 | H1 | H1 + structure | P=100000 |  | 12 | 4116 | 0.413778 |  |
| llama | 1 | H2 | H2 + structure | P=100000,R=100000 |  | 12 | 5140 | 0.393386 |  |
| llama | 1 | RCAP | Rule-conditioned attention probe | P=10000,R=1e+06 |  | nan | 2068 | 0.389654 |  |
| llama | 1 | REF_structure | Structure-only reference | none |  | nan | 20 | 0.503497 |  |
| llama | 1 | XH1 | Other generator's H1 + structure (reference) | P=10000 |  | 18 | 4116 | 0.397407 |  |
| llama | 2 | B1 | B1 | P=100000 |  | nan | 1116 | 0.0213649 |  |
| llama | 2 | B1+H1 | B1 + H1 | X=1000 | {"F_P": 100000.0} | 12 | 5212 | 0.0161154 |  |
| llama | 2 | B2 | B2 | P=100000,R=10000 |  | nan | 2140 | 0.0176067 |  |
| llama | 2 | B2+A1 | B2 + A1 | X=1000 | {"F_P": 100000.0, "F_R": 10000.0} | nan | 3164 | 0.015674 |  |
| llama | 2 | B2+H1 | B2 + H1 | X=10000 | {"F_P": 100000.0, "F_R": 10000.0} | 12 | 6236 | 0.0162033 |  |
| llama | 2 | H1 | H1 + structure | P=1000 |  | 12 | 4188 | 0.0160586 |  |
| llama | 2 | H2 | H2 + structure | P=10000,R=100000 |  | 12 | 5212 | 0.0160514 |  |
| llama | 2 | RCAP | Rule-conditioned attention probe | P=1000,R=10000 |  | nan | 2140 | 0.015657 |  |
| llama | 2 | REF_structure | Structure-only reference | none |  | nan | 92 | 0.0218296 |  |
| llama | 2 | XH1 | Other generator's H1 + structure (reference) | P=10000 |  | 18 | 4188 | 0.0168147 |  |
| qwen | 1 | B1 | B1 | P=100000 |  | nan | 1044 | 0.340422 |  |
| qwen | 1 | B1+H1 | B1 + H1 | X=100000 | {"F_P": 100000.0} | 18 | 5140 | 0.287956 |  |
| qwen | 1 | B2 | B2 | P=1e+06,R=1e+06 |  | nan | 2068 | 0.292677 |  |
| qwen | 1 | B2+A1 | B2 + A1 | X=10000 | {"F_P": 1000000.0, "F_R": 1000000.0} | nan | 3220 | 0.273442 |  |
| qwen | 1 | B2+H1 | B2 + H1 | X=100000 | {"F_P": 1000000.0, "F_R": 1000000.0} | 18 | 6164 | 0.2741 |  |
| qwen | 1 | H1 | H1 + structure | P=100000 |  | 18 | 4116 | 0.282156 |  |
| qwen | 1 | H2 | H2 + structure | P=100000,R=1e+06 |  | 18 | 5140 | 0.274071 |  |
| qwen | 1 | RCAP | Rule-conditioned attention probe | P=10000,R=1e+06 |  | nan | 2196 | 0.274078 |  |
| qwen | 1 | REF_structure | Structure-only reference | none |  | nan | 20 | 0.356035 |  |
| qwen | 1 | XH1 | Other generator's H1 + structure (reference) | P=100000 |  | 12 | 4116 | 0.289374 |  |
| qwen | 2 | B1 | B1 | P=100000 |  | nan | 1116 | 0.013931 |  |
| qwen | 2 | B1+H1 | B1 + H1 | X=1000 | {"F_P": 100000.0} | 27 | 5212 | 0.0122247 |  |
| qwen | 2 | B2 | B2 | P=100000,R=10000 |  | nan | 2140 | 0.0132242 |  |
| qwen | 2 | B2+A1 | B2 + A1 | X=1000 | {"F_P": 100000.0, "F_R": 10000.0} | nan | 3292 | 0.012514 |  |
| qwen | 2 | B2+H1 | B2 + H1 | X=1000 | {"F_P": 100000.0, "F_R": 10000.0} | 27 | 6236 | 0.0121866 |  |
| qwen | 2 | H1 | H1 + structure | P=1000 |  | 27 | 4188 | 0.012192 |  |
| qwen | 2 | H2 | H2 + structure | P=1000,R=100000 |  | 27 | 5212 | 0.0121223 |  |
| qwen | 2 | RCAP | Rule-conditioned attention probe | P=1000,R=100000 |  | nan | 2268 | 0.0121054 |  |
| qwen | 2 | REF_structure | Structure-only reference | none |  | nan | 92 | 0.0140204 |  |
| qwen | 2 | XH1 | Other generator's H1 + structure (reference) | P=10000 |  | 12 | 4188 | 0.0116078 |  |

Objective 1 loss is the pooled mean log loss per completion; Objective 2 loss is the pooled squared error of the predicted fraction. `P` = prompt-representation penalty, `R` = rule-embedding penalty, `X` = added block in comparisons 5–7 ("off" = CV did not select the block, so the gain is 0 by construction; this is not evidence that the block carries no information).

## 5. Sparse-head fallbacks that fired (Objective 1)

| model | fold | type_id | violated | compliant | groups_with_violation | reason |
|---|---|---|---|---|---|---|
| qwen | 0 | detectable_format:title | 118 | 215342 | 2 | groups_with_violation<5 |
| qwen | 1 | detectable_format:title | 118 | 205082 | 2 | groups_with_violation<5 |
| qwen | 2 | detectable_format:title | 0 | 215460 | 0 | violated<50;groups_with_violation<5 |
| qwen | 3 | detectable_format:title | 118 | 225602 | 2 | groups_with_violation<5 |
| qwen | 4 | detectable_format:title | 118 | 205082 | 2 | groups_with_violation<5 |
| qwen | final | detectable_format:title | 118 | 266642 | 2 | groups_with_violation<5 |

The decision depends only on the fitting rows' labels, so a fallback type contributes the same intercept-only prediction (v + 0.5) / (n + 1) to every method.

## 6. The seven comparisons on training CV (descriptive only)

Positive `right_minus_left` = lower training-CV loss for the left-hand predictor. These are the selection criteria themselves, with no interval and no held-out data. The planned test evaluation (Part 1 and 2 reporting plans) has not been run.

| model | objective | comparison | left | right | cv_left | cv_right | right_minus_left |
|---|---|---|---|---|---|---|---|
| llama | 1 | Primary 1 | H1 + structure | B1 | 0.413778 | 0.48204 | 0.0682624 |
| llama | 1 | Primary 2 | H2 + structure | B2 | 0.393386 | 0.401254 | 0.00786757 |
| llama | 1 | Primary 3 | Rule-conditioned attention probe | B2 | 0.389654 | 0.401254 | 0.0116001 |
| llama | 1 | Primary 7 | H2 + structure | Rule-conditioned attention probe | 0.393386 | 0.389654 | -0.00373254 |
| llama | 1 | Secondary 4 | B1 + H1 | B1 | 0.413074 | 0.48204 | 0.068966 |
| llama | 1 | Secondary 5 | B2 + H1 | B2 | 0.384945 | 0.401254 | 0.0163091 |
| llama | 1 | Secondary 6 | B2 + A1 | B2 | 0.382092 | 0.401254 | 0.0191613 |
| llama | 2 | Primary 1 | H1 + structure | B1 | 0.0160586 | 0.0213649 | 0.0053063 |
| llama | 2 | Primary 2 | H2 + structure | B2 | 0.0160514 | 0.0176067 | 0.00155537 |
| llama | 2 | Primary 3 | Rule-conditioned attention probe | B2 | 0.015657 | 0.0176067 | 0.00194978 |
| llama | 2 | Primary 7 | H2 + structure | Rule-conditioned attention probe | 0.0160514 | 0.015657 | -3.944e-04 |
| llama | 2 | Secondary 4 | B1 + H1 | B1 | 0.0161154 | 0.0213649 | 0.00524952 |
| llama | 2 | Secondary 5 | B2 + H1 | B2 | 0.0162033 | 0.0176067 | 0.00140345 |
| llama | 2 | Secondary 6 | B2 + A1 | B2 | 0.015674 | 0.0176067 | 0.00193275 |
| qwen | 1 | Primary 1 | H1 + structure | B1 | 0.282156 | 0.340422 | 0.0582658 |
| qwen | 1 | Primary 2 | H2 + structure | B2 | 0.274071 | 0.292677 | 0.0186059 |
| qwen | 1 | Primary 3 | Rule-conditioned attention probe | B2 | 0.274078 | 0.292677 | 0.0185986 |
| qwen | 1 | Primary 7 | H2 + structure | Rule-conditioned attention probe | 0.274071 | 0.274078 | 7.284e-06 |
| qwen | 1 | Secondary 4 | B1 + H1 | B1 | 0.287956 | 0.340422 | 0.0524663 |
| qwen | 1 | Secondary 5 | B2 + H1 | B2 | 0.2741 | 0.292677 | 0.0185768 |
| qwen | 1 | Secondary 6 | B2 + A1 | B2 | 0.273442 | 0.292677 | 0.0192346 |
| qwen | 2 | Primary 1 | H1 + structure | B1 | 0.012192 | 0.013931 | 0.00173898 |
| qwen | 2 | Primary 2 | H2 + structure | B2 | 0.0121223 | 0.0132242 | 0.00110189 |
| qwen | 2 | Primary 3 | Rule-conditioned attention probe | B2 | 0.0121054 | 0.0132242 | 0.00111885 |
| qwen | 2 | Primary 7 | H2 + structure | Rule-conditioned attention probe | 0.0121223 | 0.0121054 | -1.695e-05 |
| qwen | 2 | Secondary 4 | B1 + H1 | B1 | 0.0122247 | 0.013931 | 0.00170628 |
| qwen | 2 | Secondary 5 | B2 + H1 | B2 | 0.0121866 | 0.0132242 | 0.00103763 |
| qwen | 2 | Secondary 6 | B2 + A1 | B2 | 0.012514 | 0.0132242 | 7.102e-04 |

## 7. Objective 2: ordered-slot sensitivity and averaged Objective 1 predictions

| model | analysis | predictor | setting | cv_mse | n_features | note |
|---|---|---|---|---|---|---|
| llama | ordered-slot sensitivity | B2o | P=100000,R=10000 | 0.0182001 | 6236 |  |
| llama | ordered-slot sensitivity | B2o+A1 | X=1000 | 0.0159111 | 11356 |  |
| llama | ordered-slot sensitivity | B2o+H1 | X=1000 | 0.0162268 | 10332 |  |
| llama | ordered-slot sensitivity | H2o | P=1000,R=1e+06 | 0.0160592 | 9308 |  |
| llama | ordered-slot sensitivity | RCAPo | P=1000,R=10000 | 0.0158909 | 10332 |  |
| llama | averaged Objective 1 OOF predictions | B1 | Objective 1 selection | 0.0223495 | nan |  |
| llama | averaged Objective 1 OOF predictions | B1+H1 | Objective 1 selection | 0.0170335 | nan |  |
| llama | averaged Objective 1 OOF predictions | B2 | Objective 1 selection | 0.0169555 | nan |  |
| llama | averaged Objective 1 OOF predictions | B2+A1 | Objective 1 selection | 0.0154132 | nan |  |
| llama | averaged Objective 1 OOF predictions | B2+H1 | Objective 1 selection | 0.0152824 | nan |  |
| llama | averaged Objective 1 OOF predictions | H1 | Objective 1 selection | 0.0162082 | nan |  |
| llama | averaged Objective 1 OOF predictions | H2 | Objective 1 selection | 0.0149265 | nan |  |
| llama | averaged Objective 1 OOF predictions | RCAP | Objective 1 selection | 0.0156243 | nan |  |
| llama | averaged Objective 1 OOF predictions | REF_structure | Objective 1 selection | 0.0221029 | nan |  |
| llama | averaged Objective 1 OOF predictions | XH1 | Objective 1 selection | 0.0153014 | nan |  |
| qwen | ordered-slot sensitivity | B2o | P=1e+06,R=100000 | 0.0135427 | 6236 |  |
| qwen | ordered-slot sensitivity | B2o+A1 | X=1000 | 0.0124406 | 11996 |  |
| qwen | ordered-slot sensitivity | B2o+H1 | X=1000 | 0.0122031 | 10332 |  |
| qwen | ordered-slot sensitivity | H2o | P=1000,R=1e+06 | 0.0121917 | 9308 |  |
| qwen | ordered-slot sensitivity | RCAPo | P=1000,R=1e+06 | 0.0124224 | 10972 |  |
| qwen | averaged Objective 1 OOF predictions | B1 | Objective 1 selection | 0.0157417 | nan |  |
| qwen | averaged Objective 1 OOF predictions | B1+H1 | Objective 1 selection | 0.013243 | nan |  |
| qwen | averaged Objective 1 OOF predictions | B2 | Objective 1 selection | 0.0124841 | nan |  |
| qwen | averaged Objective 1 OOF predictions | B2+A1 | Objective 1 selection | 0.0117896 | nan |  |
| qwen | averaged Objective 1 OOF predictions | B2+H1 | Objective 1 selection | 0.0121951 | nan |  |
| qwen | averaged Objective 1 OOF predictions | H1 | Objective 1 selection | 0.0126394 | nan |  |
| qwen | averaged Objective 1 OOF predictions | H2 | Objective 1 selection | 0.0121497 | nan |  |
| qwen | averaged Objective 1 OOF predictions | RCAP | Objective 1 selection | 0.0117949 | nan |  |
| qwen | averaged Objective 1 OOF predictions | REF_structure | Objective 1 selection | 0.0157413 | nan |  |
| qwen | averaged Objective 1 OOF predictions | XH1 | Objective 1 selection | 0.0127836 | nan |  |

## 8. Where the outputs are

- OOF predictions (every tested setting plus `selected`): `results/v2_s3/predictions/<model>/obj<o>/<predictor>.parquet`; Objective 2 ordered-slot: `.../obj2/ordered/`; averaged Objective 1: `.../obj2/avg_obj1/`.
- Selections: `results/v2_s3/cv/<model>/obj<o>/selection.json` (+ `selection_ordered.json`, `avg_obj1_reference.json`).
- Refitted heads (all 50 TRAIN groups): `results/v2_s3/models/<model>/obj<o>/<predictor>.npz/.json`.
- Tables: `results/v2_s3/tables/`. Caches: `data/probing_features_v2/` (split-dependent ones under `split_s3/` for S3).
- Parquet, NumPy and npz files are git-ignored and stay on the Mac; JSON, CSV and Markdown are committed.
