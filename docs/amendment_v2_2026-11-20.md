# Amendment V2 to the Objective 1 / Objective 2 analysis (2026-11-20), training-only milestone

**Status:** adopted by the project owner on 2026-11-20 (spec: `manyifeval_plan_prompt_v3_obj1_obj2.md`, v3, which came out of an independent audit). This milestone is **training-only**. The 15 TEST groups stay sealed: no test-feature extraction, no test-label access, no held-out scoring and no final evaluation freeze. Those need a later, separate authorisation.

## 1. What stays unchanged

- **The 2026-11-14 freeze is preserved:** `results/prespec/analysis_plan.md` and `results/prespec/FROZEN.json` (`unblind_approved: false`, `test_runs: []`), together with every V1 cache (`data/probing_features/`), V1 CV output (`results/cv/`, `results/predictions/`, `results/models/`) and V1 table (`results/tables/`). V2 does not modify any of them. `configs/experiment.yaml`, whose sha256 the freeze records, is not edited. The V2 recipe is in a separate file, `configs/v2_recipe.yaml`.
- **Frozen inputs:** the manifest, the frozen tokenization (input ids and exact rule token spans), generations, verifier labels (`labels_prompt_rule.parquet`, `labels_prompt.parquet`) and the frozen LLM features of run `main` (Apple MPS). The five grouped folds over the 57 TRAIN groups (`results/cv/folds.json`) are also unchanged.
- **One narrow exception (authorised):** a new **max-attention** extraction on **TRAIN prompts only**, run with the original frozen models, tokenizer, token spans and backend (HF Transformers eager attention, BF16, Apple MPS). See §4.

## 2. Training-CV observations from V1 that motivated the changes

All of these come from V1 training-only CV tables (`results/tables/`). No TEST data were used.

| V1 observation (table) | V2 change |
|---|---|
| The selected hidden row was the **shallowest candidate** (row 16 / 50% depth) for Llama Objectives 1 and 2 and for Qwen Objective 1 (`cv_layer_selection.csv`) | The layer search adds rows 8 and 12 (Llama) and 9 and 14 (Qwen), all already stored |
| Many selected penalties sat **on a grid edge**, and several MLP heads chose **epoch 1** (`cv_boundary_hits.csv`, `cv_l2_edges.csv`, `cv_l2_counts.csv`) | Ridge-logistic heads only, a pre-recorded ×10 grid with a one-time boundary expansion, and a deterministic tie rule |
| Many of the 23 configurations had **identical feature sets**, e.g. H2 = H1+H2 and B1+H1 = B1+H2 (`cv_identical_features.csv`) | 7 planned comparisons with defined predictors |
| B1/B2 used one shared, rule-conditioned head while H1/H2/A1 used 15 per-type heads, so architecture differed between the two sides of a comparison (`configs.py`) | Every Objective 1 predictor has 15 per-type heads and the same structure block |
| Fine-tuned B2 cost more than 24 h on the Mac and needed Betty (`b2_projection.csv`, `open_issues.csv`) | B2 = B1 + a frozen exact-rule embedding; LoRA B2 retired |
| A sparse type passed the `≥ 50 violated` gate in one fold but had **1** violated fitting response in another fold (Qwen `detectable_format:title`, `cv_sparse_types.csv`) | Fallback also triggers on group coverage (fewer than 5 fitting groups with a violation) and on fewer than 50 compliant completions |
| A1 SUM and MEAN were within 0.2–0.6% of each other (`cv_pooling_selection.csv`); neither recovers the multi-token maximum used by SAT Probe | A1 = per-layer×head **max** over the rule span |

## 3. Disposition of the 2026-11-14 plan

| Item | Disposition |
|---|---|
| 23 configurations per objective, MLP/linear head grid, single shared L2 | **Superseded** by the V2 predictors, seven comparisons and ridge recipe |
| Fine-tuned (LoRA) B2 | **Retired** |
| Objective 2 sigmoid-bounded MSE fit | **Superseded** by the ridge fractional logit |
| A1 span SUM / MEAN | **Superseded** by max attention; V1 caches kept as historical artifacts |
| Hypothesis contrasts with Holm-adjusted p-values | **Replaced** by effect sizes with explicitly pointwise, unadjusted 95% intervals (later test evaluation) |
| Calibration (ECE, reliability curves) | **Retained for later** test evaluation |
| `english_capital`, truncation and looping sensitivity analyses | **Retained for later** |
| Evaluation 2 controlled analyses A–E | **Deferred** (separate from the Objective 2 predictor trained now) |
| Paired group bootstrap (B = 10,000, seed 20261203) | **Retained** as primary, plus a leave-one-group-out jackknife / t(14) sensitivity |
| Folds, split, frozen labels and LLM features | **Unchanged** |

## 4. Training-only max-attention extraction

`src/mife/probing/v2/maxattn.py` runs one forward pass per TRAIN prompt and generator, using `mife.introspect.load_torch_model` (the same local checkpoint, BF16, eager, MPS) on the frozen `input_ids`. For each layer l, head h and rule slot j it stores `max_{t ∈ span_j} attn_l[h, final token, t]`. The eager kernel returns BF16 probabilities, so this maximum is itself a BF16 value and is stored losslessly as raw BF16 bits. TEST prompt ids are refused. The run is resumable by shard and checks hashes against the current manifest and tokenization. torch and transformers versions must equal those of the frozen feature run, or the run stops.

Validation (`maxattn_checks.py`) was run on fixed TRAIN samples **before** the full pass, and again on the stored values **after** it. Results are in `results/v2/extraction/` and `results/v2/SUMMARY.md`.

## 5. Recorded fitting recipe

`configs/v2_recipe.yaml` was recorded (sha256 and timestamp) in `results/v2/recipe_recorded.json` **before any V2 selection ran**. The engine refuses to select if the file has changed since. In the file, `[D]` marks the owner's decisions and `[P]` marks numerical defaults proposed by Claude. The proposed defaults are:

- **Penalty grids, summed-loss scale (each step ×10):** Objective 1 uses 1e2 … 1e7 for the prompt and rule slots and a fixed structure penalty of 10. Objective 2 uses 1e0 … 1e5 and a fixed structure penalty of 1. Placement was informed by V1 training CV: V1 linear Objective 1 hidden-state heads preferred the strongest V1 penalty, l2 = 1 on the mean scale, which is about 2e5 on the summed scale.
- **Tie precision:** 1e-6 for Objective 1 mean log loss and 1e-8 for Objective 2 MSE.
- **Group-coverage fallback threshold:** 5 (accepted by the owner).
- **Boundary expansion:** one ×10 step, applied once (accepted by the owner). The order and the incremental-block edge convention are written in the recipe.
- **Exact encoding of the structural columns:** these are listed in the recipe and in the cache sidecars.
- **Solver:** L-BFGS-B, finished by damped Newton with the exact Hessian when L-BFGS has not converged.
- **Layer tie rule:** a tie in layer-search loss goes to the shallower row. The spec does not state one.
- **Ordered-slot sensitivity:** uses the main analysis' final grids and does not expand them further.

## 6. Hidden-row convention

Stored rows: 0 = token embeddings; 1..L−1 = outputs of decoder layers 1..L−1; **L = output of decoder layer L after the final RMSNorm** (HF `hidden_states[-1]`); **L+1 = the same layer's output before the final norm** (Llama L = 32, Qwen L = 36). The spec's wording ("1..L = outputs of decoder layers 1..L") is accurate for rows below L. For row L, the stored state is post-norm. No candidate row is L, so this affects documentation only.

| Generator | Row | Meaning | Depth |
|---|---|---|---|
| Llama | 8 / 12 / 16 / 24 / 33 | layer 8 / 12 / 16 / 24 / layer 32 pre-norm | 0.25 / 0.375 / 0.5 / 0.75 / 1.0 |
| Qwen | 9 / 14 / 18 / 27 / 37 | layer 9 / 14 / 18 / 27 / layer 36 pre-norm | 0.25 / 0.389 / 0.5 / 0.75 / 1.0 |

Disclosure: hidden-state methods get a 5-row layer search that the text and attention methods do not get.

## 7. Paths (versioned)

`data/probing_features_v2/` (caches with `*.meta.json` sidecars carrying `version: v2-2026-11-20`, kind and definition) and `results/v2/` (recipe record, CV selection JSONs, OOF predictions, refitted heads, tables, summary). Loaders refuse a V1 file or a mismatched version, kind or definition (`tests/test_v2.py`).

## 8. Recording log (honesty note)

- 2026-11-20 15:25 (local): recipe first recorded. Two smoke tests of the engine followed: Qwen Objective 1 REF_structure and B1 over the 6-value grid, and Llama Objective 2 B1 / B1+H1 at single settings. These produced training-CV losses, for example Qwen Objective 1 B1 at P = 1e6, loss 0.318. No grid value was changed after seeing them.
- 2026-11-20 15:45: the L-BFGS solver was far too slow on real heads. It hit its iteration cap and then needed Newton polishing, at 10–20 s per fit. It was replaced by warm-started damped Newton. Held-out losses agreed to 1e-8 on a test head. All cached CV points were deleted and recomputed with the new solver, and the recipe was re-recorded with only the `solver` line changed. Grids, structure penalties, tie precision, fallback and expansion rules are identical to the first record.
- 2026-11-20 15:48: the Llama Objective 2 warm-up stopped on a Cholesky failure. Its float32-assembled Hessian was numerically indefinite: the structure one-hots are nearly collinear under the weak structure penalty. A diagonal-jitter fallback was added to the Newton step. It is logged per fit, and the optimum is unchanged because convergence is judged on the exact float64 gradient. The recipe was re-recorded with only the `solver` line changed. No selection result had been produced at that point.
- 2026-11-20 16:00: Objective 1 throughput was too low, so the Newton solver was changed to reuse its factorised Hessian while the gradient keeps shrinking quickly. It has the same exact float64 gradient and the same stopping rule. On a test head the held-out losses matched the previous solver to within 3e-8, below the 1e-6 tie precision. Objective 1 warm-ups were restarted. The Objective 2 points already cached (H1 at each Llama and Qwen row) were computed by the previous Newton variant; both variants converge to the same optimum within the stopping tolerance. The recipe was re-recorded with only the `solver` line changed, and no selection had been run.
- 2026-11-20 16:46: two more speed changes. (a) Rule-embedding blocks are replaced, per head and fold, by their coordinates in an orthonormal basis of their row space on the fitting rows. This is exact for ridge (proof in `cv.Engine.prepare`, test `test_low_rank_rule_block_is_exact`). The Objective 2 mean rule embedding is now computed in float64 with a fixed summation order. (b) The float32 Newton Hessian can be assembled on the Apple GPU. The line search, gradient and stopping rule are unchanged, and on a test head the held-out losses matched to 1e-8. All warm-up processes were restarted; cached points not affected by (a) were kept (H1 / XH1 rows). The recipe was re-recorded with only the `solver` line changed, and no selection had been run.
- 2026-11-20 19:00 to 2026-11-21 02:30 (local): the Mac ran short of memory (about 33 GB of swap, mostly from other open applications), and the extraction slowed to between 0.6 and 1.0 s per prompt. The CV processes were deprioritised, then paused (SIGSTOP) until the Qwen extraction finished. The Qwen extraction process was restarted once and resumed from its last committed shard. No extracted value was affected: every committed shard is complete, and the post-run check compares stored values with fresh direct maxima.
- 2026-11-21 04:00–04:45: a helper shell started a duplicate final Objective 1 run for each generator. The duplicates were killed within minutes. The selection is deterministic (cached CV points, fixed tie rule), so this had no effect.
- To save time, the Objective 1 attention predictors (RCAP, and B2+A1 against B2's base-round selection) were computed in a separate cache-filling process. Every selection was then made by `cv.run` from the cache.
