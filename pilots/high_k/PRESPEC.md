# K10 dataset: pre-specification (draft v4, 2026-11-26, written before any generation)

**What this is.** This is the project's original design re-run with **up to 10 rules in the prompt instead of 5**:
ManyIFEval chains, COA orderings, 72 groups, the S3 split, the V2 probe and baseline recipe, and Evaluations 1 and 2.
Nicky made this the main dataset, not a pilot (2026-11-26, 18:15 and 18:23 UTC). The rule is **the same design,
analyses and tuning as before; the only intended change is k = 1..10**. Every place where the 10-rule version cannot
or does not match the original is listed in the alignment checklist below and marked **DEVIATION**.

**Status.** Draft awaiting Nicky's approval. Nothing has been generated yet. TEST discipline is unchanged: the 22 S3
TEST groups' prompts are built and generated, but their labels, features and predictions are not read until a TEST
pre-specification is frozen and Nicky says "go". Files live in `pilots/high_k/` (the name is historical). Nothing in
`data/`, `results/`, `src/` or `configs/` is changed.

## Why k stops at 10

ManyIFEval has 15 types, but three sets are mutually exclusive: existence/frequency, the three case types and the
three length types. The largest pairwise-compatible set is therefore 15 - 1 - 2 - 2 = **10 types**. That is also
ManyIFEval's own maximum and the size of every task's `not_conflict_rules_list`. The analogue of "sample 6, chains of
5" is **sample 10, chains of 9**, plus the full pool at k = 10.

## Design (fixed before generation)

- **Groups and split.** All 72 groups and 216 tasks, with the frozen task-family grouping of 3 tasks per group. The
  adopted S3 split is used: 50 TRAIN groups (5 CV folds of 10, `results/cv_s3/folds.json`) and 22 TEST groups.
- **Rule pools.** Each group keeps its **6 frozen exact rules** and gains 4 new ones, giving 10 pairwise-compatible
  types.
  - The 4 new rules are the missing universal types plus one member of each exclusive set the pool does not yet use.
  - Those open slots are filled to balance coverage over the 72 groups: the member used by the fewest groups so far
    wins, with a seeded tie-break.
  - New parameters come from the released code's default generators (`rule_sampler.sample_params`), with the main
    sampler's D3 contradiction proofs and a constructive witness.
  - The same 10 exact rules are used on all 3 of the group's tasks.
- **Feasibility checks** (`build.py design`, written to `data/inputs/pool_checks.csv`). For every group and task:
  - the 10 types are pairwise compatible;
  - there is no D3 contradiction proof;
  - the pool's witness passes all 10 official verifiers with that task's text. The witness satisfies all 10 rules,
    so every subset (every prompt) is satisfiable.

  This is exactly the main dataset's definition: feasibility only, with no relevance filter. For information, the
  check also counts pool types that fall outside ManyIFEval's per-task `not_conflict_rules_list`. The main dataset
  ignored that list, and so does this one. The build stops and lists any group that cannot take 10 compatible rules.
- **Chains.** A seeded random order of the 10 rules gives 10 cyclic chains with nested prefixes k = 1..9, plus the
  full pool at k = 10. Every rule appears in exactly k of the 10 size-k subsets.
- **Orderings.** Each subset gets k(k-1) orderings: 1, 2, 6, 12, 20, 30, 42, 56, 72 and 90 (`code/coa.py`). Position
  balance and immediate-neighbour balance are exact at every k.
  - k = 1..5: the main design's own templates. The build asserts they equal `mife.design.coa_template`.
  - k = 7, 8, 9: AGL(1, q) over GF(q). These are exact strength-two COAs, like k = 5.
  - k = 6 and 10: an exact COA with k(k-1) rows cannot exist. These use k - 1 stacked Williams squares with frozen
    relabelings (`code/orderings_k6_k10.json`, from `code/search_orderings.py`), which brings them close to
    strength-two balance. Ideally every position-pair × rule-pair cell holds exactly 1 ordering. That holds for 73%
    of cells at k = 6 and 59% at k = 10. The rest hold 0 or 2, and some hold 3 at k = 10.
- **TEST expansion: all subsets at every k (as in the original).** The original gave each TEST group every subset of
  its pool, C(6, k) = 6, 15, 20, 15, 6, by adding the subsets the chains miss. K10 does the same: C(10, k) = 10, 45, 120,
  210, 252, 210, 120, 45, 10, 1, i.e. all 1,023 subsets per pool. The chains already hold every subset at k = 1, 9 and
  10, so the expansion adds 35, 110, 200, 242, 200, 110 and 35 subsets at k = 2..8 (932 per pool).
  - Same treatment as the original (`src/mife/expansion_s3.py`): a new subset's base order is the chain-order rank,
    labels A..J follow the chain order, `subset_id` is `{group}-s{labels}-k{k}`, `is_original` is false.
  - Orderings (`--test-orderings williams`, adopted by Nicky 2026-11-26 18:59, **DEVIATION** from the original's full
    COA on these extra subsets only): 2k orderings per new subset, increasing with k (k = 2..8: 2, 6, 8, 10, 12, 14,
    16). Odd k: a Williams square plus its mirror. Even k: a Williams square plus its negated-label twin, because the
    mirror would repeat rows. Each rule sits twice in every position and each ordered pair is adjacent exactly twice;
    pairwise position balance beyond neighbours is not exact. At k = 2 and 3 only 2 and 6 orderings exist, so these
    equal the full COA. Chain subsets on TRAIN and TEST keep their COA orderings.
  - That is 9,250 prompts per task, **610,500 TEST prompts** over 66 TEST tasks. (Full COA would be 1,356,300; one
    ordering would be 61,512. Both remain available as `--test-orderings full|one`.)
- **Size.** 7,500 prompts per group (configurations × 3 tasks; 270 of them at k = 10) × 72 groups =
  **540,000 prompts**, plus the TEST expansion. TRAIN has 375,000. TEST has 165,000 plus 610,500, so **1,150,500 prompts**
  in all per model.
- **Models.** Gemma-2-2B-it, Llama-3.1-8B-Instruct (the frozen checkpoint) and Gemma-2-9B-it. Qwen is dropped. They
  run in that order, and Gemma 2B's TRAIN results are delivered as soon as it finishes.
- **Generation.** vLLM, BF16, the same per-sample seed scheme, at most 2,048 new tokens and **5 completions per
  prompt** (recommended; see Cost).

## Alignment checklist: original design vs K10

| Component | Original (5 rules) | K10 | Status |
|---|---|---|---|
| Source, types, verifiers | ManyIFEval pinned release, 15 types, official verifiers | same | same |
| Tasks, families, groups | 216 tasks, 72 groups of 3 | same groups | same |
| Rule sampling | 6 jointly compatible exact rules, code-default parameters, D3 feasibility on all 3 tasks | frozen 6 + 4 new, using the same sampler pieces and checks | extension |
| Same rules across a group's tasks | yes | yes | same |
| Chains | 6 cyclic chains, prefixes k = 1..5 | 10 cyclic chains, k = 1..9, plus the full pool at k = 10 | extension |
| Orderings | exact COA, k(k-1) rows | exact COA, except near-COA at k = 6 and 10 (position and neighbour balance still exact) | **DEVIATION** at k = 6 and 10 |
| Split | S3: 50 TRAIN / 22 TEST, 5 group folds | same | same |
| TEST expansion | every subset at every k, full orderings | every subset at every k (1,023 per pool); extra subsets get 2k Williams orderings | **DEVIATION** (orderings of the extra subsets) |
| Generators | Llama-3.1-8B, Qwen3-8B | Gemma-2-2B, Llama-3.1-8B, Gemma-2-9B | **DEVIATION** (Nicky's choice) |
| Decoding | per-model settings (Llama: temperature 0.6, top-p 0.9) | Llama unchanged. Gemma has no official sampling defaults, so the proposal is temperature 0.6, top-p 0.9 | **DEVIATION** for Gemma (needs a decision) |
| Completions per prompt | 20 | 5 | **DEVIATION** (cost) |
| Labels | per-rule binary verifier outcome. Objective 1 uses (violated, compliant) counts; Objective 2 uses the soft all-rules-followed fraction | same definitions; counts out of 5 | same, with coarser soft labels |
| Feature pass | one pre-generation pass, HF eager BF16, Apple MPS, all layers stored | the same pass on **Betty CUDA** (the Mac would take about 1 to 3 days per model at 1.15 M prompts). Only the 5 candidate rows (the original's tuned depths; Nicky 2026-11-26 19:57: evenly spread depths, not every layer) and max attention are stored: about 27 / 47 / 41 GB of hidden states for Gemma-2B / Llama / Gemma-9B, plus about 45 GB of attention in all | **DEVIATION** (backend and storage) |
| Hidden rows | Llama 8/12/16/24/33 (depths 0.25/0.375/0.5/0.75 and 1.0 pre-norm) | Llama the same; Gemma-2B 7/10/13/20/27; Gemma-9B 11/16/21/32/43 (same depths, rounded) | same rule |
| A1 / RCAP attention | per layer × head, the max over each rule span, from the final token | same, up to 10 slots | same |
| B1 / B2 baselines | B1 = S + P, B2 = S + P + R, frozen BGE-M3 | same | same |
| Objective 1 heads | 15 per-type ridge-logistic heads. Structure columns: k, position, companion × 15, prompt tokens, span tokens, distance | same columns; k and position now go up to 10 | same |
| Objective 2 heads | one shared ridge fractional-logit head; slot structure × 5 | same; slot structure × 10 | extension |
| Tuning recipe | `configs/v2_recipe.yaml`: grids, tie rule, fallback (50 / 50 / 5 groups), one boundary expansion, layer search, incremental rule | an identical copy, recorded before selection; candidate rows per model as above | same |
| Comparisons | primary 1–4, secondary 5–7 (plus 8 in the paper) | same | same |
| Evaluation 1 | NLL (primary), Brier, AUROC, ECE; paired 22-pool bootstrap, B = 10,000, seed 20261203 | same | same |
| Evaluation 2 | position, append (k → k+1), companion and task effects; observed effects and Δ-MSE skill | same estimands, from the all-subsets expansion | same |
| TEST discipline | sealed until a prespec is frozen and the owner says go | same | same |

## Gemma 2B early check (TRAIN only, descriptive)

As soon as Gemma 2B's generations are verified, report its TRAIN per-rule violation rate by k, by type, and by
position at k = 10, with group-bootstrap CIs. This is descriptive and gates nothing: the dataset is generated in full
regardless.

## Cost (Betty, B200, 1 GPU per array task)

Assumptions: about 400 output tokens per response, and throughputs of ~24k tok/s for Llama (measured), ~18k for
Gemma-9B (assumed) and ~65k for Gemma-2B (assumed). Features ~0.02 s per prompt. 5 completions per prompt.

| | Prompts | Responses per model | Gemma-2B | Llama-8B | Gemma-9B | Total |
|---|---|---|---|---|---|---|
| Generation | 1,150,500 | 5.75 M | ~10 GPU-h | ~27 | ~36 | **~72 GPU-h** |
| Features | 1,150,500 | | ~6.4 | ~6.4 | ~6.4 | ~19 GPU-h |

Rule checks per model: 7.27 M rule rows × 5 = 36.4 M.

Add ~3 GPU-h of start-up. With 8 concurrent array tasks, Gemma 2B takes about 2 hours wall-clock and all three models
about 12 hours. Storage: about 4 to 6 GB of generations per model and about 160 GB of features in all, which needs project (non-home) storage on Betty or shards streamed back to the Mac. Verification is CPU-only.

## Open items before submission

1. Gemma is gated, so the licence must be accepted on Hugging Face. The weights also need a way onto Betty: either an
   HF token there or a copy from the Mac to non-home storage.
2. Betty storage. Home is at about 41 of 50 GB, and the Gemma-9B weights alone are 18.5 GB. Generations take about
   3 GB per model, and features about 30 GB per model even with only 5 rows.
3. Approval of the remaining deviations marked above.
4. Betty must be reachable from the Mac (Penn VPN and a fresh login).

## Files

- `code/build.py`. `design` builds the pools, checks and prompts. `tokenize --model m --samples n` builds the
  tokenization and worklist.
- Orderings: `code/coa.py`, `code/search_orderings.py`, `code/orderings_k6_k10.json`.
- Generation: `code/generate.py`, `code/high_k.sbatch`, `code/stage.sh`.
- Analysis: `code/verify.py`, `code/analyze.py`.
- Still to write after approval: the model registry for Gemma and the CUDA feature extractor. The TEST expansion
  is in `build.py` (`test_expansion`).
