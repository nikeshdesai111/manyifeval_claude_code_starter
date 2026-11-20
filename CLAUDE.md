# Claude Code project instructions — ManyIFEval

## Current scope: probing, evaluation and results

The generation phase is complete: bulk generation, verification and pre-generation features finished on 2026-11-13 (`docs/decision_tracker.md`, §9). `PROMPT_probing_and_results.md` lifts the "stop at the generation milestone" scope below. You are authorised to build feature blocks, train and tune the probe and baseline heads (23 Objective 1 + 23 Objective 2 configurations per generator), evaluate them and write the results to `results/RESULTS.md` and `notebooks/results_explorer.ipynb`. Code lives in `src/mife/probing/`; the file layout is in `docs/probing_schema.md`.

- **Frozen inputs:** the manifest, responses, verification labels and LLM features (run `main`, Apple MPS). Do not regenerate, re-verify, re-extract or edit them. Never mix in `pilot*` or CUDA features.
- **Test-set discipline:** the 15 TEST groups (11,070 prompts) are not read (labels, descriptive statistics, feature scaling) until the pre-specification in `results/prespec/` is frozen and the project owner has said "go". Every choice comes from 5-fold GroupKFold over the 57 TRAIN groups (`results/cv/folds.json`). `tests/test_no_test_leakage.py` guards this.
- **Stop points:** B2 projected above 24 h (ask; continue non-B2 work meanwhile); always before unblinding the test set; after the results are written.
- **Honesty:** every number in `results/RESULTS.md` comes from a saved table in `results/tables/`. Synthetic data is for tests only.

## Phase 1 (complete): generation milestone

The instructions below governed the generation phase. They remain the record of how the frozen data were produced; where they conflict with the scope above, the scope above applies.

### Read first

1. Read `research_plan.md` **in full**. It is the authoritative v3.4 research specification.
2. Read `docs/decision_tracker.md` for a condensed summary. Where the tracker and research plan disagree, flag the discrepancy rather than silently choosing one.
3. Read `README.md` for project setup, and use `START_HERE_PROMPT.md` as the initial implementation request.

### Current scope: stop at the generation milestone

**Implement dataset construction, response generation, deterministic rule verification, pre-generation representation extraction, reproducible storage, and the mandatory 100-prompt pilot. Do NOT train probes, tune probes, implement final evaluation, or run the full dataset without explicit go-ahead following the pilot report.** The downstream evaluation protocol is being revised separately by the research team. Preserve all data needed to support future methods.

#### Fixed design decisions

- ManyIFEval source: 216 task descriptions, 15 instruction types / six broad rule categories; use released code, templates, allowed parameters, compatibility rules and verifiers as ground truth.
- Use **all 216 tasks**. Group them using a documented, **proposed** task-family taxonomy; do not treat task families as official dataset labels or invent final bucket counts.
- Form 72 groups of three tasks. Each group gets a newly sampled, jointly compatible pool of six exact instantiated rules that must be satisfiable for each of the group's tasks (feasibility only; no relevance filter), with **six distinct instruction types**, with parameters sampled from the **released code's default parameter generators** (official IFEval `build_description()` with no arguments, as shipped in ManyIFEval; not the published dataset's empirical values; see `docs/audit_findings.md` M2). **NO predefined frozen pool of exact rules**; exact instances remain fixed only within each group.
- Build six nested cyclic chains using a randomized ordering of the six rules, with prefixes `k=1..5`. Select six distinct subsets per k.
- COA ordering counts per subset for k=1..5: `1, 2, 6, 12, 20`; use mathematical constructions, not arbitrary random permutation subsets; test strength-two pairwise balance.
- Reuse the same 246 ordered rule configurations across each group's three tasks: **53,136 unique prompts** total; **1,026 prompt–rule instances per task**.
- Fixed grouped split: **57 training groups / 171 tasks / 42,066 prompts**; **15 test groups / 45 tasks / 11,070 prompts**. No permanent validation set. Test outputs must never inform pilot tuning or data-generation settings.
- Generators: `meta-llama/Llama-3.1-8B-Instruct` and `Qwen/Qwen3-8B`, each 20 completions/prompt (Amendment A2, 2026-11-09: restored after Amendment A1's reduction to 10 was withdrawn); **Qwen3 must have `enable_thinking=False`**. Full expected responses: **1,062,720 per model / 2,125,440 combined**.
- Mandatory **training-only pilot**: 100 unique prompts selected in advance, **20 at each k**, same prompt texts for both models, 20 completions/prompt/model = **2,000/model, 4,000 total** (executed). Stop after pilot to present findings and request authorization for bulk generation.
- Hardware: 16-inch Apple M4 Max MacBook Pro, 64 GB unified memory. Benchmark genuine local inference rather than assuming CUDA. Prefer non-quantized inference when feasible; do not silently switch to quantized weights.
- Objective 1 will eventually use **type-specific** H1/H2/A1 probe heads, never exact-rule-specific heads. Objective 2 will eventually use shared prompt-level heads. Planned configs: 23 + 23 = 46 per generator; no additional 23 averaged-per-rule-to-overall comparisons. **Do not implement these models yet.**

#### Retain inputs for later probing

Save immutable prompt texts, task labels, exact instantiated rules/parameters/types, positions, chain/subset/COA identifiers, task-group split, canonical rendered chat prompts, per-model token IDs and offsets, rule-token-span alignments, immutable checkpoint/backend/precision/template versions, seeds and all stochastic generations, verifier outputs (binary results for **every included rule**), and computed soft labels. Maintain stable IDs to join everything across files and generators.

For each unique prompt and generating model, perform **one separate pre-generation forward pass** (rather than 20 passes): collect the final-prompt-token hidden state **for every layer if practical**, plus attention from that final token to **each exact rule span**, summarized **per layer/head** by both sum and token-normalized mean, and record spans/token lengths. Preserve full attention tensors only for a small diagnostic sample if practical. Do not save a huge sequence×sequence matrix for all 53,136 prompts. Make it possible to choose layers and aggregation definitions after generation.

Check numerical consistency if generation (e.g., MLX) and introspection (e.g., Transformers/MPS) use different runtimes. Don't claim identical representations if quantization, conversion or kernels differ. Flag any mismatch.

#### Implementation and research integrity

- Separate source-of-truth ManyIFEval integration, manifest generation, model adapters, verifier, feature extractor, persistence, and pilot runner; write tests for each.
- Pin/record original repository revision, model revisions, tokenizer/chat templates, parameters, RNG seeds, Python/package versions, and schema versions.
- Before the pilot, tokenize **all 53,136** prompt manifest entries for both models; validate usable context window including output reserve; refuse silent truncation.
- Make jobs resumable, idempotent and checkpointed. Use Parquet/JSONL for tabular records and chunked NumPy/Zarr or comparable format for numerical arrays. Don't repeatedly save whole pretrained model weights per prompt.
- Clearly distinguish actual completed inference from dry-run/synthetic tests. Never fabricate generation results or substitute made-up ManyIFEval verifiers.
- Don't assume the rule sampling, task classification, or compatibility logic is resolved if the released source code doesn't support the plan; report mismatches and propose a scientifically faithful remedy before changing design.
- Don't add test-based tuning or evaluate probes, final statistical metrics, rule-position effects, or controlled Experiment A–D yet.

#### Working procedure

1. Inspect plan, repository, actual ManyIFEval implementation, and local device/toolchain; report any blockers or material ambiguities.
2. Build manifest and sampling code with reproducible unit tests; audit all expected counts.
3. Implement data schema, verifier, generator, and pre-generation introspection path; run tiny smoke tests.
4. Execute 100-training-prompt pilot on both models **if hardware/model access permits**, record real throughput, memory, feature extraction, verifier health, and runtime projections.
5. Produce a compact pilot report with pass/fail gates, cost estimates, problems, and commands to resume/run bulk inference.
6. **STOP. Do not launch bulk generation without user approval.**
