# Initial Claude Code request: implement ManyIFEval generation infrastructure and run the pilot

Read `research_plan.md` completely and follow `CLAUDE.md` before changing files. Use `docs/decision_tracker.md` as a short reference, but treat the detailed v3.4 plan as the research specification and flag any contradictions.

We are ready to **implement everything up to the point where the responses have been generated and verified**, starting with the **mandatory small pilot**. Our team is still finalizing the metrics and methodology for the eventual probe training/evaluation. **Do not implement probes, model comparisons, hyperparameter searches, or final evaluation yet.** Do not launch the full response-generation run without asking for approval after the pilot.

## Goal

Build a tested, reproducible, Apple-Silicon-compatible research pipeline that:

1. Retrieves and inspects the authoritative original ManyIFEval source, 216 task descriptions, 15 rule types, legal parameter choices, compatibility logic and deterministic verifiers. Pin its release/commit. Do not invent or approximate benchmark definitions.
2. Classifies the task descriptions under the proposed, documented task-family rubric, records ambiguities for human review, uses all 216 tasks, and groups them with fixed seeds into 72 compatible groups of three. Freeze the 57-group train / 15-group test partition *before generation or performance-based choices*. Keep related task variants together; ensure meaningful task-family and rule-type coverage.
3. For **each** three-task group independently samples **six distinct, jointly compatible ManyIFEval instruction types and exact parameter instantiations from the released code's default parameter generators**, with feasibility lookahead/backtracking. The pool must be satisfiable for each of the group's three tasks (feasibility only; no relevance filter). **Do not build a frozen inventory of 30–45 or 45–60 exact rules and do not train one probe per exact rule.** Hold each sampled instance fixed within that group and across its three tasks. Record parameters, rejection/compatibility history, rule texts/IDs/types and sampler RNG state or seeds.
4. Constructs six cyclic nested chains with prefixes k=1–5 and the valid component orthogonal array (COA) orderings 1, 2, 6, 12, 20 for k=1–5. Use mathematical COA constructions rather than sampling random permutations. Test every rule's marginal position balance and every pair's position balance. Assert six distinct k-subsets, exactly 246 unique prompt configurations/task, exactly 1,026 prompt–rule rows/task, and 53,136 prompts overall. Reuse the same 246 ordered configurations across all three tasks in each group.
5. Creates an immutable, deterministic prompt manifest with stable prompt/rule/group/chain/subset/ordering IDs and split labels. Save exact task and rule text, prompt serialization, structured parameters, original rule positions, and model-specific rendered chat template/token IDs. Produce a reproducibility summary, and audit the tokenized lengths of **all 53,136 prompts for each model** (including the reserve for expected output length) so no input truncation ever happens.
6. Implements resumable, idempotent, checkpointed generation for **Llama 3.1 8B Instruct** (`meta-llama/Llama-3.1-8B-Instruct`) and **Qwen3-8B** (`Qwen/Qwen3-8B`, with `enable_thinking=False`). Obtain 20 independent stochastic responses/prompt/model, with exact checkpoint revisions and frozen logged settings. Plan for 1,062,720 responses/model, 2,125,440 total, but **do not execute the full run yet**.
7. Implements the official ManyIFEval rule verifiers and saves a binary violation vector for **every included rule in every generation**, along with raw responses, verifier version/diagnostics and the aggregate empirical labels. Do not collapse all verification information to a single pass/fail outcome.
8. Implements **pre-generation feature extraction once per unique prompt/model**, using the model's actual prompt-processing state **before any completion tokens**: final-prompt-token hidden-state vectors across all transformer layers where feasible, plus final-query-token attention **to each included exact rule's span**, summarized per layer and head by both summed attention mass and mean/token-normalized attention. Save exact token-span alignments, masking, layer/head IDs and numerical precision. Inspect memory. Avoid storing full quadratic attention tensors across the entire dataset; a small diagnostic sample is fine. These features must support future H1/H2/A1 probes *by instruction type* without having to re-run generation. Future B1/B2 full-prompt BGE-M3 inputs must be reproducible from saved text/metadata. Do not fine-tune or train BGE-M3 now.
9. Separately confirm that if MLX is used for high-throughput local generation but another runtime is needed to expose attention tensors, both use faithful, compatible model weights, precision, token IDs and chat templates. Report any meaningful numerical mismatch; never silently mix a quantized generator with a non-quantized probe representation and claim they are the same model.
10. Persists every artifact in a documented schema suitable for later training: canonical prompt data, 20 independent raw responses each, verifier results, final-prompt-token vectors at all selected layers, per-rule-head-layer attention summaries, generation/model/tokenizer configs, deterministic seeds, prompt hashes, cache keys, storage schema versions and logs. Prefer JSONL/Parquet for records and compact chunked numerical arrays for vectors, with stable joins and atomic writes. Do not duplicate pretrained model weight files per prompt. Raw responses must remain available for any future verification updates.

## Important environment assumptions

- Computer: **16-inch M4 Max MacBook Pro, 64 GB unified memory**, macOS; do not assume an NVIDIA GPU or CUDA.
- For inference investigate **Apple MLX** or another genuine optimized Apple Silicon backend; test BF16 if feasible. **Do not silently quantize** to make throughput better; disclose and obtain approval for scientific changes to weight precision.
- Models' context: Llama 3.1 8B Instruct (~128K) and Qwen3-8B (32,768 native tokens, no YaRN unless demonstrably necessary). Confirm actual context limits using the precise tokenized input and a conservative output budget. Qwen3 **thinking MUST be disabled** for all experiments.
- Later analysis will use **per-rule probability** and **expected overall fraction violated**, 23 configurations for each objective per generator. There is **no extra family of 23 aggregate-per-rule comparison models**. This is only to motivate retained schema/features; **don't implement any of those 46 training configurations** now.

## Mandatory pilot and stop/go gate

**Start with the pilot; do not start full-scale generation.**

- After creating the complete prompt manifest and fixed train/test split, select **100 distinct prompts from TRAIN groups only** with a predeclared seed, stratified so **20 prompts come from each k=1,2,3,4,5**. Also cover several task families, instruction types and prompt lengths, including longer prompts. Use the **same 100 prompt texts for both generating models**.
- Generate **20 stochastic completions for every pilot prompt and each model**: **2,000 per LLM / 4,000 total**. Pilot outputs can count toward final totals if exact model, prompt, precision, templates, verifiers and decoding settings remain unchanged.
- Run the ManyIFEval verifiers over all included rules, save their binary outcomes, and validate all expected records and joins.
- Run and verify the pre-generation hidden-state and attention feature extraction for the pilot; do not simply assert this works because the library exposes an option.
- Benchmark realistic local batched inference separately on each model: actual time, aggregate output-token throughput, input/output token length quantiles, memory high-water mark, context and output truncation, verifier health, feature extraction runtime/storage, failures/retries and projected total generation time.
- Check that no test-group outcomes are used for pilot selection, generator optimization, or tuning.
- Produce a compact human-readable **pilot report** with pass/fail gates and actual observed measurements. Explain obstacles, propose concrete fixes and identify which changes would require human research approval.
- **STOP for my approval at the end of the pilot.** Do not launch the 1.4-million-response full run automatically.

## Requested implementation workflow

1. Audit the existing directory, both instruction files and the source ManyIFEval repository. Report precise mismatches or unresolved scientific ambiguities before you codify them.
2. Design a small, modular Python project structure, real configs and a versioned storage schema, then implement the sampling/manifests and COA tests first.
3. Implement proper response verifiers, two model adapters, response storage/resume and pre-generation feature extraction. Smoke-test on genuinely small examples before the pilot.
4. Execute the 100-prompt pilot on both models if actual hardware access, permissions and dependencies are available. Never claim it ran if it didn't.
5. Provide reproducible commands, project README and pilot report, including a clear stop/go recommendation. **Do not implement or run any final probe/evaluation pipeline.**

Prioritize scientific fidelity over quick completion. If the repository's actual task list, instruction taxonomy, compatibility behavior or verifier API disagrees with the research plan, explain the difference and stop for a decision rather than quietly modifying the experiment. Do not create synthetic/fake research data except clearly labeled unit-test fixtures.
