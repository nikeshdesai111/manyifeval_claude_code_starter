**EXPERIMENTAL DESIGN**

Predicting Per-Rule Violations  
Using ManyIFEval

Complete dataset construction, probing, and controlled evaluation plan

Working research specification • Version 3.6 (Amendment A2, 2026-11-09: 20 sampled completions per prompt restored; Amendment A1 withdrawn) • Version 3.4 base (pilot-first implementation on Apple M4 Max; all 216 tasks; 57/15 grouped train/test; compact training-only CV; two generators; 23+23 configurations) • November 9, 2026

**Purpose**

An editable end-to-end plan using all 216 tasks of the ManyIFEval source inventory, formed into 72 three-task groups using a proposed task-family taxonomy. For each three-task group, sample six jointly compatible instruction instances directly from the original ManyIFEval 15 rule types, with exact parameters drawn from the released code's default parameter generators (each type's official `build_description()` called with no arguments; see the rule-parameter sampling policy below); feasibility checks resample only provably unsatisfiable or incompatible draws. NO global frozen inventory. Keep the six sampled exact rules fixed within that group, construct six cyclic nested chains through k = 5 and COA orderings, then collect 20 generations per prompt. Objective 1 uses INSTRUCTION-TYPE-SPECIFIC probes (15 heads when all types are covered), not probes per exact rule; Objective 2 uses shared prompt-level overall heads. Each generating LLM receives 23 Objective 1 and 23 Objective 2 configurations. There is NO permanent validation split: 57 train and 15 test groups, with five-fold grouped cross-validation restricted to training for hyperparameter/early-stopping decisions. No averaged per-rule-to-overall comparison family. Implementation begins with a 100-prompt-per-generator local feasibility and throughput pilot before any bulk response generation; an expanded four-group end-to-end pilot is optional, not the required first step.

Status: The numbered design choices and counts below are proposed experimental specifications, subject to pilot validation. This document is intended to be updated as decisions are finalized. Current execution status (pilot, bulk generation, features) is kept in `docs/decision_tracker.md`.

**Rule-parameter sampling policy (decided)**

Every exact rule's parameters come from the **default parameter generators in the released code**: the IFEval instruction classes shipped in the pinned ManyIFEval repository, each called with `build_description()` and no arguments (`src/mife/rule_sampler.py`). We do **not** sample from the empirical distribution of parameter values in the published ManyIFEval dataset. Rationale:
1. **Reproducibility.** The released code defines each generator explicitly (uniform integer ranges, 50/50 relations, two distinct `WORD_LIST` keywords, a uniform lowercase letter). With fixed seeds, every draw can be reproduced exactly.
2. **No invented distributions.** The published dataset contains some values outside the code's default ranges, but the process that produced them is not documented. We therefore do not construct a distribution for them, for example uniform between the minimum and maximum observed values.
3. **Methodological consistency.** The goal is a new, controlled benchmark built on ManyIFEval's instruction types, templates and verifiers, not a reproduction of the published dataset's empirical parameter distribution.
4. **Compatibility and feasibility.** Default draws pass through checks that reject rules which are unsatisfiable or contradict the group's other rules (D1 type compatibility, D3 provable contradictions with a witness response). A rejected draw is resampled from the same default generator. Every proposal, rejection (with its proof) and resampling decision is logged per group, and per-type rejection rates are reported, because this filtering changes the final parameter distribution slightly (for example, "less than 1 sentence" never survives).

**Scope of the claim:** this follows the released code's default sampling procedure. It does **not** claim to reproduce the distribution the original authors used to generate the published dataset.

**Rule–task compatibility (decided)**

The only task-level requirement is **feasibility**: each group's six-rule pool must not be impossible to satisfy for any of the group's three tasks. There is no relevance or appropriateness filter. A pairing that is unusual but possible is kept, for example a tweet with "at least 17 sentences", a short-answer task with a title and quotation marks, or "no commas" for a task that quotes a sentence containing commas. The released ManyIFEval prompts contain the same kinds of pairings. None of the 15 official verifiers reads the task text, so satisfiability does not depend on the task. The manifest build still checks it explicitly: the pool's witness response must pass every official verifier with each task's text. Because the witness satisfies all six rules, it satisfies every subset, and so every prompt built from the pool.

**Amendment history (sampled completions per prompt)**

- **Amendment A1 (v3.5, 2026-11-09) — withdrawn.** After the 100-prompt pilot, the project owner reduced the main dataset from 20 to 10 completions per prompt to halve the projected local generation time. Under A1 all response totals would have been halved, and pilot samples 0–9 would have been retained. No main-run generation was performed under A1.
- **Amendment A2 (v3.6, 2026-11-09) — current.** The project owner reversed A1 and restored **20 completions per prompt**. Counts: **53,136 × 20 = 1,062,720 responses per LLM, 2,125,440 across both**; TRAIN 841,320 and TEST 221,400 per LLM; labels use nᵢ = 20. Pilot completions (20 per prompt per model) count within the main totals when they were generated from the frozen manifest with the bulk run's backend and decoding settings. Projected local generation time at measured pilot throughput is ≈ 27 days (Llama ≈ 12.8, Qwen ≈ 14.6). Neither amendment used test-group data or probe results.

**Runtime amendment R1 (2026-11-10)** — the generation hardware changes; the design does not. After a CUDA re-run of the 100-prompt TRAIN-only pilot on Penn PARCC Betty (one NVIDIA B200 per model; report in `docs/pilot_report_betty.md`), the project owner approved the bulk run there:
- Generation uses vLLM 0.25.0 in BF16 with no quantization, sampling with vLLM's standard order (temperature, then top-k, then top-p), which is the order the model cards assume. The Mac's mlx_lm instead filters before applying temperature.
- Pilot responses generated on Betty with the bulk settings count toward the main totals; Mac (mlx_lm) pilot responses do not.
- Pre-generation features are computed with HF Transformers (eager, BF16) on **one backend for all 53,136 prompts, never mixed**. Changed 2026-11-10 by the project owner from "CUDA only, deferred until project storage exists" to **Apple MPS on the Mac** (Betty home quota cannot hold the ≈ 37 GB of features). The Mac run reuses the frozen token IDs; no responses are regenerated. Neither backend reproduces vLLM's generation-time numerics exactly; within-backend consistency is what is required.
- Counts, checkpoints, decoding values, verifier and feature definitions are the same on both backends.
- Cross-backend checks found identical prompt tokens, hidden-state cosine ≥ 0.998 and indistinguishable violation rates per k. Representations are not identical across backends, so features must never be mixed between them.
