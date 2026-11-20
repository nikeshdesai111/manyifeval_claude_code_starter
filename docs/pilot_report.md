# Pilot report — 100 TRAIN-only prompts × 20 samples × 2 generators (4,000 responses)

Date: 2026-11-09. Hardware: 16" MacBook Pro, Apple M4 Max, 64 GB unified memory (GPU recommended working set 51.5 GB), macOS 15.7.3.
**Everything below is from actual local inference and verification.** Nothing is synthetic. Raw numbers are in `data/pilot/pilot_metrics.json`, `data/consistency/*.json` and `data/tokenized/*_audit.json`.

## Sample-count amendments (2026-11-09)

- **Amendment A1 — withdrawn.** After reading this report, the project owner briefly reduced the main dataset to 10 completions per prompt (≈ 13.7 days projected). No main-run generation happened under A1.
- **Amendment A2 — current.** **20 completions per prompt are restored**, as originally planned (`research_plan.md` v3.6, `docs/decision_tracker.md` v1.9, `configs/experiment.yaml`). Totals: **1,062,720 responses per model, 2,125,440 across both**. Projected generation: **≈ 27 days**.
- The pilot was executed at 20 per prompt, which matches the main settings. The main run nevertheless generates every manifest prompt itself, so pilot samples are not imported into it.

## Recommendation: CONDITIONAL GO, pending your approval of the generation budget and items A–C

All four technical gates pass. The binding constraint is **wall-clock time**. At 20 samples per prompt (Amendment A2), the full run needs **≈ 27 days of continuous GPU time** on this one Mac: ≈ 12.8 days for Llama plus ≈ 14.6 days for Qwen at the settings measured. Without changing any scientific setting this could drop to ≈ 24 days (see B1). **Bulk generation has not been started.**

## Gates

| Gate (plan §7.7) | Result | Evidence |
|---|---|---|
| (i) Context, no truncation, all 53,136 prompts | **PASS** | Max input is 216 tokens for Llama and 193 for Qwen. Max input + 2,048 output is 2,264, against usable budgets of 104,858 (Llama) and 26,214 (Qwen, native 32K, no YaRN). Truncation: 0. No rule-span token covers non-whitespace outside its rule. |
| (ii) Valid, verifiable responses for both generators | **PASS (with note)** | Both models: 2,000/2,000 responses, 0 empty, all 300 prompt-rule labels and 100 prompt labels joined. Qwen output contained no `<think>` tags. Output-cap hits: Llama 0; Qwen 14 (0.7%), all degenerate repetition loops (see P2). |
| (iii) Pre-generation hidden states and rule-span attention | **PASS** | 100 prompts × 2 models, one eager forward pass each. Validated: hidden states finite; mask = k; NaN only in absent slots; rule + task attention mass ≤ 1 for every head; mean = sum / span length. |
| (iv) Speed and memory make the budget feasible | **CONDITIONAL** | Feasible but slow: ≈ 27 days total. Qwen at 64 concurrent sequences ran out of GPU memory twice; it runs reliably at 32 (see P1). |
| No test-group information used | **PASS** | Pilot drawn only from the 57 train groups with predeclared seed 20261114. **0** test prompts have ever been generated, verified or introspected, including smoke and benchmark runs. Batch size was benchmarked on train pilot prompts only. Decoding settings were fixed (D4) before any generation. |

## Measured throughput and resources (MLX 0.32.4 / mlx-lm 0.32.2, BF16 weights verified, no quantization)

| | Llama-3.1-8B-Instruct | Qwen3-8B (thinking off) |
|---|---|---|
| Decoding | T 0.6, top_p 0.9 (model card / generation_config) | T 0.7, top_p 0.8, top_k 20, min_p 0 (model-card *non-thinking*; note that `generation_config.json` holds the thinking-mode defaults) |
| Concurrent sequences | 64 | 32 (64 → Metal OOM, see P1) |
| Aggregate output throughput | **333 tok/s** (692,292 tok / 2,079 s) | **231 tok/s** (261,594 tok / 1,132 s, completed segment) |
| Peak MLX memory | 36.7 GB | 39.3 GB at 32 seqs (44.4 GB before the OOM at 64) |
| Input tokens, median / max | 118 / 211 | 95 / 188 |
| Output tokens, median / p90 / p99 / max | 328 / 560 / 760 / 1,380 | 232 / 494 / 734 / 2,048 |
| Finish reason | stop 2,000 | stop 1,986, length 14 |
| Projected full generation (1,062,720 responses, 20/prompt) | ≈ 368 M tokens → **≈ 12.8 days** | ≈ 290 M tokens → **≈ 14.6 days** |
| Feature extraction (PyTorch 2.15 MPS, eager, BF16) | 0.19 s/prompt median; 0.33 MB/prompt → ≈ 2.8 h, 17.4 GB for 53,136 | 0.17 s/prompt; 0.37 MB/prompt → ≈ 2.5 h, 19.5 GB |
| Verification (CPU) | 2.4 ms/response → ≈ 0.7 h per model | same |

Throughput benchmarks on Llama, train pilot prompts only:
- Static batches: 20 → 208 tok/s, 40 → 286 tok/s, 80 → 247 tok/s. Throughput is bounded by the longest sample in each batch.
- Continuous batching: 64 → 332 tok/s, 128 → 315 tok/s. Continuous batching at 64 was adopted.

## MLX (generation) vs PyTorch-MPS (introspection) consistency

Both runtimes load the same pinned BF16 safetensors (no conversion, no quantization) and use identical token IDs from the same chat template. I compared final-prompt-token next-token distributions on 20 pilot prompts per model:

| | top-1 agreement | top-10 overlap | KL(torch‖mlx) median / max | TV max | max \|Δlogit\| median |
|---|---|---|---|---|---|
| Llama | 20/20 | 0.995 | 5.3e-4 / 9.7e-3 | 0.065 | 0.32 |
| Qwen | 20/20 | 0.970 | 4.7e-5 / 2.1e-2 | 0.093 | 0.66 |

The runtimes are **not bitwise identical**: they use different BF16 kernels. They are the same model and precision, with small kernel-level differences. Features come exclusively from PyTorch, and responses exclusively from MLX. Only final-token logits were compared; hidden states and attention were not compared across runtimes, because MLX does not expose attention probabilities.

## Verifier health and label distribution (observations only; nothing was tuned on them)

| | Llama | Qwen |
|---|---|---|
| Overall rule violation rate | 0.176 | 0.102 |
| By k = 1…5 | 0.13, 0.15, 0.15, 0.18, 0.21 | 0.03, 0.12, 0.08, 0.12, 0.11 |
| Prompt-rules all followed / mixed / all violated (of 300) | 158 / 129 / 13 | 228 / 59 / 13 |
| Hardest types | letter_frequency 0.48, number_paragraphs 0.48, bullets 0.24 | letter_frequency 0.47, number_sentences 0.36, english_capital 0.22 |
| langdetect ≠ "en" | 3 | 2 |

All 15 types were exercised. Note for the methods team: Qwen's labels are heavily zero-inflated, with 76% of prompt-rules showing no violation in 20 samples. This is a property of the generator, not a pipeline fault, but it affects Objective-1 label variance.

## Problems encountered, and what was changed

- **P1. Qwen out-of-memory at 64 concurrent sequences.** Two Metal OOMs occurred, after shards 4 and 13. Peak memory climbs because continuous batching pads every active sequence's KV cache, and Qwen has 36 layers to Llama's 32. Fixes so far:
  - A 4 GB MLX buffer-cache cap with cache clearing after each shard. This alone was insufficient.
  - Qwen resumed at 32 concurrent sequences.
  - Generation resumed from saved shards with no loss.
  - These are engineering changes only: weights, precision, template and decoding are unchanged.
  - **Consequence:** Qwen pilot responses were produced at 64 concurrent sequences (shards 0–13: 56 prompts, 1,120 responses) and 32 (shards 14–24: 44 prompts, 880 responses). Sampling settings are identical, and batch size changes only kernel-level numerics.
- **P2. Qwen degenerate repetition.** 14 responses (0.7%) hit 2,048 tokens. All are repetition loops, concentrated in 6 prompts (5/20 samples on one all-caps prompt). All 14 fell in the 32-concurrency segment, but they are prompt-concentrated; the pilot cannot separate a batch-size effect from prompt effects. They are kept and verified as-is, and flagged with `hit_max_tokens`.
- **P3. Provenance bug, now fixed.** Qwen's three attempts were all labelled `batch_id = pilot-call000` with the same segment seed, because crashed calls wrote no summary. The samples are not duplicated: each attempt generated disjoint prompts. The shard → attempt → concurrency mapping is reconstructed from logs in `data/generations/qwen/pilot/segments_provenance.json`. Segment IDs are now unique per call, and each row records its concurrency.
- **P4. Source-benchmark issues found during audit** (`docs/audit_findings.md`):
  - Implicit compatibility rule (D1).
  - 166 released arguments hand-copied from the original IFEval prompts, 90 of them outside IFEval's code-default pool; rule parameters are sampled from the code-default pool only (D2).
  - Two released prompts contain unsatisfiable "less than 1 sentences" rules; our sampler rejects this value as a provable contradiction (D3).
  - A latent keyword contradiction in task 1381.
- **P5. Verifier determinism.** The official case checkers call `langdetect`, which is random unless seeded; it is seeded with 0. The NLTK punkt data is vendored and hashed. Our wrapper matches `utils.test_instruction_following` on all tested cases.

## Decisions needed

**A. Go / no-go on bulk generation.** Running about 27 days of continuous generation (or about 24 with B1) on this Mac is a resource decision. Still awaiting your go. The pipeline is resumable at prompt granularity, so it can run in interrupted sessions. Nothing has been launched.

**B. Engineering choices (no scientific change; my recommendations):**
1. Use the new generator-recycling option (`--recycle N`) to try running Qwen at 64 concurrent sequences again (≈ 260–300 tok/s observed before the OOMs). Validate this on a train-only subset first.
2. Fix one concurrency setting per model for the entire bulk run.
3. Retain all 20 pilot samples per prompt within the full totals: model, prompts, precision, template, verifier and decoding are identical. Alternatively, regenerate Qwen's pilot if you want a single concurrency setting throughout.

**C. Scientific choices that require your approval (none are recommended now):**
- Quantization for speed (would change the generator).
- Lowering `max_new_tokens` (Llama never exceeded 1,380 tokens; the longest code-default length rule is "at least 500 words").
- A repetition penalty for Qwen's loops.
- Fewer samples per prompt. (Adopted as Amendment A1, then withdrawn by Amendment A2: 20 per prompt stands.)
- A longer queue or more hardware.

**D. Human review still outstanding:**
- Task-family labels: 68 of 216 are flagged ambiguous (`configs/task_families_v1.csv`, `docs/task_family_rubric.md`).
- Related-task clusters.
- The D3 proof list.
- Whether Qwen's degenerate cap-hit responses stay in the label set.

## Reproduce and resume

The pilot is complete. Re-running the script is a resumable no-op for completed stages.

```bash
scripts/run_pilot.sh 64 32
```

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests
```

Bulk generation, **only after approval**. The generation step is resumable, and shards are immutable. Generation draws 20 samples per prompt (Amendment A2). The main run generates every manifest prompt itself, including the 100 pilot prompts, so pilot samples are not imported (`import-pilot` remains available but is not part of the bulk procedure). Recycling is shown for Qwen as a memory safeguard over long runs.

```bash
PYTHONPATH=src TOKENIZERS_PARALLELISM=false .venv/bin/python -m mife.cli generate llama main --all-train-test-CONFIRMED --ppb 8 --cbs 64
```

```bash
PYTHONPATH=src TOKENIZERS_PARALLELISM=false .venv/bin/python -m mife.cli generate qwen main --all-train-test-CONFIRMED --ppb 8 --cbs 32 --recycle 40
```

Then run `verify <model> main` and `features <model> main --all` for each model.
