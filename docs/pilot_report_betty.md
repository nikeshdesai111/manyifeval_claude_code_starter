# Pilot report (Betty / CUDA): 100 TRAIN-only prompts × 20 samples × 2 generators

Date: 2026-11-12. Penn PARCC Betty, partition `dgx-b200`, one NVIDIA B200 (183 GB, MIG off) per model, driver 580.95.05. Slurm array job 10426227, both tasks on host `dgx027`, run id **`pilot_betty_dgx-b200`**.

**Everything below is from actual GPU inference and verification.** Nothing is synthetic. Raw numbers are in:
- `data/pilot/pilot_metrics_pilot_betty_dgx-b200.json`
- `data/consistency/{llama,qwen}_pilot_betty_dgx-b200_{vs_pilot,vllm_vs_hf}.json`
- `data/env/*job10426227.json`
- `logs/betty/mife-pilot_10426227_{0,1}.out`

**0** test-split prompts were generated, verified or introspected, and nothing was tuned on any outcome. All result files were copied to the Mac and checked against Betty by MD5: 42 of 42 files are identical.

## Result: all gates pass

Every gate passes. Prompt tokens match the Mac exactly. Hidden states agree closely across backends but are **not identical**. Two items need attention from the methods team; neither is a pipeline fault:
- An automatic attention flag fired (see below). It does not affect the bulk run, because bulk features come from one backend only.
- The official verifier's language check fails many all-caps English responses (see Verifier health).

The bulk run is Slurm job 10510610 (24 chunk tasks), submitted with `--exclude=dgx027` because one GPU on that node (`GPU-4a9de406-9073-b2a1-ab6a-fdd2fb0e4351`) returned `cudaErrorDevicesUnavailable` at engine start. Verification (10510611) runs after all chunk tasks succeed.

## Gates

| Gate | Result | Evidence |
|---|---|---|
| (i) Context, no truncation | **PASS** | Inputs: max 216 (Llama) / 193 (Qwen) tokens, plus 2,048 output, within `max_model_len` 4,096. Input truncation 0. The all-53,136 audit uses the same frozen token IDs. |
| (ii) Valid, verifiable responses | **PASS** | 2,000/2,000 responses per model, 0 empty, all 300 prompt-rule and 100 prompt labels joined. No `<think>` tags. Cap hits: Llama 4, Qwen 16 (see Verifier health). |
| (iii) Features | **PASS** | One eager BF16 forward pass per prompt on CUDA: all-layer final-token hidden states and rule-span attention sums and means. Same layout and validation as on the Mac. `hidden_states[-1]` convention identical (after the final norm). |
| (iv) Speed and memory | **PASS** | See the table below. |
| Precision / provenance | **PASS** | vLLM reports `dtype=bfloat16, quantization=None`. Weights are byte-identical to the Mac's: copied from the Mac's HF cache, with all 9 safetensors SHA-256-verified. Versions are recorded per job. |
| No test information | **PASS** | Same 100 predeclared TRAIN prompts; decoding values unchanged. |

## Throughput and resources (vLLM 0.25.0, torch 2.11.0+cu129, transformers 5.22.2)

| | Llama-3.1-8B-Instruct | Qwen3-8B (thinking off) |
|---|---|---|
| Output throughput, generation only | **24,247 tok/s** (691,895 tok / 28.5 s) | **18,776 tok/s** (542,864 tok / 28.9 s) |
| Job wall time (load + compile + generate + verify + features + checks) | 7 min 34 s | 7 min 28 s |
| Model load incl. compile (cached graphs) | 118 s | 119 s |
| GPU memory (nvidia-smi) | 153 GB | 154 GB |
| Output tokens median / p90 / p99 / max | 307 / 573 / 887 / 2,048 | 215 / 469 / 927 / 2,048 |
| Features (CUDA, eager BF16) | 0.017 s/prompt | 0.021 s/prompt |

The GPU memory figure is vLLM's up-front reservation (`gpu_memory_utilization` 0.85), not what the model needs. Throughput comes from one `generate` call of 2,000 sequences (`max_num_seqs` 512), including the tail as long responses finish. The bulk run's 10,000-sequence calls should be at least as fast.

## Bulk projection on B200 (1,062,720 responses per model, 20 per prompt)

| | Llama | Qwen | Both |
|---|---|---|---|
| Generation GPU-hours (pilot throughput × pilot length distribution) | 4.2 | 4.3 | **≈ 8.5** |
| Per chunk task (12 per model, 4,428 prompts each): generation + ~4 min start-up | ≈ 25 min | ≈ 25 min | 40-min limit |
| Verification (CPU, separate job) | ≈ 0.7 h | ≈ 0.7 h | 1.5 CPU-h |

Bulk features are computed on the Mac (Apple MPS, same extractor and layout) for all 53,136 prompts, not on Betty. Wall-clock time on Betty is dominated by the **queue**, not by compute.

## Storage projection (full run, both models)

| Item | Llama | Qwen | Where |
|---|---|---|---|
| Generations (parquet) | 1.26 GB | 0.93 GB | Betty home, then copied to the Mac |
| Verification | 0.08 GB | 0.08 GB | Betty home, then copied to the Mac |
| Features | 17.4 GB | 19.5 GB | Mac only |

Betty needs ≈ **2.4 GB** for the bulk outputs. Home held **47.31 / 50 GB** after this pilot. This pilot's CUDA features (0.22 GB) were then removed from Betty, after being copied to the Mac and checksum-verified, which leaves ≈ 2.9 GB free. Generation shards are written atomically, so a quota failure would end a chunk task cleanly, and resubmitting resumes it.

## Cross-backend consistency (Mac MPS features vs Betty CUDA run)

**Tokens: identical.** Re-tokenizing all 100 pilot prompts on Betty reproduces the Mac's token IDs, rule and task spans, and rendered-text hashes, with 0 mismatches for both models. EOS IDs are identical, and so is the transformers version (5.22.2).

**vLLM vs HF on CUDA (final-prompt-token next-token distribution, 20 prompts):**

| | Top-1 agreement | Top-10 overlap | max \|Δ log-prob\| on vLLM top-20 (median / max) |
|---|---|---|---|
| Llama | 19/20 | 0.98 | 0.22 / 0.38 |
| Qwen | 20/20 | 0.98 | 0.38 / 1.25 |

- The single Llama disagreement is a near-tie: the top token has probability 0.36 (HF) vs 0.35 (vLLM), and the top-10 sets are identical.
- These are different kernels running the same weights, not different models.

**Hidden states, Mac MPS vs CUDA (same transformers code, eager, BF16):**
- Embeddings are bit-identical.
- Across every prompt and layer, cosine similarity is ≥ **0.9993** (Llama) and ≥ **0.9982** (Qwen).
- Relative L2 difference grows with depth: median 0.2% → 2.3% (Llama) and 0.1% → 3.5% (Qwen). The worst cases are 3.7% and 6.1%, at the last layers.
- **Not identical.** Bulk features therefore come from **one backend only** (the Mac); the CUDA features here are diagnostic and are never mixed in.

**Attention, FLAGGED:**
- The largest difference in a rule-span attention *sum* is **0.039** (Llama) and **0.066** (Qwen), which exceeds the preset 0.02 threshold.
- Typical differences are small: median 0.0002 / 0.0003, p99 0.006 / 0.012.
- Token-normalized means differ by at most 0.004 / 0.006, and the NaN layout (absent rule slots) is identical.
- This looks like BF16 numerical noise on a few sharply peaked heads. Because the bulk features come from one backend, it does not affect them.

**FP32 accuracy reference:** pending. `src/mife/fp32_reference.py` compares both backends' BF16 features against an FP32 CPU pass. It needs the Mac's memory, so it runs after the bulk feature extraction finishes.

## Verifier health and label distribution (observations only; nothing was tuned on them)

| | Llama | Qwen |
|---|---|---|
| Overall rule violation rate | 0.133 | 0.079 |
| By k = 1…5 | 0.15, 0.08, 0.13, 0.12, 0.16 | 0.06, 0.06, 0.02, 0.10, 0.11 |
| Prompt-rules all followed / mixed / all violated (of 300) | 183 / 108 / 9 | 232 / 59 / 9 |
| Hardest types | capital_word_frequency 0.38, number_paragraphs 0.38, number_sentences 0.28 | capital_word_frequency 0.30, english_capital 0.25, number_words 0.18 |
| langdetect ≠ "en" | 4 | 54 |
| Output-cap hits (2,048 tokens) | 4 (0.2%), 2 prompts | 16 (0.8%), 10 prompts |

All 15 instruction types were exercised. Qwen's labels are heavily zero-inflated: 77% of prompt-rules show no violation in 20 samples, and the median prompt-level violation fraction is 0. This is a property of the generator, but it affects Objective-1 label variance.

**langdetect fails all-caps English (methods team, please review):**
- `change_case:english_capital` passes only if the response is all upper case *and* `langdetect` returns "en". This is the official ManyIFEval/IFEval checker, used unchanged.
- `langdetect` labels much all-caps English as German. Of Qwen's 54 non-"en" responses, 48 are all-caps; all 48 are detected as English once lowercased. Llama has 3 such cases.
- Effect on `english_capital` (340 instances per model): Qwen 84 violations, **48 of them only from the language check** (0.247 → 0.106 without it). Llama: 35 violations, 2 from the language check (0.103 → 0.097).
- No checker was changed. Every response and its diagnostics (`diag_is_upper`, `diag_langdetect`) are stored, so labels can be re-derived later under any rule the methods team agrees on.

**Cap hits:**
- Llama: all 4 are degenerate repetition, mostly of all-caps words. Three come from one prompt (`g58-c5-k5-o12-t2247`, k = 5, including `capital_word_frequency`), which also hits the cap twice for Qwen.
- Qwen: 8 of 16 are exact repetition loops. The rest are long formulaic text that never ends (lists, verses, an all-caps letter).
- All are kept, verified as-is, and flagged with `hit_max_tokens`.

## Problems encountered and fixes (all engineering; no scientific setting changed)

1. **Hugging Face token avoided.** Both pinned snapshots were copied from the Mac's HF cache instead of downloaded, and SHA-256-verified. `setup.sh models` skips the download when snapshots are present.
2. **Slurm CPU-per-GPU filter.** Betty requires exactly 28 / 14 / 6 CPUs per GPU (dgx-b200 / mig90 / mig45). Scripts follow this.
3. **Preflight:** `sbatch --wrap` runs under `/bin/sh`, which has no `source`, so preflight runs from `preflight.sbatch`.
4. **Preflight:** `torchcodec` (video, unused) needs FFmpeg system libraries that Betty lacks, and broke `import vllm`, so it was uninstalled. vLLM is designed to run without it. This is recorded in `setup.sh`; `pip check` notes the missing optional package.
5. **Preflight:** FlashInfer compiles Blackwell kernels on first use and needs `nvcc`, so the jobs load Betty's `cuda/12.9.0` module, which matches torch cu129. Compiled graphs are cached in home and reused (`torch.compile` 17 s).
6. **Driver:** the B200 driver supports **CUDA 13.0**. "CUDA 12.8 only" describes the toolkit modules, not the driver.
7. **Queue:** waiting dominated. The pilot was still pending at 21:48 on 2026-11-11, started at 00:14 on 2026-11-12, and ran for 7.5 minutes.
8. **Login:** SSH uses Kerberos (macOS `/usr/bin/kinit`; Anaconda's `kinit` writes a ticket cache that macOS `ssh` does not read) plus the GlobalProtect VPN.

## Bulk run settings

- Betty, vLLM 0.25.0, BF16, vLLM sampler order (temperature first), run id `main`.
- 12 contiguous manifest chunks per model, as array tasks 0–23 (`scripts/betty/bulk_array.sbatch`). Seeding: one engine seed per chunk task, derived from the generation seed, backend, model, run and chunk; it is recorded in the `batch_seed` column.
- The main run generates every manifest prompt itself, including the 100 pilot prompts.
- Verification: `scripts/betty/verify_bulk.sbatch`, after all chunk tasks succeed.
- Features: the Mac, `scripts/mac/features_main.sh`.

## Reproduce

On Betty, from `~/mife/repo` (resumable; reruns are no-ops for finished stages):

```bash
sbatch scripts/betty/pilot_array.sbatch
```

On the Mac, copy the results and recompute the metrics:

```bash
rsync -avm desai7@login.betty.parcc.upenn.edu:/vast/home/d/desai7/mife/repo/data/ ./data/ --include='*/' --include='*pilot_betty_dgx-b200*' --include='*pilot_betty_dgx-b200*/**' --include='env/**' --exclude='*'
```

```bash
PYTHONPATH=src .venv/bin/python -c "from mife import pilot_metrics as P; P.run({'llama':'pilot_betty_dgx-b200','qwen':'pilot_betty_dgx-b200'})"
```
