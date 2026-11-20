# Storage schema v1.0.0

All tabular artifacts are Parquet or JSON(L), written atomically (temp file + `os.replace`). Numerical feature arrays are chunked `.npy`/`.npz` shards with a Parquet index. Model weights are never copied; they are referenced by HF id and revision.

## Join keys

- `prompt_id = "{group_id}-c{chain}-k{k}-o{coa_row:02d}-t{task_key}"` is stable and deterministic.
- `config_id = prompt_id` without `-t{task_key}`. It is shared by the group's three tasks.
- `subset_id = "{group_id}-c{chain}-k{k}"`. `subset_key` is the sorted rule IDs, for unordered-composition matching.
- `rule_id = "{group_id}-r{sampling_index}"` identifies an exact sampled instance within a group. `exact_rule_hash = sha256(type, kwargs)[:16]` audits cross-group recurrence.
- `response_id = "{model_key}:{prompt_id}:s{sample_index:02d}"`.

## `data/manifest/` (model-independent, frozen)

| File | Grain | Key fields |
|---|---|---|
| `prompts.parquet` | prompt (53,136) | prompt_id, group_id, split, task_key, task_family, config_id, subset_id, subset_key, chain_index, k, coa_row, coa_perm, ordered_rule_ids, ordered_type_ids, task_text, task_char_span, prompt_text, prompt_sha256 |
| `prompt_rules.parquet` | prompt × included rule (221,616) | prompt_id, position (1-based), rule_id, chain_label, type_id, category, code_prefix, kwargs_json, rule_text, char_start/char_end (in prompt_text), exact_rule_hash, split |
| `rule_pools.jsonl` | group (72) | six rules (type, kwargs, text, category, hash, chain label), chain_order, task_keys, split, sampler seed, sampler_version (`rule_sampler/ifeval_code_defaults`), full proposal/rejection log, witness response |
| `tasks.parquet` | source task (216) | family, ambiguity flag, alt family, group_id, split |
| `manifest_summary.json` | — | config hash, environment and NLTK hashes, source reproduction check, family counts, groups, split, audits, pool-conformance summary, task-feasibility check (every group–task pair), manifest hash |
| `pool_audit.json` | — | every exact rule checked against IFEval's code-default pool (bounds read from the pinned `instructions.py`); per-type value histograms, relation split, D3 parameter rejections per type |

## `data/tokenized/{model}.parquet` (per model, all 53,136 prompts)

The fields are: prompt_sha256 (manifest prompt text), rendered_text (canonical chat-rendered prompt), rendered_sha256, input_ids, n_input_tokens, rule_token_spans (`[start, end)` token indices in prompt order), rule_span_lengths, rule_span_straddles, and task_token_span. `{model}_audit.json` holds the chat-template hash, length quantiles and the context gate.

## Stale-data guard (`src/mife/integrity.py`)

Prompt IDs encode design positions, not text, so every stored output carries content hashes that readers check against the current manifest and tokenization. Tokenized rows carry prompt_sha256; generation shards carry rendered_sha256; feature shard indexes and all verification outputs carry both. Resuming generation or features, importing a pilot, verifying, and every report that reads these outputs raise `StaleDataError` on a missing hash, an unknown prompt ID or a mismatch.

## `data/generations/{model}/{run}/shard_*.parquet`

One row per response. The fields are: response_id, prompt_id, sample_index, response_text (raw decoded), output_token_ids, n_output_tokens, finish_reason (`stop` | `length`), hit_max_tokens, n_input_tokens, rendered_sha256, batch_id, batch_seed, settings_hash, created_unix, completion_batch_size and source_run (`pilot` for imported rows). Pilot and main runs both hold 20 samples per prompt (Amendment A2). Rows imported from the pilot keep the pilot's settings_hash, which equals that of fresh main rows when settings are unchanged. `pilot_import.json` in the main run records the import rule (`sample_index < 20`), source shards and settings check. `run_summary_*.json` records the frozen decoding settings, weight dtypes, EOS ids, versions, throughput and peak memory.

## `data/verification/{model}/{run}/`

| File | Grain |
|---|---|
| `verification_long.parquet` | response × included rule: violated ∈ {0,1}, type_id, position, rule_id (every verification file also carries prompt_sha256 and rendered_sha256) |
| `verification_responses.parquet` | response: ordered violation_vector, fraction_violated, verifier_version, diag_* measurements |
| `labels_prompt_rule.parquet` | prompt × rule: n, v, p_hat = v/n (Objective 1 label) |
| `labels_prompt.parquet` | prompt: k, q_hat = mean p_hat (Objective 2 label) |

## `data/features/{model}/{run}/`

- `hidden_{shard}.npy` is uint16 `(n, L+2, d)` holding raw BF16 bits; decode with `introspect.bits_to_float32`. Row 0 is the embeddings and rows 1..L are `hidden_states[1..L]`. Row L+1 is the pre-final-norm output of the last layer. The convention for `hidden_states[-1]` is recorded in `features_meta.json`.
- `attn_{shard}.npz` holds:
  - `rule_sum`, `rule_mean`: float32 `(n, L, heads, 5)`, attention from the final prompt token to each rule span; slots follow prompt order and absent slots are NaN.
  - `rule_mask`: `(n, 5)`.
  - `task_sum`, `task_mean`: `(n, L, heads)`.
- `index.parquet` maps prompt_id to (shard, row) and also stores spans, span lengths and the final token id/index.
- `diag_full_attention_{prompt_id}.npz` holds full float16 `(L, heads, seq, seq)` tensors for a few diagnostic prompts only.

BGE-M3 inputs (B1/B2) are reproducible from `prompt_text` and `rule_text`; no embeddings are computed in this phase.

## CUDA runs (Amendment R1)

- `data/generations/{model}/{run}/` for a vLLM run (e.g. `pilot_betty_dgx-b200`) uses the same row schema plus `backend = "vllm"` and `stop_reason`. `completion_batch_size` holds vLLM `max_num_seqs`. `settings_hash` includes `backend` and `sampler_semantics`, so it differs from MLX rows by design. `run_summary_*.json` adds `engine` (dtype, quantization, max_model_len, gpu_memory_utilization), `environment` (torch, vLLM, transformers and CUDA runtime versions, driver, GPU, MIG mode, Slurm ids) and `peak_gpu_memory_used_gb` (nvidia-smi, which includes vLLM's pre-reservation).
- `data/features/{model}/{run}/` has the same layout; `features_meta.json` records `backend = transformers+torch-cuda`, the GPU name and the CUDA runtime.
- `data/consistency/`: `{model}_{run}_vllm_logprobs.parquet` (vLLM top-20 log-probs at the final prompt token), `{model}_{run}_vllm_vs_hf.json`, `{model}_{run}_vs_pilot.json` (cross-backend tokens / hidden states / attention / verifier, with `flags`), and `{model}_sampler_semantics.json`.
- `data/env/{model}_{run}_job{id}.json`: environment report written at job start. `env/betty_requirements.lock` and `env/betty_pip_check.txt`: the Betty environment's frozen packages.

## Feature shards (resumable runs, tracker v1.12)

`data/features/<model>/<run>/` holds:
- `hidden_NNNNN.npy` and `attn_NNNNN.npz`, as before.
- `index_NNNNN.parquet`, one per shard: the **commit marker**, written last, with each prompt's prompt_sha256 and rendered_sha256. Only shards that have one count as done, and only if their hashes match the current manifest.
- `index.parquet`, the union of all shard indexes, rebuilt on every call.
- `features_backend.json`, written before the first shard. It records the device, so a later session can't mix backends.
- `features_meta.json`, which now has a `sessions` list (per-call timing and memory) plus `n_prompts` and `n_requested_missing`.

A features directory without per-shard hashed indexes is refused.
