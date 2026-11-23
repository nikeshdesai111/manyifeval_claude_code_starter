# S3 22-group TEST expansion: verified counts, inventory and Betty job plan

Date: 2026-11-23. Basis: `main` at `cd225ab` (split S3 adopted). Handoff: `betty_handoff/claude_betty_s3_22group_generation_handoff.md` (project library), with its four approved corrections. Code: `src/mife/expansion_s3.py`. Queue status: `docs/s3_22group_betty_queue_status.md`.

## 1. Result in one paragraph

The 22 adopted S3 TEST groups (66 tasks) give **30,096 expanded prompts per generator**. All **16,236 original prompts are reproduced byte-for-byte** under their frozen IDs, and their frozen tokenizations are reproduced exactly for both generators. Every one of their **324,720 responses per generator** is on disk, valid and unique, so no original record needs generating. The exact outstanding work is the new subsets only: **277,200 responses per generator, 554,400 in total**, all at k = 2, 3 and 4. This equals the handoff's design count, now confirmed from disk rather than assumed.

## 2. Inputs and what was read

| Input | How it was read |
|---|---|
| Test groups | `mife.splits.test_groups()`, which follows `configs/active_split.yaml` (`split-s3-2026-11-23` → `results/split_s3/proposed_split.json`) and checks the `amendment` field. No hard-coded path, and the manifest's frozen `split` column is not used. |
| Frozen design | `data/manifest/prompts.parquet` (SHA-256 `f63c0aea…`, equal to `results/prespec/FROZEN.json`), `prompt_rules.parquet`, `rule_pools.jsonl`. Read only. |
| Frozen tokenization | `data/tokenized/{llama,qwen}.parquet`. Read only. |
| Existing generations | `data/generations/{llama,qwen}/main/shard_*.parquet`: **provenance columns only** (prompt ID, sample index, model, settings hash, rendered hash, finish reason, token count, seed, backend). Response text is never loaded. The 240 Mac files are MD5-identical to Betty's 240 files. |
| Not read | Response text, verification labels, features, predictions, or any TEST outcome. |

The 22 groups are g03 g04 g10 g11 g14 g17 g19 g20 g21 g24 g27 g30 g33 g36 g48 g54 g56 g58 g62 g63 g64 g65, exactly the handoff table.

## 3. Expanded design (per group: 3 tasks, same 6 exact rules)

| k | Subsets C(6,k) | COA rows | Original configs/task | Expanded configs/task | New configs/task | New prompts per generator (66 tasks) | New responses, both generators |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 1 | 6 | 6 | 0 | 0 | 0 |
| 2 | 15 | 2 | 12 | 30 | 18 | 1,188 | 47,520 |
| 3 | 20 | 6 | 36 | 120 | 84 | 5,544 | 221,760 |
| 4 | 15 | 12 | 72 | 180 | 108 | 7,128 | 285,120 |
| 5 | 6 | 20 | 120 | 120 | 0 | 0 | 0 |
| **All** | **62** | | **246** | **456** | **210** | **13,860** | **554,400** |

**Reconstruction rule.** The six subsets per k that exist in the frozen design keep their frozen chain-prefix base order, so the COA template reproduces exactly the frozen ordered prompts. This matters most at k = 4, where A4 selects 12 of 24 orders (a test shows that re-basing would pick a different 12). New subsets use the group's chain-label order (A…F) as base and the same `mife.design.coa_template(k)`. There are no full permutations and no extra k = 4 prefixes.

**Assertions passed before any input was written** (`build_manifest`, `tokenize`; per group–task in `results/audits/expansion_s3/expanded_manifest_audit.csv`):

- For every one of the 66 tasks: the 246 frozen orders are an exact subset of the 456 distinct expanded orders. No new order equals a frozen one. There are exactly 210 new configurations per task, 18 / 84 / 108 at k = 2 / 3 / 4, with none at k = 1 or 5. Every subset passes `design.check_coa`.
- All 16,236 original prompts are rebuilt with `source.serialize_prompt` and match the frozen `prompt_text`, `prompt_sha256`, `ordered_rule_ids`, `chain_index`, `coa_row`, `config_id` and `subset_id`. Their prompt-rule rows (rule text, character spans, rule hashes) match too. A mismatch would have stopped the build; there were none.
- New IDs follow `{group}-s{chain labels}-k{k}-o{row:02d}-t{task}` (for example `g03-sABD-k3-o02-t…`). They cannot collide with the frozen `{group}-c{chain}-k{k}-…` IDs, and none appears in the frozen manifest.
- Tokenization, for each generator separately: all 16,236 originals give token IDs, rule-token spans, task span and rendered hash identical to the frozen tokenization. This is a versioned, expansion-aware check, not a bypass of `mife.integrity`. New prompts have exact rule-token spans, with 0 straddling. The largest input plus 2,048 output tokens is 2,246 (Llama) and 2,223 (Qwen). That is within the 20% context-margin gate and within vLLM `max_model_len` 4,096.

## 4. Response inventory (exact, from disk)

A record counts as reusable only if all of these hold: its `(prompt_id, sample_index)` is unique; the sample index is in 0–19; the model and the vLLM settings hash match (`bf9020d626e864ec` Llama, `90f7b9ede9aa77ac` Qwen); its `rendered_sha256` matches the tokenization; its finish reason is `stop` or `length`; and its source run is `main` (or the expansion run itself).

| Quantity | Llama | Qwen |
|---|---:|---:|
| `existing_original_s3_test_responses` | 324,720 | 324,720 |
| `missing_original_s3_test_responses` | 0 | 0 |
| Partially complete original prompts | 0 | 0 |
| Invalid-provenance, duplicate or empty records | 0 / 0 / 0 | 0 / 0 / 0 |
| `new_s3_subset_prompts` | 13,860 | 13,860 |
| `new_s3_subset_responses` (expected) | 277,200 | 277,200 |
| `already_generated_for_new_subset_ids` | 0 | 0 |
| **Exact outstanding = 20 × 30,096 − valid unique records** | **277,200** | **277,200** |

The per-group, per-model breakdown is in `results/audits/expansion_s3/response_reconciliation.csv`. Every group has 14,760 original responses found and 12,600 new to generate, per model.

## 5. Generation settings (identical to `main`)

| | Llama | Qwen |
|---|---|---|
| Checkpoint | `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f2…` | `Qwen/Qwen3-8B` @ `b968826d…`, `enable_thinking=False` |
| Decoding | T 0.6, top-p 0.9, top-k off (0 → −1 in vLLM), min-p 0.0 | T 0.7, top-p 0.8, top-k 20, min-p 0.0 |
| Both | 20 samples, 2,048 max new tokens, BF16, no quantization, the same EOS stop IDs, and token-ID input from the checked tokenization (no re-rendering) | |
| Backend | vLLM 0.25.0+cu129, torch 2.11.0+cu129, transformers 5.22.2 (checked in Betty's env), module `cuda/12.9.0`, `max_num_seqs` 512, 500 prompts per `generate` call, prefix caching on | |

The settings hashes are computed from `configs/experiment.yaml` and equal the hashes stored on every `main` record, on both the Mac and Betty.

**Seeding.** Every new record has a fixed per-sample seed, `derive_seed(20261115, "vllm", model, "expansion_s3_20261123", prompt_id, sample_index) % 2^31`. This is the same rule as `generate_vllm.sample_seed`. The seed is written into the checksummed worklist before submission and stored per record (`batch_seed`). On resume, a record counts as done only if its stored seed equals the rule. Two caveats:

1. The original `main` run seeded one engine per chunk, not per sample. The expansion's seeding *mode* therefore differs from `main`, while the sampling distribution and every decoding setting are the same.
2. vLLM output is not bitwise reproducible across batch compositions. The chunk geometry (12 chunks of 1,155 prompts, 500 prompts per call) and `max_num_seqs` are recorded in each run summary so a re-run can match them as closely as vLLM allows.

## 6. Jobs

- **Worklists:** `data/evaluation_expansion_s3/worklist_{llama,qwen}.parquet` holds 277,200 rows each in 12 chunks (`prompt_id`, `sample_index`, `seed`, `chunk`), with checksums in `worklist_*.json` (copied to `results/audits/expansion_s3/`). The runner refuses a worklist or tokenization whose checksum or seeds do not match.
- **Array:** `scripts/betty/expansion_s3_array.sbatch MODEL`, one submission per generator so the two are independent. It runs `--array=0-11` on `dgx-b200` with 1 GPU, 28 CPUs and 40 min per task (12 chunks of 1,155 prompts, 23,100 records each). The geometry is sized for the new workload and for backfill: `main` took about 11–21 min of generation per 4,428 prompts with engine seeding, so a chunk needs about 3–6 min at that speed plus about 2 min of start-up. Per-sample seeds take vLLM's slower per-request sampling path, so 40 min leaves room for a slowdown of several times. A task that times out loses nothing: resubmitting resumes it per record.
- **Output:** `data/generations/<model>/expansion_s3_20261123/shard_cXX_j<jobid>_NNNNN.parquet` (atomic writes, unique per chunk and job) and `run_summary_*.json`. The runner refuses to write into `main`.
- **Preflight (TRAIN only):** `scripts/betty/expansion_s3_preflight.sbatch MODEL` runs the same runner on 20 pilot prompts from S3 TRAIN groups × 20 samples, into run `smoke_expansion_s3`. It measures per-sample-seed throughput.
- **Completeness check (prepared, not automatic):** `scripts/betty/expansion_s3_check.sbatch MODEL` (CPU). It checks that every worklist record exists exactly once with its seed. It reads provenance only and runs no verifier. **TEST verification stays locked** until it is separately authorized.
- **Staging and transfer:** `scripts/betty/expansion_s3_stage.sh code | weights MODEL | fetch MODEL` (run on the Mac). Staging copies only new files. `fetch` copies finished shards back to the Mac and compares MD5s file by file.

**Storage.** The new outputs are about 0.6 GB for both generators, based on `main`'s 2.1 GB for 2.1 M records. Betty home has 23.3 GB free of 50 GB. The pinned BF16 weights for both generators were removed from Betty after the bulk run, and each needs 15 GiB (about 16.1 GB), so only one generator's weights fit at a time. The plan is to run the generators one after the other unless more storage is obtained (see the queue-status file). No unbacked data is deleted to make room. Model weights can be re-copied from the Mac and are checked against `~/mife/model_sha256.txt`.

## 7. Safeguards kept

- The frozen manifest, tokenization, `main` generations, verification tables, feature caches, `results/prespec/` and all V1/V2 artifacts are unchanged. The sidecar lives in `data/evaluation_expansion_s3/` (version `expansion-s3-2026-11-23`).
- There is no reshuffling, no new rule pools and no new tasks. Each group keeps its 3 tasks and 6 exact rules.
- No TEST verifier label, outcome summary or prediction was read to plan this. Writing raw new TEST generations is the only TEST-side action, as authorized by the handoff.
- **Known caveats, unchanged by this run:** the 16 groups that moved from TRAIN into TEST under S3 had their outcomes used in earlier V1/V2 training and tuning. This is the accepted caveat in `docs/amendment_split_s3_2026-11-23.md`, and results should report it. Separately, the pair `capital_word_frequency` + `number_words` appears in 2 TEST pools and 0 TRAIN pools.
- A dated expansion amendment (`docs/amendment_expansion_s3_2026-11-23.md`, **proposed**) records the design, matching and verification protocol before any TEST outcome is inspected. It does not rewrite the 2026-11-14 pre-specification.

## 8. Reproduce (Mac, from the repo root)

```bash
PYTHONPATH=src .venv/bin/python -m mife.expansion_s3 build-manifest
PYTHONPATH=src .venv/bin/python -m mife.expansion_s3 tokenize llama
PYTHONPATH=src .venv/bin/python -m mife.expansion_s3 tokenize qwen
PYTHONPATH=src .venv/bin/python -m mife.expansion_s3 inventory llama
PYTHONPATH=src .venv/bin/python -m mife.expansion_s3 inventory qwen
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_expansion_s3.py
```

## 9. Queue choice (2026-11-23)

Read-only checks on Betty: the account (`zives-cis-academic-curri`) can use QOS `dgx`, `mig`, `genoa-*` and `normal`, none with a priority boost. The other GPU partitions are smaller and busier: `b200-mig90` has 1 node, fully allocated, and estimates 2026-12-31; `b200-mig45` estimates 2027-01-20. That leaves `dgx-b200` (estimate 2026-12-07 for any walltime from 20 to 90 min). Of the 1,236 jobs pending there, 927 are waiting on dependencies and only about 200 on priority. The scheduler's estimate assumes every running job uses its full limit. Earlier waits for this account were 13 min to about 22 h (`sacct`). Priority weights: age 56,000, partition 20,000, QOS 10,000, fairshare 7,000 (this account's fairshare factor is 0.20). Queueing early therefore matters most, because age dominates. The tasks are short (12 × 40 min) so they can backfill into gaps.
