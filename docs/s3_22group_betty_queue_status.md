# S3 expansion: Betty queue status

**Status: Llama SUBMITTED** (2026-11-23 06:39 UTC, approved by the owner in the project thread): preflight job **11110485**, generation array **11110486_[0-11]** on dgx-b200, both PENDING (Priority). **Qwen: NOT SUBMITTED**, because it waits for Betty storage (below).

## Checked on Betty (login node, 2026-11-23)

| Check | Result |
|---|---|
| SSH (Kerberos `desai7@UPENN.EDU`) | OK |
| Active Slurm jobs for `desai7` | none (no duplicate risk) |
| `main` generations on Betty vs Mac | 240 / 240 files MD5-identical |
| Env `~/mife/envs/mife` | vLLM 0.25.0+cu129, torch 2.11.0+cu129, transformers 5.22.2, pandas 3.1.1, pyarrow 26.0.1 |
| Code and config vs `main` | `generate_vllm.py`, `generate_mlx.py`, `config.py`, `storage.py`, `design.py`, `chat.py`, `cli.py`, `configs/experiment.yaml`, `env.sh`, `setup.sh` all MD5-identical |
| Staged | `src/mife/expansion_s3.py`, `src/mife/splits.py`, `configs/active_split.yaml`, `results/split_s3/proposed_split.json`, `scripts/betty/expansion_s3_*.sbatch`, `tests/test_expansion_s3.py`, `data/evaluation_expansion_s3/{tokenized,worklist}_*` (SHA-256 verified on Betty) |
| On Betty | `pytest tests/test_expansion_s3.py`: 9 passed. Active split `split-s3-2026-11-23`, 22 test groups. Worklists 277,200 records × 2, chunks 0–11 (12 × 23,100 records), checksums and every seed re-derived OK. Settings hashes `bf9020d626e864ec` / `90f7b9ede9aa77ac` equal `main` |
| `sbatch --test-only` | array (llama and qwen), preflight and check scripts all accepted. The scheduler's start estimate for dgx-b200 was 2026-12-07 (1,236 jobs pending in the partition). mig90 and mig45 estimates were later (2026-12-31, 2027-01-20) |
| Storage | home 26.7 / 50 GB used (about 23.3 GB free). `/vast/projects/zives` is not writable |
| **Model weights** | **Missing on Betty**: only BGE-M3 is in `~/mife/hf/hub`. Each generator needs 15 GiB, so only one fits at a time. Llama weights uploaded from the Mac on 2026-11-23 and verified against `~/mife/model_sha256.txt` (4/4 shards OK) |

## Commands (from `~/mife/repo` on Betty, after the go-ahead)

```bash
sbatch scripts/betty/expansion_s3_preflight.sbatch llama
```

```bash
sbatch --array=0-11 scripts/betty/expansion_s3_array.sbatch llama
```

```bash
sbatch --dependency=afterok:<llama_array_job_id> scripts/betty/expansion_s3_check.sbatch llama
```

Qwen runs the same three commands with `qwen`, once its weights are on Betty: either after Llama finishes, its shards are fetched and checked, and the Llama weights are removed (they can be re-copied from the Mac), or after more storage is obtained.

```bash
bash scripts/betty/expansion_s3_stage.sh fetch llama
```

```bash
bash scripts/betty/expansion_s3_stage.sh weights qwen
```

Monitoring:

```bash
squeue -u desai7 -o "%.12i %.20j %.8T %.10M %.20R"
```

```bash
sacct -u desai7 -S 2026-11-23 -X --format=JobID,JobName%22,State,Elapsed,ExitCode
```

## Job IDs

| Job | ID | Submitted (UTC) | State at submission |
|---|---|---|---|
| Llama TRAIN-only preflight | 11110485 | 2026-11-23 06:39 | PENDING (Priority) |
| Llama expansion array, chunks 0–11 | 11110486 | 2026-11-23 06:39 | PENDING |
| Qwen | — | not submitted | waiting for storage |
