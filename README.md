# ManyIFEval — per-rule violation study (plan v3.6 + Amendment R1)

## Current scope: probing, evaluation and results

The generation phase is complete: bulk generation, verification and pre-generation features finished on 2026-11-13 (`docs/decision_tracker.md`, §9). `PROMPT_probing_and_results.md` lifts the "stop at the generation milestone" scope below. You are authorised to build feature blocks, train and tune the probe and baseline heads (23 Objective 1 + 23 Objective 2 configurations per generator), evaluate them and write the results to `results/RESULTS.md` and `notebooks/results_explorer.ipynb`. Code lives in `src/mife/probing/`; the file layout is in `docs/probing_schema.md`.

- **Frozen inputs:** the manifest, responses, verification labels and LLM features (run `main`, Apple MPS). Do not regenerate, re-verify, re-extract or edit them. Never mix in `pilot*` or CUDA features.
- **Test-set discipline:** the 15 TEST groups (11,070 prompts) are not read (labels, descriptive statistics, feature scaling) until the pre-specification in `results/prespec/` is frozen and the project owner has said "go". Every choice comes from 5-fold GroupKFold over the 57 TRAIN groups (`results/cv/folds.json`). `tests/test_no_test_leakage.py` guards this.
- **Stop points:** B2 projected above 24 h (ask; continue non-B2 work meanwhile); always before unblinding the test set; after the results are written.
- **Honesty:** every number in `results/RESULTS.md` comes from a saved table in `results/tables/`. Synthetic data is for tests only.

Probing-phase commands are listed under *Probing, evaluation and results* below.

## Phase 1 (complete): generation-phase pipeline

This repository implements dataset construction, response generation, rule verification and pre-generation introspection for the ManyIFEval per-rule violation study. It stops at the **100-prompt pilot**. Probes, model comparisons, tuning and evaluation are **out of scope**. Bulk generation is **not** launched without human approval.

- Research spec: `research_plan.md` (v3.4). Decision tracker: `docs/decision_tracker.md`.
- **Source audit and approved decisions D1–D4:** `docs/audit_findings.md`.
- Task-family rubric (provisional, needs human review): `docs/task_family_rubric.md`. Labels: `configs/task_families_v1.csv`.
- Pilot report: `docs/pilot_report.md`.
- Storage schema: `docs/storage_schema.md`.
- **Guide to the dataset, the 23 configurations and the results tables:** `docs/methods_and_results_guide.md`.

## Setup (Apple Silicon, macOS)

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.lock
```

```bash
git clone https://github.com/kenoharada/Multiple-Instructions-Following third_party/Multiple-Instructions-Following && git -C third_party/Multiple-Instructions-Following checkout 5c1ba29e631e2bbee981ebd620898c2750c5ddbf
```

You also need:
- **NLTK tokenizer data**, vendored in `third_party/nltk_data`. Its hashes are recorded in `data/manifest/manifest_summary.json`.
- **Model weights** in the Hugging Face cache at the revisions pinned in `configs/experiment.yaml`. Llama requires accepting Meta's license.

## Pipeline (all commands from the repo root)

```bash
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false
```

| Step | Command | Output |
|---|---|---|
| Frozen manifest (deterministic) | `.venv/bin/python -m mife.cli manifest` | `data/manifest/` |
| Tokenize all 53,136 prompts + context gate | `.venv/bin/python -m mife.cli tokenize llama` (then `qwen`) | `data/tokenized/` |
| Pilot selection (train only) | `.venv/bin/python -m mife.cli pilot-select` | `data/pilot/` |
| Generate pilot (resumable) | `.venv/bin/python -m mife.cli generate llama pilot --pilot --ppb 8` | `data/generations/<model>/<run>/` |
| Verify + soft labels | `.venv/bin/python -m mife.cli verify llama pilot` | `data/verification/<model>/<run>/` |
| Pre-generation features | `.venv/bin/python -m mife.cli features llama pilot --pilot` | `data/features/<model>/<run>/` |
| MLX vs PyTorch consistency | `.venv/bin/python -m mife.cli consistency llama 20` | `data/consistency/` |
| Tests | `.venv/bin/python -m pytest -q tests` | |

**Rule-parameter policy.** Exact rule parameters come only from the released code's default generators (the official IFEval `build_description()` called with no arguments), filtered by compatibility/feasibility checks that resample and log every rejection. They are not sampled from the published dataset's empirical parameter values. This follows the released code's procedure and makes no claim to reproduce the original authors' dataset distribution. Rationale: `docs/audit_findings.md` M2 and `research_plan.md`.

Generation is idempotent. Re-running the same command skips prompts that already have all their samples in the run's immutable shards (20 per prompt; Amendment A2). The main run generates every manifest prompt itself, including the pilot prompts; pilot samples are not imported. **Full-dataset generation needs the flag `--all-train-test-CONFIRMED`; use it only after the pilot is approved.**

## Bulk pre-generation features on the Mac (tracker v1.12)

All bulk features come from **one backend: Apple MPS on the Mac**. Betty's home quota can't hold the ≈ 37 GB. The extractor uses the frozen token IDs in `data/tokenized/`, so it needs no responses and can run while Betty generates.

```bash
scripts/mac/features_main.sh
```

- Runs Llama, then Qwen, ≈ 5.3 h in total.
- Writes `data/features/{llama,qwen}/main/`; logs go to `logs/features_main_<model>.log`.
- It is resumable: rerun the same command after any interruption. Each shard is committed by its `index_NNNNN.parquet`, written last, so a half-written shard is recomputed.
- It refuses to add MPS features to a directory holding CUDA features (or the reverse).
- Done when `features_meta.json` shows `n_prompts` 53136 and `n_requested_missing` 0.

## Module map (`src/mife/`)

| Module | Responsibility |
|---|---|
| `source.py` | Pinned ManyIFEval integration: tasks, official rule templates, prompt serializer (reproduces all 2,160 released prompts), D1 compatibility, official instruction classes |
| `verifier.py` | Official verifiers (same semantics as `utils.test_instruction_following`, langdetect seeded), per-rule binary outcomes, diagnostics |
| `feasibility.py` | D3: explicit unsatisfiability proofs and a constructive witness checked by the official verifiers; per-task feasibility check (no relevance filter) |
| `rule_sampler.py` | Per-group six-rule sampler: parameters from IFEval's code-default pool (official `build_description()` with no arguments), type lookahead/backtracking, D3 resampling, full log |
| `pool_audit.py` | Manifest-build audit: every exact rule inside the code-default pool; per-type histograms, relation split, D3 rejections |
| `integrity.py` | Stale-data guard: generations, features and verification outputs must match the current manifest/tokenization hashes |
| `design.py` | Cyclic chains and COA templates (S2, S3, A4, AGL(1,5)) with balance checks |
| `tasks.py` | Family labels, diversity grouping of all 216 tasks, cluster-aware 57/15 split |
| `manifest.py` | Builds and audits the immutable manifest (53,136 prompts / 221,616 prompt-rule rows) |
| `chat.py`, `tokenize_audit.py` | Chat templates, token IDs, exact rule-token spans, context gate |
| `generate_mlx.py` | MLX BF16 batched generation, resumable shards |
| `verify_stage.py` | Violation vectors, per-rule and overall soft labels |
| `introspect.py` | One PyTorch-MPS eager forward per prompt: all-layer final-token hidden states and rule-span attention (sum and mean) |
| `consistency.py` | MLX vs PyTorch logit agreement on identical token IDs |
| `generate_vllm.py` | CUDA/vLLM BF16 generation from the frozen token IDs, same shard schema; refuses the Mac `pilot` run |
| `consistency_cuda.py` | vLLM vs HF-transformers (CUDA) final-token log-prob agreement |
| `xbackend.py` | CUDA run vs Mac pilot: re-tokenization, hidden-state cosine / max-abs diff, attention, verifier rates per k |
| `sampler_semantics.py` | Measures how MLX's and vLLM's definitions of the same temperature/top-p/top-k differ |

## Running the pilot on Penn PARCC Betty (CUDA)

This re-runs the **same 100 pre-selected TRAIN prompts**, one GPU per model, under a new run id (default `pilot_betty_<partition>`). The Mac pilot (`pilot`) is never overwritten: every CUDA writer refuses that run id and any directory that holds MLX/MPS output. Bulk generation is **not** part of this.

**1. Copy the repo and the frozen data from the Mac** (≈ 250 MB: code, configs, vendored ManyIFEval source and NLTK data, manifest, tokenized prompts, pilot selection, and the Mac pilot outputs used for the comparison). `<you>@<betty-login-host>` is your Betty login.

```bash
rsync -av --exclude .venv --exclude __pycache__ --exclude 'data/generations/llama/bench_*' --exclude data/generations/llama/smoke ./ <you>@<betty-login-host>:~/mife/repo/
```

**2. On the login node: environment, then models.** Everything large goes under `$PROJ` (default `~/mife`; set `PROJ` to project storage once you have it). The setup pins vLLM 0.25.0 + torch 2.11.0 built for CUDA 12.9. Newer vLLM needs CUDA 13, which a CUDA 12.8 driver cannot run; 12.9 builds run on a 12.8 driver. Pip caches are disabled or removed after install. Expected usage: ≈ 32 GB of model weights plus ≈ 8–9 GB of environment.

```bash
cd ~/mife/repo && bash scripts/betty/setup.sh env
```

```bash
source scripts/betty/env.sh && mife_activate && hf auth login
```

```bash
bash scripts/betty/setup.sh models
```

**3. Optional GPU preflight (≈ 1 min).** This prints the GPU, driver, CUDA, torch and vLLM versions.

```bash
srun -A zives-cis-academic-curri -p dgx-b200 --gpus=1 -c 28 -t 00:15:00 bash -c 'source scripts/betty/env.sh && mife_activate && python -m mife.cli env-report'
```

**4. Submit the array** (task 0 = Llama, 1 = Qwen). For a MIG slice, override `-p` and `-c` as shown in the script header. Betty's Slurm filter requires 28 / 14 / 6 CPUs per GPU on `dgx-b200` / `b200-mig90` / `b200-mig45`, with 8 GB of memory per CPU.

```bash
mkdir -p logs && sbatch scripts/betty/pilot_array.sbatch
```

Each task runs, resumably: vLLM generation → verification → HF/CUDA features → vLLM-vs-HF check → comparison against the Mac pilot. Logs are written to `logs/mife-pilot_<job>_<task>.out`.

**5. Copy the results back to the Mac** for the report (replace `RUN` with the run id, e.g. `pilot_betty_dgx-b200`):

```bash
rsync -avm <you>@<betty-login-host>:~/mife/repo/data/ ./data/ --include='*/' --include='*RUN*' --include='*RUN*/**' --include='env/**' --exclude='*'
```

Notes:
- **Sampler definitions differ.** mlx_lm applies top-p/top-k to the untempered distribution and then samples with temperature. vLLM (like HF) applies temperature first. Measured on the Mac pilot, this is ≈ 5 tokens per 512-token response that only MLX could have sampled (`data/consistency/*_sampler_semantics.json`). Cross-backend outcome differences therefore mix kernel and sampler effects.
- **Storage for a bulk run.** Full-run outputs (≈ 2.5 GB of generations, ≈ 37 GB of features for both models) will not fit next to the models in a 50 GB home. Use project storage before any bulk run.

## Probing, evaluation and results (current phase)

Spec: `PROMPT_probing_and_results.md`; file layout: `docs/probing_schema.md`. All commands run from the repo root with
`PYTHONPATH=src .venv/bin/python -m mife.cli ...`. Every step is resumable; nothing reads the TEST split before
`results/prespec/FROZEN.json` records the project owner's approval.

| Phase | Command | Notes |
|---|---|---|
| 0 | `probe-b2-bench` | B2 timing on this machine and the projection table (`results/tables/b2_*.csv`) |
| 1 | `probe-describe` | TRAIN-only descriptive tables and figures |
| 2 | `probe-features --split train` | structure, BGE-M3 and LLM feature caches (`data/probing_features/`) |
| 3 | `scripts/mac/probe_cv_all.sh` | grouped CV for every non-B2 configuration, 4 jobs in parallel |
| 3 (B2) | on Betty: `sbatch scripts/betty/b2_check.sbatch`, then `sbatch --dependency=afterok:<id> scripts/betty/b2_grid.sbatch` | full-vs-LoRA check + LR × epoch grid; copy `data/probing_features/b2_runs/` back, then `probe-cv MODEL OBJ --stage b2` |
| 3 | `probe-final MODEL OBJ` | refit selected settings on all 57 TRAIN groups |
| 4 | `python -c "from mife.probing import cvsummary, prespec; cvsummary.build(); prespec.write_plan(); prespec.freeze()"` | write and freeze the pre-specification, then STOP for approval |
| 5 | `probe-eval MODEL OBJ` | TEST evaluation, once, after approval |
| 6 | `probe-report` | tables, figures and `results/RESULTS.md`; notebook `notebooks/results_explorer.ipynb` |

Tests: `.venv/bin/python -m pytest tests/test_probing.py tests/test_no_test_leakage.py`.
