# Calibration pilot, round 1: runbook (prepared 2026-11-27, NOT started)

Design (plan "Calibrating easy, medium and hard"): 8 calibration seeds x 6 tasks = 48 tasks; 18 types x 3 levels =
54 rule instances (rule_levels_v0, TRAIN word lists), alone in the prompt (k = 1); 2,592 prompts; 5 responses per
prompt per model = 12,960 per model, 38,880 per round. Criteria are frozen in `mife.difficulty.pilot.analyze`.

Calibration seeds (data/difficulty/seeds.parquet, role = calibration): pc01, pc02b (replaces pc02, see
seed_replacements.yaml), pc03-pc08. pc01 and pc02b already have 6/6 accepted tasks (test batches).

All commands run from the worktree `~/Documents/manyifeval_difficulty` with
`export PYTHONPATH=src:.pydeps TOKENIZERS_PARALLELISM=false` and the shared `.venv` Python.

## 1. Task writing for the pilot seeds (paid: OpenRouter)

```
set -a; . ~/Documents/manyifeval_claude_code_starter/.env; set +a
python -m mife.difficulty.tasks --tasks-only --out data/difficulty/tasks_pilot \
    --seeds pc01 pc02b pc03 pc04 pc05 pc06 pc07 pc08
```
pc01 and pc02b hit the cache for their writer and judge calls only if prompts are unchanged; otherwise they are
rewritten like the others. A seed that fails after 3 rewrites must be replaced (`seeds replace`) and rerun before the
build, because `pilot build` refuses unaccepted tasks.

## 2. Build prompts, tokenization and worklists (free, ~1 min)

```
python -m mife.difficulty.pilot build --round 1
```
Writes data/difficulty/pilot/round1/inputs/{prompts,tokenized_<m>,worklist_<m>}.parquet and build_summary.json
(K10 schema; refuses prompts that would exceed 4,096 tokens with 2,048 new).

## 3. Generation (recommended: Betty, vLLM BF16, same backend as the main dataset)

K10's `pilots/high_k/code/generate.py` reads fixed paths under pilots/high_k/data/inputs. It needs a small,
reviewed change (input/output directory from an argument) before it can read this round's inputs; nothing on Betty
has been changed. Mac fallback (MLX; Gemma one response at a time because mlx-lm 0.32 batching fails on Gemma 2):
```
python -m mife.difficulty.pilot generate --round 1 --model llama --batch 32
python -m mife.difficulty.pilot generate --round 1 --model gemma2b
python -m mife.difficulty.pilot generate --round 1 --model gemma9b
```

## 4. Verify and analyze (free, minutes)

```
python -m mife.difficulty.pilot verify --round 1
python -m mife.difficulty.pilot analyze --round 1
```
