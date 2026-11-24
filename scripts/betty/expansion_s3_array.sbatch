#!/usr/bin/env bash
# S3 TEST expansion generation (handoff 2026-11-23; docs/s3_22group_expansion_generation_plan.md).
# ONE generator per submission, so Llama and Qwen queue and finish independently:
#   sbatch --array=0-11 scripts/betty/expansion_s3_array.sbatch llama
#   sbatch --array=0-11 scripts/betty/expansion_s3_array.sbatch qwen
# Array task i = worklist chunk i (12 chunks x 1,155 new prompts x 20 samples = 23,100 records each; short tasks backfill sooner).
# Generates ONLY the records in data/evaluation_expansion_s3/worklist_<model>.parquet (checksummed, per-sample seeds)
# into data/generations/<model>/expansion_s3_20261123/. Never writes into run main. Resumable per (prompt, sample):
# resubmitting a task skips records already written with their recorded seed. Before resubmitting, check that no
# task of the same chunk is still running (squeue -u $USER), so two jobs never write the same chunk at once.
# Weights on node-local disk instead of home (downloaded + checksum-verified per task):
#   sbatch --export=ALL,MIFE_NODELOCAL_WEIGHTS=1 --array=0-11 scripts/betty/expansion_s3_array.sbatch qwen
# Resubmit selected chunks:  sbatch --array=2,4 scripts/betty/expansion_s3_array.sbatch qwen
#SBATCH --job-name=mife-exp-s3
#SBATCH --account=zives-cis-academic-curri
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
M="${1:?usage: sbatch --array=0-11 scripts/betty/expansion_s3_array.sbatch llama|qwen}"
[[ "$M" == llama || "$M" == qwen ]] || { echo "model must be llama or qwen"; exit 2; }
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export MIFE_REPO="$(pwd)"
source scripts/betty/env.sh
mife_activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[[ "${MIFE_NODELOCAL_WEIGHTS:-0}" == 1 ]] && source scripts/betty/expansion_s3_nodelocal_weights.sh
C="${SLURM_ARRAY_TASK_ID:?submit as an array}"
echo "model=$M chunk=$C job=${SLURM_ARRAY_JOB_ID}/${C} host=$(hostname)"
mkdir -p data/env
python -m mife.cli env-report "data/env/${M}_expansion_s3_c$(printf %02d "$C")_job${SLURM_ARRAY_JOB_ID}.json"
python -m mife.expansion_s3 generate "$M" --chunk "$C" --ppb 500 --max-num-seqs 512 --gpu-mem 0.85 --max-model-len 4096
echo "DONE $M chunk $C"
