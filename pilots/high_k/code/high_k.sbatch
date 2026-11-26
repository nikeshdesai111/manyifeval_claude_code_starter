#!/usr/bin/env bash
# K10 generation (pilots/high_k/PRESPEC.md). Run from ~/mife/repo on Betty, weights in non-home storage:
#   export HK_HF_CACHE=/path/on/project/storage/hf/hub      # filled by stage.sh push-weights on the Mac
#   sbatch --export=ALL --array=0-47%8 pilots/high_k/code/high_k.sbatch gemma2b
#   sbatch --export=ALL --array=0-47%8 --dependency=afterany:<gemma2b job> pilots/high_k/code/high_k.sbatch llama
#   sbatch --export=ALL --array=0-47%8 --dependency=afterany:<llama job>   pilots/high_k/code/high_k.sbatch gemma9b
# Smoke (one task, 200 records): sbatch --export=ALL,MAXREC=200 --array=0 pilots/high_k/code/high_k.sbatch gemma2b
# Chunks 0..k hold TRAIN (build_summary.json train_chunks). Writes only pilots/high_k/data/. Resumable: resubmit any
# chunk (check squeue first so two jobs never write one chunk).
#SBATCH --job-name=mife-k10
#SBATCH --account=zives-cis-academic-curri
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
M="${1:?usage: sbatch --array=0-47 pilots/high_k/code/high_k.sbatch gemma2b|llama|gemma9b}"
[[ "$M" == gemma2b || "$M" == llama || "$M" == gemma9b ]] || { echo "model must be gemma2b, llama or gemma9b"; exit 2; }
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export MIFE_REPO="$(pwd)"
source scripts/betty/env.sh
mife_activate
export HF_HUB_CACHE="${HK_HF_CACHE:?set HK_HF_CACHE to the weights directory filled by stage.sh push-weights}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=pilots/high_k/code/generate.py
python $G weights "$M"
C="${SLURM_ARRAY_TASK_ID:?submit as an array}"
echo "model=$M chunk=$C job=${SLURM_ARRAY_JOB_ID}/${C} host=$(hostname)"
mkdir -p pilots/high_k/data/env logs
python -m mife.cli env-report "pilots/high_k/data/env/${M}_c${C}_job${SLURM_ARRAY_JOB_ID}.json"
python $G generate "$M" --chunk "$C" ${MAXREC:+--max-records "$MAXREC"}
echo "DONE $M chunk $C"
