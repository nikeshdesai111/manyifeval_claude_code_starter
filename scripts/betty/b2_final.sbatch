#!/usr/bin/env bash
# B2 on Betty, final refits: the CV-selected LR x epochs per (generator, objective) refitted on ALL 57 TRAIN groups
# (data/probing_features/b2_runs/final_tasks.json, written on the Mac from results/cv/*/obj*/b2_selection.json).
# Saves fp16 embeddings of all TRAIN prompts/rules and the LoRA adapter (data/probing_features/b2_runs/final/<run>/),
# so TEST prompts can later be embedded with this frozen encoder on the same backend. TRAIN-only inputs.
# Submit:  sbatch scripts/betty/b2_final.sbatch
#SBATCH --job-name=mife-b2final
#SBATCH --account=zives-cis-academic-curri
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export MIFE_REPO="$(pwd)"
source scripts/betty/env.sh
mife_activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "b2 final task=$SLURM_ARRAY_TASK_ID job=$SLURM_ARRAY_JOB_ID host=$(hostname)"
PYTHONPATH=src python -m mife.probing.b2 final "$SLURM_ARRAY_TASK_ID" data/probing_features/b2_runs
echo "DONE b2 final task $SLURM_ARRAY_TASK_ID"
