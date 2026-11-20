#!/usr/bin/env bash
# B2 on Betty, step 2 of 2: the LR x epoch grid of the method chosen by b2_check (method.json), for every
# (generator, objective, CV fold): 2 x 2 x 5 = 20 array tasks, 4 grid points each. Each grid point adapts BGE-M3
# on the fold's fitting groups, scores the held-out groups and saves fp16 embeddings of all TRAIN prompts and rules
# from the adapted encoder (data/probing_features/b2_runs/grid/<run>/). Resumable: finished grid points are skipped,
# so resubmitting a failed task resumes it:  sbatch --array=<ids> scripts/betty/b2_grid.sbatch
#SBATCH --job-name=mife-b2grid
#SBATCH --account=zives-cis-academic-curri
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --time=03:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export MIFE_REPO="$(pwd)"
source scripts/betty/env.sh
mife_activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "b2 grid task=$SLURM_ARRAY_TASK_ID job=$SLURM_ARRAY_JOB_ID host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
PYTHONPATH=src python -m mife.probing.b2 grid "$SLURM_ARRAY_TASK_ID" data/probing_features/b2_runs
echo "DONE b2 grid task $SLURM_ARRAY_TASK_ID"
