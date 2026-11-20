#!/usr/bin/env bash
# B2 (fine-tuned BGE-M3) on Betty, step 1 of 2: full-vs-LoRA training-only check, which doubles as the B200
# benchmark. Llama, Objective 1, CV fold 0, one epoch at each method's lower LR; the lower held-out NLL picks the
# method for every B2 run (tie -> full). Inputs: the TRAIN-only bundle data/probing_features/b2_bundle (no TEST
# prompt, label or feature is on Betty for this step). Writes data/probing_features/b2_runs/method.json.
# Submit:  sbatch scripts/betty/b2_check.sbatch
#          sbatch --dependency=afterok:<check_job_id> scripts/betty/b2_grid.sbatch
#SBATCH --job-name=mife-b2chk
#SBATCH --account=zives-cis-academic-curri
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export MIFE_REPO="$(pwd)"
source scripts/betty/env.sh
mife_activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "b2 check job=$SLURM_JOB_ID host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
PYTHONPATH=src python -m mife.probing.b2 check data/probing_features/b2_runs
echo "DONE b2 check"
