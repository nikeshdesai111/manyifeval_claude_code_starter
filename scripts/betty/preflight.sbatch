#!/usr/bin/env bash
# GPU preflight (~5 min of GPU): environment report + one pilot prompt x 20 samples with vLLM (Llama),
# written to the separate run id "smoke_betty". Submit from the repo root:
#   mkdir -p logs && sbatch scripts/betty/preflight.sbatch
#SBATCH --job-name=mife-preflight
#SBATCH --account=zives-cis-academic-curri
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export MIFE_REPO="$(pwd)"
source scripts/betty/env.sh
mife_activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p data/env
python -m mife.cli env-report "data/env/preflight_job${SLURM_JOB_ID}.json"
python -m mife.cli generate-vllm llama smoke_betty --pilot --max-prompts 1 --ppb 1 2>&1
echo PREFLIGHT_OK
