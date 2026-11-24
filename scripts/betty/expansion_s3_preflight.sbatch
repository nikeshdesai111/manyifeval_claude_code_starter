#!/usr/bin/env bash
# S3 expansion GPU preflight (TRAIN-only): environment report + the expansion runner on 20 pilot prompts from S3 TRAIN
# groups x 20 per-sample-seeded samples, written to the separate run smoke_expansion_s3 (never main, never TEST).
#   sbatch scripts/betty/expansion_s3_preflight.sbatch llama      (and/or qwen)
#SBATCH --job-name=mife-exp-s3-pre
#SBATCH --account=zives-cis-academic-curri
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
M="${1:?usage: sbatch scripts/betty/expansion_s3_preflight.sbatch llama|qwen}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export MIFE_REPO="$(pwd)"
source scripts/betty/env.sh
mife_activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[[ "${MIFE_NODELOCAL_WEIGHTS:-0}" == 1 ]] && source scripts/betty/expansion_s3_nodelocal_weights.sh
mkdir -p data/env
python -m mife.cli env-report "data/env/${M}_expansion_s3_preflight_job${SLURM_JOB_ID}.json"
python -m mife.expansion_s3 smoke "$M" --prompts 20
echo PREFLIGHT_OK
