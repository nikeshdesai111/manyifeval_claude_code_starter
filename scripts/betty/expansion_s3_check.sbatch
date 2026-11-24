#!/usr/bin/env bash
# CPU completeness check of the expansion run (provenance only: prompt IDs, sample indices, seeds, settings hashes).
# It does NOT verify responses and reads no TEST labels or outcomes; TEST verification stays locked until authorized.
#   sbatch --dependency=afterok:<array_job_id> scripts/betty/expansion_s3_check.sbatch llama
#SBATCH --job-name=mife-exp-s3-check
#SBATCH --account=zives-cis-academic-curri
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
M="${1:?usage: sbatch scripts/betty/expansion_s3_check.sbatch llama|qwen}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export MIFE_REPO="$(pwd)"
source scripts/betty/env.sh
mife_activate
python -m mife.expansion_s3 check-complete "$M"
