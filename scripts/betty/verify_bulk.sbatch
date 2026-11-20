#!/usr/bin/env bash
# Verify the BULK run on CPU (no GPU): task 0 = Llama, 1 = Qwen. Official ManyIFEval/IFEval checkers,
# every included rule of every response; writes data/verification/<model>/main/ (soft labels for both
# objectives). Fails loudly if any prompt-rule does not have exactly 20 verified samples.
# Submit after (or with a dependency on) the bulk array:
#   sbatch --dependency=afterok:<bulk_job_id> scripts/betty/verify_bulk.sbatch
#SBATCH --job-name=mife-verify
#SBATCH --account=zives-cis-academic-curri
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export MIFE_REPO="$(pwd)"
source scripts/betty/env.sh
mife_activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MODELS=(llama qwen)
M="${MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "verify model=$M run=main host=$(hostname)"
python -m mife.cli verify "$M" main
echo "DONE verify $M"
