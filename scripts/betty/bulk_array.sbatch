#!/usr/bin/env bash
# BULK generation (approved 2026-11-10; Amendment R1): all 53,136 manifest prompts x 20 samples per model on
# Betty with vLLM (BF16, vLLM sampler order), run id "main". Array tasks 0-11 = Llama chunks 0-11,
# 12-23 = Qwen chunks 0-11 (contiguous manifest chunks of 4,428 prompts, ~25 min each on one B200).
# The main run generates every manifest prompt itself, including the 100 pilot prompts; pilot responses
# (run pilot_betty_dgx-b200) are not imported, so every main response comes from these array tasks.
# Prompts already complete in data/generations/<model>/main (e.g. a chunk that finished before a timeout)
# are skipped, so resubmitting a failed task simply resumes it. Generation shards are written atomically.
# Shards/summaries are tagged per chunk (shard_cXX_*.parquet), so parallel tasks never collide.
#
# Submit:  mkdir -p logs && sbatch scripts/betty/bulk_array.sbatch
#          sbatch --dependency=afterok:<bulk_job_id> scripts/betty/verify_bulk.sbatch
# Resubmit selected tasks:  sbatch --array=3,15 scripts/betty/bulk_array.sbatch
#SBATCH --job-name=mife-bulk
#SBATCH --account=zives-cis-academic-curri
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:40:00
#SBATCH --array=0-23
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export MIFE_REPO="$(pwd)"
source scripts/betty/env.sh
mife_activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

N_CHUNKS=12
T=$SLURM_ARRAY_TASK_ID
MODELS=(llama qwen)
M="${MODELS[$((T / N_CHUNKS))]}"
C=$((T % N_CHUNKS))
echo "model=$M chunk=$C/$N_CHUNKS job=$SLURM_ARRAY_JOB_ID/$T host=$(hostname)"
mkdir -p data/env
python -m mife.cli env-report "data/env/${M}_main_c$(printf %02d $C)_job${SLURM_ARRAY_JOB_ID}.json"
python -m mife.cli generate-vllm "$M" main --all-train-test-CONFIRMED --chunk "$C" --n-chunks "$N_CHUNKS" \
  --ppb 500 --n-probe 0
echo "DONE $M chunk $C"
