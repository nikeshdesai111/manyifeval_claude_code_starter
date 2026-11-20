#!/usr/bin/env bash
# Re-run the 100-prompt TRAIN-only pilot on Betty: array task 0 = Llama-3.1-8B-Instruct, 1 = Qwen3-8B,
# one GPU each. Uses the SAME pre-selected pilot prompts (data/pilot/pilot_prompts.parquet) and the
# frozen token IDs from data/tokenized/. Writes under a NEW run id; the Mac pilot ("pilot") is protected.
# Every stage is resumable: resubmitting skips completed prompts.
#
# Submit from the repo root:   mkdir -p logs && sbatch scripts/betty/pilot_array.sbatch
# MIG slice instead of a full B200 (check GRES names first with `sinfo -o "%P %G"`):
#   sbatch -p b200-mig90 -c 14 --export=ALL,GPU_MEM=0.80 scripts/betty/pilot_array.sbatch   (GRES gpu:90gb)
#   sbatch -p b200-mig45 -c 6  --export=ALL,GPU_MEM=0.80 scripts/betty/pilot_array.sbatch   (GRES gpu:45gb)
# Betty's Slurm CLI filter requires exactly 28 / 14 / 6 CPUs per GPU on dgx-b200 / mig90 / mig45 and
# sets memory at 8 GB per CPU, so do not pass --mem.
#SBATCH --job-name=mife-pilot
#SBATCH --account=zives-cis-academic-curri
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export MIFE_REPO="$(pwd)"
source scripts/betty/env.sh
mife_activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # compute nodes may lack internet; models were pre-downloaded

MODELS=(llama qwen)
M="${MODELS[$SLURM_ARRAY_TASK_ID]}"
RUN_ID="${RUN_ID:-pilot_betty_${SLURM_JOB_PARTITION}}"
[[ "$RUN_ID" == pilot ]] && { echo "RUN_ID 'pilot' is the Mac pilot; refusing"; exit 2; }
GPU_MEM="${GPU_MEM:-0.85}"
PY="python -m mife.cli"
echo "model=$M run=$RUN_ID partition=$SLURM_JOB_PARTITION job=$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID host=$(hostname)"
mkdir -p data/env
$PY env-report "data/env/${M}_${RUN_ID}_job${SLURM_ARRAY_JOB_ID}.json"

# 1. generation (vLLM, BF16), 20 samples/prompt, per-sample seeds, shards resumable
$PY generate-vllm "$M" "$RUN_ID" --pilot --ppb 50 --gpu-mem "$GPU_MEM"
# 2. verifier (CPU; official ManyIFEval/IFEval checkers, vendored NLTK data)
$PY verify "$M" "$RUN_ID"
# 3. pre-generation features (HF transformers, eager attention, BF16, CUDA), one pass per prompt
$PY features "$M" "$RUN_ID" --pilot
# 4. vLLM vs HF on CUDA (final-token log-probs, same token IDs)
$PY consistency-cuda "$M" "$RUN_ID"
# 5. against the Mac pilot: token IDs/spans, hidden states, attention, verifier rates per k
$PY xbackend "$M" "$RUN_ID" --mac-run pilot
echo "DONE $M $RUN_ID"
