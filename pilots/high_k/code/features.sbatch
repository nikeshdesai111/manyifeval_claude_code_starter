#!/usr/bin/env bash
# K10 feature extraction (features.py), same weights handling as high_k.sbatch. K10 generation (pilots/high_k/PRESPEC.md). Run from ~/mife/repo on Betty. Weights: either HK_HF_CACHE (non-home
# storage filled by stage.sh push-weights) or, if unset, a per-job download to node-local disk (needs huggingface-cli login).
#   sbatch --export=ALL --array=0-47%8 pilots/high_k/code/high_k.sbatch gemma2b
#   sbatch --export=ALL --array=0-47%8 --dependency=afterany:<gemma2b job> pilots/high_k/code/high_k.sbatch llama
#   sbatch --export=ALL --array=0-47%8 --dependency=afterany:<llama job>   pilots/high_k/code/high_k.sbatch gemma9b
# Smoke (one task, 200 records): sbatch --export=ALL,MAXREC=200 --array=0 pilots/high_k/code/high_k.sbatch gemma2b
# Chunks 0..k hold TRAIN (build_summary.json train_chunks). Writes only pilots/high_k/data/. Resumable: resubmit any
# chunk (check squeue first so two jobs never write one chunk).
#SBATCH --job-name=mife-k10f
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
G=pilots/high_k/code/generate.py
if [[ -n "${HK_HF_CACHE:-}" ]]; then
  export HF_HUB_CACHE="$HK_HF_CACHE"        # weights pushed to non-home storage by stage.sh push-weights
else
  # No writable project storage: download the pinned snapshot to node-local disk (wiped at job end). Needs the
  # user's own HF read token on Betty (huggingface-cli login) for the gated models; nothing lands in home.
  LOCAL="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/mife_k10_${SLURM_JOB_ID:-$$}"
  mkdir -p "$LOCAL/hub"; trap 'rm -rf "$LOCAL"' EXIT
  export HF_HUB_CACHE="$LOCAL/hub"
  python - "$M" <<'P'
import json, sys
from huggingface_hub import snapshot_download
s = json.load(open("pilots/high_k/data/inputs/build_summary.json"))["models"][sys.argv[1]]
print(snapshot_download(s["hf_id"], revision=s["revision"],
                        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "LICENSE*", "USE_POLICY.md"]), flush=True)
P
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python $G weights "$M"
C="${SLURM_ARRAY_TASK_ID:?submit as an array}"
echo "model=$M chunk=$C job=${SLURM_ARRAY_JOB_ID}/${C} host=$(hostname)"
mkdir -p pilots/high_k/data/env logs
python -m mife.cli env-report "pilots/high_k/data/env/${M}_c${C}_job${SLURM_ARRAY_JOB_ID}.json"
PYTHONPATH=src:pilots/high_k/code python pilots/high_k/code/features.py run "$M" --chunk "$C" ${MAXREC:+--max-prompts "$MAXREC"}
echo "DONE features $M chunk $C"
