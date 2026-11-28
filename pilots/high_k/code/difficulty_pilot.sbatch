#!/usr/bin/env bash
# Difficulty calibration pilot, round R (data/difficulty/pilot/round<R>/inputs, built on the Mac by
# `python -m mife.difficulty.pilot build --round R`). Same runner, backend and decoding as K10 (generate.py with
# --inputs-dir/--outputs-dir); one task per model, chunk 0 = all 12,960 responses of that model.
#   sbatch --export=ALL pilots/high_k/code/difficulty_pilot.sbatch gemma2b 1
#   sbatch --export=ALL pilots/high_k/code/difficulty_pilot.sbatch llama 1
#   sbatch --export=ALL pilots/high_k/code/difficulty_pilot.sbatch gemma9b 1
# Writes only data/difficulty/pilot/round<R>/betty/. Resumable: resubmit (check squeue first).
#SBATCH --job-name=mife-diffpilot
#SBATCH --account=zives-cis-academic-curri
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
M="${1:?usage: sbatch difficulty_pilot.sbatch gemma2b|llama|gemma9b ROUND}"
R="${2:?round number}"
[[ "$M" == gemma2b || "$M" == llama || "$M" == gemma9b ]] || { echo "model must be gemma2b, llama or gemma9b"; exit 2; }
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export MIFE_REPO="$(pwd)"
source scripts/betty/env.sh
mife_activate
G=pilots/high_k/code/generate.py
IN="data/difficulty/pilot/round${R}/inputs"
OUT="data/difficulty/pilot/round${R}/betty"
[[ -f "$IN/build_summary.json" ]] || { echo "missing $IN/build_summary.json"; exit 3; }
if [[ -n "${HK_HF_CACHE:-}" ]]; then
  export HF_HUB_CACHE="$HK_HF_CACHE"        # the K10 weights already on non-home storage (same revisions)
else
  LOCAL="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/mife_diffpilot_${SLURM_JOB_ID:-$$}"
  mkdir -p "$LOCAL/hub"; trap 'rm -rf "$LOCAL"' EXIT
  export HF_HUB_CACHE="$LOCAL/hub" HF_XET_CACHE="$LOCAL/xet" HF_HUB_DISABLE_XET=1
  for T in "${HF_HOME:+$HF_HOME/token}" "$HOME/mife/hf/token" "$HOME/.cache/huggingface/token"; do
    [[ -n "$T" && -s "$T" ]] && { export HF_TOKEN_PATH="$T"; break; }
  done
  [[ -n "${HF_TOKEN_PATH:-}" ]] || { echo "no HF token file on Betty (gated weights)"; exit 3; }
  python - "$M" "$IN" <<'P'
import json, sys
from huggingface_hub import snapshot_download
s = json.load(open(f"{sys.argv[2]}/build_summary.json"))["models"][sys.argv[1]]
print(snapshot_download(s["hf_id"], revision=s["revision"],
                        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "LICENSE*", "USE_POLICY.md"]), flush=True)
P
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HK_MAX_GB="${HK_MAX_GB:-1.5}"
mkdir -p "$OUT/env" logs
python $G weights "$M" --inputs-dir "$IN"
python -m mife.cli env-report "$OUT/env/${M}_job${SLURM_JOB_ID}.json"
python $G generate "$M" --chunk 0 --inputs-dir "$IN" --outputs-dir "$OUT" ${MAXREC:+--max-records "$MAXREC"}
python $G check "$M" --inputs-dir "$IN" --outputs-dir "$OUT"
echo "DONE $M round $R"
