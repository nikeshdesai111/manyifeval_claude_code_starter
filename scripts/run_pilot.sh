#!/usr/bin/env bash
# Runs the mandatory 100-prompt TRAIN-only pilot end to end. Every stage is resumable/idempotent.
# Usage: scripts/run_pilot.sh [LLAMA_CBS] [QWEN_CBS]  (concurrent decoding sequences; defaults 64 / 32)
# Qwen3-8B (36 layers) OOMs at 64 concurrent sequences on 64 GB (pilot attempts 1-2); 32 is used for it.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
$PY -m mife.cli generate llama pilot --pilot --ppb 4 --cbs "${1:-64}"
$PY -m mife.cli generate qwen pilot --pilot --ppb 4 --cbs "${2:-32}"
for M in llama qwen; do
  $PY -m mife.cli verify "$M" pilot
  $PY -m mife.cli features "$M" pilot --pilot
  $PY -m mife.cli consistency "$M" 20
done
$PY -c "from mife import pilot_metrics as P; P.run({'llama':'pilot','qwen':'pilot'})"
echo "PILOT PIPELINE DONE"
