#!/usr/bin/env bash
# Training-only grouped CV for every non-B2 configuration: 2 generators x 2 objectives in parallel (4 CPU threads
# each). Resumable: finished (config, row, pooling) keys in results/cv/<model>/obj<o>/grid/ are skipped on rerun.
# Usage: scripts/mac/probe_cv_all.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/probing
for M in llama qwen; do
  for O in 1 2; do
    PYTHONPATH=src caffeinate -i .venv/bin/python -m mife.cli probe-cv "$M" "$O" >> "logs/probing/cv_${M}_obj${O}.log" 2>&1 &
  done
done
wait
echo "$(date '+%F %T') all CV jobs ended"
