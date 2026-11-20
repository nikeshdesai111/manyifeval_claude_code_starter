#!/usr/bin/env bash
# Pre-generation features for ALL 53,136 manifest prompts, on the Mac (Apple MPS, HF transformers eager BF16).
# Decision: tracker v1.12 (one feature backend for the bulk run; Betty home quota cannot hold ~37 GB).
# Writes data/features/<model>/main/ (~17.4 GB Llama, ~19.5 GB Qwen). Resumable: rerun to continue after any
# interruption; committed shards are skipped. Measured on the Mac: ~0.20 s/prompt Llama, ~0.16 s/prompt Qwen
# => ~3 h + ~2.3 h. Keep the Mac on power; caffeinate prevents idle sleep (closing the lid still sleeps it).
# Usage: scripts/mac/features_main.sh [llama|qwen|both]   (default both, sequentially)
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
WHICH="${1:-both}"
MODELS=(llama qwen); [[ "$WHICH" != both ]] && MODELS=("$WHICH")
for M in "${MODELS[@]}"; do
  echo "$(date '+%F %T') start features $M main" | tee -a "logs/features_main_$M.log"
  PYTHONPATH=src caffeinate -i .venv/bin/python -m mife.cli features "$M" main --all 2>&1 | tee -a "logs/features_main_$M.log"
  echo "$(date '+%F %T') end features $M main" | tee -a "logs/features_main_$M.log"
done
