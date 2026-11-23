#!/usr/bin/env bash
# Mac (MLX, BF16) hedge for the S3 TEST expansion generations while Betty array 11110486 (Llama) keeps running.
# Writes ONLY data/generations/<model>/expansion_s3_20261123_mac_mlx/ (never Betty's run folder). Resumable: rerun to
# continue; finished (prompt, sample) records are skipped. Chunks in reverse order so Mac and Betty overlap least.
# Graceful stop: touch data/generations/<model>/expansion_s3_20261123_mac_mlx/STOP (finishes in-flight sequences).
# Progress: PYTHONPATH=src .venv/bin/python -m mife.expansion_s3_mac status llama
# Usage: nohup scripts/mac/expansion_s3_hedge.sh llama [chunks] > /dev/null 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
M="${1:-llama}"; CHUNKS="${2:-11,10,9,8,7,6,5,4,3,2}"
echo "$(date '+%F %T') start mac hedge $M chunks $CHUNKS" >> "logs/expansion_s3_mac_$M.log"
PYTHONPATH=src caffeinate -ims .venv/bin/python -m mife.expansion_s3_mac generate "$M" --chunks "$CHUNKS" \
  >> "logs/expansion_s3_mac_$M.log" 2>&1
echo "$(date '+%F %T') end mac hedge $M" >> "logs/expansion_s3_mac_$M.log"
