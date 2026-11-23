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
# Auto-restart after a crash (e.g. a Metal out-of-memory error); finished records are skipped on restart.
# Generator recycling every RECYCLE records bounds the batched KV-cache growth that caused the first OOM.
OUT="data/generations/$M/expansion_s3_20261123_mac_mlx"
for attempt in $(seq 1 "${MAX_RESTARTS:-40}"); do
  set +e
  PYTHONPATH=src caffeinate -ims .venv/bin/python -m mife.expansion_s3_mac generate "$M" --chunks "$CHUNKS" \
    --bsz "${BSZ:-64}" --recycle-every "${RECYCLE:-640}" >> "logs/expansion_s3_mac_$M.log" 2>&1
  rc=$?
  set -e
  [[ $rc == 0 || -e "$OUT/STOP" ]] && break
  echo "$(date '+%F %T') exit code $rc (attempt $attempt); restarting in 90 s" >> "logs/expansion_s3_mac_$M.log"
  sleep 90
done
echo "$(date '+%F %T') end mac hedge $M" >> "logs/expansion_s3_mac_$M.log"
