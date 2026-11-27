#!/usr/bin/env bash
# Start the K10 head-fit stager detached (survives Claude Code restarts); PID recorded in pilots/high_k/data/loops.pid.
#   bash pilots/high_k/code/k10fit/stage_fit.sh <models: gemma2b[,llama]> <dev: cuda|cpu> [time limit, default 01:00:00]
# Stop:  kill $(awk '/^k10fit_stager/{print $2}' pilots/high_k/data/loops.pid)   (running Betty jobs keep going; a
# restart re-plans from the point cache and resumes units already on Betty)
set -euo pipefail
cd /Users/nickydesai/Documents/manyifeval_claude_code_starter
M="${1:?models}"; DEV="${2:?cuda|cpu}"; T="${3:-01:00:00}"
D=pilots/high_k/data
nohup env PYTHONPATH=src .venv/bin/python pilots/high_k/code/k10fit/stager.py --models "$M" --dev "$DEV" --time "$T" --log "$D/k10fit_g9.log" \
  >> "$D/k10fit_stager.out" 2>&1 < /dev/null &
P=$!
disown
grep -v '^k10fit_stager ' "$D/loops.pid" > "$D/loops.pid.tmp" || true
echo "k10fit_stager $P" >> "$D/loops.pid.tmp"; mv "$D/loops.pid.tmp" "$D/loops.pid"
echo "k10fit stager started: PID $P, models $M, dev $DEV, limit $T"
