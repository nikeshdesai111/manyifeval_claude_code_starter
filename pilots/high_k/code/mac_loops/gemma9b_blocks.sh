#!/usr/bin/env bash
# Build gemma9b TRAIN fitting blocks (S1, H<row> x5, A1R) under data/k10_paper/gemma9b/, low priority. TRAIN only.
cd /Users/nickydesai/Documents/manyifeval_claude_code_starter || exit 1
L=pilots/high_k/data/k10fit_g9.log
export PYTHONPATH=src VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2
echo "$(date '+%F %T') gemma9b blocks: structure" >> $L
nice -n 15 .venv/bin/python -m mife.k10paper.features structure gemma9b >> $L 2>&1 || { echo "$(date '+%F %T') ALERT k10fit_g9: structure FAILED" >> pilots/high_k/data/alerts.log; exit 1; }
echo "$(date '+%F %T') gemma9b blocks: hidden + A1R" >> $L
nice -n 15 .venv/bin/python -m mife.k10paper.features hidden gemma9b train >> $L 2>&1 || { echo "$(date '+%F %T') ALERT k10fit_g9: hidden FAILED" >> pilots/high_k/data/alerts.log; exit 1; }
ls -la data/k10_paper/gemma9b/ >> $L
echo "$(date '+%F %T') gemma9b blocks DONE" >> $L
date '+%F %T' > pilots/high_k/data/gemma9b_blocks.done
