#!/bin/bash
# Wait for the Qwen max-attention extraction to finish, then resume the paused Objective 1 CV processes.
cd "$(dirname "$0")/../.."
until grep -q '"model_key": "qwen"' logs/v2/maxattn_run.log && ! pgrep -f "maxattn[.]run[(]'" >/dev/null; do sleep 30; done
kill -CONT "$@"
echo "resumed $* at $(date -u +%H:%MZ)"
