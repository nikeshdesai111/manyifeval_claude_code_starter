#!/bin/bash
# Stop the Objective 1 warm-up once its base-round incremental fits (B2+H1) are cached, then start the final run,
# which makes every selection and expansion decision with all predictors (attention ones included).
# Usage: chain_obj1_final.sh WARMUP_PID MODEL
cd "$(dirname "$0")/../.."
pid=$1; m=$2; log=logs/v2/cv_warmup_${m}_obj1.log
n0=$(grep -c '"spec": "B2+H1"' "$log")
while kill -0 "$pid" 2>/dev/null && [ "$(grep -c '"spec": "B2+H1"' "$log")" -le "$n0" ]; do sleep 20; done
kill "$pid" 2>/dev/null; sleep 5
echo "warmup $m stopped after base incremental $(date -u +%H:%MZ)" >> logs/v2/resume_cv.log
MIFE_V2_HESS=mps THREADS=4 exec scripts/v2/run_cv.sh "$m" 1 > "logs/v2/cv_final_${m}_obj1.log" 2>&1
