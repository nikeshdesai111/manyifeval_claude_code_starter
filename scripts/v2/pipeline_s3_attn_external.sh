#!/bin/bash
# Dependency runner for the S3 rerun of one generator and objective (cache-filling pieces, then the final cv.run).
# Usage: pipeline_s3.sh MODEL OBJ      (the phase-A pieces are started separately; their logs carry "done" markers)
cd "$(dirname "$0")/../.."
m=$1; o=$2; L=logs/v2/s3
export PYTHONPATH=src MIFE_V2_HESS=${MIFE_V2_HESS:-cpu} OMP_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3 TOKENIZERS_PARALLELISM=false
has() { grep -q "'done': \['$m', $o, '$1'\]" $L/pre_${m}_o${o}_*.log 2>/dev/null; }
extracted() { PYTHONPATH=src .venv/bin/python -c "
import json,sys; from mife.probing import data
from mife.probing.v2 import maxattn
d,_=maxattn.committed('$m'); sys.exit(0 if set(data.prompt_ids('train'))<=d else 1)" 2>/dev/null; }
until has h1rows && has text; do sleep 60; done
has hdep || .venv/bin/python scripts/v2/prewarm_s3.py $m $o hdep > $L/pre_${m}_o${o}_hdep.log 2>&1 || exit 1
until extracted; do sleep 60; done
until has attn; do sleep 60; done   # attn piece started separately
until has xh1rows; do sleep 60; done
.venv/bin/python -c "from mife.probing.v2 import cv; import json; print(json.dumps({'final_done': ['$m', $o]}), flush=True); r=cv.run('$m', $o); print(json.dumps({'final_written': ['$m', $o], 'row': r['hidden_row'], 'expansions': r['expansions']}, default=str))" > $L/final_${m}_o${o}.log 2>&1 || exit 1
if [ "$o" = 2 ]; then
  .venv/bin/python -c "from mife.probing.v2 import cv; import json; r=cv.run_ordered('$m'); print(json.dumps({'ordered_written': '$m'}))" > $L/ordered_${m}.log 2>&1 || exit 1
fi
echo "pipeline $m $o complete"
