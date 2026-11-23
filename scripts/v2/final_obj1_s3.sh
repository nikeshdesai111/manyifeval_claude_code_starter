#!/bin/bash
# After the three parallel H2 pieces of MODEL finish, run the Objective 1 final selection/expansion/refits, then the
# Objective 2 averaged-Objective-1 reference for that model. Usage: final_obj1_s3.sh MODEL
cd "$(dirname "$0")/../.."
m=$1; L=logs/v2/s3
export PYTHONPATH=src MIFE_V2_HESS=mps OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 TOKENIZERS_PARALLELISM=false
until [ "$(grep -l "'done': \['$m', 1, 'h2r=" $L/pre_${m}_o1_h2r_*.log 2>/dev/null | wc -l)" -eq 3 ]; do
  grep -q Traceback $L/pre_${m}_o1_h2r_*.log && { echo "h2r piece failed"; exit 1; }
  sleep 60
done
.venv/bin/python -c "from mife.probing.v2 import cv; import json; r=cv.run('$m', 1); print(json.dumps({'final_written': ['$m', 1], 'row': r['hidden_row'], 'expansions': r['expansions']}, default=str))" > $L/final_${m}_o1.log 2>&1 || exit 1
until grep -q "ordered_written\|Traceback" $L/ordered_${m}.log 2>/dev/null && grep -q final_written $L/final_${m}_o2.log; do sleep 60; done
.venv/bin/python -c "from mife.probing.v2 import cv; import json; cv.avg_obj1_reference('$m'); print(json.dumps({'avg_written': '$m'}))" > $L/avg_${m}.log 2>&1
echo "final $m complete"
