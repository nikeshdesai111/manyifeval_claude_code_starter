#!/bin/bash
# Objective 2 ordered-slot sensitivity, then (when Objective 1 is final) nothing else. Usage: run_ordered.sh MODEL...
cd "$(dirname "$0")/../.."
export MIFE_V2_HESS=${MIFE_V2_HESS:-mps} PYTHONPATH=src OMP_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3
for m in "$@"; do
  .venv/bin/python -c "from mife.probing.v2 import cv; import json; print(json.dumps({'ordered_done': '$m', **cv.run_ordered('$m')}, default=str)[:3000])" || exit 1
done
