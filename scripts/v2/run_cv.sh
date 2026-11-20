#!/bin/bash
# V2 training-only CV for one generator and objective. Usage: scripts/v2/run_cv.sh MODEL OBJ [warmup]
cd "$(dirname "$0")/../.."
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=${THREADS:-4} VECLIB_MAXIMUM_THREADS=${THREADS:-4}
WA=True; [ "$3" = "warmup" ] && WA=False
exec .venv/bin/python -c "
from mife.probing.v2 import cv; import json
print(json.dumps(cv.run('$1', $2, with_attention=$WA, final=$WA), default=str)[:4000])"
