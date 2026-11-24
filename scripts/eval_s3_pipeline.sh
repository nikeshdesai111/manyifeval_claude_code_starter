#!/bin/bash
# Waits for the S3 TEST feature extraction (both models) to finish, then scores and writes the Objective 1 report.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
while pgrep -f "mife.eval_s3.extract" > /dev/null; do sleep 60; done
test -f data/probing_features_v2/test_s3/qwen/llm/extract_meta.json
.venv/bin/python -m mife.eval_s3.predict llama qwen
.venv/bin/python -m mife.eval_s3.run_all
.venv/bin/python -m mife.eval_s3.report
