"""Paths for the S3 TEST Objective 1 evaluation.

Unblinding of the S3 TEST set for Objective 1 was requested by the project owner in the project thread on
2026-11-24 (see the report's provenance section). Objective 2 TEST analysis is out of scope here.
"""
from __future__ import annotations

import os

from ..config import DATA_DIR, ROOT

EVAL_DIR = ROOT / "results" / "eval_s3" / os.environ.get("MIFE_EVAL_SUB", "")  # sub-run (e.g. llama_only)
EXP_DIR = DATA_DIR / "evaluation_expansion_s3"
TEST_FEAT = DATA_DIR / "probing_features_v2" / "test_s3"
EXP_RUN = "expansion_s3_20261123"
