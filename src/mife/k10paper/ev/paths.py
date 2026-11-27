"""Paths for the K10 TRAIN paper exhibits (grouped-CV OOF predictions; no TEST access)."""
from __future__ import annotations

from ...config import ROOT

import os

# MIFE_K10_OUT: pipeline tests only (synthetic predictions in a scratch folder); never set for real results
EVAL_DIR = __import__("pathlib").Path(os.environ["MIFE_K10_OUT"]) if os.environ.get("MIFE_K10_OUT") else ROOT / "results" / "k10_paper"
SPLIT = os.environ.get("MIFE_K10_SPLIT", "train")  # "test": held-out scoring under the frozen K10 TEST prespec
if SPLIT == "test" and not os.environ.get("MIFE_K10_OUT"):
    EVAL_DIR = ROOT / "results" / "k10_paper" / "test"
N_POOLS = 50 if SPLIT == "train" else 22
TRAIN_OOF_DIR = ROOT / "results" / "k10_paper" / "v2" / "predictions"
