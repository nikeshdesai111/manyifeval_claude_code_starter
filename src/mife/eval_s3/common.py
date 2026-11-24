"""Shared definitions: the seven fixed V2 comparisons, labels, data assembly."""
from __future__ import annotations

import pandas as pd

from ..config import ROOT
from .matches import FIXED_TYPES

GENS = tuple(__import__("os").environ.get("MIFE_EVAL_GENS", "llama,qwen").split(","))
GEN_LABEL = {"llama": "Llama-3.1-8B-Instruct", "qwen": "Qwen3-8B (non-thinking)"}
# (id, comparator, candidate, label, primary)
COMPS = [
    ("#1", "B1", "H1", "#1 B1 → H1 + structure", True),
    ("#2", "B2", "H2", "#2 B2 → H2 + structure", True),
    ("#3", "B2", "RCAP", "#3 B2 → rule-conditioned attention", True),
    ("#4", "H2", "RCAP", "#4 H2 + structure → attention", True),
    ("#5", "B1", "B1+H1", "#5 B1 → B1 + H", False),
    ("#6", "B2", "B2+H1", "#6 B2 → B2 + H", False),
    ("#7", "B2", "B2+A1", "#7 B2 → B2 + A", False),
]
PREDS = ["B1", "B2", "H1", "H2", "RCAP", "B1+H1", "B2+H1", "B2+A1"]
PRED_LABEL = {"B1": "B1 (S+P)", "B2": "B2 (S+P+R)", "H1": "H1 + structure", "H2": "H2 + structure",
              "RCAP": "Rule-cond. attention (S+R+A)", "B1+H1": "B1 + H", "B2+H1": "B2 + H", "B2+A1": "B2 + A",
              "REF_structure": "Structure only (ref.)"}
OOF_DIR = ROOT / "results" / "v2_s3" / "predictions"


def type_order(types) -> list[str]:
    fx = [t for t in sorted(types) if t in FIXED_TYPES]
    pa = [t for t in sorted(types) if t not in FIXED_TYPES]
    return fx + pa


def short_type(t: str) -> str:
    return t.split(":")[1]


def data(gen: str) -> pd.DataFrame:
    """115,236 S3 TEST Objective 1 rows: labels (v, n) joined one-to-one with frozen predictions."""
    from . import predict, verify
    L = verify.labels(gen)
    Pp = predict.load(gen)
    d = L.merge(Pp.drop(columns=["type_id", "group_id"]), on=["prompt_id", "position", "rule_id"], how="left",
                validate="one_to_one")
    assert d[PREDS].notna().all().all() and len(d) == 115236
    d["p_obs"] = d.v / d.n
    return d
