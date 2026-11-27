"""Shared definitions: the seven fixed V2 comparisons, labels, data assembly."""
from __future__ import annotations

import pandas as pd

from ...config import ROOT
from .matches import FIXED_TYPES

import os
GENS = tuple(os.environ.get("MIFE_K10_GENS", "llama,gemma2b").split(","))
GEN_LABEL = {"llama": "Llama-3.1-8B-Instruct", "gemma2b": "Gemma-2-2B-it"}
# (id, comparator, candidate, label, primary)
COMPS = [
    ("#1", "B1", "H1", "#1 B1 → H1 + structure", True),
    ("#2", "B2", "H2", "#2 B2 → H2 + structure", True),
    ("#3", "B2", "RCAP", "#3 B2 → A2 (rule-conditioned attention)", True),
    ("#4", "H2", "RCAP", "#4 H2 → A2", True),
    ("#5", "B1", "B1+H1", "#5 B1 → B1 + H", False),
    ("#6", "B2", "B2+H1", "#6 B2 → B2 + H", False),
    ("#7", "B2", "B2+A1", "#7 B2 → B2 + A", False),
    ("#8", "B2+H1", "B2+H1+A1", "#8 B2 + H → B2 + H + A", False),
]
PREDS = ["B1", "B2", "H1", "H2", "RCAP", "B1+H1", "B2+H1", "B2+A1", "B2+H1+A1"]
PRED_LABEL = {"B1": "B1 (S+P)", "B2": "B2 (S+P+R)", "H1": "H1 + structure", "H2": "H2 + structure",
              "RCAP": "A2: rule-cond. attention (S+R+A)", "B1+H1": "B1 + H", "B2+H1": "B2 + H", "B2+A1": "B2 + A", "B2+H1+A1": "B2 + H + A",
              "REF_structure": "Structure only (ref.)"}
from .paths import EVAL_DIR as _E, SPLIT, N_POOLS
OOF_DIR = _E / "v2" / "predictions" if SPLIT == "train" else _E / "predictions"
OBSERVED_ONLY = os.environ.get("MIFE_K10_OBSERVED_ONLY") == "1"  # observed effects only: no predictions read
if OBSERVED_ONLY:
    PREDS, COMPS = [], []


def type_order(types) -> list[str]:
    fx = [t for t in sorted(types) if t in FIXED_TYPES]
    pa = [t for t in sorted(types) if t not in FIXED_TYPES]
    return fx + pa


def short_type(t: str) -> str:
    return t.split(":")[1]


def oof(gen: str, obj: int = 1) -> pd.DataFrame:
    """V2 S3 grouped-CV OOF predictions (column `selected` = the CV-selected setting) for every predictor."""
    from .. import kdata
    allowed = kdata.allowed_ids(SPLIT)
    keys = ["prompt_id", "position", "rule_id"] if obj == 1 else ["prompt_id"]
    out = None
    for p in PREDS + ["REF_structure"]:
        x = pd.read_parquet(OOF_DIR / gen / f"obj{obj}" / f"{p}.parquet", columns=keys + ["fold", "selected"])
        assert set(x.prompt_id) <= allowed, "OOF file contains a non-TRAIN prompt"
        x = x.rename(columns={"selected": p, "fold": f"fold_{p}"})
        out = x if out is None else out.merge(x, on=keys, how="inner", validate="one_to_one")
    return out


def data(gen: str) -> pd.DataFrame:
    """S3 TRAIN Objective 1 rows (153,900): official labels (v, n) joined one-to-one with OOF predictions."""
    from .. import kdata
    from ... import splits
    L = kdata.labels_prompt_rule(gen, SPLIT)[["prompt_id", "position", "rule_id", "type_id", "v", "n"]]
    P = kdata.prompts(SPLIT)[["prompt_id", "group_id", "k", "split"]]
    NR = 2746500 if SPLIT == "train" else kdata.N_TEST_RULE_ROWS
    assert P.split.eq(SPLIT).all() and P.group_id.nunique() == N_POOLS
    assert not (set(P.group_id) & splits.test_groups())
    L = L.merge(P[["prompt_id", "group_id", "k"]], on="prompt_id", validate="many_to_one")
    if OBSERVED_ONLY:
        d = L.copy()
        d["p_obs"] = d.v / d.n
        assert len(d) == NR
        return d
    O = oof(gen, 1)
    d = L.merge(O, on=["prompt_id", "position", "rule_id"], how="left", validate="one_to_one")
    assert d[PREDS].notna().all().all() and len(d) == NR
    if SPLIT == "test":  # predictions from the frozen 50-group refits; no fold structure
        d["p_obs"] = d.v / d.n
        return d
    # every OOF prediction came from a fold that excludes the row's whole rule pool (group)
    import json
    fog = kdata.fold_of_group()
    for p in PREDS + ["REF_structure"]:
        assert (d[f"fold_{p}"].values == d.group_id.map(fog).values).all(), p
    d["p_obs"] = d.v / d.n
    return d
