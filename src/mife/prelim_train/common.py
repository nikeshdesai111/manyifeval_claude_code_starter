"""Shared definitions: the seven fixed V2 comparisons, labels, data assembly."""
from __future__ import annotations

import pandas as pd

from ..config import ROOT
from .matches import FIXED_TYPES

GENS = ("llama", "qwen")
GEN_LABEL = {"llama": "Llama-3.1-8B-Instruct", "qwen": "Qwen3-8B (non-thinking)"}
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
OOF_DIR = ROOT / "results" / "v2_s3" / "predictions"


def type_order(types) -> list[str]:
    fx = [t for t in sorted(types) if t in FIXED_TYPES]
    pa = [t for t in sorted(types) if t not in FIXED_TYPES]
    return fx + pa


def short_type(t: str) -> str:
    return t.split(":")[1]


def oof(gen: str, obj: int = 1) -> pd.DataFrame:
    """V2 S3 grouped-CV OOF predictions (column `selected` = the CV-selected setting) for every predictor."""
    from ..probing import data as pdata
    allowed = pdata.allowed_ids("train")
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
    from ..probing import data as pdata
    from .. import splits
    L = pdata.labels_prompt_rule(gen, "train")[["prompt_id", "position", "rule_id", "type_id", "v", "n"]]
    P = pdata.prompts("train")[["prompt_id", "group_id", "k", "split"]]
    assert P.split.eq("train").all() and P.group_id.nunique() == 50 and len(P) == 36900
    assert not (set(P.group_id) & splits.test_groups())
    L = L.merge(P[["prompt_id", "group_id", "k"]], on="prompt_id", validate="many_to_one")
    O = oof(gen, 1)
    d = L.merge(O, on=["prompt_id", "position", "rule_id"], how="left", validate="one_to_one")
    assert d[PREDS].notna().all().all() and len(d) == 153900
    # every OOF prediction came from a fold that excludes the row's whole rule pool (group)
    import json
    from .. import splits as _s
    fog = json.loads((_s.cv_dir() / "folds.json").read_text())["fold_of_group"]
    for p in PREDS + ["REF_structure"]:
        assert (d[f"fold_{p}"].values == d.group_id.map(fog).values).all(), p
    d["p_obs"] = d.v / d.n
    return d
