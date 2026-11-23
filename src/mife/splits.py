"""Active group split (amendment S3, 2026-11-23).

The frozen manifest keeps its 2026-11-14 `split` column. When configs/active_split.yaml names the S3 split, the
probing loaders overlay its group assignment on every manifest read (`apply`). CV folds for the active split live
in its own directory (`cv_dir`), so the V1 folds in results/cv/ stay untouched.
"""
from __future__ import annotations

import json
from functools import lru_cache

import yaml

from .config import ROOT

ACTIVE_FILE = ROOT / "configs" / "active_split.yaml"
MANIFEST_VERSION = "manifest-2026-11-14"


@lru_cache(maxsize=None)
def active() -> dict:
    if not ACTIVE_FILE.exists():
        return {"version": MANIFEST_VERSION}
    return yaml.safe_load(ACTIVE_FILE.read_text())


def version() -> str:
    return active()["version"]


@lru_cache(maxsize=None)
def split_of_group() -> dict[str, str] | None:
    """group_id -> 'train' | 'test' for the active split, or None when the manifest column is active."""
    a = active()
    if a["version"] == MANIFEST_VERSION:
        return None
    rec = json.loads((ROOT / a["assignment"]).read_text())
    if rec.get("amendment") != a["version"]:
        raise RuntimeError(f"{a['assignment']} is {rec.get('amendment')!r}, not the active split {a['version']!r}")
    out = {g: "test" for g in rec["test_groups"]} | {g: "train" for g in rec["train_groups"]}
    assert len(out) == len(rec["test_groups"]) + len(rec["train_groups"]) == 72
    return out


def apply(df):
    """Return df with its `split` column replaced by the active assignment (no-op for the manifest split)."""
    sog = split_of_group()
    if sog is None:
        return df
    df = df.copy()
    df["split"] = df["group_id"].map(sog)
    if df["split"].isna().any():
        raise RuntimeError("rows whose group_id is not in the active split assignment")
    return df


def test_groups() -> set[str]:
    sog = split_of_group()
    if sog is None:
        import pandas as pd

        from .config import DATA_DIR
        m = pd.read_parquet(DATA_DIR / "manifest" / "prompts.parquet", columns=["group_id", "split"])
        return set(m.group_id[m.split == "test"])
    return {g for g, s in sog.items() if s == "test"}


def cv_dir():
    a = active()
    return ROOT / a["cv_dir"] if "cv_dir" in a else ROOT / "results" / "cv"


def fold_group_counts() -> list[int] | None:
    return active().get("fold_group_counts")
