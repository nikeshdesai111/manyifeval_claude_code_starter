"""K10 tables by split. TRAIN: 50 S3 TRAIN groups. TEST: the 22 S3 TEST groups, opened only under the frozen K10 TEST
pre-specification (results/k10_paper/prespec/K10_TEST_PRESPEC.md, authorised by the owner 2026-11-27); every TEST read
checks that the prespec file still matches its recorded hash."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

import pandas as pd

from .. import splits
from ..config import ROOT
from .paths import FOLDS, INP, N_SAMPLES, N_TRAIN_PROMPTS, N_TRAIN_RULE_ROWS, VERIF

PRESPEC = ROOT / "results" / "k10_paper" / "prespec"
N_TEST_PROMPTS, N_TEST_RULE_ROWS = 775500, 4526940


def _check_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    assert df["split"].eq(split).all()
    tg = set(df["group_id"]) & splits.test_groups()
    if split == "train":
        assert not tg, "a TEST group reached a TRAIN loader"
    else:
        rec = (PRESPEC / "K10_TEST_PRESPEC.sha256").read_text().split()[0]
        assert hashlib.sha256((PRESPEC / "K10_TEST_PRESPEC.md").read_bytes()).hexdigest() == rec, "TEST prespec changed"
        assert set(df["group_id"]) == splits.test_groups()
    return df


@lru_cache(maxsize=2)
def _prompts(split: str) -> pd.DataFrame:
    P = pd.read_parquet(INP / "prompts.parquet")
    P = P[P["split"] == split].reset_index(drop=True)
    _check_split(P, split)
    if split == "train":
        fog = json.loads(FOLDS.read_text())["fold_of_group"]
        assert set(P.group_id) == set(fog) and len(P) == N_TRAIN_PROMPTS
    else:
        assert len(P) == N_TEST_PROMPTS
    P = P.sort_values(["group_id", "task_key", "k", "chain_index", "coa_row", "prompt_id"]).reset_index(drop=True)
    P["subset_key"] = P["subset_id"]
    return P


def prompts(split: str = "train") -> pd.DataFrame:
    return _prompts(split).copy()


def prompt_ids(split: str = "train") -> list[str]:
    return _prompts(split)["prompt_id"].tolist()


def allowed_ids(split: str = "train") -> set:
    return set(_prompts(split)["prompt_id"])


@lru_cache(maxsize=2)
def _prompt_rules(split: str) -> pd.DataFrame:
    PR = pd.read_parquet(INP / "prompt_rules.parquet")
    PR = PR[PR["split"] == split]
    _check_split(PR, split)
    order = {p: i for i, p in enumerate(_prompts(split)["prompt_id"])}
    PR = PR.assign(_o=PR["prompt_id"].map(order)).sort_values(["_o", "position"]).drop(columns="_o").reset_index(drop=True)
    assert len(PR) == (N_TRAIN_RULE_ROWS if split == "train" else N_TEST_RULE_ROWS)
    return PR


def prompt_rules(split: str = "train") -> pd.DataFrame:
    return _prompt_rules(split).copy()


@lru_cache(maxsize=4)
def _labels(model: str, split: str) -> pd.DataFrame:
    V = pd.read_parquet(VERIF / f"{model}_{split}.parquet",
                        columns=["prompt_id", "sample_index", "position", "rule_id", "group_id", "violated"])
    if split == "train":
        assert not (set(V.group_id) & splits.test_groups())
    else:
        _check_split(V.assign(split="test"), "test")
    assert V["violated"].isin([0, 1]).all()
    L = V.groupby(["prompt_id", "position", "rule_id"], sort=False).agg(v=("violated", "sum"), n=("violated", "size")).reset_index()
    assert (L["n"] == N_SAMPLES).all() and len(L) == len(_prompt_rules(split))
    return L


def labels_prompt_rule(model: str, split: str = "train") -> pd.DataFrame:
    PR = _prompt_rules(split)[["prompt_id", "position", "rule_id", "type_id"]]
    return PR.merge(_labels(model, split), on=["prompt_id", "position", "rule_id"], how="left", validate="one_to_one")


def fold_of_group() -> dict:
    return json.loads(FOLDS.read_text())["fold_of_group"]
