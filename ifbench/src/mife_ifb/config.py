"""Paths and configuration for the IFBench adaptation. Everything lives under ``ifbench/``."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

IFB_ROOT = Path(__file__).resolve().parents[2]          # .../ifbench
REPO_ROOT = IFB_ROOT.parent                             # ManyIFEval repository root (read-only for us)
CONFIG_PATH = IFB_ROOT / "configs" / "ifbench.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def source_dir() -> Path:
    return IFB_ROOT / load_config()["source"]["local_path"]


def nltk_data_dir() -> Path:
    return IFB_ROOT / load_config()["nltk"]["data_dir"]


def test_file() -> Path:
    return source_dir() / load_config()["source"]["test_file"]
