"""Project paths and configuration loading."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
DATA_DIR = ROOT / "data"


@lru_cache(maxsize=None)
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def config_sha256() -> str:
    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def source_dir() -> Path:
    return ROOT / load_config()["source"]["local_path"]


def nltk_dir() -> Path:
    return ROOT / load_config()["source"]["nltk_data"]


def derive_seed(*parts) -> int:
    """Deterministic 63-bit seed from arbitrary parts (stable across runs/platforms)."""
    h = hashlib.sha256(json.dumps([str(p) for p in parts]).encode()).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)
