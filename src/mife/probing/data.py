"""Loaders and joins for the probing phase, with the integrity guard and the test-set gate.

* Every frozen input is read through content hashes (`mife.integrity`): prompt_sha256 / rendered_sha256 must match
  the current manifest and tokenization, or StaleDataError is raised.
* TEST rows never enter memory unless the test set is unblinded: label/verification/generation files are read with a
  pyarrow filter on the allowed prompt_ids, and feature readers refuse disallowed ids.
* Only run `main` features (Apple MPS) are used; pilot and CUDA runs are refused.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from ..config import DATA_DIR, ROOT, load_config
from .. import integrity
from ..introspect import bits_to_float32

RUN = "main"
MODELS = ("llama", "qwen")
RESULTS = ROOT / "results"
PROBE_FEAT = DATA_DIR / "probing_features"
FROZEN = RESULTS / "prespec" / "FROZEN.json"


class TestSetLocked(RuntimeError):
    """Raised when code asks for TEST data before the pre-specification is frozen and unblinding approved."""


def test_unblinded() -> bool:
    if not FROZEN.exists():
        return False
    return bool(json.loads(FROZEN.read_text()).get("unblind_approved", False))


def _splits(split: str) -> tuple[str, ...]:
    if split == "train":
        return ("train",)
    if split in ("test", "all"):
        if not test_unblinded():
            raise TestSetLocked(f"split={split!r} requested but the test set is locked (no approved {FROZEN})")
        return ("train", "test") if split == "all" else ("test",)
    raise ValueError(split)


@lru_cache(maxsize=None)
def _manifest_all() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "manifest" / "prompts.parquet")


def prompts(split: str = "train") -> pd.DataFrame:
    """Manifest prompts of the requested split (TRAIN by default)."""
    m = _manifest_all()
    return m[m["split"].isin(_splits(split))].reset_index(drop=True)


def prompt_ids(split: str = "train") -> list[str]:
    return prompts(split)["prompt_id"].tolist()


def prompt_rules(split: str = "train") -> pd.DataFrame:
    pr = pd.read_parquet(DATA_DIR / "manifest" / "prompt_rules.parquet")
    return pr[pr["split"].isin(_splits(split))].reset_index(drop=True)


def allowed_ids(split: str = "train") -> set[str]:
    return set(prompt_ids(split))


def assert_allowed(ids, split: str = "train") -> None:
    bad = set(ids) - allowed_ids(split)
    if bad:
        raise TestSetLocked(f"{len(bad)} prompt_ids outside split {split!r} requested (e.g. {sorted(bad)[0]})")


def _read_filtered(path: Path, ids: list[str], columns=None) -> pd.DataFrame:
    d = ds.dataset(path, format="parquet")
    t = d.to_table(columns=columns, filter=ds.field("prompt_id").isin(ids))
    return t.to_pandas()


def verification(model: str, name: str, split: str = "train", columns=None) -> pd.DataFrame:
    """One verification output of run `main`, restricted to `split`, with hash checks."""
    assert model in MODELS
    ids = prompt_ids(split)
    cols = None if columns is None else sorted(set(columns) | {"prompt_id", "prompt_sha256", "rendered_sha256"})
    df = _read_filtered(DATA_DIR / "verification" / model / RUN / name, ids, cols)
    integrity.check_prompt(df, f"{model}/{name}")
    integrity.check_rendered(model, df, f"{model}/{name}")
    return df


def labels_prompt_rule(model: str, split: str = "train") -> pd.DataFrame:
    return verification(model, "labels_prompt_rule.parquet", split)


def labels_prompt(model: str, split: str = "train") -> pd.DataFrame:
    return verification(model, "labels_prompt.parquet", split)


def generations(model: str, split: str = "train", columns=None) -> pd.DataFrame:
    """Raw responses of run `main` for `split` (generation shards carry rendered_sha256 only)."""
    assert model in MODELS
    ids = prompt_ids(split)
    cols = None if columns is None else sorted(set(columns) | {"prompt_id", "rendered_sha256", "response_id"})
    gdir = DATA_DIR / "generations" / model / RUN
    df = pd.concat([_read_filtered(p, ids, cols) for p in sorted(gdir.glob("shard_*.parquet"))], ignore_index=True)
    integrity.check_rendered(model, df, f"{model}/generations")
    return df


# ----------------------------------------------------------------------------------------------- LLM features
def feature_dir(model: str) -> Path:
    d = DATA_DIR / "features" / model / RUN
    meta = json.loads((d / "features_meta.json").read_text())
    if meta.get("device") != "mps" or meta.get("backend") != "transformers+torch-mps":
        raise RuntimeError(f"{d} is not the Apple MPS feature run; only `main` MPS features may be used")
    return d


@lru_cache(maxsize=None)
def feature_meta(model: str) -> dict:
    return json.loads((feature_dir(model) / "features_meta.json").read_text())


@lru_cache(maxsize=None)
def feature_index(model: str) -> pd.DataFrame:
    idx = pd.read_parquet(feature_dir(model) / "index.parquet")
    integrity.check_prompt(idx, f"{model} feature index")
    integrity.check_rendered(model, idx, f"{model} feature index")
    return idx.set_index("prompt_id")


def hidden_states(model: str, ids: list[str], rows: list[int], split: str = "train") -> dict[int, np.ndarray]:
    """{row: float32 (len(ids), d)} final-prompt-token hidden states, read shard by shard via memmap."""
    assert_allowed(ids, split)
    idx = feature_index(model).loc[ids, ["shard", "row"]]
    d = feature_meta(model)["hidden_size"]
    out = {r: np.empty((len(ids), d), dtype=np.float32) for r in rows}
    pos = np.arange(len(ids))
    fdir = feature_dir(model)
    for shard, g in idx.assign(_i=pos).groupby("shard"):
        H = np.load(fdir / f"hidden_{shard:05d}.npy", mmap_mode="r")
        for r in rows:
            out[r][g["_i"].values] = bits_to_float32(np.asarray(H[g["row"].values, r, :]))
    return out


def rule_attention(model: str, ids: list[str], split: str = "train") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rule_sum, rule_mean, rule_mask): float32 (n, L, heads, 5) twice and bool (n, 5), slots in prompt order."""
    assert_allowed(ids, split)
    idx = feature_index(model).loc[ids, ["shard", "row"]]
    meta = feature_meta(model)
    L, Hh = meta["n_layers"], meta["n_heads"]
    S = np.full((len(ids), L, Hh, 5), np.nan, dtype=np.float32)
    M = np.full_like(S, np.nan)
    K = np.zeros((len(ids), 5), dtype=bool)
    pos = np.arange(len(ids))
    fdir = feature_dir(model)
    for shard, g in idx.assign(_i=pos).groupby("shard"):
        A = np.load(fdir / f"attn_{shard:05d}.npz")
        r = g["row"].values
        S[g["_i"].values] = A["rule_sum"][r]
        M[g["_i"].values] = A["rule_mean"][r]
        K[g["_i"].values] = A["rule_mask"][r]
    return S, M, K
