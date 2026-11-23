"""Version keys and guarded load/save helpers for V2 artifacts.

Every V2 array is written with a JSON sidecar `<file>.meta.json` holding at least {"version", "kind", "definition"}.
`load_array` refuses a file whose sidecar is missing or whose version/kind/definition does not match what the caller
expects, so a V1 cache (no sidecar, or a different definition) can never be read silently under a V2 name.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ... import splits
from ...config import DATA_DIR, ROOT
from ...storage import write_json, write_npy

VERSION = "v2-2026-11-20"
SPLIT = splits.version()                      # active group split (configs/active_split.yaml)
_TAG = {splits.MANIFEST_VERSION: ""}.get(SPLIT, "_s3" if SPLIT == "split-s3-2026-11-23" else None)
if _TAG is None:
    raise RuntimeError(f"no V2 path tag for split {SPLIT!r}")
FEAT_V2 = DATA_DIR / "probing_features_v2"   # split-independent per-prompt features (max attention) live here
# Split-dependent caches (row order = that split's TRAIN prompts) and all results get their own tree per split, so the
# 2026-11-14-split V2 run (results/v2/, data/probing_features_v2/{shared,llama,qwen}) stays untouched.
FEAT_SPLIT = FEAT_V2 if _TAG == "" else FEAT_V2 / "split_s3"
RESULTS_V2 = ROOT / "results" / f"v2{_TAG}"

# Definitions of every V2 block. A V1 block with the same short name has a different definition string.
DEFINITIONS = {
    "B1": "frozen BGE-M3 full-prompt dense embedding + V2 structure",
    "B2": "B1 + separate frozen BGE-M3 embedding of the exact target-rule text (same pinned encoder; no fine-tuning)",
    "A1max": "max over the exact target-rule span tokens of the final-prompt-token attention probability, per layer x head "
             "(HF eager, BF16, Apple MPS; stored as raw BF16 bits)",
    "S1v2": "Objective 1 structure: k, target position, companion-type indicators, rendered prompt tokens, target span "
            "tokens, distance from target span end to final prompt token (generator tokenizer)",
    "S2v2": "Objective 2 structure: k, rendered prompt tokens, 5 ordered slots x (type one-hot, span tokens, distance to "
            "final token) + presence mask (generator tokenizer)",
}


class VersionMismatch(RuntimeError):
    pass


def meta_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".meta.json")


def save_array(arr: np.ndarray, path: Path, kind: str, **extra) -> None:
    write_npy(arr, path)
    write_json({"version": VERSION, "split": SPLIT, "kind": kind, "definition": DEFINITIONS.get(kind, kind),
                "shape": list(arr.shape), "dtype": str(arr.dtype), **extra}, meta_path(path))


def check_meta(path: Path, kind: str) -> dict:
    mp = meta_path(path)
    if not mp.exists():
        raise VersionMismatch(f"{path}: no V2 sidecar ({mp.name}); refusing to load an unversioned (V1?) artifact")
    m = json.loads(mp.read_text())
    if m.get("version") != VERSION:
        raise VersionMismatch(f"{path}: version {m.get('version')!r} != {VERSION!r}")
    if m.get("split", splits.MANIFEST_VERSION) != SPLIT:
        raise VersionMismatch(f"{path}: built for split {m.get('split', splits.MANIFEST_VERSION)!r}, active split is {SPLIT!r}")
    if m.get("kind") != kind:
        raise VersionMismatch(f"{path}: kind {m.get('kind')!r} != expected {kind!r}")
    if m.get("definition") != DEFINITIONS.get(kind, kind):
        raise VersionMismatch(f"{path}: definition of {kind!r} differs from the current V2 definition")
    return m


def load_array(path: Path, kind: str, mmap: bool = False) -> np.ndarray:
    check_meta(path, kind)
    return np.load(path, mmap_mode="r" if mmap else None)


def write_versioned_json(obj: dict, path: Path) -> None:
    write_json({"version": VERSION, "split": SPLIT, **obj}, path)


def read_versioned_json(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    if d.get("version") != VERSION:
        raise VersionMismatch(f"{path}: version {d.get('version')!r} != {VERSION!r}")
    if d.get("split", splits.MANIFEST_VERSION) != SPLIT and not d.get("split_independent"):
        raise VersionMismatch(f"{path}: written for split {d.get('split', splits.MANIFEST_VERSION)!r}, active is {SPLIT!r}")
    return d
