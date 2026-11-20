"""Atomic, hashed persistence helpers (Parquet / JSON / JSONL / NumPy)."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _atomic(path: Path, write_fn) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        write_fn(tmp)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_parquet(df: pd.DataFrame, path) -> None:
    _atomic(path, lambda t: df.to_parquet(t, index=False))


def write_json(obj, path) -> None:
    def w(t):
        with open(t, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True, default=str)
    _atomic(path, w)


def write_jsonl(rows, path) -> None:
    def w(t):
        with open(t, "w") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    _atomic(path, w)


def write_npy(arr: np.ndarray, path) -> None:
    def w(t):
        with open(t, "wb") as f:
            np.save(f, arr)
    _atomic(path, w)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
