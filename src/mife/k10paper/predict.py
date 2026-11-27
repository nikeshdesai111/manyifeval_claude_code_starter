"""Apply the frozen 50-group refits (results/k10_paper/v2/models/<model>/obj1/*.npz) to the K10 TEST groups.

Only runs under the frozen TEST prespec (kdata checks its hash). Writes results/k10_paper/test/predictions/<model>/obj1/
<predictor>.parquet with the same keys as the TRAIN OOF files and the prediction in `selected`.
Run: PYTHONPATH=src .venv/bin/python -m mife.k10paper.predict <model>
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pandas as pd

from ..probing.v2 import ridge
from . import features as F
from . import kdata
from .cv import RESULTS_V2
from .paths import OUT

PREDS = ["REF_structure", "B1", "B2", "H1", "H2", "RCAP", "B1+H1", "B2+H1", "B2+A1", "B2+H1+A1"]
CH = 50_000


class TestBlocks:
    def __init__(self, model: str):
        self.model = model
        self.P, self.PR = kdata.prompts("test"), kdata.prompt_rules("test")
        pid = {x: i for i, x in enumerate(self.P["prompt_id"])}
        self.pix = self.PR["prompt_id"].map(pid).values
        keys = pd.read_parquet(F.d(None, "test") / "ER_keys.parquet")["rule_id"].tolist()
        assert keys == sorted(self.PR["rule_id"].unique())
        self.rix = self.PR["rule_id"].map({r: i for i, r in enumerate(keys)}).values
        self._b = {}

    def block(self, name):
        if name not in self._b:
            dm, ds = F.d(self.model, "test"), F.d(None, "test")
            f = {"S": dm / "S1.npy", "A1": dm / "A1R.npy", "EP": ds / "EP.npy", "ER": ds / "ER.npy"}.get(name, dm / f"{name}.npy")
            self._b[name] = np.load(f, mmap_mode="r")
        return self._b[name]

    def rows(self, name, rows):
        a = self.block(name)
        if name in ("S", "A1"):
            return np.asarray(a[rows], dtype=np.float64)
        if name == "ER":
            return np.asarray(a[self.rix[rows]], dtype=np.float64)
        return np.asarray(a[self.pix[rows]], dtype=np.float64)


def run(model: str) -> dict:
    B = TestBlocks(model)
    mdir = RESULTS_V2 / "models" / model / "obj1"
    out = OUT / "test" / "predictions" / model / "obj1"
    out.mkdir(parents=True, exist_ok=True)
    hashes = json.loads((mdir / "HASHES.json").read_text())
    types = B.PR["type_id"].values
    res = {}
    for name in PREDS:
        f = mdir / f"{name}.npz"
        assert hashlib.sha256(f.read_bytes()).hexdigest() == hashes[f.name], f"{f} changed after it was frozen"
        meta = json.loads((mdir / f"{name}.json").read_text())
        blocks = [b for b, _ in meta["blocks"]]
        z = np.load(f)
        p = np.full(len(B.PR), np.nan)
        for t in np.unique(types):
            rows = np.nonzero(types == t)[0]
            if z.get(f"{t}__fallback") is not None and bool(z[f"{t}__fallback"]):
                p[rows] = float(z[f"{t}__p"])
                continue
            mu, sd, w, b = z[f"{t}__mean"], z[f"{t}__scale"], z[f"{t}__w"], float(z[f"{t}__b"])
            for s in range(0, len(rows), CH):
                r = rows[s:s + CH]
                X = np.concatenate([B.rows(bl, r) for bl in blocks], 1)
                p[r] = ridge.expit(((X - mu) / sd) @ w + b)
        assert np.isfinite(p).all()
        df = B.PR[["prompt_id", "position", "rule_id", "type_id"]].copy()
        df["selected"] = p
        df.to_parquet(out / f"{name}.parquet")
        res[name] = float(p.mean())
        print(model, name, round(p.mean(), 4), flush=True)
    return res


if __name__ == "__main__":
    print(run(sys.argv[1]))
