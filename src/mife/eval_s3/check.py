"""Independent checks of the S3 TEST Objective 1 scoring for one generator (writes tables/checks_<gen>.csv).

1. Split hygiene: TEST prompts and S3 TRAIN prompts are disjoint; refits record the S3 split.
2. Labels: 115,236 rows, n = 20 everywhere; predictions finite and inside (0, 1).
3. Structure block: the TEST code path rebuilds the stored S3 TRAIN S1 block exactly.
4. Table 2 point estimates recomputed directly (NLL formula; AUROC with sklearn on expanded binary outcomes).
5. Table 4 position Δ-MSE (#2) recomputed with explicit per-row weights instead of nested means.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from ..config import DATA_DIR
from ..probing import data as pdata
from ..probing.v2 import features as F
from . import eval1, eval2, matches, predict
from .common import PREDS, data
from .paths import EVAL_DIR
from .stats import Boot


def run(gen: str) -> pd.DataFrame:
    out = []
    add = lambda name, ok, detail="": out.append({"generator": gen, "check": name, "pass": bool(ok), "detail": detail})
    d = data(gen)
    tr = set(pdata.prompt_ids("train"))
    add("TEST and S3 TRAIN prompts disjoint", not (set(d.prompt_id) & tr), f"{len(tr)} train prompts")
    js = json.loads((predict.MODELS_DIR / gen / "obj1" / "H2.json").read_text())
    add("refits fitted under split S3", js.get("split") == "split-s3-2026-11-23", js.get("split"))
    add("115,236 rows, n = 20", len(d) == 115236 and (d.n == 20).all())
    add("predictions finite in (0,1)", bool(np.all([(d[p] > 0).all() and (d[p] < 1).all() for p in PREDS])))
    # structure block on TRAIN rows vs the stored V2 S3 cache
    P, PR = pdata.prompts("train"), pdata.prompt_rules("train")
    S_new = predict.structure(gen, P, PR, tok_path=DATA_DIR / "tokenized" / f"{gen}.parquet")
    S_old = F.load_model_block(gen, "S1") if False else np.load(DATA_DIR / "probing_features_v2" / "split_s3" / gen / "S1_train.npy")
    add("TEST structure code reproduces stored TRAIN S1", np.array_equal(S_new.astype(np.float32), S_old),
        f"max diff {np.abs(S_new - S_old).max():.2e}")
    # Table 2 point estimates
    boot = Boot(sorted(d.group_id.unique()), b=200)
    e1 = eval1.run(gen, d, boot)
    from sklearn.metrics import roc_auc_score
    v, n = d.v.values, d.n.values
    y = np.r_[np.ones(v.sum()), np.zeros((n - v).sum())]
    for p in PREDS:
        q = np.clip(d[p].values, 1e-6, 1 - 1e-6)
        nll = -(v * np.log(q) + (n - v) * np.log1p(-q)).sum() / n.sum()
        s = np.r_[np.repeat(q, v), np.repeat(q, n - v)]
        auc = roc_auc_score(y, s)
        m = e1["a1_metrics"].set_index("predictor").loc[p]
        add(f"NLL/AUROC {p} recomputed directly", abs(nll - m.nll) < 1e-10 and abs(auc - m.auroc) < 1e-9,
            f"nll {nll:.6f} vs {m.nll:.6f}; auc {auc:.6f} vs {m.auroc:.6f}")
    # Position Δ-MSE (#2) with explicit weights
    pos = eval2.position(d, boot, gen, sorted(d.type_id.unique()))
    x = pos["pairs"]
    w = np.zeros(len(x))
    for k, xk in x.groupby("k"):
        G = xk.group_id.nunique()
        for g, xg in xk.groupby("group_id"):
            nt = xg.task_key.nunique()
            for t, xt in xg.groupby("task_key"):
                nc = xt.ctx.nunique()
                cnt = xt.groupby("ctx").pair_id.transform("size")
                w[xt.index] = 1 / (4 * G * nt * nc * cnt)
    est = float((w * x["dmse_#2"]).sum())
    tab = pos["skill"]
    ref = tab[(tab.quantity == "dmse_#2") & (tab.stratum == "macro")].est.iloc[0]
    add("Position Δ-MSE #2 equals explicit-weight sum", abs(est - ref) < 1e-12 and abs(w.sum() - 1) < 1e-9,
        f"{est:.3e} vs {ref:.3e}; weights sum {w.sum():.6f}")
    df = pd.DataFrame(out)
    (EVAL_DIR / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(EVAL_DIR / "tables" / f"checks_{gen}.csv", index=False)
    return df


if __name__ == "__main__":
    for g in sys.argv[1:]:
        print(run(g).to_string())
