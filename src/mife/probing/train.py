"""Final refits: each configuration's CV-selected head setting refitted on ALL 57 TRAIN groups.

Fitted heads (scaler + weights, per type or shared) go to results/models/{model}/obj{o}/{key}.pt; non-learned
references store their rates as JSON. Encoders and LLM weights are never copied. B2-family refits need the final
B2 encoder (adapted on all 57 groups), which is produced together with the test embeddings after unblinding.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from ..config import derive_seed
from ..storage import write_json
from . import configs as C
from .cv import CV_DIR, Store, pcfg
from . import data, heads
from .data import RESULTS

MODELS_DIR = RESULTS / "models"


def _grid(model, objective, key) -> dict:
    return json.loads((CV_DIR / model / f"obj{objective}" / "grid" / f"{key}.json").read_text())


def _key(config, sel):
    return config + (f"__h{sel['hidden_row']}" if C.uses_hidden(config) else "") + \
        (f"__a{sel['pooling']}" if C.uses_attention(config) else "")


def refit_config(st: Store, config: str, sel: dict) -> dict:
    o = st.objective
    key = _key(config, sel)
    out_p = MODELS_DIR / st.model / f"obj{o}" / f"{key}.pt"
    if out_p.exists():
        return {"config": config, "key": key, "skipped": True}
    g = _grid(st.model, o, key)
    kind, l2 = g["selected"].split("_l2=")
    l2 = float(l2)
    blocks = g["blocks"]
    st.cur_fold = "final" if C.uses_b2(config) else None
    t0 = time.time()
    units = sorted(set(st.types)) if C.per_type(config, o) else [None]
    saved = {}
    for t in units:
        rows = np.nonzero(st.types == t)[0] if t is not None else np.arange(len(st.y))
        X = st.X(blocks, rows, sel["hidden_row"], sel["pooling"])
        y, n = st.y[rows], None if st.n is None else st.n[rows]
        if kind == "linear":
            h = heads.fit_linear(X, y, n, o, l2)
        else:
            seed = derive_seed(pcfg()["seeds"]["heads"], st.model, o, key, "final", t, kind, l2) % (1 << 31)
            h, _ = heads.fit_mlp(X, y, n, o, l2, g["selected_epoch"], seed, lr=pcfg()["heads"]["mlp_lr"],
                                 batch=pcfg()["heads"]["mlp_batch"])
        saved[t or "shared"] = {"kind": h.kind, "l2": h.l2, "epochs": h.epochs, "scaler_mean": torch.from_numpy(h.scaler.mean),
                                "scaler_scale": torch.from_numpy(h.scaler.scale), "state": h.state}
    st.cur_fold = None
    out_p.parent.mkdir(parents=True, exist_ok=True)
    meta = {"config": config, "key": key, "objective": o, "model": st.model, "blocks": blocks,
            "hidden_row": g.get("hidden_row"), "pooling": g.get("pooling"), "per_type": C.per_type(config, o),
            "setting": g["selected"], "epochs": g.get("selected_epoch"), "n_train_units": int(len(st.y)),
            "fit_groups": int(len(set(st.groups))), "wall_s": time.time() - t0}
    torch.save({"meta": meta, "heads": saved}, out_p)
    write_json(meta, out_p.with_suffix(".json"))
    return meta


def refit_references(st: Store) -> dict:
    """Rates from all 57 TRAIN groups for REF_constant / REF_type / REF_type_k (Objective 1 labels)."""
    L = data.prompt_rules("train")[["prompt_id", "position", "type_id", "k"]].merge(
        data.labels_prompt_rule(st.model, "train")[["prompt_id", "position", "v", "n"]], on=["prompt_id", "position"])
    g1 = L.groupby("type_id")[["v", "n"]].sum()
    g2 = L.groupby(["type_id", "k"])[["v", "n"]].sum()
    const = float(L.v.sum() / L.n.sum()) if st.objective == 1 else float(st.y.mean())
    out = {"REF_constant": const, "REF_type": (g1.v / g1.n).to_dict(),
           "REF_type_k": {f"{t}|{k}": float(r) for (t, k), r in (g2.v / g2.n).items()}}
    p = MODELS_DIR / st.model / f"obj{st.objective}" / "references.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, p)
    return out


def refit(model: str, objective: int, configs: list[str] | None = None) -> dict:
    torch.set_num_threads(int(pcfg().get("torch_threads", 4)))
    sel = json.loads((CV_DIR / model / f"obj{objective}" / "selection.json").read_text())
    st = Store(model, objective)
    todo = configs or [c for c in C.CONFIGS if not C.uses_b2(c)] + sorted(C.LEARNED_REFS)
    done = []
    for c in todo:
        if C.uses_b2(c) and not (RESULTS / "cv" / model / f"obj{objective}" / "b2_selection.json").exists():
            continue
        r = refit_config(st, c, sel)
        print(json.dumps(r), flush=True)
        done.append(r)
    refit_references(st)
    return {"model": model, "objective": objective, "refitted": [d["key"] for d in done]}
