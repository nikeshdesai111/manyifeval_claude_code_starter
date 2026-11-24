"""TEST feature blocks in the V2 layouts and frozen-refit predictions for Objective 1 (no refitting, no tuning).

Blocks (rows = expanded TEST prompt_rules order):
  S   V2 Objective 1 structure (same definition and code path as features.build_structure; generator tokenizer)
  EP  frozen BGE-M3 dense embedding of prompt_text (pinned revision; mife.probing.encoder.encode)
  ER  frozen BGE-M3 embedding of the exact rule text
  H<row>  final-prompt-token hidden state (eval_s3.extract)
  A1  max attention over the target rule span, all layers x heads (eval_s3.extract)
Each frozen head: p = expit(((X - mean) / scale) @ w + b) per rule type (or its stored fallback rate).
Consistency checks against the V1/V2 caches are written to results/eval_s3/tables/feature_consistency.csv.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import DATA_DIR, ROOT
from ..probing.v2 import features as F
from ..probing.v2.ridge import expit
from . import extract
from .paths import EVAL_DIR, EXP_DIR, TEST_FEAT

MODELS_DIR = ROOT / "results" / "v2_s3" / "models"
PREDICTORS = ["REF_structure", "B1", "B2", "H1", "H2", "RCAP", "B1+H1", "B2+H1", "B2+A1"]
PRED_DIR = DATA_DIR / "eval_s3" / "predictions"


def test_frames():
    P = pd.read_parquet(EXP_DIR / "prompts.parquet")
    PR = pd.read_parquet(EXP_DIR / "prompt_rules.parquet")
    return P, PR


def structure(model: str, P, PR) -> np.ndarray:
    tok = pd.read_parquet(EXP_DIR / f"tokenized_{model}.parquet",
                          columns=["prompt_id", "n_input_tokens", "rule_token_spans"]).set_index("prompt_id").loc[P.prompt_id]
    n_tok = tok["n_input_tokens"].values.astype(np.float64)
    npm = len(P)
    pid = {x: i for i, x in enumerate(P["prompt_id"])}
    span_len = np.zeros((npm, 5)); dist = np.zeros((npm, 5))
    for i, (nt, spans) in enumerate(zip(n_tok, tok["rule_token_spans"])):
        for j, (a, b) in enumerate(spans):
            span_len[i, j] = b - a
            dist[i, j] = nt - b
    slot_type = np.full((npm, 5), -1)
    pi = PR["prompt_id"].map(pid).values
    si = PR["position"].values - 1
    slot_type[pi, si] = PR["type_id"].map(F.T_IX).values
    assert ((slot_type >= 0).sum(1) == P["k"].values).all()
    assert ((span_len > 0) == (slot_type >= 0)).all()
    slot_oh = np.zeros((npm, 5, len(F.TYPES)))
    ii, jj = np.nonzero(slot_type >= 0)
    slot_oh[ii, jj, slot_type[ii, jj]] = 1
    comp = slot_oh.sum(1)
    k = P["k"].values.astype(np.float64)
    t_oh = np.zeros((len(PR), len(F.TYPES)))
    t_oh[np.arange(len(PR)), PR["type_id"].map(F.T_IX).values] = 1
    S1 = np.concatenate([k[pi][:, None], PR["position"].values[:, None].astype(np.float64), comp[pi] - t_oh,
                         n_tok[pi][:, None], span_len[pi, si][:, None], dist[pi, si][:, None]], 1)
    assert S1.shape[1] == len(F.S1_COLS)
    return S1


def bge(P, PR):
    """Frozen BGE-M3 embeddings for the TEST prompts and exact rules (cached)."""
    from ..probing import encoder
    d = TEST_FEAT / "shared"
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "EP_test.npy").exists():
        model = encoder.load_model()
        EP = encoder.encode(P["prompt_text"].tolist(), model, batch_size=32)
        R = PR.drop_duplicates("rule_id")[["rule_id", "rule_text"]].reset_index(drop=True)
        ER = encoder.encode(R["rule_text"].tolist(), model, batch_size=32)
        np.save(d / "EP_test.npy", EP)
        np.save(d / "ER_test.npy", ER)
        P[["prompt_id"]].to_parquet(d / "EP_test_keys.parquet")
        R[["rule_id"]].to_parquet(d / "ER_test_keys.parquet")
    EP = np.load(d / "EP_test.npy")
    assert (pd.read_parquet(d / "EP_test_keys.parquet").prompt_id.values == P.prompt_id.values).all()
    ER = np.load(d / "ER_test.npy")
    rk = pd.read_parquet(d / "ER_test_keys.parquet").rule_id
    return EP, ER, {r: i for i, r in enumerate(rk)}


def consistency(model: str, P, PR, EP, H, Asum, row: int) -> list[dict]:
    """New TEST features vs the frozen V1 caches for prompts that have them (originals)."""
    from ..introspect import bits_to_float32
    out = []
    v1 = pd.read_parquet(DATA_DIR / "features" / model / "main" / "index.parquet").set_index("prompt_id")
    orig = P.index[P.is_original.values]
    sample = orig[:: max(1, len(orig) // 600)]
    ids = P.prompt_id.values[sample]
    loc = v1.loc[ids, ["shard", "row"]]
    hd_diff, sum_diff = [], []
    fdir = DATA_DIR / "features" / model / "main"
    for (sh, g) in loc.assign(_i=sample).groupby("shard"):
        Hv = np.load(fdir / f"hidden_{sh:05d}.npy", mmap_mode="r")
        At = np.load(fdir / f"attn_{sh:05d}.npz")["rule_sum"]
        for i, r in zip(g["_i"].values, g["row"].values):
            hv = bits_to_float32(np.asarray(Hv[r, row]))
            hd_diff.append(float(np.abs(hv - H[row][i]).max()))
            a = At[r]
            m = ~np.isnan(a)
            sum_diff.append(float(np.abs(a[m] - Asum[i][m]).max()))
    out.append({"model": model, "check": f"hidden row {row}: new TEST extraction vs V1 main features (original prompts)",
                "n_prompts": len(hd_diff), "max_abs_diff": max(hd_diff), "n_exact": int(sum(d == 0 for d in hd_diff))})
    out.append({"model": model, "check": "rule-span attention SUM: new extraction vs V1 main features",
                "n_prompts": len(sum_diff), "max_abs_diff": max(sum_diff), "n_exact": int(sum(d == 0 for d in sum_diff))})
    # BGE-M3: originals of the 16 groups that were TRAIN under the 2026-11-14 split have V1 frozen embeddings
    keys = pd.read_parquet(DATA_DIR / "probing_features" / "shared" / "EP_train_keys.parquet").prompt_id
    kix = {p: i for i, p in enumerate(keys)}
    both = [(i, kix[p]) for i, p in enumerate(P.prompt_id) if p in kix]
    if both:
        V1 = np.load(DATA_DIR / "probing_features" / "shared" / "EP_train.npy", mmap_mode="r")
        sel = both[:: max(1, len(both) // 600)]
        diff = [float(np.abs(V1[j] - EP[i]).max()) for i, j in sel]
        out.append({"model": "shared", "check": "BGE-M3 prompt embedding: TEST recompute vs V1 frozen cache",
                    "n_prompts": len(diff), "max_abs_diff": max(diff), "n_exact": int(sum(d == 0 for d in diff))})
    return out


def predict(model: str) -> pd.DataFrame:
    P, PR = test_frames()
    pid = {x: i for i, x in enumerate(P.prompt_id)}
    pix = PR.prompt_id.map(pid).values
    slot = PR.position.values - 1
    S1 = structure(model, P, PR)
    EP, ER, rix = bge(P, PR)
    Amax, K, H, Asum = extract.load(model, P.prompt_id.tolist())
    assert K[pix, slot].all()
    row = {"llama": 12, "qwen": 18}[model]
    checks = consistency(model, P, PR, EP, H, Asum, row)
    ri = PR.rule_id.map(rix).values
    blocks = {"S": S1, "EP": EP[pix], "ER": ER[ri], f"H{row}": H[row][pix],
              "A1": Amax[pix, :, :, slot].reshape(len(PR), -1)}
    out = PR[["prompt_id", "position", "rule_id", "type_id", "group_id"]].copy()
    for name in PREDICTORS:
        spec = json.loads((MODELS_DIR / model / "obj1" / f"{name}.json").read_text())
        z = np.load(MODELS_DIR / model / "obj1" / f"{name}.npz")
        X = np.concatenate([blocks[b] for b, _ in spec["blocks"]], axis=1).astype(np.float64)
        assert X.shape[1] == sum(spec["widths"]), (name, X.shape, spec["widths"])
        p = np.full(len(PR), np.nan)
        for t in F.TYPES:
            m = (PR.type_id == t).values
            if not m.any():
                continue
            if bool(z[f"{t}__fallback"]):
                p[m] = float(z[f"{t}__p"])
            else:
                p[m] = expit(((X[m] - z[f"{t}__mean"]) / z[f"{t}__scale"]) @ z[f"{t}__w"] + float(z[f"{t}__b"]))
        assert np.isfinite(p).all()
        out[name] = p
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PRED_DIR / f"{model}_obj1.parquet")
    (EVAL_DIR / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(EVAL_DIR / "tables" / f"feature_consistency_{model}.csv", index=False)
    return out


def load(model: str) -> pd.DataFrame:
    return pd.read_parquet(PRED_DIR / f"{model}_obj1.parquet")


if __name__ == "__main__":
    import sys
    for m in sys.argv[1:]:
        d = predict(m)
        print(m, d.shape, d[PREDICTORS].describe().T[["mean", "min", "max"]])
