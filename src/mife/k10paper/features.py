"""K10 TRAIN feature blocks under data/k10_paper/ (row order = kdata.prompts() / kdata.prompt_rules()).

shared/EP.npy        (375000, 1024) frozen BGE-M3 dense embedding of the full prompt (pinned encoder, as V2)
shared/ER.npy        (n_rules, 1024) frozen BGE-M3 embedding of each exact TRAIN rule text; ER_keys.parquet
<model>/S1.npy       Objective 1 structure (V2 columns; k and position now run to 10), rows = prompt_rules
<model>/H<row>.npy   final-prompt-token hidden state at the 5 candidate rows, float32 decoded exactly from BF16 bits
<model>/A1R.npy      max attention over the target rule's span, every layer x head, rows = prompt_rules, float32
                     decoded exactly from BF16 bits (the Betty CUDA K10 feature pass; PRESPEC DEVIATION: backend)
Only TRAIN prompts are opened: each shard's index is filtered to TRAIN prompt ids before its arrays are sliced.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from ..probing import features as v1features
from . import kdata
from .paths import FEAT, FEAT_SRC, HIDDEN_ROWS, INP, K_MAX

TYPES = v1features.TYPES
T_IX = v1features.T_IX
S1_COLS = (["k", "position"] + [f"companion:{t}" for t in TYPES]
           + ["prompt_tokens", "target_span_tokens", "target_dist_to_final"])


def d(model: str | None = None, split: str = "train"):
    p = FEAT / (model or "shared") / ("" if split == "train" else "test")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save(arr, path, **meta):
    np.save(path, arr)
    (path.parent / (path.name + ".meta.json")).write_text(json.dumps(
        {"shape": list(arr.shape), "dtype": str(arr.dtype), "rows": meta.pop("rows"), **meta}, indent=1, default=str))


def _tok(model):
    return pd.read_parquet(INP / f"tokenized_{model}.parquet",
                           columns=["prompt_id", "n_input_tokens", "rule_token_spans"]).set_index("prompt_id")


def build_structure(model: str, split: str = "train") -> None:
    P, PR = kdata.prompts(split), kdata.prompt_rules(split)
    tok = _tok(model).loc[P["prompt_id"]]
    n_tok = tok["n_input_tokens"].values.astype(np.float64)
    npm = len(P)
    pid = {x: i for i, x in enumerate(P["prompt_id"])}
    span_len = np.zeros((npm, K_MAX))
    dist = np.zeros((npm, K_MAX))
    for i, (nt, spans) in enumerate(zip(n_tok, tok["rule_token_spans"])):
        for j, (a, b) in enumerate(spans):
            span_len[i, j] = b - a
            dist[i, j] = nt - b
    slot_type = np.full((npm, K_MAX), -1)
    pi = PR["prompt_id"].map(pid).values
    si = PR["position"].values - 1
    slot_type[pi, si] = PR["type_id"].map(T_IX).values
    assert ((slot_type >= 0).sum(1) == P["k"].values).all()
    assert ((span_len > 0) == (slot_type >= 0)).all()
    slot_oh = np.zeros((npm, K_MAX, len(TYPES)))
    ii, jj = np.nonzero(slot_type >= 0)
    slot_oh[ii, jj, slot_type[ii, jj]] = 1
    comp = slot_oh.sum(1)
    k = P["k"].values.astype(np.float64)
    t_oh = np.zeros((len(PR), len(TYPES)))
    t_oh[np.arange(len(PR)), PR["type_id"].map(T_IX).values] = 1
    S1 = np.concatenate([k[pi][:, None], PR["position"].values[:, None].astype(np.float64), comp[pi] - t_oh,
                         n_tok[pi][:, None], span_len[pi, si][:, None], dist[pi, si][:, None]], 1)
    assert S1.shape[1] == len(S1_COLS)
    _save(S1.astype(np.float32), d(model, split) / "S1.npy", rows=f"kdata.prompt_rules({split!r})", columns=S1_COLS, tokenizer=model)


def build_bge(batch_size: int = 32, limit: int | None = None, split: str = "train") -> dict:
    from ..probing import encoder
    P, PR = kdata.prompts(split), kdata.prompt_rules(split)
    model = encoder.load_model()
    out = d(None, split) / "EP.npy"
    part = d(None, split) / "EP_partial.npy"
    n = len(P) if limit is None else limit
    EP = np.lib.format.open_memmap(part, mode="r+" if part.exists() else "w+", dtype=np.float32, shape=(len(P), 1024))
    done_f = d(None, split) / "EP_done.json"
    done = json.loads(done_f.read_text())["done"] if done_f.exists() else 0
    texts = P["prompt_text"].tolist()
    t0, CH = time.time(), 4096
    for s in range(done, n, CH):
        EP[s:s + CH] = encoder.encode(texts[s:s + CH], model, batch_size=batch_size)
        EP.flush()
        done_f.write_text(json.dumps({"done": min(s + CH, n)}))
        el = time.time() - t0
        print(f"EP {min(s + CH, n)}/{n} {el:.0f}s ({(min(s + CH, n) - done) / el:.1f}/s)", flush=True)
    rules = PR.drop_duplicates("rule_id")[["rule_id", "rule_text"]].sort_values("rule_id").reset_index(drop=True)
    ER = encoder.encode(rules["rule_text"].tolist(), model)
    _save(ER, d(None, split) / "ER.npy", rows="ER_keys.parquet", encoder=encoder.cfg())
    rules.to_parquet(d(None, split) / "ER_keys.parquet")
    if limit is None:
        del EP
        part.rename(out)
        (d(None, split) / "EP.npy.meta.json").write_text(json.dumps({"rows": f"kdata.prompts({split!r})", "encoder": encoder.cfg(),
                                                         "definition": "frozen BGE-M3 dense [CLS] embedding of the full prompt text",
                                                         "shape": [len(P), 1024]}, indent=1))
    return {"n_rules": len(rules)}


def _index(model: str, split: str = "train") -> pd.DataFrame:
    allowed = kdata.allowed_ids(split)
    I = pd.concat([pd.read_parquet(f, columns=["prompt_id", "row", "shard", "rule_token_spans"])
                   for f in sorted((FEAT_SRC / model).glob("index_*.parquet"))], ignore_index=True)
    I = I[I["prompt_id"].isin(allowed)]
    assert I["prompt_id"].is_unique and len(I) == len(allowed)
    return I


def build_hidden_attn(model: str, split: str = "train") -> None:
    """Decode the rows of one split from every shard into H<row>.npy (prompt order) and A1R.npy (prompt-rule order)."""
    P, PR = kdata.prompts(split), kdata.prompt_rules(split)
    pix = {x: i for i, x in enumerate(P["prompt_id"])}
    I = _index(model, split)
    D_ = d(model, split)
    tok = _tok(model)
    # spans in the feature index must equal the frozen tokenization (same prompt bytes)
    for pid, sp in zip(I.prompt_id.values[:2000], I.rule_token_spans.values[:2000]):
        assert [list(x) for x in sp] == [list(x) for x in tok.loc[pid, "rule_token_spans"]]
    rows = HIDDEN_ROWS[model]
    first = I.shard.iloc[0]
    h0 = np.load(FEAT_SRC / model / f"hidden_{first}.npy", mmap_mode="r")
    a0 = np.load(FEAT_SRC / model / f"attn_{first}.npz")["max_bits"]
    D, L, Hh = h0.shape[2], a0.shape[1], a0.shape[2]
    Hm = [np.lib.format.open_memmap(D_ / f"H{r}.npy.partial", mode="w+", dtype=np.float32, shape=(len(P), D))
          for r in rows]
    # rule-row offsets: PR is sorted by prompt order then position
    start = np.zeros(len(P) + 1, dtype=np.int64)
    np.add.at(start, PR["prompt_id"].map(pix).values + 1, 1)
    start = np.cumsum(start)
    kk = P["k"].values
    A = np.lib.format.open_memmap(D_ / "A1R.npy.partial", mode="w+", dtype=np.float32, shape=(len(PR), L * Hh))
    t0 = time.time()
    for n, (sh, g) in enumerate(I.groupby("shard", sort=True)):
        h = np.load(FEAT_SRC / model / f"hidden_{sh}.npy", mmap_mode="r")
        z = np.load(FEAT_SRC / model / f"attn_{sh}.npz")
        mx, mk = z["max_bits"], z["mask"]
        r_in = g["row"].values
        p_out = g["prompt_id"].map(pix).values
        hf = (h[r_in].astype(np.uint32) << 16).view(np.float32)  # exact BF16 -> float32
        for j in range(len(rows)):
            Hm[j][p_out] = hf[:, j]
        af = (mx[r_in].astype(np.uint32) << 16).view(np.float32)  # (m, L, H, 10)
        for a, po in enumerate(p_out):
            k = kk[po]
            assert mk[r_in[a]].sum() == k and mk[r_in[a], :k].all()
            A[start[po]:start[po] + k] = np.moveaxis(af[a, :, :, :k], 2, 0).reshape(k, -1)
        if n % 25 == 0:
            print(f"{model} shard {n}/{I.shard.nunique()} {time.time() - t0:.0f}s", flush=True)
    for m, r in zip(Hm, rows):
        m.flush()
        del m
        (D_ / f"H{r}.npy.partial").rename(D_ / f"H{r}.npy")
        (D_ / f"H{r}.npy.meta.json").write_text(json.dumps({"rows": "kdata.prompts()", "row": r, "dtype": "float32",
                                                                "source": "Betty CUDA K10 feature pass (BF16 bits)"}))
    A.flush()
    del A
    (D_ / "A1R.npy.partial").rename(D_ / "A1R.npy")
    (D_ / "A1R.npy.meta.json").write_text(json.dumps({"rows": "kdata.prompt_rules()", "layout": "layer-major (L, heads)",
                                                          "n_layers": L, "n_heads": Hh, "source": "max_bits, Betty CUDA K10"}))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "structure":
        for m in sys.argv[2].split(","):
            build_structure(m)
    elif cmd == "bge":
        print(build_bge(limit=int(sys.argv[2]) if len(sys.argv) > 2 else None))
    elif cmd == "hidden":
        build_hidden_attn(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "train")
    elif cmd == "structure_test":
        for m in sys.argv[2].split(","):
            build_structure(m, "test")
    elif cmd == "bge_test":
        print(build_bge(split="test"))


# ------------------------------------------------------------------------------------------------ loaders
def rule_index() -> dict:
    ids = sorted(kdata.prompt_rules()["rule_id"].unique())  # = ER_keys.parquet order (build_bge sorts by rule_id)
    f = d() / "ER_keys.parquet"
    if f.exists():
        assert pd.read_parquet(f)["rule_id"].tolist() == ids
    return {r: i for i, r in enumerate(ids)}


def load_shared(name: str) -> np.ndarray:
    if name in ("EP", "ER"):
        while not ((d() / "EP.npy").exists() and (d() / "ER.npy").exists()):  # the BGE-M3 job may still be running
            time.sleep(60)
    if name == "EP":
        a = np.load(d() / "EP.npy", mmap_mode="r")
        assert a.shape[0] == len(kdata.prompt_ids())
        return a
    if name == "ER":
        return np.load(d() / "ER.npy")
    raise KeyError(name)


def load_model_block(model: str, name: str) -> np.ndarray:
    f = d(model) / f"{name}.npy"
    a = np.load(f, mmap_mode="r")
    n = len(kdata.prompt_ids()) if name.startswith("H") else len(kdata.prompt_rules())
    assert a.shape[0] == n, (f, a.shape)
    if name.startswith("H"):
        assert json.loads((d(model) / f"{name}.npy.meta.json").read_text())["row"] == int(name[1:])
    return a
