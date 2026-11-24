"""One forward pass per S3 TEST prompt with the ORIGINAL frozen setup (local checkpoint, frozen tokenization, HF eager,
BF16, Apple MPS), storing exactly what the frozen V2 Objective 1/2 refits need:

  max_NNNNN.npy    (n, L, heads, 5) uint16  A1max = max over the rule span of final-token attention (raw BF16 bits)
  hid_NNNNN.npy    (n, len(ROWS), d) uint16 final-prompt-token hidden state at the selected rows (raw BF16 bits;
                   rows follow the V1 convention: r < L = output of decoder layer r)
  mask_NNNNN.npy   (n, 5) bool
  sum_NNNNN.npy    (n, L, heads, 5) float32 span SUM (V1 arithmetic) for the consistency check against V1 features
  index_NNNNN.parquet  commit marker (prompt/rendered hashes)

Prompts and tokenization come from data/evaluation_expansion_s3/ (tokenized_<model>.parquet), which holds the 16,236
original S3 TEST prompts (frozen ids, identical tokenization) and the 13,860 expansion prompts. Resumable.
Run: python -m mife.eval_s3.extract <model>
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from ..introspect import load_torch_model, torch_device, _sync
from ..storage import write_npy, write_parquet
from . import paths as A

ROWS = {"llama": [12], "qwen": [18, 27]}  # Obj1 (llama 12, qwen 18) and Obj2 (llama 12, qwen 27) selected rows
K_MAX = 5


def out_dir(model: str):
    d = A.TEST_FEAT / model / "llm"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tokenized(model: str) -> pd.DataFrame:
    return pd.read_parquet(A.EXP_DIR / f"tokenized_{model}.parquet").set_index("prompt_id")


def committed(model: str):
    out = out_dir(model)
    frames, nxt = [], 0
    for f in sorted(out.glob("index_[0-9]*.parquet")):
        n = int(f.stem.split("_")[1])
        if all((out / f"{p}_{n:05d}.npy").exists() for p in ("max", "hid", "mask", "sum")):
            frames.append(pd.read_parquet(f))
            nxt = max(nxt, n + 1)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["prompt_id"])
    return df, nxt


def _bits(t) -> np.ndarray:
    import torch
    return t.to(torch.bfloat16).contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def run(model: str, shard_size: int = 512) -> None:
    import torch
    device = torch_device()
    if device != "mps":
        raise SystemExit("must run on the original Apple MPS backend")
    tok = tokenized(model)
    done, shard = committed(model)
    done = set(done["prompt_id"])
    todo = [p for p in tok.index if p not in done]
    print(f"[extract {model}] {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return
    t0 = time.time()
    m = load_torch_model(model, device)
    L, Hh, d = m.config.num_hidden_layers, m.config.num_attention_heads, m.config.hidden_size
    rows = ROWS[model]
    per = []
    for s0 in range(0, len(todo), shard_size):
        pids = todo[s0:s0 + shard_size]
        MX = np.zeros((len(pids), L, Hh, K_MAX), np.uint16)
        SM = np.zeros((len(pids), L, Hh, K_MAX), np.float32)
        HD = np.zeros((len(pids), len(rows), d), np.uint16)
        MK = np.zeros((len(pids), K_MAX), bool)
        recs = []
        for i, pid in enumerate(pids):
            r = tok.loc[pid]
            ids = torch.tensor([list(r.input_ids)], device=device)
            t1 = time.time()
            with torch.no_grad():
                o = m(input_ids=ids, output_hidden_states=True, output_attentions=True, use_cache=False)
            last = ids.shape[1] - 1
            att = torch.stack([a[0, :, last, :] for a in o.attentions])  # (L, heads, seq) bf16
            if att.dtype != torch.bfloat16:
                raise RuntimeError(f"attention dtype {att.dtype}")
            spans = [tuple(int(v) for v in x) for x in r.rule_token_spans]
            mx = torch.zeros((L, Hh, K_MAX), dtype=att.dtype, device=att.device)
            sm = torch.zeros((L, Hh, K_MAX), dtype=torch.float32, device=att.device)
            attf = att.float()
            for j, (a, b) in enumerate(spans):
                mx[:, :, j] = att[:, :, a:b].max(-1).values
                sm[:, :, j] = attf[:, :, a:b].sum(-1)
            MX[i] = _bits(mx)
            SM[i] = sm.cpu().numpy()
            MK[i, :len(spans)] = True
            HD[i] = _bits(torch.stack([o.hidden_states[rr][0, last] for rr in rows]))
            _sync(device)
            per.append(time.time() - t1)
            recs.append({"prompt_id": pid, "shard": shard, "row": i, "prompt_sha256": r.prompt_sha256,
                         "rendered_sha256": r.rendered_sha256, "is_original": bool(r.is_original),
                         "n_input_tokens": int(r.n_input_tokens)})
            del o, att, attf
        out = out_dir(model)
        write_npy(MX, out / f"max_{shard:05d}.npy")
        write_npy(SM, out / f"sum_{shard:05d}.npy")
        write_npy(HD, out / f"hid_{shard:05d}.npy")
        write_npy(MK, out / f"mask_{shard:05d}.npy")
        write_parquet(pd.DataFrame(recs), out / f"index_{shard:05d}.parquet")
        n_done = len(done) + s0 + len(pids)
        rate = np.mean(per[-len(pids):])
        print(f"[extract {model}] shard {shard}: {n_done}/{len(tok)}, {rate:.3f} s/prompt, "
              f"eta {(len(tok) - n_done) * rate / 3600:.2f} h", flush=True)
        shard += 1
        torch.mps.empty_cache()
    meta = {"model": model, "rows": rows, "n_layers": L, "n_heads": Hh, "hidden_size": d,
            "backend": "transformers+torch-mps", "attn_implementation": "eager", "dtype": "bfloat16",
            "torch_version": torch.__version__, "transformers_version": __import__("transformers").__version__,
            "wall_s": time.time() - t0, "per_prompt_s_median": float(np.median(per)) if per else None,
            "row_convention": "r < L: output of decoder layer r (HF hidden_states[r]), as in V1 features"}
    (out_dir(model) / "extract_meta.json").write_text(json.dumps(meta, indent=2))


def load(model: str, ids: list[str]):
    """(A1max float32 (n,L,H,5), mask (n,5), hidden {row: float32 (n,d)}, span sums float32) in `ids` order."""
    from ..introspect import bits_to_float32
    idx, _ = committed(model)
    idx = idx.set_index("prompt_id").loc[ids, ["shard", "row"]]
    meta = json.loads((out_dir(model) / "extract_meta.json").read_text())
    L, Hh, d, rows = meta["n_layers"], meta["n_heads"], meta["hidden_size"], meta["rows"]
    Amax = np.empty((len(ids), L, Hh, K_MAX), np.float32)
    Asum = np.empty((len(ids), L, Hh, K_MAX), np.float32)
    K = np.zeros((len(ids), K_MAX), bool)
    H = {r: np.empty((len(ids), d), np.float32) for r in rows}
    out = out_dir(model)
    pos = np.arange(len(ids))
    for sh, g in idx.assign(_i=pos).groupby("shard"):
        rr, ii = g["row"].values, g["_i"].values
        Amax[ii] = bits_to_float32(np.load(out / f"max_{sh:05d}.npy", mmap_mode="r")[rr])
        Asum[ii] = np.load(out / f"sum_{sh:05d}.npy", mmap_mode="r")[rr]
        K[ii] = np.load(out / f"mask_{sh:05d}.npy")[rr]
        hd = np.load(out / f"hid_{sh:05d}.npy", mmap_mode="r")[rr]
        for j, r in enumerate(rows):
            H[r][ii] = bits_to_float32(np.asarray(hd[:, j]))
    return Amax, K, H, Asum


if __name__ == "__main__":

    for mk in sys.argv[1:]:
        run(mk)
