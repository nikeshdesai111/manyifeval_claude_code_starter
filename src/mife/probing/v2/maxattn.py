"""Training-only V2 max-attention extraction (A1max), authorised by the 2026-11-20 amendment.

One forward pass per TRAIN prompt and generator with the ORIGINAL frozen setup: the same local checkpoint, the
frozen tokenization (`data/tokenized/<model>.parquet`: input ids and exact rule token spans), HF Transformers eager
attention, BF16 weights, Apple MPS (`mife.introspect.load_torch_model`). Generations and verifier outcomes are
reused, not touched.

For every layer l, head h and rule slot j (prompt order):
    A1max[l, h, j] = max_{t in span_j} attn_l[h, final_token, t]
The attention probabilities returned by the eager kernel are BF16, so the maximum is itself a BF16 value: it is
stored losslessly as raw BF16 bits (uint16; decode with `mife.introspect.bits_to_float32`). Absent slots: 0 bits
with mask False.

Layout: data/probing_features_v2/<model>/maxattn/
    max_NNNNN.npy (n, L, heads, 5) uint16, mask_NNNNN.npy (n, 5) bool, index_NNNNN.parquet (commit marker, last)
    maxattn_meta.json (version, backend, timings, storage).
Resumable and idempotent: committed shards are skipped; hashes are checked against the current manifest/tokenization.
TEST prompts are refused (`data.assert_allowed(..., "train")`).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ...config import DATA_DIR
from ... import integrity
from ...introspect import bits_to_float32, load_torch_model, torch_device, _sync
from ...storage import write_json, write_npy, write_parquet
from .. import data
from .versioning import FEAT_V2, VERSION, check_meta, write_versioned_json

K_MAX = 5
KIND = "A1max"


def out_dir(model_key: str):
    d = FEAT_V2 / model_key / "maxattn"
    d.mkdir(parents=True, exist_ok=True)
    return d


def span_stats(att, spans):
    """att: torch (L, heads, seq) attention row of the final token (model dtype). Returns per-slot (max, sum_fp32)."""
    import torch
    L, H, _ = att.shape
    mx = torch.zeros((L, H, K_MAX), dtype=att.dtype, device=att.device)
    sm = torch.zeros((L, H, K_MAX), dtype=torch.float32, device=att.device)
    for j, (a, b) in enumerate(spans):
        mx[:, :, j] = att[:, :, a:b].max(-1).values
        sm[:, :, j] = att[:, :, a:b].float().sum(-1)  # identical arithmetic to the V1 sum (introspect.run)
    return mx, sm


def _bits(t) -> np.ndarray:
    import torch
    return t.to(torch.bfloat16).contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def _tok(model_key: str) -> pd.DataFrame:
    integrity.tokenized_hashes(model_key)
    return pd.read_parquet(DATA_DIR / "tokenized" / f"{model_key}.parquet").set_index("prompt_id")


def committed(model_key: str) -> tuple[set[str], int]:
    out = out_dir(model_key)
    done, nxt, frames = set(), 0, []
    for f in sorted(out.glob("index_[0-9]*.parquet")):
        n = int(f.stem.split("_")[1])
        if (out / f"max_{n:05d}.npy").exists() and (out / f"mask_{n:05d}.npy").exists():
            frames.append(pd.read_parquet(f))
            nxt = max(nxt, n + 1)
    if frames:
        df = pd.concat(frames, ignore_index=True)
        if set(df["version"]) != {VERSION}:
            raise RuntimeError(f"{out}: shards from another version {set(df['version'])}")
        integrity.check_prompt(df, str(out))
        integrity.check_rendered(model_key, df, str(out))
        done = set(df.prompt_id)
    return done, nxt


def forward_rows(model, ids_tensor):
    """Final-token attention rows (L, heads, seq) in the model's dtype."""
    import torch
    with torch.no_grad():
        o = model(input_ids=ids_tensor, output_attentions=True, use_cache=False)
    last = ids_tensor.shape[1] - 1
    return torch.stack([a[0, :, last, :] for a in o.attentions])


def run(model_key: str, shard_size: int = 1024, max_prompts: int | None = None, ids: list[str] | None = None) -> dict:
    import torch
    device = torch_device()
    if device != "mps":
        raise SystemExit(f"V2 max-attention must use the original Apple MPS backend (got {device})")
    ids = ids or data.prompt_ids("train")
    data.assert_allowed(ids, "train")  # TRAIN prompts only
    tok = _tok(model_key)
    out = out_dir(model_key)
    done, shard = committed(model_key)
    todo = [p for p in ids if p not in done]
    if max_prompts is not None:
        todo = todo[:max_prompts]
    t0 = time.time()
    model = load_torch_model(model_key, device) if todo else None
    load_s = time.time() - t0
    per = []
    if model is not None:
        L, Hh = model.config.num_hidden_layers, model.config.num_attention_heads
    for s0 in range(0, len(todo), shard_size):
        pids = todo[s0:s0 + shard_size]
        MX = np.zeros((len(pids), L, Hh, K_MAX), dtype=np.uint16)
        MK = np.zeros((len(pids), K_MAX), dtype=bool)
        rows = []
        for i, pid in enumerate(pids):
            r = tok.loc[pid]
            t1 = time.time()
            att = forward_rows(model, torch.tensor([list(r.input_ids)], device=device))
            if att.dtype != torch.bfloat16:
                raise RuntimeError(f"attention dtype {att.dtype}; expected bfloat16 (frozen backend)")
            spans = [tuple(int(v) for v in x) for x in r.rule_token_spans]
            mx, _ = span_stats(att, spans)
            MX[i] = _bits(mx)
            MK[i, :len(spans)] = True
            _sync(device)
            per.append(time.time() - t1)
            rows.append({"prompt_id": pid, "model_key": model_key, "shard": shard, "row": i,
                         "prompt_sha256": r.prompt_sha256, "rendered_sha256": r.rendered_sha256,
                         "n_input_tokens": int(r.n_input_tokens), "rule_token_spans": [list(x) for x in spans],
                         "version": VERSION})
            del att
        write_npy(MX, out / f"max_{shard:05d}.npy")
        write_npy(MK, out / f"mask_{shard:05d}.npy")
        write_parquet(pd.DataFrame(rows), out / f"index_{shard:05d}.parquet")  # commit marker
        print(f"[maxattn {model_key}] shard {shard}: {len(pids)} prompts, {len(done) + s0 + len(pids)}/{len(ids)}, "
              f"{np.mean(per[-len(pids):]):.3f} s/prompt", flush=True)
        shard += 1
        torch.mps.empty_cache()
    all_done, _ = committed(model_key)
    mp = out / "maxattn_meta.json"
    prev = json.loads(mp.read_text()) if mp.exists() else {}
    sessions = prev.get("sessions", [])
    if per:
        sessions.append({"started_unix": t0, "model_load_s": load_s, "wall_s": time.time() - t0, "n_prompts": len(per),
                         "per_prompt_s_quantiles": {str(q): float(np.quantile(per, q)) for q in (0, .5, .9, 1)}})
    nbytes = sum(f.stat().st_size for f in out.glob("*_[0-9]*.*"))
    meta = {"kind": KIND, "model_key": model_key, "backend": "transformers+torch-mps", "device": device,
            "attn_implementation": "eager", "dtype": "bfloat16", "split": "train",
            "n_layers": prev.get("n_layers") or (L if model is not None else None),
            "n_heads": prev.get("n_heads") or (Hh if model is not None else None),
            "n_prompts": len(all_done), "n_train_prompts": len(data.prompt_ids("train")),
            "n_train_prompts_missing": len(set(data.prompt_ids("train")) - all_done),
            "storage_bytes": nbytes, "sessions": sessions,
            "status": "complete" if set(data.prompt_ids("train")) <= all_done else "partial",
            "torch_version": torch.__version__, "transformers_version": __import__("transformers").__version__,
            "frozen_features_meta": {k: data.feature_meta(model_key)[k] for k in ("torch_version", "transformers_version", "backend", "dtype")},
            "definition": "A1max[l,h,j] = max_{t in rule span j} attention_l[h, final prompt token, t]; raw BF16 bits",
            "split_independent": True,
            "note": "per-prompt values do not depend on the split; n_prompts counts every committed prompt, including "
                    "prompts extracted while they were TRAIN under an earlier split (never read while they are TEST)"}
    for k in ("torch_version", "transformers_version"):
        if meta[k] != meta["frozen_features_meta"][k]:
            raise SystemExit(f"{k} {meta[k]} differs from the frozen feature run ({meta['frozen_features_meta'][k]})")
    write_versioned_json(meta, mp)
    return meta


# --------------------------------------------------------------------------------------------- loading
def load(model_key: str, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(A1max float32 (n, L, heads, 5), mask bool (n, 5)) in the order of `ids` (TRAIN only)."""
    data.assert_allowed(ids, "train")
    out = FEAT_V2 / model_key / "maxattn"
    meta = json.loads((out / "maxattn_meta.json").read_text())
    if meta.get("version") != VERSION or meta.get("kind") != KIND:
        from .versioning import VersionMismatch
        raise VersionMismatch(f"{out}: not a {VERSION} {KIND} extraction")
    idx = pd.concat([pd.read_parquet(f) for f in sorted(out.glob("index_[0-9]*.parquet"))], ignore_index=True)
    integrity.check_prompt(idx, str(out))
    integrity.check_rendered(model_key, idx, str(out))
    idx = idx.set_index("prompt_id").loc[ids, ["shard", "row"]]
    L, Hh = meta["n_layers"], meta["n_heads"]
    A = np.empty((len(ids), L, Hh, K_MAX), dtype=np.float32)
    K = np.zeros((len(ids), K_MAX), dtype=bool)
    pos = np.arange(len(ids))
    for shard, g in idx.assign(_i=pos).groupby("shard"):
        mx = np.load(out / f"max_{shard:05d}.npy", mmap_mode="r")
        mk = np.load(out / f"mask_{shard:05d}.npy")
        A[g["_i"].values] = bits_to_float32(np.asarray(mx[g["row"].values]))
        K[g["_i"].values] = mk[g["row"].values]
    return A, K
