"""K10 pre-generation features on Betty CUDA (pilots/high_k/PRESPEC.md): one forward pass per prompt and model.

    python pilots/high_k/code/features.py run gemma2b --chunk 0     # from ~/mife/repo, PYTHONPATH=src
    python pilots/high_k/code/features.py check gemma2b
Same definitions as the main pipeline (mife.introspect / probing.v2.maxattn), HF Transformers eager attention, BF16,
batch size 1 (no padding), but CUDA instead of Apple MPS (DEVIATION, in the PRESPEC alignment table) and only:
  * hidden: final-prompt-token states at 5 rows = the original's tuned relative depths 0.25/0.375/0.5/0.75 of
    num_hidden_layers (rows of `hidden_states`, 0 = embeddings) plus the pre-final-norm last-layer output
    (row L+1), as raw BF16 bits (uint16). Llama 8/12/16/24/33; Gemma-2B 7/10/13/20/27; Gemma-9B 11/16/21/32/43.
  * A1max[l, h, j] = max over rule span j of the final token's attention, every layer x head, up to 10 slots, raw
    BF16 bits; A1sum the same as fp16 sums (for RCAP-style summaries); mask (n, 10).
Shards of 1,000 prompts: hidden_*.npy, attn_*.npz, then index_*.parquet (commit marker, kept on Betty after the
heavy files are copied back and removed, so resumption never recomputes a fetched shard). Writes only
pilots/high_k/data/features/<model>/. TEST prompts are included (features are pre-generation and read no labels;
they are not opened for analysis until the TEST prespec is frozen).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from generate import INP, PILOT, load_llm as _unused, model_path, space_guard, spec  # noqa: F401

DEPTHS = (0.25, 0.375, 0.5, 0.75)
K_MAX = 10
SHARD = 1000


def rows_for(n_layers: int) -> list[int]:
    return [int(d * n_layers + 0.5) for d in DEPTHS] + [n_layers + 1]


def bits(t) -> np.ndarray:
    import torch
    return t.to(torch.bfloat16).contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def committed(out) -> set[str]:
    done = set()
    for f in out.glob("index_*.parquet"):
        done |= set(pd.read_parquet(f, columns=["prompt_id"]).prompt_id)
    return done


def run(model_key: str, chunk: int, max_prompts: int | None = None) -> dict:
    import torch
    from transformers import AutoModelForCausalLM
    assert torch.cuda.is_available(), "CUDA required"
    tok = pd.read_parquet(INP / f"tokenized_{model_key}.parquet").set_index("prompt_id")
    work = pd.read_parquet(INP / f"worklist_{model_key}.parquet", columns=["prompt_id", "chunk"])
    pids = list(dict.fromkeys(work[work.chunk == chunk].prompt_id))
    out = PILOT / "data" / "features" / model_key
    out.mkdir(parents=True, exist_ok=True)
    done = committed(out)
    todo = [p for p in pids if p not in done][:max_prompts]
    if not todo:
        return {"model_key": model_key, "chunk": chunk, "todo": 0}
    space_guard()
    t0 = time.time()
    p = str(model_path(model_key))
    model = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    load_s = time.time() - t0
    cfg = model.config
    L, Hh, D = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size
    rows = rows_for(L)
    cap = {}
    model.model.norm.register_forward_hook(lambda m, i, o: cap.update(pre=i[0]))
    tag = f"c{chunk:02d}_j{os.environ.get('SLURM_JOB_ID', 'local')}"
    per = []
    for s0 in range(0, len(todo), SHARD):
        space_guard()
        sp = todo[s0:s0 + SHARD]
        H = np.zeros((len(sp), len(rows), D), dtype=np.uint16)
        MX = np.zeros((len(sp), L, Hh, K_MAX), dtype=np.uint16)
        SM = np.zeros((len(sp), L, Hh, K_MAX), dtype=np.float16)
        MK = np.zeros((len(sp), K_MAX), dtype=bool)
        idx = []
        for i, pid in enumerate(sp):
            r = tok.loc[pid]
            ids = torch.tensor([list(r.input_ids)], device="cuda")
            t1 = time.time()
            with torch.no_grad():
                o = model(input_ids=ids, output_hidden_states=True, output_attentions=True, use_cache=False)
            last = ids.shape[1] - 1
            hs = list(o.hidden_states) + [cap["pre"]]
            assert len(o.hidden_states) == L + 1 and len(o.attentions) == L
            H[i] = bits(torch.stack([hs[k][0, last] for k in rows]))
            att = torch.stack([a[0, :, last, :] for a in o.attentions])  # (L, heads, seq), bf16
            assert att.dtype == torch.bfloat16
            rs = att.float().sum(-1)
            assert torch.allclose(rs, torch.ones_like(rs), atol=2e-2), float((rs - 1).abs().max())
            spans = [tuple(int(v) for v in x) for x in r.rule_token_spans]
            for j, (a, b) in enumerate(spans):
                MX[i, :, :, j] = bits(att[:, :, a:b].max(-1).values)
                SM[i, :, :, j] = att[:, :, a:b].float().sum(-1).cpu().numpy().astype(np.float16)
                MK[i, j] = True
            torch.cuda.synchronize()
            per.append(time.time() - t1)
            idx.append({"prompt_id": pid, "row": i, "rendered_sha256": r.rendered_sha256,
                        "n_input_tokens": int(r.n_input_tokens), "rule_token_spans": [list(x) for x in spans]})
            del o, hs, att
        n = f"{tag}_{s0 // SHARD:04d}"
        np.save(out / f"hidden_{n}.npy", H)
        np.savez(out / f"attn_{n}.npz", max_bits=MX, sum_fp16=SM, mask=MK)
        pd.DataFrame(idx).assign(shard=n, model_key=model_key).to_parquet(out / f"index_{n}.parquet")  # commit last
        print(f"[features {model_key} c{chunk:02d}] shard {n}: {len(sp)} prompts, {np.mean(per[-len(sp):]):.4f} s/prompt", flush=True)
    meta = {"model_key": model_key, "chunk": chunk, "tag": tag, "backend": "transformers+torch-cuda", "dtype": "bfloat16",
            "attn_implementation": "eager", "n_layers": L, "n_heads": Hh, "hidden_size": D, "hidden_rows": rows,
            "hidden_rows_meaning": f"rows of hidden_states (0=embeddings); {L + 1}=pre-final-norm last-layer output",
            "prompts": len(todo), "model_load_s": load_s, "per_prompt_s_median": float(np.median(per)),
            "torch": torch.__version__, "transformers": __import__("transformers").__version__,
            "gpu": torch.cuda.get_device_name(0), "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
    (out / f"meta_{tag}_{int(time.time())}.json").write_text(json.dumps(meta, indent=1))
    return meta


def check(model_key: str) -> dict:
    out = PILOT / "data" / "features" / model_key
    work = pd.read_parquet(INP / f"worklist_{model_key}.parquet", columns=["prompt_id"])
    want = set(work.prompt_id)
    got = pd.concat([pd.read_parquet(f, columns=["prompt_id"]) for f in out.glob("index_*.parquet")]) if list(out.glob("index_*.parquet")) else pd.DataFrame(columns=["prompt_id"])
    r = {"model_key": model_key, "expected": len(want), "committed": got.prompt_id.nunique(),
         "duplicates": int(got.prompt_id.duplicated().sum()), "missing": len(want - set(got.prompt_id))}
    print(json.dumps(r))
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "check"])
    ap.add_argument("model", choices=["gemma2b", "llama", "gemma9b"])
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--max-prompts", type=int)
    a = ap.parse_args()
    print(json.dumps(run(a.model, a.chunk, a.max_prompts) if a.cmd == "run" else check(a.model), default=str))
