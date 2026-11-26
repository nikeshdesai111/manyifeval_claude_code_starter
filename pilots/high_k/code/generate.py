"""Generate the K10 worklist for one model and chunk on Betty (vLLM, BF16, per-sample seeds).

    python pilots/high_k/code/generate.py generate gemma2b --chunk 0   # from ~/mife/repo, PYTHONPATH=src
    python pilots/high_k/code/generate.py check gemma2b                # completeness / seed check
    python pilots/high_k/code/generate.py weights gemma2b              # print and sha256 the snapshot in use
Same backend, sampling semantics and record schema as the S3 expansion runner (src/mife/expansion_s3), with the
models and decoding in code/models.yaml as recorded in data/inputs/build_summary.json. Reads pilots/high_k/data/inputs/
and writes only pilots/high_k/data/generations/<model>/. Resumable per (prompt, sample).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd

from mife import generate_vllm as GV
from mife.config import ROOT, derive_seed, load_config
from mife.generate_mlx import GEN_SCHEMA, SAMPLER_SEMANTICS

PILOT = ROOT / "pilots" / "high_k"
INP = PILOT / "data" / "inputs"
RUN_NAME = "pilot_high_k_20261126"
VERSION = "pilot-high-k-2026-11-26"


def summary() -> dict:
    return json.loads((INP / "build_summary.json").read_text())


def spec(model_key: str) -> dict:
    return summary()["models"][model_key]


def gen_settings(model_key: str) -> dict:
    s, cfg = spec(model_key), load_config()
    return {"model_key": model_key, "hf_id": s["hf_id"], "revision": s["revision"], "precision": cfg["precision"],
            "quantization": None, "backend": "vllm", "chat_template_kwargs": s["chat_template_kwargs"],
            **s["decoding"], "n_samples": s["n_samples"], "sampler_semantics": SAMPLER_SEMANTICS["vllm"]}


def settings_hash(model_key: str) -> str:
    return hashlib.sha256(json.dumps(gen_settings(model_key), sort_keys=True).encode()).hexdigest()[:16]


def model_path(model_key: str) -> Path:
    """Pinned snapshot in HF_HUB_CACHE; every safetensors file named in the index (or the single file) must exist."""
    from huggingface_hub.constants import HF_HUB_CACHE
    s = spec(model_key)
    p = Path(HF_HUB_CACHE) / f"models--{s['hf_id'].replace('/', '--')}" / "snapshots" / s["revision"]
    idx = p / "model.safetensors.index.json"
    files = set(json.loads(idx.read_text())["weight_map"].values()) if idx.exists() else {"model.safetensors"}
    missing = [f for f in files if not (p / f).exists()]
    if missing:
        raise FileNotFoundError(f"{model_key}: missing weight files {missing} in {p}")
    return p


def weights(model_key: str) -> dict:
    p = model_path(model_key)
    h = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.glob("*.safetensors"))}
    r = {"model_key": model_key, "path": str(p), "safetensors_sha256": h}
    print(json.dumps(r))
    return r


def eos_token_ids(model_key: str) -> list[int]:
    from transformers import AutoTokenizer
    p = model_path(model_key)
    ids: set[int] = set()
    for f in ("config.json", "generation_config.json"):
        if (p / f).exists():
            v = json.loads((p / f).read_text()).get("eos_token_id")
            ids |= set(v if isinstance(v, list) else [v] if v is not None else [])
    t = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
    if t.eos_token_id is not None:
        ids.add(int(t.eos_token_id))
    return sorted(int(i) for i in ids)


def load(model_key: str):
    s = spec(model_key)
    for f, key in ((f"worklist_{model_key}.parquet", "worklist_sha256"), (f"tokenized_{model_key}.parquet", "tokenized_sha256")):
        got = hashlib.sha256((INP / f).read_bytes()).hexdigest()
        if got != s[key]:
            raise SystemExit(f"{f}: sha256 {got} != recorded {s[key]}")
    work = pd.read_parquet(INP / f"worklist_{model_key}.parquet")
    tok = pd.read_parquet(INP / f"tokenized_{model_key}.parquet").set_index("prompt_id")
    return work, tok, s


def done(out_dir: Path, shash: str) -> set[tuple[str, int]]:
    """Completed (prompt, sample) keys. Read from the small done_*.parquet ledgers written next to every shard, so a
    shard that was copied back to the Mac and then deleted on Betty is not regenerated."""
    have = set()
    for f in out_dir.glob("done_*.parquet"):
        d = pd.read_parquet(f)
        have |= {(p, int(i)) for p, i, h in zip(d.prompt_id, d.sample_index, d.settings_hash) if h == shash}
    return have


def space_guard(limit_gb: float | None = None, wait_s: int = 1200) -> None:
    """Betty home has only a few GB free: wait (up to wait_s) until data/generations + data/features hold less than HK_MAX_GB
    (default 4 GB), i.e. until the Mac has copied finished shards back and removed them. Exits non-zero otherwise."""
    limit = float(limit_gb or os.environ.get("HK_MAX_GB", 4)) * 1e9
    root = PILOT / "data"
    t0 = time.time()
    while True:
        used = sum(f.stat().st_size for k in ("generations", "features") for f in (root / k).rglob("*") if f.is_file())
        if used < limit:
            return
        if time.time() - t0 > wait_s:
            raise SystemExit(f"space guard: {used / 1e9:.1f} GB in {root} >= limit after {wait_s}s; resubmit later")
        print(f"space guard: {used / 1e9:.1f} GB in use, waiting for copy-back", flush=True)
        time.sleep(60)


def load_llm(model_key: str, max_model_len: int, gpu_mem: float, max_num_seqs: int, seed: int):
    from vllm import LLM
    p = str(model_path(model_key))
    llm = LLM(model=p, tokenizer=p, dtype="bfloat16", quantization=None, seed=seed, max_model_len=max_model_len,
              gpu_memory_utilization=gpu_mem, max_num_seqs=max_num_seqs, enable_prefix_caching=True,
              generation_config="vllm", trust_remote_code=False)
    mc = getattr(llm.llm_engine, "model_config", None)
    info = {"model_dir": p, "vllm_dtype": str(getattr(mc, "dtype", "unknown")),
            "vllm_quantization": getattr(mc, "quantization", "unknown"), "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_mem, "max_num_seqs": max_num_seqs, "enable_prefix_caching": True}
    if info["vllm_quantization"] not in (None, "unknown"):
        raise RuntimeError(f"vLLM reports quantization {info['vllm_quantization']}; refusing")
    if "bfloat16" not in info["vllm_dtype"] and info["vllm_dtype"] != "unknown":
        raise RuntimeError(f"vLLM dtype {info['vllm_dtype']} is not bfloat16; refusing")
    return llm, info


def generate(model_key: str, chunk: int, prompts_per_shard: int = 500, gpu_mem: float = 0.85,
             max_num_seqs: int = 512, max_records: int | None = None) -> dict:
    from vllm import SamplingParams

    work, tok, wmeta = load(model_key)
    max_model_len = summary()["max_model_len"]
    work = work[work.chunk == chunk]
    out_dir = PILOT / "data" / "generations" / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    st = gen_settings(model_key)
    shash = settings_hash(model_key)
    assert int(tok.loc[work.prompt_id.unique(), "n_input_tokens"].max()) + st["max_new_tokens"] <= max_model_len
    have = done(out_dir, shash)
    todo = work[[(p, int(s)) not in have for p, s in zip(work.prompt_id, work.sample_index)]]
    if max_records is not None:
        todo = todo.iloc[:max_records]
    stop = eos_token_ids(model_key)
    wsha = weights(model_key)["safetensors_sha256"]
    engine_seed = derive_seed(load_config()["seeds"]["generation"], "vllm", model_key, RUN_NAME, f"c{chunk:02d}") % (1 << 31)
    tag = f"c{chunk:02d}_j{os.environ.get('SLURM_JOB_ID', 'local')}"
    space_guard()
    t0 = time.time()
    llm, info = load_llm(model_key, max_model_len, gpu_mem, max_num_seqs, engine_seed)
    load_s = time.time() - t0
    skw = GV.sampling_kwargs(st)
    tokz = llm.get_tokenizer()
    pids = list(dict.fromkeys(todo.prompt_id))
    log, total, wall = [], 0, 0.0
    with GV.GpuMemSampler() as mem:
        for b0 in range(0, len(pids), prompts_per_shard):
            bp = set(pids[b0:b0 + prompts_per_shard])
            rows_in = todo[todo.prompt_id.isin(bp)]
            reqs, params, meta = [], [], []
            for p, s, seed in zip(rows_in.prompt_id, rows_in.sample_index, rows_in.seed):
                reqs.append({"prompt_token_ids": [int(x) for x in tok.loc[p, "input_ids"]]})
                params.append(SamplingParams(n=1, seed=int(seed), stop_token_ids=stop, skip_special_tokens=False,
                                             detokenize=False, **skw))
                meta.append((p, int(s), int(seed)))
            t1 = time.time()
            outs = llm.generate(reqs, params, use_tqdm=False)
            el = time.time() - t1
            wall += el
            rows = []
            for (p, s, seed), o in zip(meta, outs):
                c = o.outputs[0]
                ids = GV.strip_stop(list(c.token_ids), c.finish_reason, set(stop))
                rows.append({
                    "response_id": f"{model_key}:{p}:s{s:02d}", "prompt_id": p, "model_key": model_key,
                    "sample_index": s, "response_text": tokz.decode(ids), "output_token_ids": ids,
                    "n_output_tokens": len(ids), "finish_reason": c.finish_reason,
                    "hit_max_tokens": c.finish_reason == "length", "stop_reason": str(c.stop_reason),
                    "n_input_tokens": int(tok.loc[p, "n_input_tokens"]), "rendered_sha256": tok.loc[p, "rendered_sha256"],
                    "batch_id": f"{RUN_NAME}-vllm-{tag}", "batch_seed": seed, "insert_order": len(rows),
                    "completion_batch_size": max_num_seqs, "source_run": RUN_NAME, "backend": "vllm",
                    "settings_hash": shash, "schema": GEN_SCHEMA, "created_unix": time.time(),
                    "pilot_version": VERSION, "worklist_sha256": wmeta["worklist_sha256"]})
            n_out = sum(r["n_output_tokens"] for r in rows)
            total += n_out
            idx = len(list(out_dir.glob(f"shard_{tag}_[0-9]*.parquet")))
            pd.DataFrame(rows).to_parquet(out_dir / f"shard_{tag}_{idx:05d}.parquet")
            pd.DataFrame(rows)[["prompt_id", "sample_index", "settings_hash"]].to_parquet(
                out_dir / f"done_{tag}_{idx:05d}.parquet")  # ledger, written after the shard; kept on Betty
            log.append({"shard": f"shard_{tag}_{idx:05d}", "n_records": len(rows), "output_tokens": n_out,
                        "generate_s": el, "peak_gpu_memory_used_gb": mem.peak})
            print(f"[{model_key}/highk c{chunk:02d}] {len(rows)} records, {n_out} tok in {el:.0f}s "
                  f"({n_out / max(el, 1e-9):.0f} tok/s)", flush=True)
    out = {"model_key": model_key, "run_name": RUN_NAME, "version": VERSION, "chunk": chunk, "tag": tag,
           "backend": "vllm", "settings": st, "settings_hash": shash, "weights_sha256": wsha,
           "sampling_params": {**skw, "stop_token_ids": stop, "n": 1, "seeding": "per-sample"},
           "engine": info, "environment": GV.cuda_env(), "max_num_seqs": max_num_seqs,
           "worklist_sha256": wmeta["worklist_sha256"], "tokenized_sha256": wmeta["tokenized_sha256"],
           "chunk_records": len(work), "records_generated_this_call": int(sum(b["n_records"] for b in log)),
           "model_load_s": load_s, "generation_wall_s": wall, "output_tokens": total,
           "aggregate_output_tok_per_s": total / wall if wall else None, "batches": log}
    (out_dir / f"run_summary_{tag}_{int(time.time())}.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def check(model_key: str) -> dict:
    work, _, _ = load(model_key)
    out_dir = PILOT / "data" / "generations" / model_key
    d = pd.concat([pd.read_parquet(f, columns=["prompt_id", "sample_index", "batch_seed", "settings_hash",
                                               "n_output_tokens", "hit_max_tokens"])
                   for f in sorted(out_dir.glob("shard_*.parquet"))])
    key = d.prompt_id + "#" + d.sample_index.astype(str)
    exp = work.prompt_id + "#" + work.sample_index.astype(str)
    seeds = dict(zip(exp, work.seed))
    r = {"model_key": model_key, "expected": len(work), "records": len(d), "duplicates": int(key.duplicated().sum()),
         "missing": int((~exp.isin(set(key))).sum()), "outside_worklist": int((~key.isin(set(exp))).sum()),
         "seed_mismatch": int(sum(seeds.get(k) != s for k, s in zip(key, d.batch_seed))),
         "settings_hashes": sorted(d.settings_hash.unique()), "empty": int((d.n_output_tokens == 0).sum()),
         "hit_max_tokens": int(d.hit_max_tokens.sum())}
    print(json.dumps(r))
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "check", "weights"])
    ap.add_argument("model", choices=["gemma2b", "llama", "gemma9b"])
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--max-records", type=int)
    a = ap.parse_args()
    if a.cmd == "generate":
        generate(a.model, a.chunk, max_records=a.max_records)
    elif a.cmd == "check":
        check(a.model)
    else:
        weights(a.model)
