"""Resumable, idempotent batched generation on Apple Silicon with MLX (BF16, no quantization).

Work unit = one prompt x n samples (design.n_responses_per_prompt = 20; Amendment A2). Each batch is written atomically as an immutable shard
(parquet). On restart, prompts already holding all samples in existing shards are skipped.
Sampling RNG: mx.random.seed(segment_seed) once per generation call (continuous batching);
the seed, insertion order and shard composition are logged per response.
Bitwise reproducibility of individual samples is NOT guaranteed (batch composition, GPU kernels);
the stored generations are the experimental record.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, derive_seed, load_config
from . import integrity
from .storage import sha256_text, write_json, write_parquet

GEN_SCHEMA = "generations/v1"


def model_path(model_key: str) -> Path:
    """Pinned local snapshot dir; verifies every safetensors shard named in the index exists."""
    from huggingface_hub.constants import HF_HUB_CACHE

    m = load_config()["models"][model_key]
    p = Path(HF_HUB_CACHE) / f"models--{m['hf_id'].replace('/', '--')}" / "snapshots" / m["revision"]
    index = json.loads((p / "model.safetensors.index.json").read_text())
    missing = [f for f in set(index["weight_map"].values()) if not (p / f).exists()]
    if missing:
        raise FileNotFoundError(f"{model_key}: missing weight shards {missing}")
    return p


# How each backend turns (temperature, top_k, top_p, min_p) into a sampling distribution.
# mlx_lm 0.32.x applies top_p, then min_p, then top_k to the UNTEMPERED log-probabilities and only then
# samples with temperature. vLLM (like HF transformers) applies temperature first, then top_k, then top_p.
# The two definitions differ whenever top_p < 1 and temperature != 1 (or top_k and top_p are combined).
SAMPLER_SEMANTICS = {
    "mlx_lm": "filters(top_p->min_p->top_k) on untempered logprobs, then categorical(logprobs / T)",
    "vllm": "logits / T, then top_k, then top_p (on tempered, renormalised probs), then min_p",
}


def gen_settings(model_key: str, backend: str = "mlx_lm") -> dict:
    cfg = load_config()
    m = cfg["models"][model_key]
    st = {"model_key": model_key, "hf_id": m["hf_id"], "revision": m["revision"],
          "precision": cfg["precision"], "quantization": None, "backend": backend,
          "chat_template_kwargs": m["chat_template_kwargs"], **m["decoding"],
          "n_samples": cfg["design"]["n_responses_per_prompt"]}
    if backend != "mlx_lm":  # MLX settings dict (and hash) kept exactly as used for the Mac pilot
        st["sampler_semantics"] = SAMPLER_SEMANTICS[backend]
    return st


def settings_hash(model_key: str, backend: str = "mlx_lm") -> str:
    return sha256_text(json.dumps(gen_settings(model_key, backend), sort_keys=True))[:16]


def eos_token_ids(model_key: str) -> list[int]:
    """Union of eos ids in config.json, generation_config.json and tokenizer_config (what mlx_lm stops on)."""
    p = model_path(model_key)
    ids: set[int] = set()
    for f in ("config.json", "generation_config.json"):
        if (p / f).exists():
            v = json.loads((p / f).read_text()).get("eos_token_id")
            ids |= set(v if isinstance(v, list) else [v] if v is not None else [])
    from .chat import tokenizer
    if tokenizer(model_key).eos_token_id is not None:
        ids.add(int(tokenizer(model_key).eos_token_id))
    return sorted(int(i) for i in ids)


def load_model(model_key: str):
    import mlx.core as mx
    from mlx_lm import load

    model, tok = load(str(model_path(model_key)))
    dtypes = {str(v.dtype) for _, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(model.parameters())}
    if any("int" in d or "uint" in d for d in dtypes) or dtypes != {"mlx.core.bfloat16"}:
        raise RuntimeError(f"unexpected MLX weight dtypes {dtypes}; refusing (no silent quantization)")
    eos = sorted(tok.eos_token_ids)
    return model, tok, {"weight_dtypes": sorted(dtypes), "eos_token_ids": eos, "mlx_version": mx.__version__}


def done_prompts(out_dir: Path, n_samples: int, model_key: str) -> set[str]:
    """Prompts with all samples already stored. Refuses (StaleDataError) if any stored shard belongs to
    another manifest/tokenization, so resuming can never silently reuse outputs of a different prompt."""
    shards = sorted(out_dir.glob("shard_*.parquet"))
    if not shards:
        return set()
    df = pd.concat([pd.read_parquet(s, columns=["prompt_id", "sample_index", "rendered_sha256"]) for s in shards])
    integrity.check_rendered(model_key, df, str(out_dir))
    c = df.drop_duplicates().groupby("prompt_id").size()
    return set(c[c >= n_samples].index)


def run(model_key: str, prompt_ids: list[str], run_name: str, prompts_per_batch: int = 4,
        completion_batch_size: int | None = None, max_prompts: int | None = None,
        cache_limit_gb: float = 4.0, recycle_every_prompts: int | None = None) -> dict:
    import mlx.core as mx
    import mlx_lm
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler

    cfg = load_config()
    st = gen_settings(model_key)
    n_s = st["n_samples"]
    out_dir = DATA_DIR / "generations" / model_key / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_df = pd.read_parquet(DATA_DIR / "tokenized" / f"{model_key}.parquet",
                             columns=["prompt_id", "input_ids", "n_input_tokens", "rendered_sha256"]).set_index("prompt_id")
    integrity.tokenized_hashes(model_key)  # tokenization must match the current manifest
    done = done_prompts(out_dir, n_s, model_key)
    todo = [p for p in prompt_ids if p not in done]
    if max_prompts is not None:
        todo = todo[:max_prompts]

    t_load = time.time()
    model, tok, info = load_model(model_key)
    load_s = time.time() - t_load
    mx.reset_peak_memory()
    # Memory management only (no effect on sampling): cap the MLX buffer cache and clear it per shard,
    # otherwise freed KV buffers accumulate until the macOS GPU working-set limit is hit.
    mx.set_cache_limit(int(cache_limit_gb * 1e9))
    dev = mx.device_info()
    info["max_recommended_working_set_gb"] = dev.get("max_recommended_working_set_size", 0) / 1e9
    sampler = make_sampler(temp=st["temperature"], top_p=st["top_p"], min_p=st["min_p"], top_k=st["top_k"])
    stop = [[t] for t in info["eos_token_ids"]]
    bsz = completion_batch_size or 64
    prompts_per_shard = prompts_per_batch

    # Continuous batching: one generator, a rolling queue of prompts (20 sequences each), at most
    # `bsz` active sequences. A shard is written atomically whenever `prompts_per_shard` prompts
    # have all their samples finished.
    # unique per call even if an earlier call crashed before writing its summary
    segment_id = f"{run_name}-seg{time.strftime('%Y%m%dT%H%M%S')}-{len(list(out_dir.glob('shard_*.parquet'))):05d}"
    seed = derive_seed(cfg["seeds"]["generation"], model_key, segment_id)
    mx.random.seed(seed % (1 << 32))
    def new_gen():
        return BatchGenerator(model, max_tokens=st["max_new_tokens"], stop_tokens=stop, sampler=sampler,
                              completion_batch_size=bsz, prefill_batch_size=min(bsz, 16))

    gen = new_gen()
    inserted_in_gen = 0  # prompts inserted into the current generator (for optional recycling)
    queue = list(todo)
    uid_meta, toks, fin, remaining = {}, {}, {}, {}
    finished_prompts: list[str] = []
    batch_log = []
    t_run = time.time()
    total_out = 0
    insert_order = 0

    def feed():
        nonlocal insert_order, inserted_in_gen
        while queue and len(uid_meta) - len(fin) < 2 * bsz:
            if recycle_every_prompts and inserted_in_gen >= recycle_every_prompts:
                return  # let this generator drain; a fresh one is created below
            inserted_in_gen += 1
            pid = queue.pop(0)
            ids = [int(x) for x in tok_df.loc[pid, "input_ids"]]
            uids = gen.insert([ids] * n_s, [st["max_new_tokens"]] * n_s)
            remaining[pid] = n_s
            for sidx, u in enumerate(uids):
                uid_meta[u] = (pid, sidx, insert_order)
                toks[u] = []
                insert_order += 1

    def flush(pids, t_shard):
        nonlocal total_out
        shard_idx = len(list(out_dir.glob("shard_*.parquet")))
        rows = []
        for u, (pid, sidx, order) in uid_meta.items():
            if pid not in pids:
                continue
            ids = toks[u]
            rows.append({
                "response_id": f"{model_key}:{pid}:s{sidx:02d}", "prompt_id": pid, "model_key": model_key,
                "sample_index": sidx, "response_text": tok.decode(ids), "output_token_ids": ids,
                "n_output_tokens": len(ids), "finish_reason": fin[u], "hit_max_tokens": fin[u] == "length",
                "n_input_tokens": int(tok_df.loc[pid, "n_input_tokens"]),
                "rendered_sha256": tok_df.loc[pid, "rendered_sha256"],
                "batch_id": segment_id, "batch_seed": seed, "insert_order": order,
                "completion_batch_size": bsz, "source_run": run_name,
                "settings_hash": settings_hash(model_key), "schema": GEN_SCHEMA, "created_unix": time.time(),
            })
        n_out = sum(r["n_output_tokens"] for r in rows)
        total_out += n_out
        write_parquet(pd.DataFrame(rows), out_dir / f"shard_{shard_idx:05d}.parquet")
        mx.clear_cache()
        for u in [u for u, m in uid_meta.items() if m[0] in pids]:
            del uid_meta[u], toks[u], fin[u]
        el = time.time() - t_run
        batch_log.append({"shard": shard_idx, "prompts": list(pids), "output_tokens": n_out, "elapsed_s": el,
                          "peak_mem_gb": mx.get_peak_memory() / 1e9})
        print(f"[{model_key}] shard {shard_idx}: {len(pids)} prompts, {n_out} tok; cumulative "
              f"{total_out} tok in {el:.0f}s = {total_out / el:.0f} tok/s, peak {mx.get_peak_memory() / 1e9:.1f} GB",
              flush=True)

    feed()
    while True:
        responses = gen.next_generated()
        if not responses:
            if not queue:
                break
            if recycle_every_prompts and inserted_in_gen >= recycle_every_prompts:
                gen.close()
                mx.clear_cache()
                gen = new_gen()
                inserted_in_gen = 0
            feed()
            continue
        for r in responses:
            if r.finish_reason != "stop":
                toks[r.uid].append(r.token)
            if r.finish_reason is not None:
                fin[r.uid] = r.finish_reason
                pid = uid_meta[r.uid][0]
                remaining[pid] -= 1
                if remaining[pid] == 0:
                    finished_prompts.append(pid)
        if len(finished_prompts) >= prompts_per_shard:
            flush(set(finished_prompts), time.time())
            finished_prompts = []
        feed()
    if finished_prompts:
        flush(set(finished_prompts), time.time())
    gen.close()

    run_s = time.time() - t_run
    summary = {"model_key": model_key, "run_name": run_name, "settings": st, "settings_hash": settings_hash(model_key),
               "model_info": info, "mlx_lm_version": mlx_lm.__version__, "platform": platform.platform(),
               "prompts_per_batch": prompts_per_batch, "completion_batch_size": bsz, "segment_id": segment_id,
               "segment_seed": seed, "cache_limit_gb": cache_limit_gb, "recycle_every_prompts": recycle_every_prompts, "batching": "continuous (mlx_lm BatchGenerator, rolling prompt queue)",
               "n_prompts_generated_this_call": len(todo), "n_prompts_skipped_done": len(done),
               "model_load_s": load_s, "generation_wall_s": run_s, "output_tokens": total_out,
               "aggregate_output_tok_per_s": total_out / run_s if run_s else None,
               "peak_mlx_memory_gb": mx.get_peak_memory() / 1e9, "batches": batch_log}
    write_json(summary, out_dir / f"run_summary_{int(time.time())}.json")
    return summary


PILOT_IMPORT = "pilot_import.json"


def import_pilot(model_key: str, src_dir: Path | None = None, dst_dir: Path | None = None,
                 settings: dict | None = None, n_keep: int | None = None) -> dict:
    """Copy each pilot prompt's sample_index 0..n_keep-1 into the main run.

    n_keep defaults to the main design value (20 under Amendment A2, i.e. every pilot sample).
    If n_keep is smaller than the pilot count, indices are assigned at insertion, before any
    output exists, so keeping the lowest indices is outcome-independent.
    Requires that every decoding setting except n_samples matches the pilot run summaries.
    Idempotent: does nothing if the import marker already exists in dst_dir.
    """
    src_dir = src_dir or DATA_DIR / "generations" / model_key / "pilot"
    dst_dir = dst_dir or DATA_DIR / "generations" / model_key / "main"
    marker = dst_dir / PILOT_IMPORT
    if marker.exists():
        return {**json.loads(marker.read_text()), "status": "already imported"}
    strip = lambda d: {k: v for k, v in d.items() if k != "n_samples"}
    sums = [json.loads(p.read_text()) for p in sorted(src_dir.glob("run_summary_*.json"))]
    if not sums:
        raise RuntimeError(f"{src_dir}: no run summaries; cannot confirm pilot settings")
    # compare with current settings for the backend the pilot used (MLX pilot vs MLX, vLLM pilot vs vLLM)
    settings = settings or gen_settings(model_key, sums[-1]["settings"].get("backend", "mlx_lm"))
    n_keep = n_keep or settings["n_samples"]
    # never mix backends inside one main run
    for f in dst_dir.glob("run_summary_*.json"):
        b = json.loads(f.read_text()).get("settings", {}).get("backend")
        if b and b != settings["backend"]:
            raise RuntimeError(f"{dst_dir} already holds {b} generations; refusing to import a {settings['backend']} pilot")
    bad = [s["settings"] for s in sums if strip(s["settings"]) != strip(settings)]
    if bad:
        raise RuntimeError(f"pilot decoding settings differ from current settings; regenerate instead: {bad[0]}")
    shards = sorted(src_dir.glob("shard_*.parquet"))
    df = pd.concat([pd.read_parquet(s).assign(source_shard=s.name) for s in shards], ignore_index=True)
    assert df["response_id"].is_unique
    integrity.check_rendered(model_key, df, str(src_dir))
    have = df.groupby("prompt_id")["sample_index"].apply(set)
    short = [p for p, s in have.items() if not set(range(n_keep)) <= s]
    if short:
        raise RuntimeError(f"{len(short)} pilot prompts lack sample indices 0..{n_keep - 1}")
    kept = df[df["sample_index"] < n_keep].copy()
    kept["source_run"] = "pilot"
    dst_dir.mkdir(parents=True, exist_ok=True)
    start = len(list(dst_dir.glob("shard_*.parquet")))
    written = []
    for i, (shard, part) in enumerate(kept.groupby("source_shard", sort=True)):
        name = f"shard_{start + i:05d}.parquet"
        write_parquet(part.drop(columns="source_shard").reset_index(drop=True), dst_dir / name)
        written.append({"dst": name, "src": shard, "rows": len(part)})
    info = {"model_key": model_key, "src": str(src_dir), "n_keep": n_keep,
            "rule": f"sample_index < {n_keep}", "n_prompts": int(kept["prompt_id"].nunique()),
            "n_rows_kept": len(kept), "n_rows_excluded": int(len(df) - len(kept)),
            "pilot_settings_hash_in_rows": sorted(kept["settings_hash"].unique().tolist()),
            "main_settings": settings, "shards": written, "created_unix": time.time()}
    write_json(info, marker)
    return info


def load_generations(model_key: str, run_name: str, columns: list[str] | None = None) -> pd.DataFrame:
    out_dir = DATA_DIR / "generations" / model_key / run_name
    cols = None if columns is None else sorted(set(columns) | {"response_id", "prompt_id", "rendered_sha256"})
    df = pd.concat([pd.read_parquet(s, columns=cols) for s in sorted(out_dir.glob("shard_*.parquet"))],
                   ignore_index=True)
    assert df["response_id"].is_unique, "duplicate responses across shards"
    integrity.check_rendered(model_key, df, str(out_dir))
    return df
