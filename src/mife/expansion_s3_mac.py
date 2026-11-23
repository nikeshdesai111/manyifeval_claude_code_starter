"""Local Mac (MLX) hedge for the S3 TEST expansion generations (2026-11-23).

Betty generates the expansion worklist with vLLM (`mife.expansion_s3 generate`, run `expansion_s3_20261123`). This
module generates the SAME worklist records on the Mac with MLX, as a hedge in case Betty stalls. It is deliberately
kept apart from the Betty output:

* Output run `expansion_s3_20261123_mac_mlx` (data/generations/<model>/expansion_s3_20261123_mac_mlx/). It is NOT in
  `expansion_s3.SOURCE_RUNS`, so nothing downstream picks these records up automatically. Using them in place of
  Betty's records is a methods decision (backend mixing) that needs the project owner's approval.
* Same worklist (checksums verified), same chunking (12 chunks of 1,155 prompts), same per-sample seeds, same
  generation schema plus `backend`, `sampler_semantics`, `rng` and `expansion_version` metadata columns.
* Sampling follows vLLM's definition, not mlx_lm's default: logits / T, then top-k (off for Llama), then top-p on the
  tempered and renormalised distribution, then min-p (0). mlx_lm's `make_sampler` applies top-p to the untempered
  distribution, which differs whenever T != 1 (see generate_mlx.SAMPLER_SEMANTICS).
* Every sequence has its own RNG stream: mx.random.key(worklist seed), split once per sampled token. The seed is the
  worklist's per-sample seed, so a record's seed is meaningful, but MLX and vLLM random streams differ, and BF16
  batched kernels depend on batch composition, so samples are not bitwise reproducible across backends or batches.
* Resumable per (prompt, sample): records already written with the right settings hash and seed are skipped.
  Chunks are processed in the order given (default 11 down to 2, the reverse of Betty's order).

It never reads response text from other runs, verification labels, features or any TEST outcome.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import time
from pathlib import Path

import pandas as pd

from . import expansion_s3 as E
from .config import DATA_DIR
from .storage import sha256_text, write_json, write_parquet

MAC_RUN = "expansion_s3_20261123_mac_mlx"
BACKEND = "mlx_lm"
SAMPLER_SEMANTICS = ("vLLM order on MLX: logits / T, then top_k (0 = off), then top_p on the tempered, renormalised "
                     "probs (keep tokens whose preceding cumulative mass < top_p), then min_p; categorical draw")
RNG = "per-sequence mx.random.key(worklist seed), mx.random.split once per sampled token"


def mac_settings(model_key: str) -> dict:
    from .generate_mlx import gen_settings

    st = gen_settings(model_key, "mlx_lm")
    st["backend"] = BACKEND
    st["sampler_semantics"] = SAMPLER_SEMANTICS
    st["rng"] = RNG
    return st


def mac_settings_hash(model_key: str) -> str:
    return sha256_text(json.dumps(mac_settings(model_key), sort_keys=True))[:16]


def vllm_hash(model_key: str) -> str:
    from .generate_mlx import settings_hash

    return settings_hash(model_key, "vllm")


def out_dir_for(model_key: str, run_name: str = MAC_RUN) -> Path:
    if run_name in E.SOURCE_RUNS or run_name == "main":
        raise SystemExit(f"refusing to write the Mac hedge into run {run_name}")
    return DATA_DIR / "generations" / model_key / run_name


# ---------------------------------------------------------------------------------------------- sampler
class SeededVllmSampler:
    """One per sequence. Called once per sampled token with logprobs of shape (1, V)."""

    def __init__(self, seed: int, temp: float, top_p: float, top_k: int, min_p: float):
        import mlx.core as mx

        self.key = mx.random.key(int(seed))
        self.temp, self.top_p, self.top_k, self.min_p = float(temp), float(top_p), int(top_k), float(min_p)

    def __call__(self, logprobs):
        import mlx.core as mx

        k, self.key = mx.random.split(self.key)
        x = logprobs.astype(mx.float32) / self.temp
        x = x - mx.logsumexp(x, axis=-1, keepdims=True)
        order = mx.argsort(-x, axis=-1)
        xs = mx.take_along_axis(x, order, axis=-1)
        if self.top_k > 0:
            ranks = mx.arange(xs.shape[-1])[None, :]
            xs = mx.where(ranks < self.top_k, xs, -mx.inf)
            xs = xs - mx.logsumexp(xs, axis=-1, keepdims=True)
        if 0.0 < self.top_p < 1.0:
            p = mx.exp(xs)
            before = mx.cumsum(p, axis=-1) - p
            xs = mx.where(before < self.top_p, xs, -mx.inf)
            xs = xs - mx.logsumexp(xs, axis=-1, keepdims=True)
        if self.min_p > 0.0:
            xs = mx.where(xs >= xs[:, :1] + mx.log(mx.array(self.min_p)), xs, -mx.inf)
        idx = mx.random.categorical(xs, axis=-1, key=k)
        return mx.take_along_axis(order, idx[:, None], axis=-1)[:, 0]


# ---------------------------------------------------------------------------------------------- helpers
def done_records(out_dir: Path, model_key: str, shash: str) -> set[tuple[str, int]]:
    done = set()
    for f in sorted(out_dir.glob("shard_*.parquet")):
        d = pd.read_parquet(f, columns=["prompt_id", "sample_index", "settings_hash", "batch_seed", "model_key"])
        for p, s, h, seed, mk in d.itertuples(index=False):
            if h == shash and mk == model_key and int(seed) == E.sample_seed(model_key, p, int(s)):
                done.add((p, int(s)))
    return done


def thermal_state() -> dict:
    """pmset thermal readout (no sudo). Empty dict when nothing is recorded."""
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return {}
    st = {}
    for key, pat in (("cpu_speed_limit", r"CPU_Speed_Limit\s*=\s*(\d+)"),
                     ("thermal_level", r"[Tt]hermal warning level\s*(?:set to|=)?\s*(\d+)"),
                     ("performance_level", r"[Pp]erformance warning level\s*(?:set to|=)?\s*(\d+)")):
        m = re.search(pat, out)
        if m:
            st[key] = int(m.group(1))
    return st


def hot(st: dict) -> bool:
    return st.get("cpu_speed_limit", 100) < 100 or st.get("thermal_level", 0) > 0 or st.get("performance_level", 0) > 0


def stop_requested(out_dir: Path) -> bool:
    return (out_dir / "STOP").exists()


# ---------------------------------------------------------------------------------------------- generation
def generate_chunk(model, tok, info, model_key: str, chunk: int, work: pd.DataFrame, tok_df: pd.DataFrame,
                   wmeta: dict, out_dir: Path, run_name: str, bsz: int, records_per_shard: int,
                   recycle_every: int, cooldown_s: int, max_records: int | None = None) -> dict:
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator

    from .generate_mlx import GEN_SCHEMA

    st = mac_settings(model_key)
    shash = mac_settings_hash(model_key)
    work = work[work.chunk == chunk]
    done = done_records(out_dir, model_key, shash)
    todo = work[[(p, int(s)) not in done for p, s in zip(work.prompt_id, work.sample_index)]]
    if max_records is not None:
        todo = todo.iloc[:max_records]
    if todo.empty:
        print(f"[{model_key}/mac c{chunk:02d}] nothing to do ({len(work)} records already present)", flush=True)
        return {"chunk": chunk, "chunk_records": len(work), "generated": 0}
    assert int(tok_df.loc[todo.prompt_id.unique(), "n_input_tokens"].max()) + st["max_new_tokens"] <= 131072

    stop = [[t] for t in info["eos_token_ids"]]
    segment = f"c{chunk:02d}_mac{time.strftime('%Y%m%dT%H%M%S')}"

    def new_gen():
        return BatchGenerator(model, max_tokens=st["max_new_tokens"], stop_tokens=stop,
                              completion_batch_size=bsz, prefill_batch_size=min(bsz, 16))

    gen, inserted_in_gen = new_gen(), 0
    queue = list(zip(todo.prompt_id, todo.sample_index.astype(int), todo.seed.astype(int)))
    meta, toks, fin, finished = {}, {}, {}, []
    order = 0
    t0, total_out, n_written, shards, thermal_log = time.time(), 0, 0, [], []

    def feed():
        nonlocal order, inserted_in_gen
        batch = []
        while queue and (len(meta) - len(fin)) + len(batch) < 2 * bsz and inserted_in_gen < recycle_every:
            batch.append(queue.pop(0))
            inserted_in_gen += 1
        if not batch:
            return
        samplers = [SeededVllmSampler(seed, st["temperature"], st["top_p"], st["top_k"], st["min_p"])
                    for _, _, seed in batch]
        prompts = [[int(x) for x in tok_df.loc[p, "input_ids"]] for p, _, _ in batch]
        uids = gen.insert(prompts, [st["max_new_tokens"]] * len(batch), samplers=samplers)
        for u, (p, s, seed) in zip(uids, batch):
            meta[u] = (p, s, seed, order)
            toks[u] = []
            order += 1

    def flush(uids):
        nonlocal total_out, n_written
        rows = []
        for u in uids:
            p, s, seed, o = meta[u]
            ids = toks[u]
            rows.append({
                "response_id": f"{model_key}:{p}:s{s:02d}", "prompt_id": p, "model_key": model_key,
                "sample_index": s, "response_text": tok.decode(ids), "output_token_ids": ids,
                "n_output_tokens": len(ids), "finish_reason": fin[u], "hit_max_tokens": fin[u] == "length",
                "stop_reason": None, "n_input_tokens": int(tok_df.loc[p, "n_input_tokens"]),
                "rendered_sha256": tok_df.loc[p, "rendered_sha256"], "batch_id": f"{run_name}-mlx-{segment}",
                "batch_seed": seed, "insert_order": o, "completion_batch_size": bsz, "source_run": run_name,
                "backend": BACKEND, "sampler_semantics": SAMPLER_SEMANTICS, "rng": RNG, "settings_hash": shash,
                "schema": GEN_SCHEMA, "created_unix": time.time(), "expansion_version": E.VERSION,
                "worklist_sha256": wmeta["worklist_sha256"], "worklist_chunk": chunk})
        n_out = sum(r["n_output_tokens"] for r in rows)
        total_out += n_out
        n_written += len(rows)
        idx = len(list(out_dir.glob(f"shard_{segment}_[0-9]*.parquet")))
        name = f"shard_{segment}_{idx:05d}.parquet"
        write_parquet(pd.DataFrame(rows), out_dir / name)
        for u in uids:
            del meta[u], toks[u], fin[u]
        mx.clear_cache()
        el = time.time() - t0
        th = thermal_state()
        shards.append({"shard": name, "n_records": len(rows), "output_tokens": n_out, "elapsed_s": el,
                       "peak_mem_gb": mx.get_peak_memory() / 1e9, "thermal": th})
        print(f"[{model_key}/mac c{chunk:02d}] {name}: {len(rows)} rec; chunk {n_written}/{len(todo)} new; "
              f"{total_out} tok in {el:.0f}s = {total_out / el:.0f} tok/s, {n_written / el * 3600:.0f} rec/h; "
              f"peak {mx.get_peak_memory() / 1e9:.1f} GB; therm {th or 'nominal'}", flush=True)
        return th

    feed()
    while True:
        resp = gen.next_generated()
        if not resp:
            if not queue:
                break
            if inserted_in_gen >= recycle_every:
                gen.close()
                mx.clear_cache()
                gen, inserted_in_gen = new_gen(), 0
            feed()
            continue
        for r in resp:
            if r.finish_reason != "stop":
                toks[r.uid].append(r.token)
            if r.finish_reason is not None:
                fin[r.uid] = r.finish_reason
                finished.append(r.uid)
        if len(finished) >= records_per_shard:
            th = flush(finished)
            finished = []
            if hot(th) and cooldown_s:
                thermal_log.append({"t": time.time(), **th})
                print(f"[{model_key}/mac] thermal pressure {th}; draining is not possible mid-batch, "
                      f"pausing feed for {cooldown_s}s", flush=True)
                time.sleep(cooldown_s)
            if stop_requested(out_dir):
                queue.clear()  # finish in-flight sequences, then stop
        feed()
    if finished:
        flush(finished)
    gen.close()
    el = time.time() - t0
    return {"chunk": chunk, "segment": segment, "chunk_records": len(work), "records_already_done": len(work) - len(todo),
            "generated": n_written, "output_tokens": total_out, "wall_s": el,
            "records_per_hour": n_written / el * 3600 if el else None, "tok_per_s": total_out / el if el else None,
            "thermal_pauses": thermal_log, "shards": shards}


def run(model_key: str, chunks: list[int], run_name: str = MAC_RUN, bsz: int = 64, records_per_shard: int = 400,
        recycle_every: int = 2000, cooldown_s: int = 300, max_records: int | None = None, smoke: int = 0,
        cache_limit_gb: float = 4.0) -> dict:
    import mlx.core as mx
    import mlx_lm

    from .generate_mlx import load_model

    if smoke:
        run_name = "smoke_expansion_s3_mac_mlx"
        work, tok_df, wmeta = E.smoke_work(model_key, smoke)
        work = work.copy()
        work["seed"] = [E.sample_seed(model_key, p, int(s)) for p, s in zip(work.prompt_id, work.sample_index)]
        chunks = [0]
    else:
        work, tok_df, wmeta = E.check_worklist(model_key)
        wmeta.setdefault("seeding", "per-sample worklist seeds")
    out_dir = out_dir_for(model_key, run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_load = time.time()
    model, tok, info = load_model(model_key)
    load_s = time.time() - t_load
    mx.set_cache_limit(int(cache_limit_gb * 1e9))
    mx.reset_peak_memory()
    results = []
    for c in chunks:
        if stop_requested(out_dir):
            print("STOP file present; not starting another chunk", flush=True)
            break
        r = generate_chunk(model, tok, info, model_key, c, work, tok_df, wmeta, out_dir, run_name, bsz,
                           records_per_shard, recycle_every, cooldown_s, max_records)
        results.append(r)
        write_json({"model_key": model_key, "run_name": run_name, "version": E.VERSION, "backend": BACKEND,
                    "settings": mac_settings(model_key), "settings_hash": mac_settings_hash(model_key),
                    "vllm_settings_hash_on_betty": vllm_hash(model_key),
                    "model_info": info, "mlx_version": mx.__version__, "mlx_lm_version": mlx_lm.__version__,
                    "platform": platform.platform(), "model_load_s": load_s, "completion_batch_size": bsz,
                    "cache_limit_gb": cache_limit_gb, "recycle_every_records": recycle_every,
                    "worklist_sha256": wmeta["worklist_sha256"],
                    "tokenized_parquet_sha256": wmeta.get("tokenized_parquet_sha256"),
                    "note": "Mac hedge for Betty array 11110486; separate run, never merged automatically. "
                            "Using these records in place of Betty's mixes backends and needs owner approval.",
                    "chunk_result": r}, out_dir / f"run_summary_c{c:02d}_{int(time.time())}.json")
    return {"run_name": run_name, "chunks": results}


def status(model_key: str, run_name: str = MAC_RUN) -> dict:
    work, _, _ = E.check_worklist(model_key)
    out_dir = out_dir_for(model_key, run_name)
    done = done_records(out_dir, model_key, mac_settings_hash(model_key)) if out_dir.exists() else set()
    w = work.assign(done=[(p, int(s)) in done for p, s in zip(work.prompt_id, work.sample_index)])
    per = w.groupby("chunk").done.agg(["sum", "size"])
    return {"records_done": int(w.done.sum()), "records_total": len(w),
            "chunks": {int(c): f"{int(r['sum'])}/{int(r['size'])}" for c, r in per.iterrows()}}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mife.expansion_s3_mac")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("model", choices=E.MODELS)
    g.add_argument("--chunks", default="11,10,9,8,7,6,5,4,3,2")
    g.add_argument("--bsz", type=int, default=64)
    g.add_argument("--records-per-shard", type=int, default=400)
    g.add_argument("--recycle-every", type=int, default=2000)
    g.add_argument("--cooldown-s", type=int, default=300)
    g.add_argument("--max-records", type=int)
    g.add_argument("--smoke", type=int, default=0, help="TRAIN-only benchmark on N pilot prompts (separate run)")
    s = sub.add_parser("status")
    s.add_argument("model", choices=E.MODELS)
    a = ap.parse_args(argv)
    if a.cmd == "generate":
        out = run(a.model, [int(c) for c in a.chunks.split(",")], bsz=a.bsz, records_per_shard=a.records_per_shard,
                  recycle_every=a.recycle_every, cooldown_s=a.cooldown_s, max_records=a.max_records, smoke=a.smoke)
        print(json.dumps({k: v for k, v in out.items() if k != "chunks"}, default=str))
    else:
        print(json.dumps(status(a.model), indent=2))


if __name__ == "__main__":
    main()
