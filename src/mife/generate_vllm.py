"""Resumable, idempotent generation on NVIDIA GPUs with vLLM (BF16, no quantization).

Same work unit, shard schema and resume logic as generate_mlx. Prompts are submitted as the exact
token IDs stored in data/tokenized/{model}.parquet (no re-rendering inside vLLM), so the chat template
and Qwen3 enable_thinking=False are identical by construction. Each (prompt, sample_index) is its own
request. Seeding: one engine seed per run by default (as the Mac run used one seed per segment). Optional
per-sample seeds (derived from generation seed, model, run, prompt, sample index) route vLLM through its
slower per-request-generator sampling path, so they are off by default to keep throughput representative.
Neither mode is bitwise reproducible across batch compositions.

Not identical to the Mac pilot: different kernels and RNG, and a different order of temperature vs
top-p/top-k filtering (see generate_mlx.SAMPLER_SEMANTICS). Outputs are a separate run, never mixed
into the MLX pilot directory.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import threading
import time
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, derive_seed, load_config
from . import integrity
from .generate_mlx import GEN_SCHEMA, done_prompts, eos_token_ids, gen_settings, model_path, settings_hash
from .storage import write_json, write_parquet

BACKEND = "vllm"
PROTECTED_RUNS = {"pilot"}  # the Mac/MLX pilot; CUDA runs must use a new run id


def check_run_name(model_key: str, run_name: str, kind: str = "generations") -> Path:
    """Refuse to write a CUDA run into the Mac pilot (or any directory produced by another backend)."""
    if run_name in PROTECTED_RUNS:
        raise SystemExit(f"run '{run_name}' is the Mac/MLX pilot and is protected; use a new run id")
    out = DATA_DIR / kind / model_key / run_name
    for f in list(out.glob("run_summary_*.json")) + list(out.glob("features_meta.json")):
        d = json.loads(f.read_text())
        b = str(d.get("backend") or d.get("settings", {}).get("backend") or "")
        if b == "mlx_lm" or "mps" in b:
            raise SystemExit(f"{out} already holds output from backend {b}; refusing to mix backends")
    return out


def chunk_ids(prompt_ids: list[str], chunk: int, n_chunks: int) -> list[str]:
    """Deterministic contiguous partition of the manifest order into n_chunks parts (bulk job arrays)."""
    if not 0 <= chunk < n_chunks:
        raise ValueError(f"chunk {chunk} not in [0, {n_chunks})")
    size = -(-len(prompt_ids) // n_chunks)
    return prompt_ids[chunk * size:(chunk + 1) * size]


def sample_seed(model_key: str, run_name: str, prompt_id: str, sample_index: int) -> int:
    base = load_config()["seeds"]["generation"]
    return derive_seed(base, BACKEND, model_key, run_name, prompt_id, sample_index) % (1 << 31)


def sampling_kwargs(st: dict) -> dict:
    """Map frozen decoding settings onto vLLM SamplingParams (top_k 0 = disabled = -1 in vLLM)."""
    return {"temperature": st["temperature"], "top_p": st["top_p"],
            "top_k": st["top_k"] if st["top_k"] and st["top_k"] > 0 else -1,
            "min_p": st["min_p"], "max_tokens": st["max_new_tokens"],
            "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}


def strip_stop(token_ids: list[int], finish_reason: str, stop_ids: set[int]) -> list[int]:
    """Match the MLX record: the terminating EOS token is not part of the stored response."""
    ids = list(token_ids)
    if finish_reason == "stop" and ids and ids[-1] in stop_ids:
        ids = ids[:-1]
    return ids


class GpuMemSampler:
    """Polls nvidia-smi for memory.used of the visible GPU(s); peak over the generation phase."""

    def __init__(self, every_s: float = 2.0):
        self.every, self.peak, self._stop = every_s, 0.0, threading.Event()
        self.t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout
                self.peak = max([self.peak] + [float(x) / 1024 for x in out.split() if x.strip()])
            except Exception:
                pass
            self._stop.wait(self.every)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.t.join(timeout=15)


def cuda_env() -> dict:
    """Backend, precision and version provenance for CUDA runs."""
    info = {"hostname": platform.node(), "platform": platform.platform(),
            **{k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                                              "SLURM_JOB_PARTITION", "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE")}}
    for mod in ("torch", "vllm", "transformers", "tokenizers", "flashinfer", "triton"):
        try:
            info[f"{mod}_version"] = __import__(mod).__version__
        except Exception as e:  # noqa: BLE001
            info[f"{mod}_version"] = f"unavailable: {type(e).__name__}"
    try:
        import torch
        info.update({"torch_cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                     "cuda_available": torch.cuda.is_available()})
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update({"gpu_name": p.name, "gpu_total_memory_gb": p.total_memory / 1e9,
                         "gpu_capability": list(torch.cuda.get_device_capability(0))})
    except Exception as e:  # noqa: BLE001
        info["torch_error"] = repr(e)
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,mig.mode.current",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        info["nvidia_smi"] = q
        hdr = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout
        info["nvidia_smi_cuda_version"] = next((l.split("CUDA Version:")[1].split("|")[0].strip()
                                                for l in hdr.splitlines() if "CUDA Version:" in l), None)
        info["nvidia_smi_L"] = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        info["nvidia_smi_error"] = repr(e)
    return info


def load_llm(model_key: str, max_model_len: int, gpu_memory_utilization: float, max_num_seqs: int, seed: int):
    from vllm import LLM

    p = str(model_path(model_key))
    llm = LLM(model=p, tokenizer=p, dtype="bfloat16", quantization=None, seed=seed,
              max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
              max_num_seqs=max_num_seqs, enable_prefix_caching=True,
              generation_config="vllm",  # do NOT pull sampling defaults from generation_config.json
              trust_remote_code=False)
    mc = getattr(llm.llm_engine, "model_config", None)
    info = {"model_dir": p, "vllm_dtype": str(getattr(mc, "dtype", "unknown")),
            "vllm_quantization": getattr(mc, "quantization", "unknown"),
            "max_model_len": max_model_len, "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs, "enable_prefix_caching": True}
    if info["vllm_quantization"] not in (None, "unknown"):
        raise RuntimeError(f"vLLM reports quantization {info['vllm_quantization']}; refusing (no silent quantization)")
    if "bfloat16" not in info["vllm_dtype"] and info["vllm_dtype"] != "unknown":
        raise RuntimeError(f"vLLM dtype {info['vllm_dtype']} is not bfloat16; refusing")
    return llm, info


def run(model_key: str, prompt_ids: list[str], run_name: str, prompts_per_shard: int = 50,
        max_prompts: int | None = None, gpu_memory_utilization: float = 0.85, max_num_seqs: int = 512,
        max_model_len: int = 4096, n_probe: int = 20, per_sample_seeds: bool = False,
        chunk_tag: str | None = None) -> dict:
    """chunk_tag (e.g. "c03") makes shard/summary names and the engine seed unique per parallel job, so
    several jobs can write into the same run directory without colliding."""
    from vllm import SamplingParams

    out_dir = check_run_name(model_key, run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    st = gen_settings(model_key, BACKEND)
    n_s = st["n_samples"]
    tok_df = pd.read_parquet(DATA_DIR / "tokenized" / f"{model_key}.parquet",
                             columns=["prompt_id", "input_ids", "n_input_tokens", "rendered_sha256"]).set_index("prompt_id")
    assert int(tok_df.loc[prompt_ids, "n_input_tokens"].max()) + st["max_new_tokens"] <= max_model_len, \
        "max_model_len too small for input + max_new_tokens (no truncation allowed)"
    integrity.tokenized_hashes(model_key)  # tokenization must match the current manifest
    done = done_prompts(out_dir, n_s, model_key)
    todo = [p for p in prompt_ids if p not in done]
    if max_prompts is not None:
        todo = todo[:max_prompts]

    stop = eos_token_ids(model_key)
    env = cuda_env()
    seed_parts = (cfg["seeds"]["generation"], BACKEND, model_key, run_name) + ((chunk_tag,) if chunk_tag else ())
    engine_seed = derive_seed(*seed_parts) % (1 << 31)
    prefix = f"shard_{chunk_tag}_" if chunk_tag else "shard_"
    t_load = time.time()
    llm, info = load_llm(model_key, max_model_len, gpu_memory_utilization, max_num_seqs, engine_seed)
    load_s = time.time() - t_load
    skw = sampling_kwargs(st)
    shash = settings_hash(model_key, BACKEND)

    batch_log, total_out, gen_wall = [], 0, 0.0
    with GpuMemSampler() as mem:
        for s0 in range(0, len(todo), prompts_per_shard):
            pids = todo[s0:s0 + prompts_per_shard]
            reqs, params, meta = [], [], []
            for pid in pids:
                ids = [int(x) for x in tok_df.loc[pid, "input_ids"]]
                for sidx in range(n_s):
                    seed = sample_seed(model_key, run_name, pid, sidx) if per_sample_seeds else engine_seed
                    reqs.append({"prompt_token_ids": ids})
                    params.append(SamplingParams(n=1, seed=seed if per_sample_seeds else None, stop_token_ids=stop,
                                                 skip_special_tokens=False, detokenize=False, **skw))
                    meta.append((pid, sidx, seed))
            t0 = time.time()
            outs = llm.generate(reqs, params, use_tqdm=False)
            el = time.time() - t0
            gen_wall += el
            tok = llm.get_tokenizer()
            rows = []
            for (pid, sidx, seed), o in zip(meta, outs):
                c = o.outputs[0]
                ids = strip_stop(list(c.token_ids), c.finish_reason, set(stop))
                rows.append({
                    "response_id": f"{model_key}:{pid}:s{sidx:02d}", "prompt_id": pid, "model_key": model_key,
                    "sample_index": sidx, "response_text": tok.decode(ids), "output_token_ids": ids,
                    "n_output_tokens": len(ids), "finish_reason": c.finish_reason,
                    "hit_max_tokens": c.finish_reason == "length", "stop_reason": str(c.stop_reason),
                    "n_input_tokens": int(tok_df.loc[pid, "n_input_tokens"]),
                    "rendered_sha256": tok_df.loc[pid, "rendered_sha256"],
                    "batch_id": f"{run_name}-{BACKEND}-{chunk_tag or 'shard'}", "batch_seed": seed, "insert_order": len(rows),
                    "completion_batch_size": max_num_seqs, "source_run": run_name, "backend": BACKEND,
                    "settings_hash": shash, "schema": GEN_SCHEMA, "created_unix": time.time()})
            n_out = sum(r["n_output_tokens"] for r in rows)
            total_out += n_out
            shard_idx = len(list(out_dir.glob(f"{prefix}[0-9]*.parquet")))
            write_parquet(pd.DataFrame(rows), out_dir / f"{prefix}{shard_idx:05d}.parquet")
            batch_log.append({"shard": shard_idx, "prompts": pids, "output_tokens": n_out, "generate_s": el,
                              "peak_gpu_memory_used_gb": mem.peak})
            print(f"[{model_key}/{BACKEND}] shard {shard_idx}: {len(pids)} prompts, {n_out} tok in {el:.0f}s "
                  f"({n_out / el:.0f} tok/s); cumulative {total_out / max(gen_wall, 1e-9):.0f} tok/s", flush=True)
        probe = final_token_logprobs(llm, model_key, prompt_ids[:n_probe], run_name) if n_probe else None

    summary = {"model_key": model_key, "run_name": run_name, "backend": BACKEND, "settings": st,
               "settings_hash": shash, "sampling_params": {**skw, "stop_token_ids": stop, "n": 1,
                                                           "seeding": ("per-sample: derive_seed(generation, vllm, model, run, prompt, sample) % 2^31"
                                                                       if per_sample_seeds else "engine seed only (batch_seed column = engine seed)")},
               "engine": info, "engine_seed": engine_seed, "chunk_tag": chunk_tag, "environment": env,
               "prompts_per_shard": prompts_per_shard, "completion_batch_size": max_num_seqs,
               "n_prompts_generated_this_call": len(todo), "n_prompts_skipped_done": len(done),
               "model_load_s": load_s, "generation_wall_s": gen_wall, "output_tokens": total_out,
               "aggregate_output_tok_per_s": total_out / gen_wall if gen_wall else None,
               "peak_gpu_memory_used_gb": mem.peak,
               "peak_gpu_memory_note": "nvidia-smi memory.used; vLLM pre-reserves gpu_memory_utilization of the GPU "
                                       "for weights + KV cache, so this reflects the reservation, not the minimum need",
               "logprob_probe_file": probe, "batches": batch_log}
    write_json(summary, out_dir / f"run_summary_{chunk_tag + '_' if chunk_tag else ''}{int(time.time())}.json")
    return summary


def final_token_logprobs(llm, model_key: str, prompt_ids: list[str], run_name: str, k: int = 20) -> str:
    """Top-k next-token log-probabilities at the final prompt token, for the vLLM-vs-HF check."""
    from vllm import SamplingParams

    tok_df = pd.read_parquet(DATA_DIR / "tokenized" / f"{model_key}.parquet",
                             columns=["prompt_id", "input_ids"]).set_index("prompt_id")
    reqs = [{"prompt_token_ids": [int(x) for x in tok_df.loc[p, "input_ids"]]} for p in prompt_ids]
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=1, logprobs=k, detokenize=False)
    outs = llm.generate(reqs, sp, use_tqdm=False)
    rows = []
    for pid, o in zip(prompt_ids, outs):
        lp = o.outputs[0].logprobs[0]  # dict token_id -> Logprob
        items = sorted(((int(t), float(v.logprob)) for t, v in lp.items()), key=lambda x: -x[1])[:k]
        rows.append({"prompt_id": pid, "token_ids": [t for t, _ in items], "logprobs": [v for _, v in items]})
    path = DATA_DIR / "consistency" / f"{model_key}_{run_name}_vllm_logprobs.parquet"
    write_parquet(pd.DataFrame(rows), path)
    return str(path.relative_to(DATA_DIR))
