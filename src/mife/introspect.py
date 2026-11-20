"""Pre-generation introspection: ONE forward pass per unique prompt and model (PyTorch, BF16, eager).

Device: Apple MPS on the Mac, CUDA on GPU clusters (auto-detected; override with MIFE_TORCH_DEVICE).

Per prompt we store:
  * final-prompt-token hidden state for every entry of `hidden_states` (embeddings + every layer),
    plus the pre-final-norm output of the last layer, as raw BF16 bits (uint16, lossless);
  * attention from the final prompt token to each included rule's exact token span, per layer x head:
    SUM of attention probabilities over the span and token-normalised MEAN (sum / span length);
    also the same statistics for the task-text span;
  * the exact token spans, span lengths and final token id/index.
Full attention tensors are kept only for a small diagnostic sample.
The final prompt token is the last token of the rendered chat prompt (after the assistant header),
i.e. the state from which the first response token is sampled.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .config import DATA_DIR, load_config
from .generate_mlx import model_path
from . import integrity
from .storage import write_json, write_npy, write_parquet

FEAT_SCHEMA = "features/v1"
K_MAX = 5


def torch_device() -> str:
    import os

    import torch
    d = os.environ.get("MIFE_TORCH_DEVICE")
    if d:
        return d
    if torch.cuda.is_available():
        return "cuda"
    return "mps"


def _sync(device: str):
    import torch
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


def load_torch_model(model_key: str, device: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM

    device = device or torch_device()
    model = AutoModelForCausalLM.from_pretrained(str(model_path(model_key)), dtype=torch.bfloat16,
                                                 attn_implementation="eager").to(device).eval()
    dtypes = {p.dtype for p in model.parameters()}
    assert dtypes == {torch.bfloat16}, dtypes
    return model


def _bf16_bits(t) -> np.ndarray:
    import torch
    return t.to(torch.bfloat16).contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def bits_to_float32(a: np.ndarray) -> np.ndarray:
    """Decode stored BF16 bit patterns to float32 exactly."""
    return (a.astype(np.uint32) << 16).view(np.float32)


def _write_npz(path, **arrays) -> None:
    from .storage import _atomic

    def w(t):
        with open(t, "wb") as f:
            np.savez(f, **arrays)
    _atomic(path, w)


def done_features(out, model_key: str) -> tuple[set[str], int]:
    """Prompts already committed in `out`, and the next free shard number.

    A shard is committed only once its index_NNNNN.parquet exists; it is written last, after the hidden and attention
    files, so an interrupted shard is simply recomputed. Every committed shard's prompt and rendered hashes are
    checked against the current manifest/tokenization (StaleDataError on any mismatch).
    """
    idx = sorted(out.glob("index_[0-9]*.parquet"))
    if not idx and (out / "index.parquet").exists():
        raise integrity.StaleDataError(f"{out}: features without per-shard hashed indexes; cannot verify them")
    done, nxt, frames = set(), 0, []
    for f in idx:
        n = int(f.stem.split("_")[1])
        if (out / f"hidden_{n:05d}.npy").exists() and (out / f"attn_{n:05d}.npz").exists():
            frames.append(pd.read_parquet(f))
            nxt = max(nxt, n + 1)
    if frames:
        df = pd.concat(frames, ignore_index=True)
        integrity.check_prompt(df, str(out))
        integrity.check_rendered(model_key, df, str(out))
        done = set(df.prompt_id)
    return done, nxt


def _check_backend(out, device: str) -> None:
    for name in ("features_backend.json", "features_meta.json"):
        f = out / name
        if f.exists():
            prev = json.loads(f.read_text()).get("backend", "")
            if ("cuda" in device) != ("cuda" in prev):
                raise SystemExit(f"{out} holds features from {prev}; refusing to add features from device {device}")


def run(model_key: str, run_name: str, prompt_ids: list[str], n_diag: int = 3, shard_size: int = 512,
        max_prompts: int | None = None) -> dict:
    """Resumable: prompts already committed in data/features/<model>/<run>/ are skipped."""
    import torch

    device = torch_device()
    backend = f"transformers+torch-{device.split(':')[0]}"
    integrity.tokenized_hashes(model_key)  # tokenization must match the current manifest
    tokdf = pd.read_parquet(DATA_DIR / "tokenized" / f"{model_key}.parquet").set_index("prompt_id")
    out = DATA_DIR / "features" / model_key / run_name
    _check_backend(out, device)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "features_backend.json").exists():
        write_json({"backend": backend, "device": device, "model_key": model_key, "schema": FEAT_SCHEMA},
                   out / "features_backend.json")
    diag_ids = set(prompt_ids[:n_diag])
    done, shard = done_features(out, model_key)
    todo = [p for p in prompt_ids if p not in done]
    if max_prompts is not None:
        todo = todo[:max_prompts]
    t0 = time.time()
    model = load_torch_model(model_key, device) if todo else None
    load_s = time.time() - t0
    convention, per_prompt_s, n_layers = None, [], None
    if model is not None:
        norm = model.model.norm
        captured = {}
        hook = norm.register_forward_hook(lambda m, i, o: captured.update(pre=i[0], post=o))
        cfg = model.config
        n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
    for s0 in range(0, len(todo), shard_size):
        pids = todo[s0:s0 + shard_size]
        index_rows = []
        H = np.zeros((len(pids), n_layers + 2, cfg.hidden_size), dtype=np.uint16)
        A_sum = np.full((len(pids), n_layers, n_heads, K_MAX), np.nan, dtype=np.float32)
        A_mean = np.full_like(A_sum, np.nan)
        T_sum = np.zeros((len(pids), n_layers, n_heads), dtype=np.float32)
        T_mean = np.zeros_like(T_sum)
        mask = np.zeros((len(pids), K_MAX), dtype=bool)
        for i, pid in enumerate(pids):
            r = tokdf.loc[pid]
            ids = torch.tensor([list(r.input_ids)], device=device)
            t1 = time.time()
            with torch.no_grad():
                o = model(input_ids=ids, output_hidden_states=True, output_attentions=True, use_cache=False)
            hs = o.hidden_states
            assert len(hs) == n_layers + 1 and len(o.attentions) == n_layers
            last = ids.shape[1] - 1
            if convention is None:
                post_eq = torch.equal(hs[-1][0, last], captured["post"][0, last])
                convention = ("hidden_states[-1] is AFTER final norm" if post_eq else "hidden_states[-1] is BEFORE final norm")
            stack = torch.stack([h[0, last] for h in hs] + [captured["pre"][0, last]])
            H[i] = _bf16_bits(stack)
            att = torch.stack([a[0, :, last, :] for a in o.attentions]).float()  # (L, heads, seq)
            spans = [tuple(x) for x in r.rule_token_spans]
            for j, (a, b) in enumerate(spans):
                s = att[:, :, a:b].sum(-1)
                A_sum[i, :, :, j] = s.cpu().numpy()
                A_mean[i, :, :, j] = (s / (b - a)).cpu().numpy()
                mask[i, j] = True
            ta, tb = r.task_token_span
            ts = att[:, :, ta:tb].sum(-1)
            T_sum[i] = ts.cpu().numpy()
            T_mean[i] = (ts / (tb - ta)).cpu().numpy()
            # attention rows must sum to 1 over the causal prefix
            rowsum = att.sum(-1)
            assert torch.allclose(rowsum, torch.ones_like(rowsum), atol=2e-2), float((rowsum - 1).abs().max())
            if pid in diag_ids and not (out / f"diag_full_attention_{pid}.npz").exists():
                full = torch.stack([a[0] for a in o.attentions]).to(torch.float16).cpu().numpy()
                _write_npz(out / f"diag_full_attention_{pid}.npz", attention=full)
            _sync(device)
            per_prompt_s.append(time.time() - t1)
            index_rows.append({"prompt_id": pid, "model_key": model_key, "shard": shard, "row": i,
                               "prompt_sha256": r.prompt_sha256, "rendered_sha256": r.rendered_sha256,
                               "n_input_tokens": int(r.n_input_tokens), "final_token_index": last,
                               "final_token_id": int(r.input_ids[-1]),
                               "rule_token_spans": [list(x) for x in spans], "rule_span_lengths": [b - a for a, b in spans],
                               "task_token_span": list(r.task_token_span), "schema": FEAT_SCHEMA})
            del o, hs, att
        write_npy(H, out / f"hidden_{shard:05d}.npy")
        _write_npz(out / f"attn_{shard:05d}.npz", rule_sum=A_sum, rule_mean=A_mean, rule_mask=mask,
                   task_sum=T_sum, task_mean=T_mean)
        write_parquet(pd.DataFrame(index_rows), out / f"index_{shard:05d}.parquet")  # commit marker, written last
        print(f"[features {model_key}/{run_name}] shard {shard}: {len(pids)} prompts, "
              f"{len(done) + s0 + len(pids)}/{len(prompt_ids)} done, {np.mean(per_prompt_s[-len(pids):]):.3f} s/prompt",
              flush=True)
        shard += 1
        if device == "mps":
            torch.mps.empty_cache()
    if model is not None:
        hook.remove()
    # combined index over every committed shard (rebuilt each call; per-shard files remain the source of truth)
    shard_idx = sorted(out.glob("index_[0-9]*.parquet"))
    if shard_idx:
        write_parquet(pd.concat([pd.read_parquet(f) for f in shard_idx], ignore_index=True), out / "index.parquet")
    all_done, _ = done_features(out, model_key)
    prev = json.loads((out / "features_meta.json").read_text()) if (out / "features_meta.json").exists() else {}
    if model is None:
        prev.update({"n_prompts": len(all_done), "n_requested_missing": len(set(prompt_ids) - all_done)})
        write_json(prev, out / "features_meta.json")
        return prev
    peak = (torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda")
            else torch.mps.driver_allocated_memory() / 1e9 if device.startswith("mps") else None)
    session = {"started_unix": t0, "n_prompts_this_call": len(per_prompt_s), "n_prompts_skipped_done": len(done),
               "model_load_s": load_s, "wall_s": time.time() - t0,
               "per_prompt_s_quantiles": {str(q): float(np.quantile(per_prompt_s, q)) for q in (0, .5, .9, 1)},
               ("cuda_max_memory_allocated_gb" if device.startswith("cuda") else "mps_driver_allocated_gb_end"): peak}
    meta = {
        "model_key": model_key, "schema": FEAT_SCHEMA, "backend": backend, "device": device,
        "attn_implementation": "eager", "dtype": "bfloat16", "n_layers": n_layers, "n_heads": n_heads,
        "n_kv_heads": getattr(cfg, "num_key_value_heads", None), "hidden_size": cfg.hidden_size,
        "hidden_rows": f"0=embeddings, 1..{n_layers}=hidden_states[1..{n_layers}], {n_layers + 1}=pre-final-norm last-layer output",
        "hidden_states_convention": convention, "hidden_storage": "uint16 raw BF16 bits; decode with bits_to_float32",
        "attention_stats": "probabilities from the final prompt token (softmax in fp32, cast to bf16 by the model), summed in fp32; "
                           "rule_sum[n, layer, head, slot], rule_mean = rule_sum / span_len; slots in prompt order; "
                           "absent slots NaN with rule_mask False",
        "shards": "hidden_NNNNN.npy / attn_NNNNN.npz / index_NNNNN.parquet (commit marker); index.parquet = all shards",
        "n_prompts": len(all_done), "n_requested_missing": len(set(prompt_ids) - all_done),
        "diag_full_attention_prompts": sorted(diag_ids),
        "sessions": prev.get("sessions", []) + [session],
        # kept for compatibility with readers of single-session runs (last session's values)
        "model_load_s": load_s, "per_prompt_s_quantiles": session["per_prompt_s_quantiles"],
        "n_diag_full_attention": min(n_diag, len(prompt_ids)),
        "torch_version": torch.__version__, "transformers_version": __import__("transformers").__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "platform": __import__("platform").platform(),
    }
    for k in ("torch_version", "transformers_version"):
        if prev.get(k) and prev[k] != meta[k]:
            raise SystemExit(f"{k} changed between sessions ({prev[k]} -> {meta[k]}); refusing to mix")
    write_json(meta, out / "features_meta.json")
    return meta
