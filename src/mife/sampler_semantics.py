"""Quantify the difference between the MLX and vLLM/HF definitions of the same decoding settings.

mlx_lm 0.32: top_p -> min_p -> top_k on UNTEMPERED log-probs, then sample from softmax(logprobs / T).
vLLM / HF:   logits / T -> top_k -> top_p (on tempered, renormalised probs) -> min_p, then sample.
For each pilot prompt we teacher-force one stored Mac response through the HF model and compare the two
next-token distributions at every response position (total variation, and probability mass that one
definition puts on tokens the other excludes). This is a measurement on existing pilot data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_DIR, load_config
from .storage import write_json


def _nucleus_keep(probs, top_p: float):
    """Keep a token iff the probability mass of strictly-more-likely tokens is < top_p (>=1 token kept)."""
    import torch
    sp, si = torch.sort(probs, dim=-1, descending=True)
    above = torch.cumsum(sp, -1) - sp
    keep_sorted = above < top_p
    keep_sorted[..., 0] = True
    keep = torch.zeros_like(keep_sorted)
    return keep.scatter(-1, si, keep_sorted)


def _topk_keep(x, k: int):
    import torch
    if not k or k <= 0 or k >= x.shape[-1]:
        return torch.ones_like(x, dtype=torch.bool)
    th = torch.topk(x, k, dim=-1).values[..., -1:]
    return x >= th


def _minp_keep(probs, min_p: float):
    import torch
    if not min_p:
        return torch.ones_like(probs, dtype=torch.bool)
    return probs >= min_p * probs.max(-1, keepdim=True).values


def mlx_distribution(logits, temperature: float, top_p: float, top_k: int, min_p: float):
    import torch
    lp = torch.log_softmax(logits.float(), -1)
    keep = torch.ones_like(lp, dtype=torch.bool)
    if 0 < top_p < 1:
        keep &= _nucleus_keep(lp.exp(), top_p)
    lp = lp.masked_fill(~keep, float("-inf"))
    keep &= _minp_keep(lp.exp(), min_p)
    lp = lp.masked_fill(~keep, float("-inf"))
    keep &= _topk_keep(lp, top_k)
    lp = lp.masked_fill(~keep, float("-inf"))
    return torch.softmax(lp / temperature, -1)


def vllm_distribution(logits, temperature: float, top_p: float, top_k: int, min_p: float):
    import torch
    z = logits.float() / temperature
    keep = _topk_keep(z, top_k)
    z = z.masked_fill(~keep, float("-inf"))
    if 0 < top_p < 1:
        keep &= _nucleus_keep(torch.softmax(z, -1), top_p)
        z = z.masked_fill(~keep, float("-inf"))
    keep &= _minp_keep(torch.softmax(z, -1), min_p)
    z = z.masked_fill(~keep, float("-inf"))
    return torch.softmax(z, -1)


def compare(pm, pv) -> dict:
    """pm, pv: (positions, vocab) distributions."""
    tv = 0.5 * (pm - pv).abs().sum(-1)
    m_out = (pm * (pv == 0)).sum(-1)   # MLX mass on tokens vLLM/HF can never sample
    v_out = (pv * (pm == 0)).sum(-1)
    supp_m, supp_v = (pm > 0).sum(-1).float(), (pv > 0).sum(-1).float()
    return {"tv": tv, "mlx_mass_outside_vllm_support": m_out, "vllm_mass_outside_mlx_support": v_out,
            "support_mlx": supp_m, "support_vllm": supp_v}


def run(model_key: str, n_prompts: int = 20, max_response_tokens: int = 512) -> dict:
    import torch

    from .generate_mlx import load_generations
    from .introspect import load_torch_model, torch_device

    st = load_config()["models"][model_key]["decoding"]
    args = (st["temperature"], st["top_p"], st["top_k"], st["min_p"])
    pilot = list(pd.read_parquet(DATA_DIR / "pilot" / "pilot_prompts.parquet").prompt_id)
    pids = pilot[:: max(1, len(pilot) // n_prompts)][:n_prompts]
    gens = load_generations(model_key, "pilot").set_index("response_id")
    tokdf = _tok_checked(model_key).set_index("prompt_id")
    device = torch_device()
    model = load_torch_model(model_key, device)
    per_resp, all_tv = [], []
    for pid in pids:
        g = gens.loc[f"{model_key}:{pid}:s00"]
        p_ids = [int(x) for x in tokdf.loc[pid, "input_ids"]]
        r_ids = [int(x) for x in g.output_token_ids][:max_response_tokens]
        ids = torch.tensor([p_ids + r_ids], device=device)
        with torch.no_grad():
            lg = model(input_ids=ids, use_cache=False).logits[0, len(p_ids) - 1: len(p_ids) - 1 + len(r_ids) + 1]
        c = compare(mlx_distribution(lg, *args), vllm_distribution(lg, *args))
        c = {k: v.cpu().numpy() for k, v in c.items()}
        all_tv.append(c["tv"])
        per_resp.append({"prompt_id": pid, "positions": len(c["tv"]), "tv_mean": float(c["tv"].mean()),
                         "frac_positions_tv_gt_0.01": float((c["tv"] > 0.01).mean()),
                         "expected_mlx_only_tokens_per_response": float(c["mlx_mass_outside_vllm_support"].sum()),
                         "expected_vllm_only_tokens_per_response": float(c["vllm_mass_outside_mlx_support"].sum()),
                         "support_mlx_median": float(np.median(c["support_mlx"])),
                         "support_vllm_median": float(np.median(c["support_vllm"]))})
    df = pd.DataFrame(per_resp)
    tv = np.concatenate(all_tv)
    rep = {"model_key": model_key, "decoding": st, "device": device, "n_responses": len(df),
           "n_positions": int(len(tv)), "max_response_tokens": max_response_tokens,
           "teacher_forced_on": "Mac pilot response s00 (MLX-generated) for each prompt",
           "tv_mean": float(tv.mean()), "tv_quantiles": {str(q): float(np.quantile(tv, q)) for q in (.5, .9, .99, 1)},
           "frac_positions_tv_gt_0.01": float((tv > 0.01).mean()),
           "frac_positions_tv_gt_0.1": float((tv > 0.1).mean()),
           "mean_expected_mlx_only_tokens_per_response": float(df.expected_mlx_only_tokens_per_response.mean()),
           "mean_expected_vllm_only_tokens_per_response": float(df.expected_vllm_only_tokens_per_response.mean()),
           "per_response": per_resp}
    write_json(rep, DATA_DIR / "consistency" / f"{model_key}_sampler_semantics.json")
    return rep


def _tok_checked(model_key: str):
    """Tokenized prompts, after confirming they match the current manifest (StaleDataError otherwise)."""
    from .integrity import tokenized_hashes
    tokenized_hashes(model_key)
    return pd.read_parquet(DATA_DIR / "tokenized" / f"{model_key}.parquet")
