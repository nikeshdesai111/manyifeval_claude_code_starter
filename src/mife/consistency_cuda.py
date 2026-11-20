"""vLLM (generation) vs HF Transformers on CUDA (introspection): final-prompt-token next-token agreement.

The generation job stores vLLM's top-20 log-probabilities at the final prompt token for the first
pilot prompts. Here the HF eager BF16 model (the one used for features) scores the same token IDs.
Same safetensors, same token IDs, different kernels: agreement is expected to be close, not bitwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .storage import write_json


def compare_topk(hf_logprobs: np.ndarray, vllm_ids: list[int], vllm_lps: list[float]) -> dict:
    """hf_logprobs: full-vocabulary log-softmax from HF; vLLM gives only its top-k."""
    k = len(vllm_ids)
    hf_top = np.argsort(-hf_logprobs)[:k]
    diffs = np.abs(hf_logprobs[np.array(vllm_ids)] - np.array(vllm_lps))
    return {"top1_agree": bool(hf_top[0] == vllm_ids[0]),
            "top10_overlap": len(set(hf_top[:10].tolist()) & set(vllm_ids[:10])) / 10,
            "topk_overlap": len(set(hf_top.tolist()) & set(vllm_ids)) / k,
            "max_abs_logprob_diff_on_vllm_topk": float(diffs.max()),
            "abs_logprob_diff_top1": float(diffs[0]),
            "vllm_top1_prob": float(np.exp(vllm_lps[0])), "hf_top1_prob": float(np.exp(hf_logprobs[hf_top[0]]))}


def run(model_key: str, run_name: str) -> dict:
    import torch

    from .introspect import load_torch_model, torch_device

    vl = pd.read_parquet(DATA_DIR / "consistency" / f"{model_key}_{run_name}_vllm_logprobs.parquet")
    tokdf = _tok_checked(model_key).set_index("prompt_id")
    device = torch_device()
    model = load_torch_model(model_key, device)
    rows = []
    for r in vl.itertuples():
        ids = [int(x) for x in tokdf.loc[r.prompt_id, "input_ids"]]
        with torch.no_grad():
            lg = model(input_ids=torch.tensor([ids], device=device), use_cache=False).logits[0, -1].float()
        lp = torch.log_softmax(lg, -1).cpu().numpy()
        rows.append({"prompt_id": r.prompt_id, **compare_topk(lp, list(r.token_ids), list(r.logprobs))})
    df = pd.DataFrame(rows)
    rep = {"model_key": model_key, "run": run_name, "comparison": "vLLM BF16 vs HF transformers eager BF16, same device type",
           "device": device, "n_prompts": len(df), "same_safetensors": True, "same_token_ids": True,
           "top1_agreement_rate": float(df.top1_agree.mean()), "top10_overlap_mean": float(df.top10_overlap.mean()),
           "topk_overlap_mean": float(df.topk_overlap.mean()),
           "max_abs_logprob_diff_quantiles": df.max_abs_logprob_diff_on_vllm_topk.quantile([.5, .9, 1]).to_dict(),
           "flags": (["top-1 disagreement on some prompts"] if df.top1_agree.mean() < 1 else []),
           "per_prompt": rows}
    write_json(rep, DATA_DIR / "consistency" / f"{model_key}_{run_name}_vllm_vs_hf.json")
    return rep


def _tok_checked(model_key: str):
    """Tokenized prompts, after confirming they match the current manifest (StaleDataError otherwise)."""
    from .integrity import tokenized_hashes
    tokenized_hashes(model_key)
    return pd.read_parquet(DATA_DIR / "tokenized" / f"{model_key}.parquet")
