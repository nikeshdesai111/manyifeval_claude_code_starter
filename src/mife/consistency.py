"""MLX (generation) vs PyTorch-MPS (introspection) numerical consistency on identical token IDs.

Both runtimes load the SAME pinned BF16 safetensors (no conversion, no quantization). Kernels differ,
so bitwise equality is not expected; we report next-token logit agreement at the final prompt token.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .storage import write_json


def run(model_key: str, prompt_ids: list[str]) -> dict:
    import mlx.core as mx
    import torch

    from .generate_mlx import load_model
    from .introspect import load_torch_model

    tokdf = _tok_checked(model_key).set_index("prompt_id")
    mlx_model, _, info = load_model(model_key)
    tmodel = load_torch_model(model_key)
    rows = []
    for pid in prompt_ids:
        ids = [int(x) for x in tokdf.loc[pid, "input_ids"]]
        lm = np.array(mlx_model(mx.array([ids]))[0, -1].astype(mx.float32))
        with torch.no_grad():
            lt = tmodel(input_ids=torch.tensor([ids], device="mps"), use_cache=False).logits[0, -1].float().cpu().numpy()
        pm, pt = _softmax(lm), _softmax(lt)
        top_m, top_t = np.argsort(-lm)[:10], np.argsort(-lt)[:10]
        rows.append({"prompt_id": pid, "max_abs_logit_diff": float(np.abs(lm - lt).max()),
                     "mean_abs_logit_diff": float(np.abs(lm - lt).mean()),
                     "top1_agree": bool(top_m[0] == top_t[0]), "top10_overlap": len(set(top_m) & set(top_t)) / 10,
                     "kl_torch_mlx": float(np.sum(pt * (np.log(pt + 1e-30) - np.log(pm + 1e-30)))),
                     "tv_distance": float(0.5 * np.abs(pm - pt).sum())})
    df = pd.DataFrame(rows)
    rep = {"model_key": model_key, "n_prompts": len(df), "mlx_weight_dtypes": info["weight_dtypes"],
           "torch_dtype": "bfloat16", "same_safetensors": True, "same_token_ids": True,
           "top1_agreement_rate": float(df.top1_agree.mean()), "top10_overlap_mean": float(df.top10_overlap.mean()),
           "kl_quantiles": df.kl_torch_mlx.quantile([.5, .9, 1]).to_dict(),
           "tv_quantiles": df.tv_distance.quantile([.5, .9, 1]).to_dict(),
           "max_abs_logit_diff_quantiles": df.max_abs_logit_diff.quantile([.5, .9, 1]).to_dict(),
           "per_prompt": rows}
    write_json(rep, DATA_DIR / "consistency" / f"{model_key}.json")
    return rep


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _tok_checked(model_key: str):
    """Tokenized prompts, after confirming they match the current manifest (StaleDataError otherwise)."""
    from .integrity import tokenized_hashes
    tokenized_hashes(model_key)
    return pd.read_parquet(DATA_DIR / "tokenized" / f"{model_key}.parquet")
