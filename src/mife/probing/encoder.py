"""BGE-M3 (BAAI/bge-m3, pinned revision) for the prompt-only baselines.

Dense embedding = L2-normalised final-layer [CLS] state, the model's released dense output (FlagEmbedding's
BGEM3FlagModel dense_vecs). Inputs are never truncated: `encode` raises if any text exceeds `max_length` tokens.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from ..config import load_config


def cfg() -> dict:
    return load_config()["probing"]["encoder"]


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=None)
def tokenizer():
    from transformers import AutoTokenizer
    c = cfg()
    return AutoTokenizer.from_pretrained(c["hf_id"], revision=c["revision"])


def load_model(dev: str | None = None, dtype=torch.float32):
    from transformers import AutoModel
    c = cfg()
    m = AutoModel.from_pretrained(c["hf_id"], revision=c["revision"], dtype=dtype, add_pooling_layer=False)
    return m.to(dev or device())


def token_lengths(texts: list[str]) -> np.ndarray:
    tok = tokenizer()
    return np.array([len(x) for x in tok(texts, add_special_tokens=True, truncation=False)["input_ids"]])


def check_lengths(texts: list[str]) -> np.ndarray:
    n = token_lengths(texts)
    lim = cfg()["max_length"]
    if (n > lim).any():
        raise ValueError(f"{int((n > lim).sum())} texts exceed the BGE-M3 limit of {lim} tokens; refusing to truncate")
    return n


def length_audit() -> dict:
    """BGE-M3 token lengths of EVERY manifest prompt (train and test) and every exact rule text.

    Tokenization only: no labels, outputs or features are read. Raises if any text exceeds the encoder limit."""
    import pandas as pd
    from ..config import DATA_DIR
    p = pd.read_parquet(DATA_DIR / "manifest" / "prompts.parquet", columns=["prompt_id", "prompt_text"])
    r = pd.read_parquet(DATA_DIR / "manifest" / "prompt_rules.parquet", columns=["rule_id", "rule_text"]).drop_duplicates()
    n_p, n_r = check_lengths(p["prompt_text"].tolist()), check_lengths(r["rule_text"].tolist())
    q = lambda a: {str(x): int(np.quantile(a, x)) for x in (0, 0.5, 0.9, 0.99, 1)}
    return {"encoder": cfg(), "limit": cfg()["max_length"], "n_prompts": len(p), "prompt_tokens": q(n_p),
            "n_rule_texts": len(r), "n_rule_ids": int(r["rule_id"].nunique()), "rule_tokens": q(n_r),
            "n_over_limit": 0, "truncation": "none"}


def dense(model, batch: dict) -> torch.Tensor:
    out = model(**batch).last_hidden_state[:, 0]
    return torch.nn.functional.normalize(out, dim=-1)


@torch.no_grad()
def encode(texts: list[str], model=None, batch_size: int = 32, dev: str | None = None) -> np.ndarray:
    """Frozen dense embeddings, float32 (n, 1024). Sorted by length for batching; returned in input order."""
    dev = dev or device()
    model = model or load_model(dev)
    model.eval()
    tok = tokenizer()
    check_lengths(texts)
    order = np.argsort([len(t) for t in texts])[::-1]
    out = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    for s in range(0, len(texts), batch_size):
        ix = order[s:s + batch_size]
        b = tok([texts[i] for i in ix], padding=True, truncation=False, return_tensors="pt").to(dev)
        out[ix] = dense(model, b).float().cpu().numpy()
    return out
