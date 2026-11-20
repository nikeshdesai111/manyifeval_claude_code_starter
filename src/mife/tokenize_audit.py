"""Tokenize all 53,136 manifest prompts per model; persist token IDs/spans; context-fit audit."""
from __future__ import annotations

import pandas as pd

from . import chat
from .config import DATA_DIR, load_config
from .manifest import MANIFEST_DIR
from .storage import sha256_text, write_json, write_parquet


def run(model_key: str) -> dict:
    cfg = load_config()
    m = cfg["models"][model_key]
    prompts = pd.read_parquet(MANIFEST_DIR / "prompts.parquet")
    assert (prompts["prompt_sha256"] == prompts["prompt_text"].map(sha256_text)).all(), "manifest text/hash mismatch"
    rules = pd.read_parquet(MANIFEST_DIR / "prompt_rules.parquet").sort_values(["prompt_id", "position"])
    spans = rules.groupby("prompt_id")[["char_start", "char_end"]].apply(lambda d: list(zip(d.char_start, d.char_end)))
    rows = []
    for p in prompts.itertuples():
        e = chat.encode(model_key, p.prompt_text, spans[p.prompt_id], tuple(p.task_char_span))
        rows.append({"prompt_id": p.prompt_id, "model_key": model_key, "prompt_sha256": p.prompt_sha256,
                     "rendered_sha256": sha256_text(e["rendered"]),
                     "rendered_text": e["rendered"], "input_ids": e["input_ids"], "n_input_tokens": e["n_input_tokens"],
                     "rule_token_spans": e["rule_token_spans"], "rule_span_straddles": e["rule_span_straddles"],
                     "rule_span_lengths": [b - a for a, b in e["rule_token_spans"]],
                     "task_token_span": e["task_token_span"]})
    df = pd.DataFrame(rows)
    out = DATA_DIR / "tokenized" / f"{model_key}.parquet"
    write_parquet(df, out)
    n = df["n_input_tokens"]
    budget = (1 - cfg["context_safety_margin"]) * m["native_context"]
    worst = int(n.max()) + m["decoding"]["max_new_tokens"]
    report = {
        **chat.template_info(model_key), "n_prompts": len(df),
        "input_tokens_quantiles": {str(q): float(n.quantile(q)) for q in (0, .5, .9, .99, 1)},
        "max_input_plus_output": worst, "native_context": m["native_context"],
        "usable_budget_with_margin": budget, "context_gate_pass": bool(worst <= budget),
        "n_prompts_with_straddling_rule_spans": int(df["rule_span_straddles"].apply(any).sum()),
        "n_rule_spans_straddling": int(df["rule_span_straddles"].apply(sum).sum()),
        "truncation": "none (no truncation applied anywhere; every prompt tokenized in full)",
    }
    write_json(report, DATA_DIR / "tokenized" / f"{model_key}_audit.json")
    return report
