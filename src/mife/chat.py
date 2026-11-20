"""Model-specific chat rendering, tokenization and exact rule-token-span alignment."""
from __future__ import annotations

import hashlib
from functools import lru_cache

from .config import load_config


@lru_cache(maxsize=None)
def tokenizer(model_key: str):
    from transformers import AutoTokenizer

    m = load_config()["models"][model_key]
    return AutoTokenizer.from_pretrained(m["hf_id"], revision=m["revision"], local_files_only=True)


def template_info(model_key: str) -> dict:
    tok = tokenizer(model_key)
    tmpl = tok.chat_template if isinstance(tok.chat_template, str) else str(tok.chat_template)
    m = load_config()["models"][model_key]
    return {"model_key": model_key, "hf_id": m["hf_id"], "revision": m["revision"],
            "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest(),
            "chat_template_kwargs": m["chat_template_kwargs"],
            "tokenizer_class": type(tok).__name__, "vocab_size": len(tok)}


def render(model_key: str, prompt_text: str) -> str:
    m = load_config()["models"][model_key]
    return tokenizer(model_key).apply_chat_template(
        [{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True,
        **m["chat_template_kwargs"])


def encode(model_key: str, prompt_text: str, rule_char_spans: list[tuple[int, int]],
           task_char_span: tuple[int, int]) -> dict:
    """Render, tokenize and map prompt-relative character spans to token index spans.

    A token belongs to a span iff its character range overlaps the span. Tokens that also
    extend outside the span (e.g. a merged '.\\n') are flagged as straddling.
    """
    tok = tokenizer(model_key)
    rendered = render(model_key, prompt_text)
    base = rendered.find(prompt_text)
    assert base >= 0 and rendered.find(prompt_text, base + 1) < 0, "user content not uniquely located"
    enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    m = load_config()["models"][model_key]
    ref = tok.apply_chat_template([{"role": "user", "content": prompt_text}], tokenize=True,
                                  add_generation_prompt=True, **m["chat_template_kwargs"])
    if not isinstance(ref, list):
        ref = ref["input_ids"]
    assert list(ids) == list(ref), "offset tokenization differs from chat-template tokenization"

    def to_tokens(a, b):
        a, b = a + base, b + base
        idx = [i for i, (s, e) in enumerate(offs) if s < b and e > a and e > s]
        assert idx, "empty token span"
        lo, hi = idx[0], idx[-1] + 1
        assert idx == list(range(lo, hi))
        # straddling = a boundary token also covers NON-whitespace characters outside the span
        outside = rendered[offs[lo][0]:a] + rendered[b:offs[hi - 1][1]]
        straddle = bool(outside.strip())
        return lo, hi, straddle

    spans = [to_tokens(a, b) for a, b in rule_char_spans]
    for (l1, h1, _), (l2, h2, _) in zip(spans, spans[1:]):
        assert h1 <= l2, "rule token spans overlap"
    t = to_tokens(*task_char_span)
    return {"rendered": rendered, "input_ids": list(ids), "n_input_tokens": len(ids),
            "rule_token_spans": [[lo, hi] for lo, hi, _ in spans],
            "rule_span_straddles": [s for _, _, s in spans],
            "task_token_span": [t[0], t[1]], "user_content_char_offset": base}
