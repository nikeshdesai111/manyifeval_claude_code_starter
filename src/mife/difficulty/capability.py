"""Stage 4: can the three study models represent and produce emoji, hashtags and leading-space/tab indentation?

1. Tokenizer round-trip at the K10 revisions: encode -> decode for test strings, both with the tokenizer's default
   decode (what K10's generate.py uses: tokz.decode(ids)) and with clean_up_tokenization_spaces=False; then run every
   checker on both decodings and report any checker whose verdict differs.
2. K10 code audit is reported in the summary (see README notes in the report).
3. Local generations (MLX, BF16 weights from the HF cache, K10 decoding: temperature 0.6, top-p 0.9) for a few
   emoji / hashtag / indentation prompts per model, checked with the difficulty checkers.

Usage: PYTHONPATH=src:.pydeps python -m mife.difficulty.capability {tokenizer,generate} [--models ...]
"""
from __future__ import annotations

import argparse
import json
import random
import time

from . import catalog as K
from . import checkers as C

OUT = K.ROOT / "data" / "difficulty" / "capability"
MODELS = {
    "gemma2b": ("google/gemma-2-2b-it", "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"),
    "llama": ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659"),
    "gemma9b": ("google/gemma-2-9b-it", "11c9b309abf73637e4b6f9a3fa1e92e615547819"),
}
STRINGS = {
    "emoji_basic": "Pack early 😀. Then relax! 🎉 Done? 🤔",
    "emoji_skin_zwj": "Great job 👍🏽 team 👨‍👩‍👧‍👦 and love ❤️ plus 1️⃣ keycap 🇺🇸 flag.",
    "hashtags": "Plan ahead #Budget #SaveMore #Tip2 and #move-day #mom's",
    "leading_spaces": "Start\n  two\n    four\n      six\n        eight",
    "tabs": "\tone tab\n\t\ttwo tabs\n\t  tab+2",
    "leading_space_first": "   three leading spaces",
    "space_before_punct": "Wait , really ? Yes ! It 's fine . Don 't worry",
    "section_markup": "SECTION 1\n*Plan* ( [ { nested } ] )\n\nSECTION 2\n**Done** .",
    "curly": "It’s “fine” — really…",
}


def _checker_suite() -> list[tuple[str, dict]]:
    rng = random.Random(0)
    out = []
    for t in K.types():
        for lvl in K.LEVELS:
            out.append((t, K.instantiate(t, lvl, rng, "train", set())))
    return out


def tokenizer_check() -> dict:
    from transformers import AutoTokenizer
    res = {}
    suite = _checker_suite()
    for key, (hf, rev) in MODELS.items():
        tok = AutoTokenizer.from_pretrained(hf, revision=rev)
        rows = {}
        for name, s in STRINGS.items():
            ids = tok.encode(s, add_special_tokens=False)
            d_default = tok.decode(ids)
            d_raw = tok.decode(ids, clean_up_tokenization_spaces=False)
            diffs = [f"{t}:{json.dumps(kw)}" for t, kw in suite
                     if C.check(t, d_default, kw).passed != C.check(t, d_raw, kw).passed]
            rows[name] = {"n_tokens": len(ids), "roundtrip_default": d_default == s, "roundtrip_no_cleanup": d_raw == s,
                          "default_decode": d_default if d_default != s else None, "checker_verdict_diffs": diffs,
                          "unk": tok.unk_token_id is not None and tok.unk_token_id in ids}
        # generation-like: response ids taken from a rendered chat (prompt + assistant text), then decoded alone
        msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": STRINGS["leading_spaces"]}]
        full = tok.apply_chat_template(msgs, tokenize=False)
        res[key] = {"revision": rev, "clean_up_tokenization_spaces_default": bool(tok.clean_up_tokenization_spaces),
                    "strings": rows,
                    "chat_template_keeps_leading_ws_in_assistant": STRINGS["leading_spaces"] in full}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tokenizer_roundtrip.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


GEN_PROMPTS = [
    ("emoji_ends", {"scope": "every_sentence"},
     "My electricity bill rose this month after I started working from home. Why might that be?"),
    ("hashtags", {"n": 3}, "I have two weeks before my exams. How should I plan my study time?"),
    ("stair_indent", {"lines": 3}, "Which should I fix first at home: a leaking tap or a broken door lock?"),
]


def generate(models: list[str], samples: int, max_tokens: int) -> list[dict]:
    from huggingface_hub import snapshot_download
    from mlx_lm import generate as mgen
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler
    rows = []
    for key in models:
        hf, rev = MODELS[key]
        path = snapshot_download(hf, revision=rev, local_files_only=True)
        t0 = time.time()
        model, tok = load(path)
        load_s = time.time() - t0
        for t, kw, task in GEN_PROMPTS:
            prompt_text = f"{task}\n\n{K.rule_text(t, kw)}"
            ids = tok.apply_chat_template([{"role": "user", "content": prompt_text}], add_generation_prompt=True)
            for s in range(samples):
                import mlx.core as mx
                mx.random.seed(1000 + s)
                t1 = time.time()
                out = mgen(model, tok, ids, max_tokens=max_tokens, sampler=make_sampler(temp=0.6, top_p=0.9))
                el = time.time() - t1
                r = C.check(t, out, kw)
                rows.append({"model": key, "type_id": t, "kwargs": kw, "sample": s, "passed": r.passed,
                             "detail": r.detail, "seconds": round(el, 1), "load_s": round(load_s, 1),
                             "response": out})
                print(key, t, s, r.passed, r.detail, f"{el:.0f}s", flush=True)
        del model
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "generations.jsonl").open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["tokenizer", "generate"])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=400)
    a = ap.parse_args()
    if a.step == "tokenizer":
        print(json.dumps(tokenizer_check(), indent=1, ensure_ascii=False))
    else:
        generate(a.models, a.samples, a.max_tokens)


if __name__ == "__main__":
    main()
