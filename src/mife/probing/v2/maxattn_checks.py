"""Validation of the V2 max-attention extraction on TRAIN examples (run before any CV).

Checks, per generator, on a fixed sample of TRAIN prompts (every k, multi-token rules, and spans whose boundary token
straddles the rule text):
  1. span alignment: the decoded tokens of each indexed span contain the exact rule text; for a non-straddling span,
     dropping its first or last token loses part of the rule text (the span is tight); for a straddling span the
     extra characters come only from the boundary token(s);
  2. max: for every layer and head, the extractor's value equals the direct maximum over the span taken from the full
     attention row (exact equality, independent NumPy code path), and (post-run) equals the stored shard value;
  3. sum: the recomputed span SUM reproduces the cached V1 sum (data/features/<model>/main attn shards) within
     SUM_ATOL (documented tolerance).
Writes results/v2/extraction/<model>_checks.json and results/v2/tables/extraction_checks.csv.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ...introspect import bits_to_float32, load_torch_model, torch_device
from .. import data
from . import maxattn
from .versioning import RESULTS_V2, write_versioned_json

SUM_ATOL = 1e-3  # documented tolerance on |sum_V2 - sum_V1| (attention probabilities, per layer x head x span)
N_PER_K = 4


def _offsets(tokenizer, r):
    enc = tokenizer(r.rendered_text, add_special_tokens=False, return_offsets_mapping=True)
    if list(enc["input_ids"]) != list(r.input_ids):
        raise AssertionError("re-tokenization of the rendered text differs from the frozen input ids")
    return enc["offset_mapping"]


def _boundary(rendered, offs, span, a, b):
    """Characters of the span's boundary tokens that fall outside the rule's character range [a, b)."""
    lo, hi = span
    return rendered[offs[lo][0]:a] + rendered[b:offs[hi - 1][1]]


def sample_ids(model_key: str, tokenizer, seed: int = 20261120) -> list[str]:
    """Fixed TRAIN sample: N_PER_K prompts per k (seeded) plus prompts whose rule span has a straddling boundary token.

    The frozen tokenization flags a span as straddling only if a boundary token covers NON-whitespace characters
    outside the rule text; no TRAIN span does (checked here). We therefore also add prompts in which a boundary token
    covers whitespace outside the rule text (e.g. a merged ' .\\n' / leading-space token), found by a seeded scan."""
    P = data.prompts("train")
    tok = maxattn._tok(model_key)
    PR = data.prompt_rules("train")
    rng = np.random.default_rng(seed)
    ids = []
    for k in range(1, 6):
        ids += list(rng.choice(P.loc[P.k == k, "prompt_id"].values, N_PER_K, replace=False))
    tr = tok.loc[P["prompt_id"]]
    n_flag = int(sum(int(np.sum(x)) for x in tr["rule_span_straddles"]))
    found = []
    pr_by = PR.set_index("prompt_id")
    for pid in rng.permutation(P["prompt_id"].values):
        r = tok.loc[pid]
        offs = _offsets(tokenizer, r)
        rr = pr_by.loc[[pid]].sort_values("position")
        prompt_text = _PT(pid)
        base = r.rendered_text.find(prompt_text)
        for (_, q), sp in zip(rr.iterrows(), r.rule_token_spans):
            if _boundary(r.rendered_text, offs, tuple(sp), base + q.char_start, base + q.char_end):
                found.append(pid)
                break
        if len(found) >= 3:
            break
    _STATS[model_key] = {"n_train_spans_flagged_nonwhitespace_straddle": n_flag}
    return ids + [x for x in found if x not in ids]


_STATS: dict = {}
_PT_CACHE: dict = {}


def _PT(pid):
    if not _PT_CACHE:
        _PT_CACHE.update(data.prompts("train").set_index("prompt_id")["prompt_text"].to_dict())
    return _PT_CACHE[pid]


def _alignment(tokenizer, r, j, rule, prompt_text) -> dict:
    """Recompute the token span of rule j from character offsets and compare it with the frozen span."""
    offs = _offsets(tokenizer, r)
    base = r.rendered_text.find(prompt_text)
    a, b = base + int(rule.char_start), base + int(rule.char_end)
    if r.rendered_text[a:b] != rule.rule_text:
        return {"span_len": None, "ok": False, "why": "rule text not at its character span"}
    idx = [i for i, (s, e) in enumerate(offs) if s < b and e > a and e > s]
    recomputed = (idx[0], idx[-1] + 1)
    lo, hi = (int(v) for v in r.rule_token_spans[j])
    decoded = tokenizer.decode(list(r.input_ids[lo:hi]))
    outside = _boundary(r.rendered_text, offs, (lo, hi), a, b)
    ok = recomputed == (lo, hi) and rule.rule_text.strip() in decoded
    return {"span_len": hi - lo, "span_equals_recomputed": recomputed == (lo, hi),
            "contains_rule_text": rule.rule_text.strip() in decoded, "straddles_flagged": bool(r.rule_span_straddles[j]),
            "boundary_outside_chars": repr(outside), "straddles_whitespace": bool(outside) and not outside.strip(),
            "ok": bool(ok)}


def run(model_key: str, post_run: bool = False) -> dict:
    import torch
    from transformers import AutoTokenizer
    from ...generate_mlx import model_path
    device = torch_device()
    assert device == "mps"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path(model_key)))
    ids = sample_ids(model_key, tokenizer)
    data.assert_allowed(ids, "train")
    tok = maxattn._tok(model_key)
    PR = data.prompt_rules("train").set_index(["prompt_id", "position"])
    S_v1, _, K_v1 = data.rule_attention(model_key, ids, "train")  # cached V1 sums, slots in prompt order
    if post_run:
        A_stored, K_stored = maxattn.load(model_key, ids)
    t0 = time.time()
    model = load_torch_model(model_key, device)
    rows, worst_sum, max_ok, stored_ok, n_spans_multi = [], 0.0, True, True, 0
    for i, pid in enumerate(ids):
        r = tok.loc[pid]
        att = maxattn.forward_rows(model, torch.tensor([list(r.input_ids)], device=device))
        spans = [tuple(int(v) for v in x) for x in r.rule_token_spans]
        mx, sm = maxattn.span_stats(att, spans)
        mx = bits_to_float32(maxattn._bits(mx))
        sm = sm.cpu().numpy()
        full = att.float().cpu().numpy()  # (L, H, seq), exact upcast of BF16
        for j, (a, b) in enumerate(spans):
            direct = full[:, :, a:b].max(-1)  # independent NumPy path
            eq = bool(np.array_equal(direct, mx[:, :, j]))
            max_ok &= eq
            d_sum = float(np.abs(sm[:, :, j] - S_v1[i, :, :, j]).max())
            worst_sum = max(worst_sum, d_sum)
            st_eq = None
            if post_run:
                st_eq = bool(np.array_equal(A_stored[i, :, :, j], mx[:, :, j])) and bool(K_stored[i, j])
                stored_ok &= st_eq
            al = _alignment(tokenizer, r, j, PR.loc[(pid, j + 1)], _PT(pid))
            n_spans_multi += al["span_len"] > 1
            rows.append({"model": model_key, "prompt_id": pid, "k": len(spans), "position": j + 1,
                         "type_id": PR.loc[(pid, j + 1), "type_id"], **al, "max_equals_direct": eq,
                         "stored_equals_direct": st_eq, "sum_max_abs_diff_vs_v1": d_sum,
                         "v1_sum_range": [float(S_v1[i, :, :, j].min()), float(S_v1[i, :, :, j].max())]})
        assert int(K_v1[i].sum()) == len(spans)
        del att
    df = pd.DataFrame(rows)
    res = {"model": model_key, "n_prompts": len(ids), "n_spans": len(df), "n_multi_token_spans": int(n_spans_multi),
           "n_straddling_spans_whitespace": int(df.straddles_whitespace.sum()),
           "n_straddling_spans_flagged": int(df.straddles_flagged.sum()), **_STATS.get(model_key, {}),
           "alignment_all_ok": bool(df.ok.all()),
           "alignment_failures": df.loc[~df.ok, ["prompt_id", "position"]].to_dict("records"),
           "max_equals_direct_all": bool(max_ok), "stored_equals_direct_all": (bool(stored_ok) if post_run else None),
           "sum_tolerance_abs": SUM_ATOL, "sum_max_abs_diff_vs_v1": worst_sum, "sum_within_tolerance": worst_sum <= SUM_ATOL,
           "wall_s": time.time() - t0, "post_run": post_run, "sample_ids": ids}
    out = RESULTS_V2 / "extraction"
    out.mkdir(parents=True, exist_ok=True)
    tag = "post" if post_run else "pre"
    write_versioned_json(res, out / f"{model_key}_checks_{tag}.json")
    (RESULTS_V2 / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_V2 / "tables" / f"extraction_checks_{model_key}_{tag}.csv", index=False)
    return res
