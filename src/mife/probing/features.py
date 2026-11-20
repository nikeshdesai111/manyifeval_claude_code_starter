"""Feature blocks for the probing phase, cached under data/probing_features/.

Blocks (names used by configs.py):
  S1   structure, Objective 1 row: k, target type, position, typed target parameters, companion types and parameters,
       task family, prompt/task lengths. Exact rule IDs are never inputs.
  S2   structure, Objective 2 prompt: k, ordered 5-slot types + typed parameters + presence mask, type composition,
       task family, lengths.
  M2   H1/H2 Objective 2 metadata: k, ordered 5-slot rule types + presence mask, type composition (no exact-rule info).
  EP   frozen BGE-M3 full-prompt embedding (1,024).        ER   frozen BGE-M3 exact target-rule embedding (1,024).
  ER5  ordered frozen BGE-M3 embeddings of all rules, 5 zero-padded slots (5 x 1,024).
  H    generating-LLM final-prompt-token hidden state at one candidate row (4,096).
  A    final-token attention to the target rule span, all layers x heads, SUM or MEAN (Objective 1).
  A5   the same for all 5 slots in prompt order, absent slots zero, plus the presence mask (Objective 2).
Row orders: prompt-level arrays follow `data.prompts(split)`; Objective 1 arrays follow `data.prompt_rules(split)`.
Every cache file has a sidecar `*_keys.parquet` that is checked on load.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import load_config
from ..storage import write_json
from . import data

TYPES = [
    "change_case:capital_word_frequency", "change_case:english_capital", "change_case:english_lowercase",
    "detectable_content:number_placeholders", "detectable_format:number_bullet_lists", "detectable_format:title",
    "keywords:existence", "keywords:forbidden_words", "keywords:frequency", "keywords:letter_frequency",
    "length_constraints:number_paragraphs", "length_constraints:number_sentences",
    "length_constraints:number_words", "punctuation:no_comma", "startend:quotation"]
T_IX = {t: i for i, t in enumerate(TYPES)}
FAMILIES = ["ANALYTICAL", "CREATIVE", "INFO", "PERSUASIVE", "PRACTICAL", "TRANSFORM"]
NUMERIC = [("change_case:capital_word_frequency", "capital_frequency"),
           ("detectable_content:number_placeholders", "num_placeholders"),
           ("detectable_format:number_bullet_lists", "num_bullets"), ("keywords:frequency", "frequency"),
           ("keywords:letter_frequency", "let_frequency"), ("length_constraints:number_paragraphs", "num_paragraphs"),
           ("length_constraints:number_sentences", "num_sentences"), ("length_constraints:number_words", "num_words")]
RELATION = [("change_case:capital_word_frequency", "capital_relation"), ("keywords:frequency", "relation"),
            ("keywords:letter_frequency", "let_relation"), ("length_constraints:number_sentences", "relation"),
            ("length_constraints:number_words", "relation")]
PARAM_COLS = [f"param_log1p:{t.split(':')[1]}" for t, _ in NUMERIC] + [f"rel_lessthan:{t.split(':')[1]}" for t, _ in RELATION]


def cache_dir(model: str | None = None):
    d = data.PROBE_FEAT / (model or "shared")
    d.mkdir(parents=True, exist_ok=True)
    return d


def param_block(type_id: str, kwargs_json: str) -> np.ndarray:
    """Typed parameters of one exact rule: log1p of its numeric value in its type's column; relation as +1
    ('less than') / -1 ('at least') in its type's column; zeros elsewhere (types are distinct within a prompt)."""
    kw = json.loads(kwargs_json)
    v = np.zeros(len(PARAM_COLS), dtype=np.float32)
    for j, (t, key) in enumerate(NUMERIC):
        if t == type_id:
            v[j] = np.log1p(float(kw[key]))
    for j, (t, key) in enumerate(RELATION):
        if t == type_id:
            rel = kw[key]
            if rel not in ("less than", "at least"):
                raise ValueError(f"unexpected relation {rel!r} for {type_id}")
            v[len(NUMERIC) + j] = 1.0 if rel == "less than" else -1.0
    return v


def _onehot(ix, n):
    o = np.zeros((len(ix), n), dtype=np.float32)
    o[np.arange(len(ix)), ix] = 1.0
    return o


def _prompt_level(split: str):
    p = data.prompts(split)
    pr = data.prompt_rules(split)
    pr = pr.assign(pb=[param_block(t, k) for t, k in zip(pr.type_id, pr.kwargs_json)])
    return p, pr


def _lengths(p: pd.DataFrame) -> np.ndarray:
    from . import encoder
    return np.stack([encoder.token_lengths(p["prompt_text"].tolist()), p["prompt_text"].str.len().values,
                     p["task_text"].str.len().values], axis=1).astype(np.float32)


def build_structure(split: str = "train") -> None:
    p, pr = _prompt_level(split)
    pid = {x: i for i, x in enumerate(p["prompt_id"])}
    npm = len(p)
    k = p["k"].values
    fam = _onehot(p["task_family"].map(FAMILIES.index).values, len(FAMILIES))
    lens = _lengths(p)
    # per-prompt slot tables
    slot_type = np.full((npm, 5), -1, dtype=np.int64)
    slot_pb = np.zeros((npm, 5, len(PARAM_COLS)), dtype=np.float32)
    pi = pr["prompt_id"].map(pid).values
    si = pr["position"].values - 1
    slot_type[pi, si] = pr["type_id"].map(T_IX).values
    slot_pb[pi, si] = np.stack(pr["pb"].values)
    present = (slot_type >= 0).astype(np.float32)
    slot_oh = np.zeros((npm, 5, len(TYPES)), dtype=np.float32)
    ii, jj = np.nonzero(slot_type >= 0)
    slot_oh[ii, jj, slot_type[ii, jj]] = 1.0
    comp = slot_oh.sum(1)
    k_oh = _onehot(k - 1, 5)
    # ---- S2 / M2 (prompt level)
    S2 = np.concatenate([k[:, None].astype(np.float32), k_oh, slot_oh.reshape(npm, -1), slot_pb.reshape(npm, -1),
                         present, comp, fam, lens], axis=1)
    c_slot = [f"slot{s + 1}_type:{t}" for s in range(5) for t in TYPES]
    c_pb = [f"slot{s + 1}_{c}" for s in range(5) for c in PARAM_COLS]
    S2_cols = (["k"] + [f"k={i}" for i in range(1, 6)] + c_slot + c_pb + [f"slot{s + 1}_present" for s in range(5)]
               + [f"has:{t}" for t in TYPES] + [f"family:{f}" for f in FAMILIES] + ["prompt_tokens_bge", "prompt_chars", "task_chars"])
    M2 = np.concatenate([k[:, None].astype(np.float32), k_oh, slot_oh.reshape(npm, -1), present, comp], axis=1)
    M2_cols = ["k"] + [f"k={i}" for i in range(1, 6)] + c_slot + [f"slot{s + 1}_present" for s in range(5)] + [f"has:{t}" for t in TYPES]
    # ---- S1 (prompt-rule row)
    t_oh = _onehot(pr["type_id"].map(T_IX).values, len(TYPES))
    pos = pr["position"].values.astype(np.float32)
    kk = k[pi].astype(np.float32)
    own_pb = np.stack(pr["pb"].values)
    comp_types = comp[pi] - t_oh
    comp_pb = slot_pb[pi].sum(1) - own_pb
    S1 = np.concatenate([kk[:, None], k_oh[pi], t_oh, pos[:, None], (pos / kk)[:, None], (pos == 1)[:, None].astype(np.float32),
                         (pos == kk)[:, None].astype(np.float32), own_pb, comp_types, comp_pb, fam[pi], lens[pi]], axis=1)
    S1_cols = (["k"] + [f"k={i}" for i in range(1, 6)] + [f"type:{t}" for t in TYPES]
               + ["position", "position_over_k", "is_first", "is_last"] + [f"target_{c}" for c in PARAM_COLS]
               + [f"companion:{t}" for t in TYPES] + [f"companion_{c}" for c in PARAM_COLS]
               + [f"family:{f}" for f in FAMILIES] + ["prompt_tokens_bge", "prompt_chars", "task_chars"])
    d = cache_dir()
    for name, X, cols, keys in (("S1", S1, S1_cols, pr[["prompt_id", "rule_id", "position", "type_id"]]),
                                ("S2", S2, S2_cols, p[["prompt_id"]]), ("M2", M2, M2_cols, p[["prompt_id"]])):
        assert X.shape[1] == len(cols), (name, X.shape, len(cols))
        np.save(d / f"{name}_{split}.npy", X)
        keys.reset_index(drop=True).to_parquet(d / f"{name}_{split}_keys.parquet")
        write_json({"columns": cols}, d / f"{name}_columns.json")
    np.save(d / f"slot_type_{split}.npy", slot_type)


def build_bge(split: str = "train") -> None:
    from . import encoder
    p, pr = _prompt_level(split)
    d = cache_dir()
    rules = pr.drop_duplicates("rule_id")[["rule_id", "rule_text"]].reset_index(drop=True)
    if not (d / f"EP_{split}.npy").exists():
        model = encoder.load_model()
        np.save(d / f"ER_rules_{split}.npy", encoder.encode(rules["rule_text"].tolist(), model))
        rules.to_parquet(d / f"ER_rules_{split}_keys.parquet")
        EP = encoder.encode(p["prompt_text"].tolist(), model, batch_size=64)
        p[["prompt_id"]].to_parquet(d / f"EP_{split}_keys.parquet")
        np.save(d / f"EP_{split}.npy", EP)


def build_llm(model: str, split: str = "train") -> None:
    """Cache the candidate hidden rows and the rule-span attention for `split` (one pass over the shards)."""
    p = data.prompts(split)
    ids = p["prompt_id"].tolist()
    d = cache_dir(model)
    rows = load_config()["probing"]["hidden_rows"][model]
    todo = [r for r in rows if not (d / f"H{r}_{split}.npy").exists()]
    if todo:
        H = data.hidden_states(model, ids, todo, split)
        for r in todo:
            np.save(d / f"H{r}_{split}.npy", H[r])
    if not (d / f"Asum_{split}.npy").exists():
        S, M, K = data.rule_attention(model, ids, split)
        assert (K.sum(1) == p["k"].values).all()
        np.save(d / f"Asum_{split}.npy", np.nan_to_num(S, nan=0.0))
        np.save(d / f"Amean_{split}.npy", np.nan_to_num(M, nan=0.0))
        np.save(d / f"Amask_{split}.npy", K)
    p[["prompt_id"]].to_parquet(d / f"prompt_keys_{split}.parquet")


# ------------------------------------------------------------------------------------------------- loading
def _check_keys(path, expect: pd.Series):
    got = pd.read_parquet(path)["prompt_id"]
    if len(got) != len(expect) or not (got.values == expect.values).all():
        raise RuntimeError(f"{path}: cached row order does not match the current {expect.name or 'prompt'} order")


def load_prompt_block(name: str, split: str = "train", model: str | None = None, mmap: bool = True) -> np.ndarray:
    """Prompt-level block in data.prompts(split) order: S2, M2, EP, ER5, H<row>, A5sum/A5mean."""
    p = data.prompts(split)
    if name in ("S2", "M2", "EP"):
        d = cache_dir()
        _check_keys(d / f"{name}_{split}_keys.parquet", p["prompt_id"])
        return np.load(d / f"{name}_{split}.npy", mmap_mode="r" if mmap else None)
    if name == "ER5":
        d = cache_dir()
        er, rk = np.load(d / f"ER_rules_{split}.npy"), pd.read_parquet(d / f"ER_rules_{split}_keys.parquet")
        rix = {r: i for i, r in enumerate(rk["rule_id"])}
        out = np.zeros((len(p), 5, er.shape[1]), dtype=np.float32)
        pid = {x: i for i, x in enumerate(p["prompt_id"])}
        pr = data.prompt_rules(split)
        out[pr["prompt_id"].map(pid).values, pr["position"].values - 1] = er[pr["rule_id"].map(rix).values]
        return out.reshape(len(p), -1)
    d = cache_dir(model)
    _check_keys(d / f"prompt_keys_{split}.parquet", p["prompt_id"])
    if name.startswith("H"):
        return np.load(d / f"{name}_{split}.npy", mmap_mode="r" if mmap else None)
    if name in ("A5sum", "A5mean"):
        A = np.load(d / f"A{name[2:]}_{split}.npy", mmap_mode="r")  # (n, L, H, 5)
        mask = np.load(d / f"Amask_{split}.npy").astype(np.float32)
        a = np.ascontiguousarray(np.moveaxis(np.asarray(A), 3, 1)).reshape(len(p), -1)  # slot-major
        return np.concatenate([a, mask], axis=1)
    raise KeyError(name)


def load_rule_block(name: str, split: str = "train", model: str | None = None) -> np.ndarray:
    """Objective 1 block in data.prompt_rules(split) order: S1, ER, EP, H<row>, Asum/Amean."""
    pr = data.prompt_rules(split)
    if name == "S1":
        d = cache_dir()
        k = pd.read_parquet(d / f"S1_{split}_keys.parquet")
        if not ((k["prompt_id"].values == pr["prompt_id"].values).all() and (k["rule_id"].values == pr["rule_id"].values).all()):
            raise RuntimeError("S1 cache row order does not match prompt_rules")
        return np.load(d / f"S1_{split}.npy")
    if name == "ER":
        d = cache_dir()
        er, rk = np.load(d / f"ER_rules_{split}.npy"), pd.read_parquet(d / f"ER_rules_{split}_keys.parquet")
        return er[pr["rule_id"].map({r: i for i, r in enumerate(rk["rule_id"])}).values]
    p = data.prompts(split)
    pix = pr["prompt_id"].map({x: i for i, x in enumerate(p["prompt_id"])}).values
    if name in ("EP",) or name.startswith("H"):
        return np.asarray(load_prompt_block(name, split, model))[pix]
    if name in ("Asum", "Amean"):
        d = cache_dir(model)
        _check_keys(d / f"prompt_keys_{split}.parquet", p["prompt_id"])
        A = np.load(d / f"{name}_{split}.npy", mmap_mode="r")
        slot = pr["position"].values - 1
        out = np.empty((len(pr), A.shape[1] * A.shape[2]), dtype=np.float32)
        for s in range(5):
            m = slot == s
            out[m] = np.asarray(A[pix[m], :, :, s]).reshape(m.sum(), -1)
        return out
    raise KeyError(name)
