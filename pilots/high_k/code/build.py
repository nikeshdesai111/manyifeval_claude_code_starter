"""Build the high-k pilot prompts, per-model tokenization and worklists (pilots/high_k/PRESPEC.md).

    PYTHONPATH=src python pilots/high_k/code/build.py design                         # Mac, repo root
    PYTHONPATH=src python pilots/high_k/code/build.py tokenize --model gemma2b --samples 5   # then llama, gemma9b
Reads only the pinned ManyIFEval source, the frozen rule pools (data/manifest/rule_pools.jsonl) and the S3 split/fold
files. Reads no response, label or feature. Writes nothing outside pilots/high_k/data/inputs/.

Design: all 72 groups (S3 split, 50 TRAIN / 22 TEST); each group keeps its 6 frozen exact rules and gains 4 new code-default rules so that the pool holds
10 pairwise-compatible types (the ManyIFEval maximum: 15 types minus the 3 mutually exclusive groups). Ten cyclic
chains of a random 10-order give nested prefixes k = 1..9 ("sample 10, choose 9", the analogue of 6/5), plus the full
pool at k = 10. Orderings (code/coa.py), k(k-1) per subset (1, 2, 6, 12, 20, 30, 42, 56, 72, 90): exact strength-two
COAs at k <= 5 (the main design's templates) and k = 7, 8, 9 (AGL(1, q)); at k = 6 and 10, where no such COA exists,
k - 1 stacked Williams squares with frozen relabelings (exact position and immediate-neighbour balance, near COA).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys

import pandas as pd

from mife import design, feasibility, rule_sampler
from mife import source as S
from mife.config import ROOT, derive_seed, load_config

import coa as COA  # pilots/high_k/code/coa.py (script directory is on sys.path)

PILOT = ROOT / "pilots" / "high_k"
OUT = PILOT / "data" / "inputs"
RUN_NAME = "pilot_high_k_20261126"
VERSION = "pilot-high-k-2026-11-26"
SEED = 2026112610
N_POOL = 10
N_CHUNKS = 48


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def select_groups(pools: dict) -> tuple[list[str], dict[str, str]]:
    """All 72 groups, with the adopted S3 split (50 TRAIN / 22 TEST). The manifest's own split column is the old split."""
    train = set(json.loads((ROOT / "results/split_s3/proposed_split.json").read_text())["train_groups"])
    folds = json.loads((ROOT / "results/cv_s3/folds.json").read_text())["folds"]
    assert set().union(*map(set, folds)) == train and len(train) == 50
    groups = sorted(pools)
    assert len(groups) == 72
    split = {g: ("train" if g in train else "test") for g in groups}
    assert sum(v == "test" for v in split.values()) == 22
    return groups, split


def balance_open_slots(groups: list[str], pools: dict) -> dict[str, dict[int, str]]:
    """For each exclusive set a group's frozen pool leaves open, pick the member used by the fewest chosen groups so far
    (frozen uses counted first; ties broken by a seeded draw). Evens out coverage of the 8 non-universal types."""
    sets = [sorted(g) for g in S.EXCLUSIVE_GROUPS]
    count = {t: 0 for s in sets for t in s}
    for g in groups:
        for t in {r["type_id"] for r in pools[g]["rules"]}:
            if t in count:
                count[t] += 1
    rng = random.Random(derive_seed(SEED, "balance"))
    out = {}
    for g in groups:
        have = {r["type_id"] for r in pools[g]["rules"]}
        out[g] = {}
        for i, s in enumerate(sets):
            if not (set(s) & have):
                low = min(count[t] for t in s)
                pick = rng.choice([t for t in s if count[t] == low])
                count[pick] += 1
                out[g][i] = pick
    return out


def extend_pool(pool: dict, task_texts: dict[int, str], preferred: dict[int, str] | None = None) -> dict:
    """Add 4 code-default rules to a frozen 6-rule pool: 10 pairwise-compatible types, jointly feasible for all 3 tasks.

    `preferred` maps an open exclusive set to the member chosen by balance_open_slots; it is used for the first 10 attempts
    and replaced by a seeded draw only if that attempt cannot be made feasible (logged)."""
    rng = random.Random(derive_seed(SEED, "extend", pool["group_id"]))
    base = [(r["type_id"], r["kwargs"]) for r in pool["rules"]]
    have = [t for t, _ in base]
    universal = [t for t in S.TYPES if not any(t in g for g in S.EXCLUSIVE_GROUPS)]
    need_univ = [t for t in universal if t not in have]
    sets = [sorted(g) for g in S.EXCLUSIVE_GROUPS]
    open_idx = [i for i, g in enumerate(sets) if not (set(g) & set(have))]
    preferred = preferred or {}
    log = []
    for _attempt in range(50):
        new_types = list(need_univ) + [preferred[i] if (_attempt < 10 and i in preferred) else rng.choice(sets[i])
                                       for i in open_idx]
        assert len(have) + len(new_types) == N_POOL, (have, new_types)
        rng.shuffle(new_types)
        chosen = list(base)
        ok = True
        for t in new_types:
            for _try in range(rule_sampler.MAX_PARAM_TRIES):
                kw = rule_sampler.sample_params(t, rng)
                try:
                    res = feasibility.check_pool(chosen + [(t, kw)], seed=rng.randrange(1 << 30))
                except feasibility.UnresolvedFeasibility as e:
                    log.append({"event": "params_unresolved", "type": t, "kwargs": kw, "detail": str(e)[:200]})
                    continue
                if res["feasible"]:
                    chosen.append((t, kw))
                    log.append({"event": "accepted", "type": t, "kwargs": kw})
                    break
                log.append({"event": "params_rejected", "type": t, "kwargs": kw, "proofs": res["proofs"]})
            else:
                ok = False
                log.append({"event": "type_exhausted_restart", "type": t})
                break
        if not ok:
            continue
        final = feasibility.check_pool(chosen, seed=rng.randrange(1 << 30))
        bad = {k: v for k, v in feasibility.check_tasks(chosen, final["witness"], task_texts).items() if any(v)}
        if final["feasible"] and not bad:
            break
        log.append({"event": "final_check_failed_restart", "tasks": sorted(bad)})
    else:
        raise RuntimeError(f"{pool['group_id']}: could not extend pool")
    types = [t for t, _ in chosen]
    assert len(set(types)) == N_POOL and not any(S.conflicts(a, b) for i, a in enumerate(types) for b in types[i + 1:])
    gid = pool["group_id"]
    rules = [dict(r, origin="frozen") for r in pool["rules"]]
    for j, (t, kw) in enumerate(chosen[len(base):]):
        rules.append({"type_id": t, "kwargs": kw, "rule_text": S.render_rule_text(t, kw), "category": S.category_of(t),
                      "code_prefix": S.code_prefix_of(t), "exact_rule_hash": rule_sampler.rule_hash(t, kw),
                      "rule_id": f"{gid}-x{j + 1}", "origin": "new"})
    return {"group_id": gid, "task_keys": pool["task_keys"], "rules": rules, "witness": final["witness"],
            "extension_log": log}


def configurations(order: list[str], rng: random.Random) -> list[dict]:
    """10 cyclic chains, prefixes k=1..9, plus the full pool at k=10; exact strength-two COA orderings per subset."""
    n = len(order)
    subsets = [(c + 1, k, [order[(c + i) % n] for i in range(k)]) for c in range(n) for k in range(1, n)]
    subsets.append((0, n, list(order)))
    out = []
    for chain, k, sub in subsets:
        for row, sigma in enumerate(COA.coa_rows(k)):  # as in the main design: position p gets chain element sigma[p]
            out.append({"chain_index": chain, "k": k, "subset": sub, "coa_row": row,
                        "ordered": [sub[s] for s in sigma]})
    return out


def williams_rows(k: int) -> list[tuple[int, ...]]:
    """Orderings for the extra TEST subsets (Nicky 2026-11-26 18:59): 2k rows per subset with exact position balance
    (each rule twice per position) and exact immediate-neighbour balance (each ordered pair adjacent twice).
    Odd k: a Williams square plus its mirror. Even k: a Williams square plus its negated-label twin (the mirror would
    duplicate rows at even k). k = 2 and 3 have only 2 and 6 distinct orderings, which equal the full COA."""
    if k <= 3:
        return [tuple(r) for r in COA.coa_rows(k)]
    base = [0]
    for j in range(1, k):
        base.append((j + 1) // 2 if j % 2 else (k - j // 2) % k)
    sq = [tuple((b + i) % k for b in base) for i in range(k)]
    twin = [tuple(r[::-1]) for r in sq] if k % 2 else [tuple((-b + i) % k for b in base) for i in range(k)]
    rows = sq + twin
    assert len(set(rows)) == 2 * k
    for p in range(k):
        assert sorted(r[p] for r in rows) == sorted(list(range(k)) * 2)
    adj = {}
    for r in rows:
        for a, b in zip(r, r[1:]):
            adj[(a, b)] = adj.get((a, b), 0) + 1
    assert len(adj) == k * (k - 1) and set(adj.values()) == {2}
    return rows


def test_expansion(order: list[str], mode: str) -> list[dict]:
    """TEST only, mirroring the original's expansion (src/mife/expansion_s3.py): every size-k subset of the pool that
    no chain already holds, k = 2..8 (k = 1, 9 and 10 are already complete), so all C(10, k) subsets exist at every k.
    As in the original, a new subset's base order is the chain-order rank and labels A..J follow the chain order.
    mode "williams" (adopted): 2k balanced orderings (williams_rows); "full": the same k(k-1) orderings as every chain
    subset; "one": COA row 0 only."""
    from itertools import combinations
    n = len(order)
    chain_sets = {frozenset(order[(c + i) % n] for i in range(k)) for c in range(n) for k in range(1, n)}
    label = {r: "ABCDEFGHIJ"[i] for i, r in enumerate(order)}
    out = []
    for k in range(2, n - 1):
        new = [list(sub) for sub in combinations(order, k) if frozenset(sub) not in chain_sets]  # rank order kept
        assert len(new) == math.comb(n, k) - n, (k, len(new))
        rows = {"full": COA.coa_rows(k), "one": COA.coa_rows(k)[:1], "williams": williams_rows(k)}[mode]
        for sub in new:
            lab = "".join(label[r] for r in sub)
            for row, sigma in enumerate(rows):
                out.append({"chain_index": -1, "k": k, "subset": sub, "coa_row": row, "ordered": [sub[s] for s in sigma],
                            "subset_label": lab, "is_original": False})
    return out


def check_design(cfgs: list[dict], order: list[str]) -> None:
    n = len(order)
    for k in range(1, 6):  # identical templates to the frozen main design at k <= 5
        assert [tuple(r) for r in design.coa_template(k)] == COA.coa_rows(k)
    for k in range(1, n + 1):
        COA.check(COA.coa_rows(k), k)
    by = pd.DataFrame(cfgs)
    assert len(by) == sum(10 * len(COA.coa_rows(k)) for k in range(1, n)) + len(COA.coa_rows(n))
    for (c, k), d in by.groupby(["chain_index", "k"]):  # balances on the actual rule ids of every subset
        assert len({tuple(x) for x in d.ordered}) == len(d) == len(COA.coa_rows(k))
        for p in range(k):
            assert pd.Series([x[p] for x in d.ordered]).value_counts().nunique() == 1
        if k > 1:
            adj = pd.Series([(x[i], x[i + 1]) for x in d.ordered for i in range(k - 1)]).value_counts()
            assert len(adj) == k * (k - 1) and adj.nunique() == 1
        if k in COA.EXACT and k > 1:
            for p1 in range(k):
                for p2 in range(k):
                    if p1 != p2:
                        cnt = pd.Series([(x[p1], x[p2]) for x in d.ordered]).value_counts()
                        assert len(cnt) == k * (k - 1) and cnt.eq(1).all(), (c, k, p1, p2)
    for c in range(1, n + 1):  # nesting
        subs = [set(by[(by.chain_index == c) & (by.k == k)].subset.iloc[0]) for k in range(1, n)]
        assert all(a < b for a, b in zip(subs, subs[1:]))
    for k in range(1, n):  # each rule in exactly k of the 10 chains' size-k subsets
        cnt = pd.Series([r for s in by[(by.k == k) & (by.coa_row == 0)].subset for r in s]).value_counts()
        assert len(cnt) == n and cnt.eq(k).all(), (k, cnt)


def design_stage(test_orderings: str = "williams") -> None:
    """Pools, feasibility checks, prompts (no tokenizer, no worklist)."""
    OUT.mkdir(parents=True, exist_ok=True)
    pools = {json.loads(l)["group_id"]: json.loads(l) for l in (ROOT / "data/manifest/rule_pools.jsonl").open()}
    groups, split = select_groups(pools)
    TT = {k: v["task_text"] for k, v in S.tasks().items()}
    ext, prompts, rules = {}, [], []
    pref = balance_open_slots(groups, pools)
    failed = {}
    for gid in groups:  # extend every pool first; report all failures before building any prompt
        try:
            ext[gid] = extend_pool(pools[gid], {k: TT[k] for k in pools[gid]["task_keys"]}, pref[gid])
            ext[gid]["split"] = split[gid]
        except RuntimeError as err:
            failed[gid] = str(err)
    if failed:
        (OUT / "design_failures.json").write_text(json.dumps(failed, indent=1))
        raise SystemExit(f"{len(failed)} group(s) cannot take 10 compatible rules: {sorted(failed)}")
    for gid in groups:
        e = ext[gid]
        rmap = {r["rule_id"]: r for r in e["rules"]}
        rng = random.Random(derive_seed(SEED, "chain", gid))
        order = [r["rule_id"] for r in e["rules"]]
        rng.shuffle(order)
        e["chain_order"] = order
        cfgs = configurations(order, rng)
        check_design(cfgs, order)
        if split[gid] == "test":
            cfgs = [{**c, "is_original": True} for c in cfgs] + test_expansion(order, test_orderings)
        for c in cfgs:
            stem = f"{gid}-s{c['subset_label']}" if "subset_label" in c else f"{gid}-c{c['chain_index']}"
            cid = f"{stem}-k{c['k']:02d}-o{c['coa_row']:02d}"
            texts = [rmap[r]["rule_text"] for r in c["ordered"]]
            for tkey in e["task_keys"]:
                task = TT[tkey]
                text = S.serialize_prompt(task, texts)
                pos = len(task) + len(S.PROMPT_HEADER)
                spans = []
                for j, t in enumerate(texts):
                    pos += (1 if j else 0) + 2
                    spans.append((pos, pos + len(t)))
                    pos += len(t)
                assert all(text[a:b] == t for (a, b), t in zip(spans, texts)) and pos == len(text)
                pid = f"hk-{cid}-t{tkey}"
                prompts.append({"prompt_id": pid, "group_id": gid, "split": split[gid], "task_key": tkey, "config_id": cid,
                                "subset_id": f"{stem}-k{c['k']:02d}", "chain_index": c["chain_index"],
                                "is_original": c.get("is_original", True),
                                "k": c["k"], "coa_row": c["coa_row"], "ordered_rule_ids": c["ordered"],
                                "ordered_type_ids": [rmap[r]["type_id"] for r in c["ordered"]],
                                "n_new_rules": sum(rmap[r]["origin"] == "new" for r in c["ordered"]),
                                "task_text": task, "task_char_span": [0, len(task)], "prompt_text": text,
                                "prompt_sha256": sha(text), "n_chars": len(text), "version": VERSION})
                for p, ((a, b), r) in enumerate(zip(spans, c["ordered"])):
                    rr = rmap[r]
                    rules.append({"prompt_id": pid, "position": p + 1, "k": c["k"], "rule_id": r, "type_id": rr["type_id"],
                                  "category": rr["category"], "origin": rr["origin"],
                                  "kwargs_json": json.dumps(rr["kwargs"], sort_keys=True), "rule_text": rr["rule_text"],
                                  "char_start": a, "char_end": b, "exact_rule_hash": rr["exact_rule_hash"],
                                  "group_id": gid, "split": split[gid], "task_key": tkey, "version": VERSION})
    prompts, rules = pd.DataFrame(prompts), pd.DataFrame(rules)
    assert prompts.prompt_id.is_unique and prompts.prompt_sha256.is_unique
    prompts.to_parquet(OUT / "prompts.parquet")
    rules.to_parquet(OUT / "prompt_rules.parquet")
    (OUT / "pools.jsonl").write_text("".join(json.dumps(e, default=str) + "\n" for e in ext.values()))

    checks = pool_checks(ext, TT)
    checks.to_csv(OUT / "pool_checks.csv", index=False)
    summary = {"version": VERSION, "run_name": RUN_NAME, "seed": SEED, "groups": groups, "test_orderings": test_orderings,
               "test_expansion_prompts": int((~prompts.is_original).sum()),
               "subsets_per_k_test_group": prompts[prompts.split == "test"].groupby(["group_id", "k"]).subset_id.nunique()
                                           .groupby("k").agg(["min", "max"]).to_dict(),
               "split_counts": pd.Series(split).value_counts().to_dict(),
               "n_prompts": len(prompts), "n_rule_rows": len(rules),
               "prompts_per_k": prompts.k.value_counts().sort_index().to_dict(),
               "prompts_per_split": prompts.split.value_counts().to_dict(),
               "groups_per_type": pd.Series([r["type_id"] for e in ext.values() for r in e["rules"]]).value_counts().to_dict(),
               "groups_per_type_by_split": {s: pd.Series([r["type_id"] for e in ext.values() if e["split"] == s
                                                          for r in e["rules"]]).value_counts().to_dict() for s in ("train", "test")},
               "new_types": pd.Series([r["type_id"] for e in ext.values() for r in e["rules"] if r["origin"] == "new"]).value_counts().to_dict(),
               "pool_checks": {"all_pass": bool(checks["all_pass"].all()), "n_group_task_pairs": len(checks),
                               "outside_task_not_conflict_list": int(checks.n_types_outside_task_list.gt(0).sum())},
               "extension_restarts": {g: sum(x["event"].endswith("restart") for x in e["extension_log"]) for g, e in ext.items()},
               "prompts_sha256": hashlib.sha256((OUT / "prompts.parquet").read_bytes()).hexdigest(),
               "prompt_rules_sha256": hashlib.sha256((OUT / "prompt_rules.parquet").read_bytes()).hexdigest()}
    assert summary["pool_checks"]["all_pass"], checks[~checks.all_pass]
    (OUT / "design_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: summary[k] for k in ("split_counts", "n_prompts", "n_rule_rows", "prompts_per_k",
                                              "prompts_per_split", "groups_per_type", "pool_checks")}, default=str))


def pool_checks(ext: dict, TT: dict) -> pd.DataFrame:
    """Independent re-check of every 10-rule pool, per (group, task): pairwise type compatibility, no D3 contradiction
    proof, and the pool's witness passing all 10 official verifiers with that task's text (so every subset of the
    pool, i.e. every prompt, is satisfiable for that task). Also reports, for information only, how many of the pool's
    types fall outside ManyIFEval's own per-task not_conflict_rules_list (the main dataset ignored that list too)."""
    rows = []
    tasks = S.tasks()
    for gid, e in ext.items():
        pool = [(r["type_id"], r["kwargs"]) for r in e["rules"]]
        types = [t for t, _ in pool]
        pair_ok = len(set(types)) == N_POOL and not any(S.conflicts(a, b) for i, a in enumerate(types) for b in types[i + 1:])
        proofs = feasibility.proofs(pool)
        for tkey in e["task_keys"]:
            v = feasibility.check_tasks(pool, e["witness"], {tkey: TT[tkey]})[tkey]
            outside = sorted(set(types) - set(tasks[tkey]["source_type_list"]))
            rows.append({"group_id": gid, "split": e["split"], "task_key": tkey, "types": "|".join(types),
                         "pairwise_compatible": pair_ok, "n_contradiction_proofs": len(proofs),
                         "witness_violations": int(sum(v)), "n_types_outside_task_list": len(outside),
                         "types_outside_task_list": "|".join(outside),
                         "all_pass": pair_ok and not proofs and not any(v)})
    return pd.DataFrame(rows)


MODELS = __import__("yaml").safe_load((PILOT / "code" / "models.yaml").read_text())
MAX_MODEL_LEN = 4096


def hf_tokenizer(key: str, revision: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODELS[key]["hf_id"], revision=revision)


def encode_with(tok, kw: dict, prompt_text: str, rule_char_spans, task_char_span) -> dict:
    """Same rendering and span alignment as mife.chat.encode, for tokenizers outside configs/experiment.yaml."""
    msgs = [{"role": "user", "content": prompt_text}]
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    base = rendered.find(prompt_text)
    assert base >= 0 and rendered.find(prompt_text, base + 1) < 0, "user content not uniquely located"
    enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    ref = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kw)
    if not isinstance(ref, list):
        ref = ref["input_ids"]
    assert list(ids) == list(ref), "offset tokenization differs from chat-template tokenization"

    def to_tokens(a, b):
        a, b = a + base, b + base
        idx = [i for i, (s0, e0) in enumerate(offs) if s0 < b and e0 > a and e0 > s0]
        assert idx, "empty token span"
        lo, hi = idx[0], idx[-1] + 1
        assert idx == list(range(lo, hi))
        outside = rendered[offs[lo][0]:a] + rendered[b:offs[hi - 1][1]]
        return lo, hi, bool(outside.strip())

    spans = [to_tokens(a, b) for a, b in rule_char_spans]
    for (l1, h1, _), (l2, h2, _) in zip(spans, spans[1:]):
        assert h1 <= l2, "rule token spans overlap"
    t = to_tokens(*task_char_span)
    return {"rendered": rendered, "input_ids": list(ids), "n_input_tokens": len(ids),
            "rule_token_spans": [[lo, hi] for lo, hi, _ in spans], "rule_span_straddles": [s for _, _, s in spans],
            "task_token_span": [t[0], t[1]]}


def tokenize_stage(model_key: str, n_samples: int) -> None:
    """Per-model tokenization and worklist (run on the Mac after design_stage). TRAIN prompts come first in the
    worklist, so the low chunks hold TRAIN and early results need no TEST data."""
    from huggingface_hub import HfApi
    m, spec = model_key, MODELS[model_key]
    prompts = pd.read_parquet(OUT / "prompts.parquet")
    rules = pd.read_parquet(OUT / "prompt_rules.parquet")
    prompts = prompts.assign(_o=prompts.split.map({"train": 0, "test": 1})).sort_values(["_o", "prompt_id"]).drop(columns="_o")
    rev = spec["revision"] or HfApi().model_info(spec["hf_id"]).sha
    tok = hf_tokenizer(m, rev)
    summary = json.loads((OUT / "build_summary.json").read_text()) if (OUT / "build_summary.json").exists() else {}
    psha = hashlib.sha256((OUT / "prompts.parquet").read_bytes()).hexdigest()
    if summary.get("prompts_sha256") not in (None, psha):
        raise SystemExit("prompts.parquet changed since another model was tokenized; retokenize all models")
    summary.update(version=VERSION, run_name=RUN_NAME, prompts_sha256=psha, n_chunks=N_CHUNKS, max_model_len=MAX_MODEL_LEN)
    rules_by = dict(tuple(rules.groupby("prompt_id", sort=False)))
    tk = []
    for r in prompts.itertuples():
        rr = rules_by[r.prompt_id].sort_values("position")
        e = encode_with(tok, spec["chat_template_kwargs"], r.prompt_text, list(zip(rr.char_start, rr.char_end)),
                        tuple(r.task_char_span))
        tk.append({"prompt_id": r.prompt_id, "input_ids": e["input_ids"], "n_input_tokens": e["n_input_tokens"],
                   "rendered_sha256": sha(e["rendered"]), "rule_token_spans": e["rule_token_spans"],
                   "rule_span_straddles": e["rule_span_straddles"], "task_token_span": e["task_token_span"]})
    tk = pd.DataFrame(tk)
    mx = int(tk.n_input_tokens.max()) + spec["decoding"]["max_new_tokens"]
    assert mx <= min(MAX_MODEL_LEN, spec["native_context"]), mx
    tk.to_parquet(OUT / f"tokenized_{m}.parquet")
    base = load_config()["seeds"]["generation"]
    w = pd.DataFrame([{"prompt_id": p, "sample_index": s, "seed": derive_seed(base, "vllm", m, RUN_NAME, p, s) % (1 << 31)}
                      for p in prompts.prompt_id for s in range(n_samples)])
    order = {p: i for i, p in enumerate(prompts.prompt_id)}
    w["chunk"] = [order[p] * N_CHUNKS // len(order) for p in w.prompt_id]
    w.to_parquet(OUT / f"worklist_{m}.parquet")
    tmpl = tok.chat_template if isinstance(tok.chat_template, str) else str(tok.chat_template)
    kk = prompts.set_index("prompt_id").k
    sp = prompts.set_index("prompt_id").split
    summary.setdefault("models", {})[m] = {
        **spec, "revision": rev, "n_samples": n_samples, "chat_template_sha256": sha(tmpl),
        "tokenizer_class": type(tok).__name__,
        "example_rendered": tok.apply_chat_template([{"role": "user", "content": "X"}], tokenize=False,
                                                    add_generation_prompt=True, **spec["chat_template_kwargs"]),
        "max_input_tokens_by_k": tk.assign(k=tk.prompt_id.map(kk)).groupby("k").n_input_tokens.max().to_dict(),
        "records": len(w), "train_chunks": sorted(w[w.prompt_id.map(sp) == "train"].chunk.unique().tolist()),
        "worklist_sha256": hashlib.sha256((OUT / f"worklist_{m}.parquet").read_bytes()).hexdigest(),
        "tokenized_sha256": hashlib.sha256((OUT / f"tokenized_{m}.parquet").read_bytes()).hexdigest()}
    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    s = summary["models"][m]
    print(json.dumps({k: s[k] for k in ("revision", "records", "max_input_tokens_by_k", "train_chunks")}, default=str))
    print(repr(s["example_rendered"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["design", "tokenize"])
    ap.add_argument("--model")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--test-orderings", choices=["williams", "full", "one"], default="williams")
    a = ap.parse_args()
    design_stage(a.test_orderings) if a.stage == "design" else tokenize_stage(a.model, a.samples)


if __name__ == "__main__":
    sys.exit(main())
