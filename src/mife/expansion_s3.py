"""S3 TEST expansion (handoff 2026-11-23): all C(6,k) subsets x COA template on the 22 adopted S3 TEST groups.

What this module does, in order (each step is a CLI sub-command, see `main`):
  build-manifest   Reconstruct every frozen 246-configuration design for the 22 S3 TEST groups exactly (same prompt
                   IDs, text and hashes) and append the 210 new COA configurations per task (k = 2, 3, 4 only), as a
                   versioned sidecar manifest in data/evaluation_expansion_s3/. The frozen data/manifest/ is read only.
  tokenize MODEL   Tokenize all 30,096 expanded prompts with the frozen chat template. Original prompts must reproduce
                   the frozen data/tokenized/<model>.parquet token IDs, spans and rendered hashes exactly.
  inventory MODEL  Count valid, unique (prompt_id, sample_index) generation records already on disk for the 30,096
                   prompts (run `main` plus any earlier expansion records), and write the outstanding worklist with one
                   deterministic per-sample seed per record.
  generate         (Betty, vLLM) generate exactly the worklist records of one chunk, resumably, into run
                   `expansion_s3_20261123`. Never writes into `main`.

What it reads: manifest metadata and prompt text, the active split (mife.splits), the frozen tokenization, and
generation-shard *provenance* columns (prompt_id, sample_index, settings hash, seeds, finish reason, token counts).
It never reads response text, verification labels, features, predictions or any TEST outcome.
"""
from __future__ import annotations

import argparse
import json
import math
import string
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

from . import design
from .config import DATA_DIR, derive_seed, load_config
from .storage import sha256_file, sha256_text, write_json, write_parquet

VERSION = "expansion-s3-2026-11-23"
RUN_NAME = "expansion_s3_20261123"
SPLIT_VERSION = "split-s3-2026-11-23"
OUT = DATA_DIR / "evaluation_expansion_s3"
MANIFEST_DIR = DATA_DIR / "manifest"
FROZEN_PROMPTS_SHA256 = "f63c0aeaeec8872ec22df70f3feccbb4ecf464ee24368a001aace0cc5ccf1b14"  # results/prespec/FROZEN.json
K_MAX = 5
N_TEST_GROUPS = 22
N_ORIGINAL, N_EXPANDED = 246, 456
NEW_PER_K = {1: 0, 2: 18, 3: 84, 4: 108, 5: 0}
MODELS = ("llama", "qwen")
SOURCE_RUNS = ("main", RUN_NAME)  # runs whose records count as reusable for the expanded design


# ============================================================================================ design (pure)
def expanded_configs(chain_order: list[str]) -> list[dict]:
    """All k-subsets (k = 1..5) of the six rules x the k-specific COA template.

    Subsets that exist in the frozen design keep their chain-order base ordering (the chain prefix), so every frozen
    ordered configuration is reproduced exactly: for k = 4 the A4 template picks 12 of 24 orders and a different base
    would pick a different 12. New subsets use the group's chain-label order (A..F) as base.
    """
    rank = {r: i for i, r in enumerate(chain_order)}
    label = {r: string.ascii_uppercase[i] for i, r in enumerate(chain_order)}
    existing = {}
    for c_idx, chain in enumerate(design.chains(chain_order), start=1):
        for k in range(1, K_MAX + 1):
            existing.setdefault(frozenset(chain[:k]), (tuple(chain[:k]), c_idx))
    out = []
    for k in range(1, K_MAX + 1):
        tmpl = design.coa_template(k)
        for sub in combinations(chain_order, k):
            s = frozenset(sub)
            if s in existing:
                base, chain_idx = existing[s]
            else:
                base, chain_idx = tuple(sorted(sub, key=rank.get)), None
            labels = "".join(label[r] for r in sorted(sub, key=rank.get))
            for row, sigma in enumerate(tmpl):
                out.append({"k": k, "subset": s, "subset_labels": labels, "base_order": list(base),
                            "chain_index": chain_idx, "coa_row": row, "coa_perm": list(sigma),
                            "order": tuple(base[x] for x in sigma), "is_original": s in existing})
    return out


def check_expanded(cfgs: list[dict], frozen_orders: set[tuple]) -> dict:
    """The invariants the handoff requires before any generation input is written (raises on failure)."""
    orders = [c["order"] for c in cfgs]
    assert len(orders) == N_EXPANDED and len(set(orders)) == N_EXPANDED, "expanded orders must be 456 distinct"
    orig = {c["order"] for c in cfgs if c["is_original"]}
    assert orig == frozen_orders and len(orig) == N_ORIGINAL, "original 246 orders must equal the frozen design"
    new = [c for c in cfgs if not c["is_original"]]
    per_k = Counter(c["k"] for c in new)
    assert {k: per_k.get(k, 0) for k in range(1, 6)} == NEW_PER_K, per_k
    assert not (set(c["order"] for c in new) & frozen_orders), "a new order duplicates a frozen one"
    for k in range(1, K_MAX + 1):
        subs = {c["subset"] for c in cfgs if c["k"] == k}
        assert len(subs) == math.comb(6, k)
        for s in subs:
            design.check_coa([tuple(c["coa_perm"]) for c in cfgs if c["subset"] == s], k)
    return {f"new_k{k}": per_k.get(k, 0) for k in range(1, 6)} | {"expanded": len(orders), "original": len(orig)}


def subset_id(gid: str, c: dict) -> str:
    """Frozen subsets keep `{g}-c{chain}-k{k}`; new subsets are `{g}-s{labels}-k{k}` (cannot collide)."""
    if c["is_original"]:
        return f"{gid}-c{c['chain_index']}-k{c['k']}"
    return f"{gid}-s{c['subset_labels']}-k{c['k']}"


# ============================================================================================ manifest
def _test_groups() -> list[str]:
    from . import splits

    if splits.version() != SPLIT_VERSION:
        raise SystemExit(f"active split is {splits.version()!r}, not {SPLIT_VERSION!r}")
    g = sorted(splits.test_groups())
    assert len(g) == N_TEST_GROUPS, g
    return g


def build_manifest(out_dir: Path = OUT) -> dict:
    from . import source as S

    assert sha256_file(MANIFEST_DIR / "prompts.parquet") == FROZEN_PROMPTS_SHA256, "frozen manifest changed"
    groups = _test_groups()
    P = pd.read_parquet(MANIFEST_DIR / "prompts.parquet")
    R = pd.read_parquet(MANIFEST_DIR / "prompt_rules.parquet")
    pools = {}
    for line in open(MANIFEST_DIR / "rule_pools.jsonl"):
        p = json.loads(line)
        pools[p["group_id"]] = p
    P = P[P.group_id.isin(groups)].copy()
    assert len(P) == 22 * 738 == 16236
    prompt_rows, rule_rows, audit_rows = [], [], []
    for gid in groups:
        pool = pools[gid]
        rmap = {r["rule_id"]: r for r in pool["rules"]}
        Pg = P[P.group_id == gid]
        frozen_orders = {tuple(o) for o in Pg.drop_duplicates("config_id").ordered_rule_ids}
        assert frozen_orders == {tuple(c["ordered_rule_labels"]) for c in design.configurations(pool["chain_order"])}
        cfgs = expanded_configs(pool["chain_order"])
        counts = check_expanded(cfgs, frozen_orders)
        assert sorted(Pg.task_key.unique()) == sorted(pool["task_keys"])
        by_pid = Pg.set_index("prompt_id")
        n_text_identical = 0
        for c in cfgs:
            sid = subset_id(gid, c)
            cid = f"{sid}-o{c['coa_row']:02d}"
            texts = [rmap[r]["rule_text"] for r in c["order"]]
            for tkey in pool["task_keys"]:
                task_text = S.tasks()[tkey]["task_text"]
                prompt = S.serialize_prompt(task_text, texts)
                pid = f"{cid}-t{tkey}"
                if c["is_original"]:
                    f = by_pid.loc[pid]  # KeyError = a frozen prompt is not reproduced under its own ID: STOP
                    if not (f.prompt_text == prompt and f.prompt_sha256 == sha256_text(prompt)
                            and list(f.ordered_rule_ids) == list(c["order"]) and int(f.coa_row) == c["coa_row"]
                            and int(f.chain_index) == c["chain_index"] and f.config_id == cid and f.subset_id == sid):
                        raise SystemExit(f"STOP: frozen prompt {pid} is not reproduced byte-for-byte")
                    n_text_identical += 1
                spans, cursor = [], len(task_text) + len(S.PROMPT_HEADER)
                for pos, (rid, txt) in enumerate(zip(c["order"], texts), start=1):
                    start = cursor + len(S.rule_line_prefix())
                    assert prompt[start:start + len(txt)] == txt
                    spans.append((start, start + len(txt)))
                    r = rmap[rid]
                    rule_rows.append({
                        "prompt_id": pid, "position": pos, "k": c["k"], "rule_id": rid, "chain_label": r["chain_label"],
                        "type_id": r["type_id"], "category": r["category"], "code_prefix": r["code_prefix"],
                        "kwargs_json": json.dumps(r["kwargs"], sort_keys=True), "rule_text": txt, "char_start": start,
                        "char_end": start + len(txt), "exact_rule_hash": r["exact_rule_hash"], "group_id": gid,
                        "split": "test"})
                    cursor = start + len(txt) + 1
                prompt_rows.append({
                    "prompt_id": pid, "original_prompt_id": pid if c["is_original"] else None,
                    "new_prompt_id": None if c["is_original"] else pid, "is_original": c["is_original"],
                    "group_id": gid, "split": "test", "split_version": SPLIT_VERSION, "task_key": tkey,
                    "config_id": cid, "subset_id": sid, "subset_labels": c["subset_labels"],
                    "subset_key": "|".join(sorted(c["subset"])), "chain_index": c["chain_index"], "k": c["k"],
                    "base_order": c["base_order"], "coa_row": c["coa_row"], "coa_perm": c["coa_perm"],
                    "ordered_rule_ids": list(c["order"]), "ordered_type_ids": [rmap[r]["type_id"] for r in c["order"]],
                    "task_text": task_text, "task_char_span": [0, len(task_text)], "prompt_text": prompt,
                    "prompt_sha256": sha256_text(prompt), "n_chars": len(prompt),
                    "source_manifest": "data/manifest (frozen 2026-11-14)" if c["is_original"] else VERSION,
                    "expansion_version": VERSION})
        assert n_text_identical == 738
        for tkey in pool["task_keys"]:
            audit_rows.append({"group_id": gid, "task_key": tkey, **counts, "original_ids_text_identical": 246,
                               "original_subset_of_expanded": True, "new_orders_unique": True,
                               "new_k1": 0, "new_k5": 0})
    prompts = pd.DataFrame(prompt_rows)
    rules = pd.DataFrame(rule_rows)
    # global invariants
    assert len(prompts) == 22 * 3 * N_EXPANDED == 30096 and prompts.prompt_id.is_unique
    assert prompts.prompt_sha256.is_unique
    assert int(prompts.is_original.sum()) == 16236 and int((~prompts.is_original).sum()) == 13860
    assert set(prompts.prompt_id[prompts.is_original]) == set(P.prompt_id)
    assert not set(prompts.prompt_id[~prompts.is_original]) & set(pd.read_parquet(MANIFEST_DIR / "prompts.parquet",
                                                                                     columns=["prompt_id"]).prompt_id)
    assert (prompts.groupby("task_key").size() == N_EXPANDED).all() and prompts.task_key.nunique() == 66
    fr = R[R.prompt_id.isin(P.prompt_id)].sort_values(["prompt_id", "position"]).reset_index(drop=True)
    mine = rules[rules.prompt_id.isin(P.prompt_id)].sort_values(["prompt_id", "position"]).reset_index(drop=True)
    for col in ("rule_id", "rule_text", "char_start", "char_end", "exact_rule_hash", "kwargs_json"):
        assert (fr[col].values == mine[col].values).all(), f"original prompt_rules differ in {col}"
    by_k = prompts[~prompts.is_original].groupby("k").size().to_dict()
    assert by_k == {2: 1188, 3: 5544, 4: 7128}, by_k
    write_parquet(prompts, out_dir / "prompts.parquet")
    write_parquet(rules, out_dir / "prompt_rules.parquet")
    pd.DataFrame(audit_rows).to_csv(out_dir / "expanded_manifest_audit.csv", index=False)
    summary = {
        "version": VERSION, "split_version": SPLIT_VERSION, "test_groups": _test_groups(),
        "frozen_manifest_prompts_sha256": FROZEN_PROMPTS_SHA256,
        "n_prompts": len(prompts), "n_original": int(prompts.is_original.sum()), "n_new": int((~prompts.is_original).sum()),
        "n_new_by_k": {str(k): int(v) for k, v in by_k.items()},
        "n_prompt_rule_rows": len(rules),
        "expanded_manifest_sha256": sha256_text("".join(prompts.prompt_sha256) + "".join(prompts.prompt_id)),
        "prompts_parquet_sha256": sha256_file(out_dir / "prompts.parquet"),
        "prompt_rules_parquet_sha256": sha256_file(out_dir / "prompt_rules.parquet"),
        "base_ordering": "frozen subsets: frozen chain-prefix base; new subsets: chain-label order (A..F)",
        "coa_version": design.COA_VERSION, "new_prompt_id_scheme": "{group}-s{chain labels}-k{k}-o{coa_row:02d}-t{task}",
        "reads": ["data/manifest/{prompts,prompt_rules}.parquet", "data/manifest/rule_pools.jsonl",
                  "configs/active_split.yaml via mife.splits", "pinned ManyIFEval task texts"],
        "not_read": ["generations", "verification", "features", "predictions", "any TEST outcome"],
        "created_unix": time.time()}
    write_json(summary, out_dir / "manifest_summary.json")
    return summary


# ============================================================================================ tokenize
def tokenize(model_key: str, out_dir: Path = OUT) -> dict:
    from . import chat

    cfg = load_config()
    m = cfg["models"][model_key]
    summ = json.loads((out_dir / "manifest_summary.json").read_text())
    assert sha256_file(out_dir / "prompts.parquet") == summ["prompts_parquet_sha256"], "expansion manifest changed"
    prompts = pd.read_parquet(out_dir / "prompts.parquet")
    rules = pd.read_parquet(out_dir / "prompt_rules.parquet").sort_values(["prompt_id", "position"])
    spans = rules.groupby("prompt_id")[["char_start", "char_end"]].apply(lambda d: list(zip(d.char_start, d.char_end)))
    frozen = pd.read_parquet(DATA_DIR / "tokenized" / f"{model_key}.parquet").set_index("prompt_id")
    rows, n_orig_identical = [], 0
    for p in prompts.itertuples():
        e = chat.encode(model_key, p.prompt_text, spans[p.prompt_id], tuple(p.task_char_span))
        row = {"prompt_id": p.prompt_id, "model_key": model_key, "is_original": p.is_original,
               "prompt_sha256": p.prompt_sha256, "rendered_sha256": sha256_text(e["rendered"]),
               "rendered_text": e["rendered"], "input_ids": e["input_ids"], "n_input_tokens": e["n_input_tokens"],
               "input_ids_sha256": sha256_text(json.dumps(e["input_ids"])),
               "rule_token_spans": e["rule_token_spans"], "rule_span_straddles": e["rule_span_straddles"],
               "rule_span_lengths": [b - a for a, b in e["rule_token_spans"]], "task_token_span": e["task_token_span"]}
        if p.is_original:
            f = frozen.loc[p.prompt_id]
            same = (f.prompt_sha256 == row["prompt_sha256"] and f.rendered_sha256 == row["rendered_sha256"]
                    and list(f.input_ids) == row["input_ids"]
                    and [list(x) for x in f.rule_token_spans] == row["rule_token_spans"]
                    and list(f.task_token_span) == row["task_token_span"])
            if not same:
                raise SystemExit(f"STOP: {model_key} tokenization of frozen prompt {p.prompt_id} differs")
            n_orig_identical += 1
        rows.append(row)
    df = pd.DataFrame(rows)
    assert n_orig_identical == 16236
    path = out_dir / f"tokenized_{model_key}.parquet"
    write_parquet(df, path)
    n = df.n_input_tokens
    budget = (1 - cfg["context_safety_margin"]) * m["native_context"]
    worst = int(n.max()) + m["decoding"]["max_new_tokens"]
    report = {**chat.template_info(model_key), "version": VERSION, "n_prompts": len(df),
              "n_original_identical_to_frozen_tokenization": n_orig_identical,
              "n_new": int((~df.is_original).sum()),
              "input_tokens_quantiles": {str(q): float(n.quantile(q)) for q in (0, .5, .9, .99, 1)},
              "max_input_plus_output": worst, "native_context": m["native_context"],
              "usable_budget_with_margin": budget, "context_gate_pass": bool(worst <= budget),
              "fits_vllm_max_model_len_4096": bool(worst <= 4096),
              "n_new_prompts_with_straddling_rule_spans": int(df.rule_span_straddles[~df.is_original].apply(any).sum()),
              "truncation": "none", "tokenized_parquet_sha256": sha256_file(path),
              "expanded_manifest_sha256": summ["expanded_manifest_sha256"]}
    assert report["context_gate_pass"] and report["fits_vllm_max_model_len_4096"]
    write_json(report, out_dir / f"tokenized_{model_key}_audit.json")
    return report


# ============================================================================================ inventory (pure core)
def sample_seed(model_key: str, prompt_id: str, sample_index: int, run_name: str = RUN_NAME, base: int | None = None) -> int:
    """Same rule as generate_vllm.sample_seed with run name expansion_s3_20261123."""
    base = load_config()["seeds"]["generation"] if base is None else base
    return derive_seed(base, "vllm", model_key, run_name, prompt_id, sample_index) % (1 << 31)


def valid_records(gen: pd.DataFrame, expected: pd.DataFrame, model_key: str, settings_hash: str,
                  n_samples: int = 20) -> tuple[pd.DataFrame, dict]:
    """Filter generation provenance rows to valid, reusable records.

    gen: rows with prompt_id, sample_index, model_key, settings_hash, rendered_sha256, finish_reason, source_run,
         n_output_tokens (and batch_seed for expansion rows). expected: prompt_id -> rendered_sha256 (+ seed table for
         expansion rows is checked by the caller). Returns unique (prompt_id, sample_index) valid rows and counts.
    """
    g = gen[gen.prompt_id.isin(expected.index)].copy()
    stats = {"rows_for_expanded_prompts": len(g), "rows_other_prompts": int(len(gen) - len(g))}
    ok = ((g.model_key == model_key) & (g.settings_hash == settings_hash)
          & g.sample_index.between(0, n_samples - 1) & g.finish_reason.isin(["stop", "length"])
          & g.source_run.isin(SOURCE_RUNS)
          & (g.rendered_sha256.values == expected.loc[g.prompt_id, "rendered_sha256"].values))
    stats["rows_invalid_provenance"] = int((~ok).sum())
    g = g[ok]
    dup = g.duplicated(["prompt_id", "sample_index"], keep="first")
    stats["rows_duplicate_prompt_sample"] = int(dup.sum())
    g = g[~dup]
    stats["rows_valid_unique"] = len(g)
    stats["rows_empty_response"] = int((g.n_output_tokens == 0).sum())
    return g, stats


def outstanding(expected_ids: list[str], have: pd.DataFrame, n_samples: int = 20) -> pd.DataFrame:
    """Every (prompt_id, sample_index) in expected_ids x range(n_samples) not present in `have`, in manifest order."""
    got = set(zip(have.prompt_id, have.sample_index.astype(int)))
    rows = [(p, s) for p in expected_ids for s in range(n_samples) if (p, s) not in got]
    return pd.DataFrame(rows, columns=["prompt_id", "sample_index"])


def assign_chunks(work: pd.DataFrame, n_chunks: int) -> pd.Series:
    """Contiguous chunks by prompt (a prompt's samples share a chunk, for prefix caching)."""
    pids = list(dict.fromkeys(work.prompt_id))
    size = max(1, -(-len(pids) // n_chunks))
    ch = {p: i // size for i, p in enumerate(pids)}
    return work.prompt_id.map(ch).astype(int)


GEN_COLS = ["prompt_id", "sample_index", "model_key", "settings_hash", "rendered_sha256", "finish_reason",
            "source_run", "n_output_tokens", "batch_seed", "backend"]


def read_provenance(model_key: str, runs=SOURCE_RUNS) -> pd.DataFrame:
    """Provenance columns only (never response text) from every shard of the given runs."""
    frames = []
    for run in runs:
        for f in sorted((DATA_DIR / "generations" / model_key / run).glob("shard_*.parquet")):
            d = pd.read_parquet(f, columns=GEN_COLS)
            d["shard_file"] = f"{run}/{f.name}"
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=GEN_COLS + ["shard_file"])


def inventory(model_key: str, n_chunks: int = 12, out_dir: Path = OUT) -> dict:
    from .generate_mlx import settings_hash

    tok = pd.read_parquet(out_dir / f"tokenized_{model_key}.parquet",
                          columns=["prompt_id", "rendered_sha256", "is_original"]).set_index("prompt_id")
    prompts = pd.read_parquet(out_dir / "prompts.parquet", columns=["prompt_id", "group_id", "k", "is_original"])
    shash = settings_hash(model_key, "vllm")
    gen = read_provenance(model_key)
    have, stats = valid_records(gen, tok, model_key, shash)
    # expansion-run records must carry their recorded per-sample seed
    exp = have.source_run == RUN_NAME
    if exp.any():
        want = [sample_seed(model_key, p, int(s)) for p, s in zip(have.prompt_id[exp], have.sample_index[exp])]
        bad = have[exp].batch_seed.astype(int).values != pd.Series(want).values
        stats["expansion_rows_seed_mismatch"] = int(bad.sum())
        if bad.any():
            raise SystemExit(f"{model_key}: {int(bad.sum())} expansion records carry an unexpected seed")
    work = outstanding(list(prompts.prompt_id), have)
    work["seed"] = [sample_seed(model_key, p, int(s)) for p, s in zip(work.prompt_id, work.sample_index)]
    work["chunk"] = assign_chunks(work, n_chunks) if len(work) else pd.Series(dtype=int)
    work["model_key"] = model_key
    work["run_name"] = RUN_NAME
    work["version"] = VERSION
    meta = prompts.set_index("prompt_id")
    work["is_original"] = meta.loc[work.prompt_id, "is_original"].values
    path = out_dir / f"worklist_{model_key}.parquet"
    write_parquet(work, path)
    # reconciliation by group
    hv = have.assign(group_id=meta.loc[have.prompt_id, "group_id"].values,
                     is_original=meta.loc[have.prompt_id, "is_original"].values)
    rec = []
    for gid, pg in prompts.groupby("group_id"):
        n_o, n_n = int(pg.is_original.sum()), int((~pg.is_original).sum())
        h = hv[hv.group_id == gid]
        w = work[work.prompt_id.isin(pg.prompt_id)]
        rec.append({"model": model_key, "group_id": gid,
                    "original_prompts": n_o, "original_expected": 20 * n_o,
                    "original_found_reusable": int(h.is_original.sum()),
                    "original_missing": int(w.is_original.sum()),
                    "new_prompts": n_n, "new_expected": 20 * n_n,
                    "new_already_found": int((~h.is_original).sum()),
                    "new_missing": int((~w.is_original).sum()),
                    "to_generate": len(w)})
    R = pd.DataFrame(rec)
    assert (R.original_found_reusable + R.original_missing == R.original_expected).all()
    assert (R.new_already_found + R.new_missing == R.new_expected).all()
    partial = have.groupby("prompt_id").size()
    summary = {
        "version": VERSION, "model_key": model_key, "run_name": RUN_NAME, "settings_hash": shash,
        "source_runs_checked": list(SOURCE_RUNS), "shards_read": int(gen.shard_file.nunique()) if len(gen) else 0,
        **stats,
        f"existing_original_s3_test_responses_{model_key}": int(R.original_found_reusable.sum()),
        f"missing_original_s3_test_responses_{model_key}": int(R.original_missing.sum()),
        f"new_s3_subset_prompts_{model_key}": int(R.new_prompts.sum()),
        f"new_s3_subset_responses_{model_key}": int(R.new_expected.sum()),
        f"already_generated_for_new_subset_ids_{model_key}": int(R.new_already_found.sum()),
        "outstanding_responses": len(work),
        "outstanding_formula_check": 20 * len(prompts) - len(have) == len(work),
        "original_prompts_partially_complete": int(((partial > 0) & (partial < 20)).sum()),
        "n_chunks": n_chunks, "chunk_prompt_counts": work.groupby("chunk").prompt_id.nunique().to_dict(),
        "seeding": "per-sample: derive_seed(seeds.generation=20261115, 'vllm', model, 'expansion_s3_20261123', "
                   "prompt_id, sample_index) % 2^31 (= generate_vllm.sample_seed)",
        "worklist_parquet": path.name, "worklist_sha256": sha256_file(path),
        "worklist_records_sha256": sha256_text("".join(f"{p}:{s}:{d};" for p, s, d in
                                                       zip(work.prompt_id, work.sample_index, work.seed))),
        "tokenized_parquet_sha256": sha256_file(out_dir / f"tokenized_{model_key}.parquet"),
        "not_read": ["response_text", "verification", "features", "predictions", "any TEST outcome"]}
    write_json(summary, out_dir / f"worklist_{model_key}.json")
    R.to_csv(out_dir / f"reconciliation_{model_key}.csv", index=False)
    return {"summary": summary, "reconciliation": R}


# ============================================================================================ generation (Betty)
def check_worklist(model_key: str, out_dir: Path = OUT) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    meta = json.loads((out_dir / f"worklist_{model_key}.json").read_text())
    assert meta["version"] == VERSION and meta["run_name"] == RUN_NAME
    wl_path, tok_path = out_dir / f"worklist_{model_key}.parquet", out_dir / f"tokenized_{model_key}.parquet"
    if sha256_file(wl_path) != meta["worklist_sha256"]:
        raise SystemExit(f"{wl_path} does not match its recorded checksum")
    if sha256_file(tok_path) != meta["tokenized_parquet_sha256"]:
        raise SystemExit(f"{tok_path} does not match the checksum recorded with the worklist")
    work = pd.read_parquet(wl_path)
    for p, s, d in zip(work.prompt_id, work.sample_index, work.seed):
        if sample_seed(model_key, p, int(s)) != int(d):
            raise SystemExit(f"worklist seed for {p}/{s} does not follow the recorded rule")
    tok = pd.read_parquet(tok_path, columns=["prompt_id", "input_ids", "n_input_tokens", "rendered_sha256",
                                             "input_ids_sha256"]).set_index("prompt_id")
    return work, tok, meta


def done_records(out_dir: Path, model_key: str, shash: str, run_name: str = RUN_NAME) -> set[tuple[str, int]]:
    """(prompt_id, sample_index) already written by this run with the right settings and seed (per-sample resume)."""
    done = set()
    for f in sorted(out_dir.glob("shard_*.parquet")):
        d = pd.read_parquet(f, columns=["prompt_id", "sample_index", "settings_hash", "batch_seed", "model_key"])
        for p, s, h, seed, mk in d.itertuples(index=False):
            if h == shash and mk == model_key and int(seed) == sample_seed(model_key, p, int(s), run_name):
                done.add((p, int(s)))
    return done


SMOKE_RUN = "smoke_expansion_s3"


def smoke_work(model_key: str, n_prompts: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """TRAIN-only preflight work: the first n pilot prompts whose group is S3 TRAIN, x 20 samples, frozen tokenization."""
    from . import splits

    sog = splits.split_of_group()
    pilot = pd.read_parquet(DATA_DIR / "pilot" / "pilot_prompts.parquet")
    pids = [p for p in pilot.prompt_id if sog[p.split("-")[0]] == "train"][:n_prompts]
    assert pids and all(sog[p.split("-")[0]] == "train" for p in pids)
    tok = pd.read_parquet(DATA_DIR / "tokenized" / f"{model_key}.parquet",
                          columns=["prompt_id", "input_ids", "n_input_tokens", "rendered_sha256"]).set_index("prompt_id")
    work = pd.DataFrame([(p, s) for p in pids for s in range(20)], columns=["prompt_id", "sample_index"])
    work["seed"] = [sample_seed(model_key, p, int(s), SMOKE_RUN) for p, s in zip(work.prompt_id, work.sample_index)]
    work["chunk"] = 0
    meta = {"worklist_sha256": "smoke (TRAIN pilot prompts, S3 train groups)", "tokenized_parquet_sha256": "frozen",
            "seeding": f"per-sample: derive_seed(20261115, 'vllm', model, '{SMOKE_RUN}', prompt_id, sample_index) % 2^31"}
    return work, tok.loc[pids], meta


def generate(model_key: str, chunk: int, prompts_per_shard: int = 500, gpu_memory_utilization: float = 0.85,
             max_num_seqs: int = 512, max_model_len: int = 4096, max_records: int | None = None,
             run_name: str = RUN_NAME, smoke_prompts: int = 20) -> dict:
    """Generate the worklist records of one chunk with per-sample seeds (Betty / vLLM)."""
    import os

    if run_name not in (RUN_NAME, SMOKE_RUN):
        raise SystemExit(f"run name must be {RUN_NAME} (or {SMOKE_RUN} for the TRAIN-only preflight)")
    from vllm import SamplingParams

    from . import generate_vllm as GV
    from .generate_mlx import GEN_SCHEMA, eos_token_ids, gen_settings, settings_hash

    if run_name == SMOKE_RUN:
        work, tok, wmeta = smoke_work(model_key, smoke_prompts)
    else:
        work, tok, wmeta = check_worklist(model_key)
    work = work[work.chunk == chunk]
    out_dir = GV.check_run_name(model_key, run_name)
    if out_dir.name == "main":
        raise SystemExit("refusing to write into run main")
    out_dir.mkdir(parents=True, exist_ok=True)
    st = gen_settings(model_key, "vllm")
    shash = settings_hash(model_key, "vllm")
    assert int(tok.loc[work.prompt_id.unique(), "n_input_tokens"].max()) + st["max_new_tokens"] <= max_model_len
    done = done_records(out_dir, model_key, shash, run_name)
    todo = work[[(p, int(s)) not in done for p, s in zip(work.prompt_id, work.sample_index)]]
    if max_records is not None:
        todo = todo.iloc[:max_records]
    stop = eos_token_ids(model_key)
    env = GV.cuda_env()
    engine_seed = derive_seed(load_config()["seeds"]["generation"], "vllm", model_key, run_name, f"c{chunk:02d}") % (1 << 31)
    tag = f"c{chunk:02d}_j{os.environ.get('SLURM_JOB_ID', 'local')}"
    t_load = time.time()
    llm, info = GV.load_llm(model_key, max_model_len, gpu_memory_utilization, max_num_seqs, engine_seed)
    load_s = time.time() - t_load
    skw = GV.sampling_kwargs(st)
    tokz = llm.get_tokenizer()
    pids = list(dict.fromkeys(todo.prompt_id))
    batch_log, total_out, gen_wall = [], 0, 0.0
    with GV.GpuMemSampler() as mem:
        for b0 in range(0, len(pids), prompts_per_shard):
            bp = set(pids[b0:b0 + prompts_per_shard])
            rows_in = todo[todo.prompt_id.isin(bp)]
            reqs, params, meta = [], [], []
            for p, s, seed in zip(rows_in.prompt_id, rows_in.sample_index, rows_in.seed):
                reqs.append({"prompt_token_ids": [int(x) for x in tok.loc[p, "input_ids"]]})
                params.append(SamplingParams(n=1, seed=int(seed), stop_token_ids=stop, skip_special_tokens=False,
                                             detokenize=False, **skw))
                meta.append((p, int(s), int(seed)))
            t0 = time.time()
            outs = llm.generate(reqs, params, use_tqdm=False)
            el = time.time() - t0
            gen_wall += el
            rows = []
            for (p, s, seed), o in zip(meta, outs):
                c = o.outputs[0]
                ids = GV.strip_stop(list(c.token_ids), c.finish_reason, set(stop))
                rows.append({
                    "response_id": f"{model_key}:{p}:s{s:02d}", "prompt_id": p, "model_key": model_key,
                    "sample_index": s, "response_text": tokz.decode(ids), "output_token_ids": ids,
                    "n_output_tokens": len(ids), "finish_reason": c.finish_reason,
                    "hit_max_tokens": c.finish_reason == "length", "stop_reason": str(c.stop_reason),
                    "n_input_tokens": int(tok.loc[p, "n_input_tokens"]), "rendered_sha256": tok.loc[p, "rendered_sha256"],
                    "batch_id": f"{run_name}-vllm-{tag}", "batch_seed": seed, "insert_order": len(rows),
                    "completion_batch_size": max_num_seqs, "source_run": run_name, "backend": "vllm",
                    "settings_hash": shash, "schema": GEN_SCHEMA, "created_unix": time.time(),
                    "expansion_version": VERSION, "worklist_sha256": wmeta["worklist_sha256"]})
            n_out = sum(r["n_output_tokens"] for r in rows)
            total_out += n_out
            idx = len(list(out_dir.glob(f"shard_{tag}_[0-9]*.parquet")))
            write_parquet(pd.DataFrame(rows), out_dir / f"shard_{tag}_{idx:05d}.parquet")
            batch_log.append({"shard": f"shard_{tag}_{idx:05d}", "n_prompts": len(bp), "n_records": len(rows),
                              "output_tokens": n_out, "generate_s": el, "peak_gpu_memory_used_gb": mem.peak})
            print(f"[{model_key}/expansion c{chunk:02d}] {len(rows)} records, {n_out} tok in {el:.0f}s "
                  f"({n_out / max(el, 1e-9):.0f} tok/s)", flush=True)
    summary = {"model_key": model_key, "run_name": run_name, "version": VERSION, "chunk": chunk, "tag": tag,
               "backend": "vllm", "settings": st, "settings_hash": shash,
               "sampling_params": {**skw, "stop_token_ids": stop, "n": 1, "seeding": wmeta["seeding"]},
               "engine": info, "engine_seed_unused_for_sampling": engine_seed, "environment": env,
               "prompts_per_shard": prompts_per_shard, "max_num_seqs": max_num_seqs,
               "worklist_sha256": wmeta["worklist_sha256"], "tokenized_parquet_sha256": wmeta["tokenized_parquet_sha256"],
               "chunk_records": len(work), "records_already_done": len(work) - len(todo) if max_records is None else None,
               "records_generated_this_call": int(sum(b["n_records"] for b in batch_log)),
               "model_load_s": load_s, "generation_wall_s": gen_wall, "output_tokens": total_out,
               "aggregate_output_tok_per_s": total_out / gen_wall if gen_wall else None,
               "reproducibility_note": "per-sample seeds; vLLM output is not bitwise reproducible across batch "
                                       "compositions, so chunk geometry and max_num_seqs are recorded",
               "batches": batch_log}
    write_json(summary, out_dir / f"run_summary_{tag}_{int(time.time())}.json")
    return summary


def check_complete(model_key: str, out_dir: Path = OUT) -> dict:
    """After the jobs: every worklist record exists exactly once with its seed (no TEST outcomes are read)."""
    from .generate_mlx import settings_hash

    work, tok, _ = check_worklist(model_key, out_dir)
    gen = read_provenance(model_key, runs=(RUN_NAME,))
    have, stats = valid_records(gen, tok, model_key, settings_hash(model_key, "vllm"))
    miss = outstanding(list(dict.fromkeys(work.prompt_id)), have)
    want = set(zip(work.prompt_id, work.sample_index.astype(int)))
    missing = [(p, s) for p, s in zip(miss.prompt_id, miss.sample_index) if (p, s) in want]
    extra = len(set(zip(have.prompt_id, have.sample_index.astype(int))) - want)
    return {"model_key": model_key, **stats, "worklist_records": len(work), "missing": len(missing),
            "records_outside_worklist": extra, "complete": not missing}


# ============================================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mife.expansion_s3")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-manifest")
    t = sub.add_parser("tokenize"); t.add_argument("model", choices=MODELS)
    i = sub.add_parser("inventory"); i.add_argument("model", choices=MODELS); i.add_argument("--n-chunks", type=int, default=12)
    g = sub.add_parser("generate"); g.add_argument("model", choices=MODELS); g.add_argument("--chunk", type=int, required=True)
    g.add_argument("--ppb", type=int, default=500); g.add_argument("--gpu-mem", type=float, default=0.85)
    g.add_argument("--max-num-seqs", type=int, default=512); g.add_argument("--max-model-len", type=int, default=4096)
    g.add_argument("--max-records", type=int, default=None)
    sm = sub.add_parser("smoke", help="TRAIN-only GPU preflight (S3 TRAIN pilot prompts, run smoke_expansion_s3)")
    sm.add_argument("model", choices=MODELS); sm.add_argument("--prompts", type=int, default=20)
    c = sub.add_parser("check-complete"); c.add_argument("model", choices=MODELS)
    a = ap.parse_args(argv)
    if a.cmd == "build-manifest":
        out = build_manifest()
    elif a.cmd == "tokenize":
        out = tokenize(a.model)
    elif a.cmd == "inventory":
        out = inventory(a.model, a.n_chunks)
        out = out["summary"]
    elif a.cmd == "generate":
        out = generate(a.model, a.chunk, a.ppb, a.gpu_mem, a.max_num_seqs, a.max_model_len, a.max_records)
        out.pop("batches", None)
    elif a.cmd == "smoke":
        out = generate(a.model, 0, run_name=SMOKE_RUN, smoke_prompts=a.prompts)
        out.pop("batches", None)
    else:
        out = check_complete(a.model)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
