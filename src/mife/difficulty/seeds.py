"""Stage 1: seed requests from WildChat-1M (Dolly-15k fallback) for the 8 domains x 5 task types grid.

Steps (each writes to data/difficulty/seeds_work/ and is resumable):
  filter    English, non-toxic, not redacted, unflagged first user turns; 40-1500 chars; no code / role-play /
            creative-writing / SEO templates; regex PII screen (emails, phones, URLs, addresses, "my name is", handles).
  embed     BGE-M3 dense embeddings of the kept turns and of the 40 cell descriptions; top-N per cell -> candidates.
  classify  local LLM (MLX, configs/difficulty/seeds.yaml) labels domain, task type, concrete situation, sensitive,
            personal data. Cached per text hash.
  select    1 seed + 2 spares per cell, then 8 calibration + 8 confirmation pilot seeds; one per conversation and per
            hashed user overall; no pair at cosine >= 0.85; Dolly tops up thin cells (flagged).
Output: data/difficulty/seeds.parquet (+ seeds_coverage.csv, seeds_summary.json).

Usage: PYTHONPATH=src:.pydeps python -m mife.difficulty.seeds {filter,embed,classify,select,all}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from .catalog import ROOT

CFG = yaml.safe_load((ROOT / "configs" / "difficulty" / "seeds.yaml").read_text())
DATA = ROOT / "data" / "difficulty"
RAW = DATA / "raw"
WORK = DATA / "seeds_work"
DOMAINS = list(CFG["domains"])
TYPES = list(CFG["task_types"])
CELLS = [(d, t) for d in DOMAINS for t in TYPES]

# ------------------------------------------------------------------------------------------------ filters

CODE_RE = re.compile(r"```|\bdef \w+\(|#include|\bimport \w+|\bpublic static\b|console\.log|\bSELECT\b.+\bFROM\b|"
                     r"</?\w+>|\{\s*\n|;\s*\n|=>|\bfunction\s*\(|\bvar \w+ =|\$\w+\s*=|\bpython\b|\bjava(script)?\b|"
                     r"\bc\+\+|\bhtml\b|\bcss\b|\bsql\b|\bregex\b|\bexcel formula\b", re.I)
EXCLUDE_RE = re.compile(
    r"role[- ]?play|\bpretend\b|\bact as\b|\byou are (a|an|now)\b|\bimagine you\b|\bstory\b|\bpoem\b|\bfanfic|"
    r"\bscript\b|\bepisode\b|\bcharacter\b|\bnovel\b|\blyrics\b|\bsong\b|\brap\b|\bessay\b|\barticle\b|\bblog\b|"
    r"\bseo\b|\bkeywords?\b|\bmidjourney\b|\bprompt\b|\bdan\b|\bjailbreak|\bignore (all|previous)\b|\btranslat|"
    r"\brewrite\b|\bparaphrase\b|\bproofread|\bsummari[sz]e\b|\bwrite (a|an|me|the)\b|\bgenerate\b|\bcaption\b|"
    r"\btweet\b|\binstagram\b|\byoutube\b|\bdescription for\b|\bcover letter\b|\bresume\b|\bcv\b|\bemail (to|for)\b|"
    r"\bletter to\b|\bsex|\bnsfw\b|\bkill|\bsuicide|\bdrug|\bweapon|\bmedical\b|\bdoctor\b|\blawyer\b|\blawsuit\b|"
    r"\bdiagnos|\bsymptom|\bpregnan|\btherap|\bmedication|\bcrypto|\bstocks?\b|\binvest", re.I)
PII_RE = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+|"                                   # email
    r"(?:\+?\d[\d\s().-]{8,}\d)|"                                 # phone-like / long digit runs
    r"https?://|www\.|"                                           # URLs
    r"\b\d{1,5}\s+\w+(?:\s\w+)?\s(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr|blvd|boulevard|court|ct)\b|"
    r"\bmy name is\b|\bi am called\b|\bmy (?:address|phone|email|number) is\b|(?<!\w)@\w{3,}|"
    r"\bdear [A-Z][a-z]+|\b(?:regards|sincerely|thanks),?\s*\n\s*[A-Z][a-z]+", re.I)


def _first_user_turn(conv) -> dict | None:
    if conv is None or len(conv) == 0:
        return None
    t = conv[0]
    return t if t.get("role") == "user" else None


def keep_text(s: str) -> str | None:
    """Reason a first turn is dropped, or None if kept (shared by WildChat and Dolly)."""
    f = CFG["filters"]
    if not (f["min_chars"] <= len(s) <= f["max_chars"]):
        return "length"
    if sum(ord(c) > 127 for c in s) / max(len(s), 1) > f["max_non_ascii_frac"]:
        return "non_ascii"
    if CODE_RE.search(s):
        return "code"
    if EXCLUDE_RE.search(s):
        return "excluded_genre_or_sensitive"
    if PII_RE.search(s):
        return "pii_regex"
    return None


def step_filter() -> pd.DataFrame:
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / "filtered.parquet"
    if out.exists():
        return pd.read_parquet(out)
    rows, reasons = [], {}
    cols = ["conversation_hash", "model", "timestamp", "conversation", "language", "toxic", "redacted",
            "hashed_ip", "openai_moderation", "detoxify_moderation", "country"]
    n_total = 0
    for shard in sorted((RAW / "wildchat" / "data").glob("train-*.parquet")):
        pf = pq.ParquetFile(shard)
        for rg in range(pf.num_row_groups):
            df = pf.read_row_group(rg, columns=cols).to_pandas()
            n_total += len(df)
            for r in df.itertuples(index=False):
                why = None
                t = _first_user_turn(r.conversation)
                if r.language != "English":
                    why = "language"
                elif r.toxic or r.redacted:
                    why = "toxic_or_redacted_flag"
                elif t is None or t.get("language") != "English" or t.get("toxic") or t.get("redacted"):
                    why = "first_turn_flags"
                elif len(r.openai_moderation) and r.openai_moderation[0]["flagged"]:
                    why = "openai_moderation"
                elif len(r.detoxify_moderation) and r.detoxify_moderation[0]["toxicity"] > CFG["filters"]["max_detoxify_toxicity"]:
                    why = "detoxify"
                else:
                    why = keep_text(t["content"])
                if why:
                    reasons[why] = reasons.get(why, 0) + 1
                    continue
                rows.append({"source": "wildchat", "conversation_hash": r.conversation_hash,
                             "user_hash": r.hashed_ip, "model": r.model, "timestamp": str(r.timestamp),
                             "country": r.country, "text": t["content"].strip(), "shard": shard.name})
        print(shard.name, len(rows), flush=True)
    df = pd.DataFrame(rows)
    n0 = len(df)
    df["text_hash"] = df.text.map(lambda s: hashlib.sha256(re.sub(r"\s+", " ", s.lower()).encode()).hexdigest()[:16])
    df = df.drop_duplicates("text_hash").drop_duplicates("conversation_hash").reset_index(drop=True)
    reasons["exact_duplicate"] = n0 - len(df)
    df.to_parquet(out)
    (WORK / "filter_summary.json").write_text(json.dumps({"n_conversations": n_total, "n_kept": len(df),
                                                          "dropped": reasons}, indent=1))
    return df


def load_dolly() -> pd.DataFrame:
    rows = []
    for i, line in enumerate((RAW / "dolly" / "databricks-dolly-15k.jsonl").read_text().splitlines()):
        r = json.loads(line)
        if r["category"] not in CFG["dolly"]["categories"] or r.get("context"):
            continue
        s = r["instruction"].strip()
        if keep_text(s) is None or (len(s) >= 25 and not CODE_RE.search(s) and not PII_RE.search(s)
                                    and not EXCLUDE_RE.search(s)):
            rows.append({"source": "dolly", "conversation_hash": f"dolly-{i:05d}", "user_hash": f"dolly-{i:05d}",
                         "model": None, "timestamp": None, "country": None, "text": s, "shard": "dolly",
                         "text_hash": hashlib.sha256(s.lower().encode()).hexdigest()[:16],
                         "dolly_category": r["category"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ embeddings

def cell_description(d: str, t: str) -> str:
    return (f"A person asks an assistant for help with their own situation in {CFG['domains'][d]}. "
            f"They want the assistant to {CFG['task_types'][t]}, for example {CFG['cell_examples'][d][t]}.")


def embed_texts(texts: list[str], batch: int = 32) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer
    name = CFG["embedding"]["model"]
    tok = AutoTokenizer.from_pretrained(name)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float16).to(dev).eval()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(texts), batch):
            idx = order[b:b + batch]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
            h = model(**enc).last_hidden_state[:, 0]
            h = torch.nn.functional.normalize(h.float(), dim=-1)
            out[idx] = h.cpu().numpy()
            if b // batch % 200 == 0:
                print(f"embedded {b}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
    return out


def step_embed() -> None:
    df = step_filter()
    dolly = load_dolly()
    for name, d in (("wildchat", df), ("dolly", dolly)):
        f = WORK / f"emb_{name}.npy"
        if not f.exists():
            np.save(f, embed_texts(d.text.tolist()))
    if not (WORK / "emb_cells.npy").exists():
        np.save(WORK / "emb_cells.npy", embed_texts([cell_description(d, t) for d, t in CELLS]))
    dolly.to_parquet(WORK / "dolly_filtered.parquet")
    cells = np.load(WORK / "emb_cells.npy")
    for name, d in (("wildchat", df), ("dolly", dolly)):
        E = np.load(WORK / f"emb_{name}.npy")
        S = E @ cells.T
        top = CFG["embedding"]["top_per_cell"] if name == "wildchat" else 40
        cand = {}
        for j, (dom, typ) in enumerate(CELLS):
            for i in np.argsort(-S[:, j])[:top]:
                cand.setdefault(int(i), []).append((f"{dom}|{typ}", float(S[i, j])))
        rows = [{"row": i, "retrieved_for": [c for c, _ in v], "best_retrieval_cos": max(s for _, s in v)}
                for i, v in cand.items()]
        pd.DataFrame(rows).to_parquet(WORK / f"candidates_{name}.parquet")
        print(name, "candidates", len(rows))


TYPE_HINT_RE = {
    "explanation": re.compile(r"\bwhy\b|\bexplain\b|\breason\b|\bhow come\b|\bcaused?\b", re.I),
    "comparison": re.compile(r"\bvs\.?\b|\bversus\b|\bcompar|\bbetter\b|\bwhich (one|option|is)\b|\bor should\b|"
                             r"\bchoose between\b|\bpros and cons\b|\btwo (options|offers|plans|choices)\b", re.I),
    "planning": re.compile(r"\bplan\b|\bschedule\b|\bitinerary\b|\borgani[sz]e\b|\bprepare\b|\btimeline\b", re.I),
    "troubleshooting": re.compile(r"\bnot working\b|\bproblem\b|\bissue\b|\bkeeps?\b|\bwrong\b|\bfix\b|\bwon'?t\b|"
                                  r"\bfailed\b|\bnever arrived\b|\bstill\b", re.I),
    "prioritization": re.compile(r"\bpriorit|\bfirst\b|\bmost important\b|\burgent\b|\bin what order\b|"
                                 r"\bwhich should i\b|\btoo many\b|\boverwhelmed\b|\bdeadlines?\b", re.I),
}


def step_expand(thin_below: int = 3, per_cell: int = 80) -> None:
    """Targeted retrieval for thin cells: rank by cell-specific similarity (cos to the cell minus the mean cos to all
    cells), keep texts matching the cell's task-type hint, skip rows already sent to the classifier."""
    base = pd.read_parquet(WORK / "filtered.parquet")
    E, C = np.load(WORK / "emb_wildchat.npy"), np.load(WORK / "emb_cells.npy")
    S = E @ C.T
    spec = S - S.mean(axis=1, keepdims=True)
    cand = pd.read_parquet(WORK / "candidates_wildchat.parquet")
    have = set(cand.row)
    acc = _accepted("wildchat")
    counts = acc[acc.accepted].cell.value_counts().to_dict()
    new_rows = []
    for j, (d, t) in enumerate(CELLS):
        if counts.get(f"{d}|{t}", 0) >= thin_below:
            continue
        hint = TYPE_HINT_RE[t]
        k = 0
        for i in np.argsort(-(spec[:, j] + 0.5 * S[:, j])):
            if int(i) in have or not hint.search(base.text.iloc[i]):
                continue
            new_rows.append({"row": int(i), "retrieved_for": [f"{d}|{t}"], "best_retrieval_cos": float(S[i, j])})
            have.add(int(i))
            k += 1
            if k >= per_cell:
                break
    print("expand: thin cells", sum(counts.get(f"{d}|{t}", 0) < thin_below for d, t in CELLS), "new rows", len(new_rows))
    pd.concat([cand, pd.DataFrame(new_rows)], ignore_index=True).to_parquet(WORK / "candidates_wildchat.parquet")


# ------------------------------------------------------------------------------------------------ classifier

CLS_PROMPT = """You label real requests that people sent to a chatbot. Read the request and answer with one line of JSON.

Domains: {domains}
Task types: {types}

Fields:
- "domain": one of the domain names above, or "other" if none fits well.
- "task_type": the one task type above that best matches what the person wants, or "other".
- "concrete": true if the person describes a concrete situation of their own (their own work, money, trip, school, home, business, purchase or event), false for general knowledge or hypothetical questions.
- "sensitive": true if it involves medical, legal, mental-health, sexual, violent, political or other sensitive matters.
- "personal_data": true if it contains a real person's full name, contact details, address, account or order number, or other identifying data.

Request:
<<<
{text}
>>>

Answer with JSON only, for example {{"domain": "travel", "task_type": "planning", "concrete": true, "sensitive": false, "personal_data": false}}"""


def _cls_prompt(text: str) -> str:
    doms = "; ".join(f"{k} ({v})" for k, v in CFG["domains"].items())
    typs = "; ".join(f"{k} ({v})" for k, v in CFG["task_types"].items())
    return CLS_PROMPT.format(domains=doms, types=typs, text=text[:1500])


def _parse_cls(s: str) -> dict:
    m = re.search(r"\{.*\}", s, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    return {"domain": d.get("domain") if d.get("domain") in DOMAINS else "other",
            "task_type": d.get("task_type") if d.get("task_type") in TYPES else "other",
            "concrete": bool(d.get("concrete", False)), "sensitive": bool(d.get("sensitive", True)),
            "personal_data": bool(d.get("personal_data", True)), "parsed": bool(d)}


def step_classify(sources=("wildchat", "dolly")) -> None:
    from mlx_lm import batch_generate, load
    cache_f = WORK / "classify_cache.jsonl"
    cache = {}
    if cache_f.exists():
        for line in cache_f.read_text().splitlines():
            r = json.loads(line)
            cache[r["text_hash"]] = r
    model, tok = load(CFG["classifier"]["model"])
    bs = CFG["classifier"]["batch_size"]
    for name in sources:
        base = pd.read_parquet(WORK / ("filtered.parquet" if name == "wildchat" else "dolly_filtered.parquet"))
        cand = pd.read_parquet(WORK / f"candidates_{name}.parquet")
        todo = [i for i in cand.row if base.text_hash.iloc[i] not in cache]
        print(name, "to classify", len(todo), "cached", len(cand) - len(todo), flush=True)
        t0 = time.time()
        with cache_f.open("a") as fh:
            for b in range(0, len(todo), bs):
                idx = todo[b:b + bs]
                prompts = [tok.apply_chat_template([{"role": "user", "content": _cls_prompt(base.text.iloc[i])}],
                                                   add_generation_prompt=True) for i in idx]
                res = batch_generate(model, tok, prompts, max_tokens=CFG["classifier"]["max_tokens"])
                for i, out in zip(idx, res.texts):
                    r = {"text_hash": base.text_hash.iloc[i], "raw": out, **_parse_cls(out)}
                    cache[r["text_hash"]] = r
                    fh.write(json.dumps(r) + "\n")
                fh.flush()
                if b // bs % 10 == 0:
                    el = time.time() - t0
                    print(f"{name} {b + len(idx)}/{len(todo)} {el:.0f}s ({(b + len(idx)) / max(el, 1e-9):.1f}/s)",
                          flush=True)


# ------------------------------------------------------------------------------------------------ selection

def _accepted(name: str) -> pd.DataFrame:
    base = pd.read_parquet(WORK / ("filtered.parquet" if name == "wildchat" else "dolly_filtered.parquet"))
    cand = pd.read_parquet(WORK / f"candidates_{name}.parquet")
    cache = {}
    for line in (WORK / "classify_cache.jsonl").read_text().splitlines():
        r = json.loads(line)
        cache[r["text_hash"]] = r
    E = np.load(WORK / f"emb_{name}.npy")
    C = np.load(WORK / "emb_cells.npy")
    rows = []
    for r in cand.itertuples(index=False):
        b = base.iloc[r.row]
        c = cache.get(b.text_hash)
        if c is None:
            continue
        cell = f"{c['domain']}|{c['task_type']}"
        ok = (c["domain"] != "other" and c["task_type"] != "other" and c["concrete"] and not c["sensitive"]
              and not c["personal_data"])
        j = CELLS.index((c["domain"], c["task_type"])) if ok else None
        rows.append({**b.to_dict(), "row": r.row, "cell": cell, "accepted": ok,
                     "cell_cos": float(E[r.row] @ C[j]) if ok else np.nan, "retrieved_for": list(r.retrieved_for),
                     "cls_raw": c["raw"], "cls_concrete": c["concrete"], "cls_sensitive": c["sensitive"],
                     "cls_personal_data": c["personal_data"]})
    df = pd.DataFrame(rows)
    df["emb_source"] = name
    return df


def step_select() -> pd.DataFrame:
    sel = CFG["selection"]
    rng = random.Random(sel["seed"])
    wc, do = _accepted("wildchat"), _accepted("dolly")
    E = {"wildchat": np.load(WORK / "emb_wildchat.npy"), "dolly": np.load(WORK / "emb_dolly.npy")}
    pool = {"wildchat": wc[wc.accepted].copy(), "dolly": do[do.accepted].copy()}
    chosen: list[dict] = []
    vecs: list[np.ndarray] = []
    used_users, used_convs = set(), set()

    def try_take(r, role, cell) -> bool:
        if r.user_hash in used_users or r.conversation_hash in used_convs:
            return False
        v = E[r.emb_source][r.row]
        if vecs and float(np.max(np.stack(vecs) @ v)) >= CFG["embedding"]["near_duplicate_cosine"]:
            return False
        used_users.add(r.user_hash)
        used_convs.add(r.conversation_hash)
        vecs.append(v)
        chosen.append({**{k: getattr(r, k) for k in ("source", "conversation_hash", "user_hash", "model",
                                                     "timestamp", "country", "text", "text_hash", "cell_cos",
                                                     "cls_raw")},
                       "domain": cell[0], "task_type": cell[1], "role": role,
                       "dolly_fill": r.source == "dolly"})
        return True

    def by_cell(name, cell):
        p = pool[name]
        return p[p.cell == f"{cell[0]}|{cell[1]}"].sort_values("cell_cos", ascending=False)

    counts = {c: len(by_cell("wildchat", c)) for c in CELLS}
    # main seeds first (thinnest cells first), then spares, so scarce candidates go to main seeds
    for role_i, role in enumerate(["seed", "spare1", "spare2"]):
        for cell in sorted(CELLS, key=lambda c: (counts[c], CELLS.index(c))):
            taken = False
            for r in by_cell("wildchat", cell).itertuples(index=False):
                if try_take(r, role, cell):
                    taken = True
                    break
            if not taken:  # WildChat exhausted for this slot: top up from Dolly (flagged dolly_fill)
                for r in by_cell("dolly", cell).itertuples(index=False):
                    if try_take(r, role, cell):
                        taken = True
                        break
            if not taken:
                chosen.append({"domain": cell[0], "task_type": cell[1], "role": role, "source": None,
                               "dolly_fill": False, "text": None, "missing": True})
    # pilot seeds: 1 per domain, task type rotated; calibration and confirmation use different rotations
    for k, role in ((0, "calibration"), (2, "confirmation")):
        for i, d in enumerate(DOMAINS):
            pref = [TYPES[(i + k + s) % len(TYPES)] for s in range(len(TYPES))]
            taken = False
            for t in pref:
                for name in ("wildchat", "dolly"):
                    for r in by_cell(name, (d, t)).itertuples(index=False):
                        if try_take(r, role, (d, t)):
                            taken = True
                            break
                    if taken:
                        break
                if taken:
                    break
            if not taken:
                chosen.append({"domain": d, "task_type": None, "role": role, "source": None, "text": None,
                               "missing": True, "dolly_fill": False})
    out = pd.DataFrame(chosen)
    out["missing"] = out.get("missing", False)
    out["missing"] = out["missing"].fillna(False).astype(bool)
    order = {r: i for i, r in enumerate(["seed", "spare1", "spare2", "calibration", "confirmation"])}
    out["_o"] = out.role.map(order)
    out = out.sort_values(["_o", "domain", "task_type"], na_position="last").drop(columns="_o").reset_index(drop=True)
    ids = {(d, t): f"sc{i + 1:02d}" for i, (d, t) in enumerate(CELLS)}
    out["seed_id"] = [
        (ids[(r.domain, r.task_type)] + ("" if r.role == "seed" else f"-{r.role}")) if r.role in ("seed", "spare1", "spare2")
        else None for r in out.itertuples()]
    for role, pre in (("calibration", "pc"), ("confirmation", "pf")):
        m = out.role == role
        out.loc[m, "seed_id"] = [f"{pre}{i + 1:02d}" for i in range(int(m.sum()))]
    out["provenance"] = [
        (f"WildChat-1M@{CFG['wildchat']['revision'][:12]} conversation_hash={r.conversation_hash}" if r.source == "wildchat"
         else f"Dolly-15k@{CFG['dolly']['revision'][:12]} {r.conversation_hash}" if r.source == "dolly" else None)
        for r in out.itertuples()]
    out["seeds_version"] = CFG["version"]
    out = apply_replacements(out)
    out.to_parquet(DATA / "seeds.parquet")
    # coverage: accepted WildChat candidates per cell, and what was picked
    cov = pd.DataFrame([{"domain": d, "task_type": t, "wildchat_accepted": counts[(d, t)],
                         "dolly_accepted": len(by_cell("dolly", (d, t))),
                         "picked_wildchat": int(((out.domain == d) & (out.task_type == t) & (out.source == "wildchat")
                                                 & out.role.isin(["seed", "spare1", "spare2"])).sum()),
                         "picked_dolly": int(((out.domain == d) & (out.task_type == t) & (out.source == "dolly")
                                              & out.role.isin(["seed", "spare1", "spare2"])).sum())}
                        for d, t in CELLS])
    cov.to_csv(DATA / "seeds_coverage.csv", index=False)
    summ = {"version": CFG["version"], "n_rows": len(out), "by_role_source": out.groupby(["role", "source"], dropna=False)
            .size().reset_index().values.tolist(),
            "missing": out[out.missing][["role", "domain", "task_type"]].values.tolist(),
            "wildchat_classified": len(wc), "wildchat_accepted": int(wc.accepted.sum()),
            "dolly_classified": len(do), "dolly_accepted": int(do.accepted.sum()),
            "unique_users": int(out.user_hash.dropna().nunique()), "n_with_user": int(out.user_hash.notna().sum())}
    (DATA / "seeds_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))
    return out


REPLACEMENTS = ROOT / "configs" / "difficulty" / "seed_replacements.yaml"


def _replacement_log() -> list[dict]:
    return (yaml.safe_load(REPLACEMENTS.read_text()) or {}).get("replacements", []) if REPLACEMENTS.exists() else []


def _row_vec(r) -> np.ndarray:
    name = "wildchat" if r["source"] == "wildchat" else "dolly"
    base = pd.read_parquet(WORK / ("filtered.parquet" if name == "wildchat" else "dolly_filtered.parquet"))
    i = int(np.flatnonzero(base.conversation_hash.values == r["conversation_hash"])[0])
    return np.load(WORK / f"emb_{name}.npy")[i]


def pick_replacement(out: pd.DataFrame, seed_id: str) -> dict:
    """Next-best accepted candidate for the replaced seed's cell: a user and conversation not used by any seed row
    (including replaced ones) and cosine < threshold to every seed row."""
    old = out[out.seed_id == seed_id].iloc[0]
    cell = f"{old.domain}|{old.task_type}"
    used_u, used_c = set(out.user_hash.dropna()), set(out.conversation_hash.dropna())
    vecs = np.stack([_row_vec(r) for r in out[out.source.notna()].to_dict("records")])
    for name in ("wildchat", "dolly"):
        acc = _accepted(name)
        acc = acc[acc.accepted & (acc.cell == cell)].sort_values("cell_cos", ascending=False)
        E = np.load(WORK / f"emb_{name}.npy")
        for r in acc.itertuples(index=False):
            if r.user_hash in used_u or r.conversation_hash in used_c:
                continue
            if float(np.max(vecs @ E[r.row])) >= CFG["embedding"]["near_duplicate_cosine"]:
                continue
            return {k: getattr(r, k) for k in ("source", "conversation_hash", "user_hash", "model", "timestamp",
                                               "country", "text", "text_hash", "cell_cos", "cls_raw")}
    raise RuntimeError(f"no replacement candidate for {seed_id} ({cell})")


def apply_replacements(out: pd.DataFrame) -> pd.DataFrame:
    """Re-apply configs/difficulty/seed_replacements.yaml: the old row keeps its id with role 'replaced', the new
    row takes the old role under new_seed_id and the logged conversation."""
    for rep in _replacement_log():
        m = out.seed_id == rep["seed_id"]
        if not m.any() or (out.seed_id == rep["new_seed_id"]).any():
            continue
        old = out[m].iloc[0].to_dict()
        out.loc[m, "role"] = "replaced"
        name = rep["new_source"]
        acc = _accepted("wildchat" if name == "wildchat" else "dolly")
        r = acc[acc.conversation_hash == rep["new_conversation_hash"]].iloc[0]
        new = {**old, **{k: r[k] for k in ("source", "conversation_hash", "user_hash", "model", "timestamp", "country",
                                            "text", "text_hash", "cell_cos", "cls_raw")},
               "seed_id": rep["new_seed_id"], "role": old["role"], "dolly_fill": name == "dolly", "missing": False,
               "replaces": rep["seed_id"]}
        new["provenance"] = (f"WildChat-1M@{CFG['wildchat']['revision'][:12]} conversation_hash={new['conversation_hash']}"
                             if name == "wildchat" else f"Dolly-15k@{CFG['dolly']['revision'][:12]} {new['conversation_hash']}")
        out = pd.concat([out, pd.DataFrame([new])], ignore_index=True)
    return out


def step_replace(seed_id: str, reason: str) -> dict:
    out = pd.read_parquet(DATA / "seeds.parquet")
    old = out[out.seed_id == seed_id].iloc[0]
    new = pick_replacement(out, seed_id)
    log = _replacement_log()
    stem = seed_id.rstrip("bcdefgh") if seed_id[-1] in "bcdefgh" and seed_id[:-1][-1].isdigit() else seed_id
    n = sum(r["seed_id"].rstrip("bcdefgh") == stem or r["seed_id"] == stem for r in log)
    rep = {"seed_id": seed_id, "new_seed_id": f"{stem}{'bcdefgh'[n]}", "cell": f"{old.domain}/{old.task_type}",
           "role": old.role, "old_conversation_hash": old.conversation_hash, "new_source": new["source"],
           "new_conversation_hash": new["conversation_hash"], "reason": reason,
           "date": time.strftime("%Y-%m-%d")}
    log.append(rep)
    REPLACEMENTS.write_text("# Seed replacements (difficulty dataset). Applied by `seeds select` after selection.\n"
                            + yaml.safe_dump({"replacements": log}, sort_keys=False, width=120))
    out = apply_replacements(out)
    out.to_parquet(DATA / "seeds.parquet")
    print(json.dumps(rep, indent=1))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["filter", "embed", "classify", "expand", "select", "replace", "all"])
    ap.add_argument("--seed-id")
    ap.add_argument("--reason")
    a = ap.parse_args()
    if a.step in ("filter", "all"):
        step_filter()
    if a.step in ("embed", "all"):
        step_embed()
    if a.step in ("classify", "all"):
        step_classify()
    if a.step == "replace":
        step_replace(a.seed_id, a.reason)
    if a.step == "expand":
        step_expand()
    if a.step in ("select", "all"):
        step_select()


if __name__ == "__main__":
    main()
