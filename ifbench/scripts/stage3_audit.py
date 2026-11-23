"""Stage 3: pairwise witness audit (D1) over all pairs of the 42 candidate types.

For each pair: 50 parameter draws from the released default generators; for each draw the
builder gets up to 24 seeded attempts and stops at the first witness. A pair is compatible if
any draw yields a witness (official verifiers + frozen plain-meaning checks). Records the
success share and the failure reasons of every draw.

ratio:overlap pairs use provisional base tasks (released prompts with the constraint text
stripped, verbatim rows only; texts, no labels), one task per draw, recorded per task. Stage 4
replaces these with the cleaned task list.

Resumable: results append to data/stage3/<run>.jsonl and finished pairs are skipped.
Usage: python scripts/stage3_audit.py timing|full|rerun|final [--workers 4]
The report uses the final run (one builder version for every pair); full/rerun were investigation passes.
"""
import argparse
import itertools
import json
import multiprocessing as mp
import random
import sys
import time
from collections import Counter
from pathlib import Path

IFB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IFB / "src"))
from mife_ifb import inventory, source, witness  # noqa: E402

SEED = 20261124
DRAWS, ATTEMPTS = 50, 24
HYPOTHESES = [  # expected conflicts written in the plan (Step 3), used to mix the timing sample
    ("format:list", "format:sub-bullets"), ("format:list", "format:no_bullets_bullets"),
    ("format:sub-bullets", "format:no_bullets_bullets"), ("ratio:sentence_type", "ratio:sentence_balance"),
    ("words:alphabet", "words:vowel"), ("words:alphabet", "words:prime_lengths"), ("words:vowel", "words:consonants"),
    ("words:prime_lengths", "words:odd_even_syllables"), ("words:no_consecutive", "words:alphabet"),
    ("format:newline", "format:line_indent"), ("format:newline", "format:no_whitespace"), ("format:newline", "words:vowel"),
    ("format:output_template", "words:alphabet"), ("format:output_template", "words:vowel"),
    ("words:repeats", "count:keywords_multiple"), ("format:no_whitespace", "count:unique_word_count"),
    ("count:numbers", "words:consonants"), ("words:odd_even_syllables", "words:consonants"),
    ("format:line_indent", "format:no_whitespace"), ("words:vowel", "format:line_indent"),
    ("format:output_template", "words:consonants"), ("format:output_template", "words:prime_lengths"),
    ("words:alphabet", "words:odd_even_syllables"), ("count:numbers", "words:alphabet"), ("words:vowel", "words:prime_lengths"),
]


def provisional_tasks() -> list[str]:
    rows = source.source_rows()
    out = []
    for r in rows:
        p, ok = r["prompt"], True
        for iid, kw in zip(r["instruction_id_list"], r["kwargs"]):
            try:
                d = source.instruction(iid).build_description(**source._clean_kwargs(kw))
            except Exception:
                ok = False
                break
            if d not in p:
                ok = False
                break
            p = p.replace(d, "", 1)
        p = " ".join(p.split())
        if ok and len(p.split()) >= 5:
            out.append(p)
    return sorted(set(out))


def all_pairs():
    c = inventory.candidates()
    return [tuple(sorted(p)) for p in itertools.combinations(c, 2)]


def run_pair(args):
    i, pair = args
    tasks = provisional_tasks() if "ratio:overlap" in pair else None
    t0 = time.time()
    draws = []
    example = None
    for d in range(DRAWS):
        rules = {t: source.sample_default_kwargs(t, rng_seed=SEED + i * 1000 + d * 10 + k) for k, t in enumerate(pair)}
        for kw in rules.values():
            kw.pop("reference_text", None)
        task = tasks[(i * 7 + d) % len(tasks)] if tasks else None
        td = time.time()
        try:
            r = witness.build_witness(rules, task_text=task, seed=d, attempts=ATTEMPTS)
        except Exception as e:  # a builder bug must not stop the audit; it is recorded and fixed
            r = witness.Result(ok=False)
            r.reasons[f"BUILDER ERROR {type(e).__name__}: {e}"] += 1
        rec = {"d": d, "ok": r.ok, "attempt": r.attempt, "sec": round(time.time() - td, 2),
               "params": rules, "reasons": r.reasons.most_common(3)}
        if task:
            rec["task_index"] = tasks.index(task)
        draws.append(rec)
        if r.ok and example is None:
            example = {"params": rules, "text": r.text, "task": task}
    n_ok = sum(x["ok"] for x in draws)
    reasons = Counter()
    for x in draws:
        for k, v in x["reasons"]:
            reasons[k] += v
    import hashlib
    bsha = hashlib.sha256((IFB / "src" / "mife_ifb" / "witness.py").read_bytes()).hexdigest()[:12]
    return {"pair": list(pair), "index": i, "builder_sha": bsha, "draws": DRAWS, "success": n_ok, "share": n_ok / DRAWS,
            "compatible": n_ok > 0, "seconds": round(time.time() - t0, 1), "top_reasons": reasons.most_common(5),
            "draw_records": draws, "example": example}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["timing", "full", "rerun", "final"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    pairs = all_pairs()
    index = {p: i for i, p in enumerate(pairs)}
    if a.mode == "timing":
        hyp = [tuple(sorted(p)) for p in HYPOTHESES]
        rest = [p for p in pairs if p not in hyp]
        random.Random(SEED).shuffle(rest)
        chosen = hyp + rest[:50 - len(hyp)]
    elif a.mode == "rerun":
        # pairs touched by the builder extensions made while investigating no-witness pairs
        # (ordinal numbers, thesis punctuation, overlap placement, unique-word sizing) or hit by a builder error
        touched = {"ratio:overlap", "count:numbers", "format:thesis", "count:unique_word_count"}
        full = [json.loads(l) for l in (IFB / "data" / "stage3" / "full.jsonl").open()]
        err = {tuple(r["pair"]) for r in full if any("BUILDER ERROR" in k for k, _ in r["top_reasons"])}
        chosen = [p for p in pairs if touched & set(p) or p in err]
    elif a.mode == "final":
        # every pair with the final builder, except the written proofs (stage3_proofs.PROOFS)
        from mife_ifb import stage3_proofs
        chosen = [p for p in pairs if p not in stage3_proofs.proven()]
    else:
        chosen = pairs
    out = IFB / "data" / "stage3" / f"{a.mode}.jsonl"
    done = set()
    if out.exists():
        done = {tuple(json.loads(l)["pair"]) for l in out.open()}
    todo = [(index[p], p) for p in chosen if p not in done]
    print(f"{len(todo)} pairs to run ({len(done)} done)", flush=True)
    t0 = time.time()
    with mp.Pool(a.workers) as pool, out.open("a") as f:
        for k, res in enumerate(pool.imap_unordered(run_pair, todo), 1):
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{k}/{len(todo)} {time.time() - t0:7.0f}s] {res['pair'][0]} + {res['pair'][1]}: "
                  f"{res['success']}/{DRAWS} in {res['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
