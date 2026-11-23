"""Stage 3: six-rule feasibility pilot (an estimate, not exhaustive).

Samples random pools of six distinct candidate types that avoid every pair with no witness in
the pairwise audit (sequential uniform proposals, as ManyIFEval samples types), draws their
parameters from the released defaults, and asks the builder for one joint six-rule witness
(24 attempts). Reports acceptance per pool and per type.
Usage: python scripts/stage3_six_pilot.py [n_pools] [--workers 4]
"""
import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

IFB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IFB / "src"))
sys.path.insert(0, str(IFB / "scripts"))
from mife_ifb import inventory, source, witness  # noqa: E402
import stage3_audit as A  # noqa: E402

SEED = 20261125


def blocked_pairs():
    rows = [json.loads(l) for l in (IFB / "data" / "stage3" / "full.jsonl").open()]
    return {frozenset(r["pair"]) for r in rows if not r["compatible"]}


def sample_pool(rng, types, blocked):
    while True:
        pool = []
        cand = types[:]
        rng.shuffle(cand)
        for t in cand:
            if all(frozenset((t, u)) not in blocked for u in pool):
                pool.append(t)
            if len(pool) == 6:
                return pool


def run_pool(args):
    i, pool, task = args
    rules = {t: source.sample_default_kwargs(t, rng_seed=SEED + i * 100 + k) for k, t in enumerate(pool)}
    for kw in rules.values():
        kw.pop("reference_text", None)
    t0 = time.time()
    r = witness.build_witness(rules, task_text=task, seed=i, attempts=24)
    return {"i": i, "pool": pool, "params": rules, "ok": r.ok, "attempt": r.attempt, "sec": round(time.time() - t0, 2),
            "reasons": r.reasons.most_common(3), "text": r.text, "task": task,
            "words": len(r.text.split()) if r.text else None, "chars": len(r.text) if r.text else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?", default=300)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    rng = random.Random(SEED)
    blocked = blocked_pairs()
    types = inventory.candidates()
    tasks = A.provisional_tasks()
    jobs = []
    for i in range(a.n):
        pool = sample_pool(rng, types, blocked)
        jobs.append((i, pool, rng.choice(tasks) if "ratio:overlap" in pool else None))
    out = IFB / "data" / "stage3" / "six_pilot.jsonl"
    done = {json.loads(l)["i"] for l in out.open()} if out.exists() else set()
    jobs = [j for j in jobs if j[0] not in done]
    t0 = time.time()
    with mp.Pool(a.workers) as p, out.open("a") as f:
        for k, res in enumerate(p.imap_unordered(run_pool, jobs), 1):
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{k}/{len(jobs)} {time.time() - t0:6.0f}s] ok={res['ok']} {res['sec']}s", flush=True)


if __name__ == "__main__":
    main()
