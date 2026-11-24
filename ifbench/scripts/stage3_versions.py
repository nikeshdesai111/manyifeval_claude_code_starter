"""Stage 3: apply the version rule (Nicky, 2026-11-24) to the pairwise audit.

Rule: every *version* of a pair gets exactly 24 builder attempts.
- Both rules fixed (no parameters): one version.
- At least one parameterised rule with fewer than 50 unique parameter combinations: one version
  per combination (all of them).
- Otherwise: 50 versions (parameter draws), as before.

The builder is deterministic given (rules, parameters, task text, seed), and witness.py is unchanged
since the final run, so a version already run in data/stage3/final.jsonl is replayed from its record:
- fixed pairs: the final run's draw 0 (seed 0);
- small pairs: for each combination, the first final-run draw that used it (its seed);
- other pairs: the 50 final-run draws.
Combinations the final run never drew are run now with seed 50 + their index in the sorted list
of combinations (no earlier draw of the pair used those seeds). For every compatible pair the first
successful version is rebuilt to store its witness text and to confirm the replay.

Builder change after the final run (Nicky approved 2026-11-24): the thesis full stop stays inside
</i> when words:last_first is present. The change is inside the thesis layout branch of
witness.compose, so it can only alter texts for pools that contain format:thesis. Every pair with
format:thesis is therefore rerun with the current builder (same versions, same seeds, same task
texts); every other pair is replayed from the final run, whose builder differs only in that branch.

Output: data/stage3/versions.jsonl (git-ignored, resumable).
Usage: python scripts/stage3_versions.py [--workers 4]
"""
import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

IFB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IFB / "src"))
sys.path.insert(0, str(IFB / "scripts"))
from mife_ifb import inventory, source, stage3_proofs as P, witness  # noqa: E402
import stage3_audit as A  # noqa: E402

D = IFB / "data" / "stage3"
ATTEMPTS = 24
FINAL_SHA = "1ae17eb3ee01"   # witness.py of the final run
RERUN_TYPES = {"format:thesis"}  # pairs whose texts the post-final builder change can alter
# Every value the released default generator can draw, for the types with fewer than 50 (checked
# against 20,000 seeded draws in tests/test_stage3_versions.py).
SMALL_VALUES = {
    "format:list": [{"sep": s} for s in ("-", "...", "!?!?", "SEPARATOR")],
    "count:conjunctions": [{"small_n": n} for n in range(2, 7)],
    "words:repeats": [{"small_n": n} for n in range(1, 6)],
    "sentence:increment": [{"small_n": n} for n in range(1, 6)],
    "count:numbers": [{"N": n} for n in range(1, 7)],
    "count:pronouns": [{"N": n} for n in range(1, 26)],
}


def n_combinations(pair, big=10 ** 6):
    n = 1
    for t in pair:
        if t in inventory.parameterized():
            n *= len(SMALL_VALUES[t]) if t in SMALL_VALUES else big
    return n


def category(pair):
    if not set(pair) & set(inventory.parameterized()):
        return "fixed"
    return "all_combinations" if n_combinations(pair) < 50 else "draws"


def combinations(pair):
    """Sorted list of {type: kwargs} for a pair with fewer than 50 combinations."""
    lists = [SMALL_VALUES[t] if t in SMALL_VALUES else [{}] for t in pair]
    out = [dict(zip(pair, c)) for c in itertools.product(*lists)]
    return sorted(out, key=lambda r: json.dumps(r, sort_keys=True))


def key(rules):
    return json.dumps(rules, sort_keys=True)


def run_new(args):
    pair, rules, seed, task_index = args
    task = A.provisional_tasks()[task_index] if task_index is not None else None
    t0 = time.time()
    try:
        r = witness.build_witness(rules, task_text=task, seed=seed, attempts=ATTEMPTS)
    except Exception as e:  # recorded, not hidden
        r = witness.Result(ok=False)
        r.reasons[f"BUILDER ERROR {type(e).__name__}: {e}"] += 1
    return {"pair": pair, "params": rules, "seed": seed, "ok": r.ok, "attempt": r.attempt,
            "sec": round(time.time() - t0, 2), "reasons": r.reasons.most_common(3), "source": "new",
            "text": r.text, **({"task_index": task_index} if task_index is not None else {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    final = {tuple(json.loads(l)["pair"]): json.loads(l) for l in (D / "final.jsonl").open()}
    proven = P.proven()
    bsha = hashlib.sha256((IFB / "src" / "mife_ifb" / "witness.py").read_bytes()).hexdigest()[:12]
    shas = {r["builder_sha"] for r in final.values()}
    assert shas == {FINAL_SHA}, f"unexpected final-run builder: {shas}"
    out = D / "versions.jsonl"
    done = {tuple(json.loads(l)["pair"]) for l in out.open()} if out.exists() else set()

    plans, jobs = {}, []
    for p in A.all_pairs():
        if p in proven or p in done:
            continue
        cat = category(p)
        r = final.get(p)
        if cat == "draws":
            plans[p] = (cat, [dict(rec, seed=rec["d"], source="replay") for rec in r["draw_records"]] if r else None)
            continue
        if cat == "fixed":
            d0 = [x for x in r["draw_records"] if x["d"] == 0][0]
            assert all(v == {} for v in d0["params"].values())
            plans[p] = (cat, [dict(d0, seed=0, source="replay")])
            continue
        first = {}
        for rec in sorted(r["draw_records"], key=lambda x: x["d"]):
            first.setdefault(key(rec["params"]), rec)
        vers = []
        for i, combo in enumerate(combinations(p)):
            rec = first.get(key(combo))
            if rec is not None:
                vers.append(dict(rec, seed=rec["d"], source="replay"))
            else:
                vers.append(None)
                jobs.append((p, combo, 50 + i, None))
        plans[p] = (cat, vers)
    # pairs the builder change can affect: every version is run again with the current builder
    for p, (cat, vers) in list(plans.items()):
        if not (set(p) & RERUN_TYPES) or vers is None:
            continue
        combos = combinations(p) if cat == "all_combinations" else None
        jobs = [j for j in jobs if j[0] != p]
        for i, v in enumerate(vers):
            if v is None:
                jobs.append((p, combos[i], 50 + i, None))
            else:
                jobs.append((p, v["params"], v["seed"], v.get("task_index")))
        plans[p] = (cat, [None] * len(vers))
    jobs = [j if len(j) == 4 else j + (None,) for j in jobs]
    print(f"{len(plans)} pairs, {len(jobs)} versions to run with the current builder", flush=True)

    t0 = time.time()
    new = {}
    with mp.Pool(a.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(run_new, jobs), 1):
            new[(tuple(res["pair"]), key(res["params"]), res["seed"])] = res
            if k % 20 == 0 or k == len(jobs):
                print(f"[{k}/{len(jobs)} {time.time() - t0:5.0f}s]", flush=True)

    with out.open("a") as f:
        for p, (cat, vers) in plans.items():
            if vers is None:  # the timed-out pair: no record exists, none is invented
                continue
            if any(v is None for v in vers):
                mine = [j for j in jobs if j[0] == p]
                if all(v is None for v in vers):
                    vers = [new[(p, key(j[1]), j[2])] for j in mine]
                else:
                    it = iter(mine)
                    vers = [v if v is not None else new[(p, key(j[1]), j[2])] for v, j in
                            ((v, next(it) if v is None else None) for v in vers)]
            rerun = bool(set(p) & RERUN_TYPES)
            ok = [v for v in vers if v["ok"]]
            example = None
            if ok:
                if rerun:
                    v = ok[0]
                    example = {"params": v["params"], "text": v["text"], "seed": v["seed"],
                               "task": A.provisional_tasks()[v["task_index"]] if "task_index" in v else None}
                elif cat == "draws":
                    example = final[p]["example"]
                else:
                    v = ok[0]
                    w = witness.build_witness(v["params"], task_text=None, seed=v["seed"], attempts=ATTEMPTS)
                    assert w.ok, f"replay of {p} {v['params']} seed {v['seed']} did not reproduce"
                    # the replay used the current builder: this is also a check that it reproduces the old result
                    example = {"params": v["params"], "text": w.text, "seed": v["seed"]}
            rec = {"pair": list(p), "category": cat, "builder_sha": bsha if rerun else FINAL_SHA,
                   "rerun_with_current_builder": rerun, "versions": len(vers),
                   "success": len(ok), "share": len(ok) / len(vers), "compatible": bool(ok),
                   "new_versions": sum(v["source"] == "new" for v in vers),
                   "seconds": round(sum(v["sec"] for v in vers), 1),
                   "version_records": [{k: v[k] for k in ("params", "seed", "ok", "attempt", "sec", "reasons", "source")
                                        if k in v} | ({"task_index": v["task_index"]} if "task_index" in v else {})
                                       for v in vers],
                   "example": example}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
