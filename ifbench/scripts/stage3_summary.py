"""Stage 3: build the pairwise compatibility matrix and summary from the final run + written proofs.

Inputs: data/stage3/final.jsonl (one builder version for every non-proven pair),
src/mife_ifb/stage3_proofs.py, data/stage3/six_pilot.jsonl (if present), data/stage3/timing.jsonl.
Outputs: freeze/stage3_pairs.json (per-pair status and success share) and
freeze/stage3_summary.json (counts, lists, per-type tables, pilot acceptance).
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

IFB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IFB / "src"))
sys.path.insert(0, str(IFB / "scripts"))
from mife_ifb import inventory, stage3_proofs as P  # noqa: E402
import stage3_audit as A  # noqa: E402

D = IFB / "data" / "stage3"
# Pairs whose 50 draws did not finish: the run for this pair went for more than an hour of builder
# time (2026-11-23, 20:08-21:08 UTC) without completing, so it counts as unsupported by the
# builder. No result row is invented for it.
TIMED_OUT = {("ratio:overlap", "words:keywords_specific_position")}


def load(name):
    f = D / name
    return [json.loads(l) for l in f.open()] if f.exists() else []


def main():
    final = {tuple(r["pair"]): r for r in load("final.jsonl")}
    full = {tuple(r["pair"]): r for r in load("full.jsonl")}
    pairs = A.all_pairs()
    proven, hand, loop = P.proven(), P.hand_witnesses(), P.loophole_only()
    shas = Counter(r["builder_sha"] for r in final.values())
    assert len(shas) == 1, f"final run mixes builder versions: {shas}"
    missing = [p for p in pairs if p not in final and p not in proven and p not in TIMED_OUT]
    assert not missing, f"pairs with no result: {missing[:5]}"
    # a proof must never meet a witness in any run
    for p in proven:
        for run in (final, full):
            assert not (p in run and run[p]["compatible"]), f"proven pair {p} has a witness"

    rows = []
    for p in pairs:
        if p in proven:
            basis, arg = proven[p]
            rows.append({"pair": list(p), "status": "proven_conflict", "basis": basis, "argument": arg})
            continue
        if p in TIMED_OUT:
            rows.append({"pair": list(p), "status": "unsupported_by_builder", "why": "timed_out",
                         "top_reasons": [["50 draws did not finish within an hour", 0]]})
            continue
        r = final[p]
        row = {"pair": list(p), "success": r["success"], "draws": r["draws"], "share": r["share"],
               "seconds": r["seconds"]}
        if r["compatible"]:
            row["status"] = "compatible"
        else:
            row["status"] = "unsupported_by_builder"
            row["why"] = ("hand_witness" if p in hand else "loophole_only" if p in loop else "no_witness_no_proof")
            row["top_reasons"] = r["top_reasons"][:3]
            if p in loop:
                row["note"] = loop[p]
        rows.append(row)

    st = Counter(r["status"] for r in rows)
    comp = [r for r in rows if r["status"] == "compatible"]
    shares = [r["share"] for r in comp]
    types = inventory.candidates()
    per_type = {}
    for t in types:
        mine = [r for r in rows if t in r["pair"]]
        c = [r for r in mine if r["status"] == "compatible"]
        per_type[t] = {"compatible": len(c), "proven": sum(r["status"] == "proven_conflict" for r in mine),
                       "unsupported": sum(r["status"] == "unsupported_by_builder" for r in mine),
                       "mean_share": round(mean(r["share"] for r in c), 3) if c else None}

    # overlap: success per provisional task
    ov = defaultdict(lambda: [0, 0])
    for p, r in final.items():
        if "ratio:overlap" in p:
            for d in r["draw_records"]:
                ov[d["task_index"]][0] += d["ok"]
                ov[d["task_index"]][1] += 1
    ov_tasks = {"tasks_tried": len(ov), "tasks_with_any_witness": sum(1 for a, b in ov.values() if a),
                "draw_success_share": round(sum(a for a, _ in ov.values()) / max(1, sum(b for _, b in ov.values())), 3)}

    six = load("six_pilot.jsonl")
    pilot = None
    if six:
        acc = Counter()
        tot = Counter()
        for r in six:
            for t in r["pool"]:
                tot[t] += 1
                acc[t] += r["ok"]
        reasons = Counter()
        for r in six:
            if not r["ok"]:
                for k, v in r["reasons"]:
                    reasons[k.split(":")[0] if k.startswith("verify") else k] += v
        ok = [r for r in six if r["ok"]]
        pilot = {"pools": len(six), "accepted": len(ok), "acceptance": round(len(ok) / len(six), 3),
                 "per_type": {t: {"pools": tot[t], "accepted": acc[t], "acceptance": round(acc[t] / tot[t], 3)}
                              for t in sorted(tot)},
                 "types_never_sampled": sorted(set(types) - set(tot)),
                 "witness_words": {"median": median(r["words"] for r in ok), "max": max(r["words"] for r in ok)} if ok else None,
                 "seconds": {"median": median(r["sec"] for r in six), "max": max(r["sec"] for r in six),
                             "total": round(sum(r["sec"] for r in six))},
                 "top_failure_reasons": reasons.most_common(10)}

    lists = {s: sorted([r["pair"] for r in rows if r["status"] == s]) for s in ("proven_conflict", "unsupported_by_builder")}
    summary = {
        "builder_sha": next(iter(shas)),
        "witness_py_sha256": hashlib.sha256((IFB / "src" / "mife_ifb" / "witness.py").read_bytes()).hexdigest(),
        "seed": A.SEED, "draws": A.DRAWS, "attempts": A.ATTEMPTS,
        "pairs": len(pairs), "counts": dict(st),
        "proven_by_basis": dict(Counter(r["basis"] for r in rows if r["status"] == "proven_conflict")),
        "unsupported_by_why": dict(Counter(r["why"] for r in rows if r["status"] == "unsupported_by_builder")),
        "compatible_share": {"median": median(shares), "min": min(shares),
                             "below_0.1": sum(s < 0.1 for s in shares), "below_0.5": sum(s < 0.5 for s in shares),
                             "all_50": sum(s == 1 for s in shares)},
        "final_run_seconds": round(sum(r["seconds"] for r in final.values())),
        "overlap_tasks": ov_tasks,
        "proven_conflicts": lists["proven_conflict"],
        "unsupported_by_builder": [{"pair": r["pair"], "why": r["why"], "top_reasons": r["top_reasons"]}
                                   for r in rows if r["status"] == "unsupported_by_builder"],
        "low_share_pairs": sorted(([r["pair"], r["success"]] for r in comp if r["share"] < 0.1), key=lambda x: x[1]),
        "per_type": per_type,
        "six_rule_pilot": pilot,
    }
    (IFB / "freeze" / "stage3_pairs.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    (IFB / "freeze" / "stage3_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("counts", "proven_by_basis", "unsupported_by_why", "compatible_share",
                                              "overlap_tasks")}, indent=1))
    if pilot:
        print(json.dumps({k: pilot[k] for k in ("pools", "accepted", "acceptance", "types_never_sampled", "seconds",
                                                "witness_words")}, indent=1))


if __name__ == "__main__":
    main()
