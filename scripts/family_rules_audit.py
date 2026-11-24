"""Run the rule-based task-family rubric on all 216 tasks and compare with the hand labels.

    PYTHONPATH=src python scripts/family_rules_audit.py

Reads task texts from the pinned ManyIFEval data file (hash-checked against
configs/experiment.yaml) and the hand labels from configs/task_families_v1.csv. Writes:

    audits/family_rules/rule_labels.csv   per-task rule label, deciding cue, hand label
    audits/family_rules/agreement.json    accuracy, Cohen's kappa, confusion matrix, counts

It never modifies configs/task_families_v1.csv; the pipeline keeps using the hand labels.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from mife.family_rules import PRECEDENCE, RULES_VERSION, classify  # noqa: E402

OUT = ROOT / "audits" / "family_rules"


def load_task_texts() -> dict[int, str]:
    cfg = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text())["source"]
    path = ROOT / cfg["local_path"] / cfg["data_file"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != cfg["data_sha256"]:
        raise RuntimeError("ManyIFEval data hash mismatch")
    out: dict[int, str] = {}
    for line in raw.decode().splitlines():
        if line.strip():
            r = json.loads(line)
            out.setdefault(r["key"], r["task_prompt"].strip())
    return out


def load_hand_labels() -> dict[int, dict]:
    with open(ROOT / "configs" / "task_families_v1.csv") as f:
        return {int(r["task_key"]): r for r in csv.DictReader(f)}


def cohen_kappa(a: list[str], b: list[str], labels: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in labels) / n**2
    return (po - pe) / (1 - pe)


def main() -> dict:
    texts = load_task_texts()
    hand = load_hand_labels()
    assert set(texts) == set(hand) and len(texts) == 216
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for k in sorted(texts):
        a = classify(texts[k])
        h = hand[k]
        rows.append({
            "task_key": k, "rule_family": a.family, "rule": a.rule, "matched": a.matched,
            "families_with_cues": "|".join(a.all_families),
            "hand_family": h["family"], "hand_ambiguous": h["ambiguous"], "hand_alt_family": h["alt_family"],
            "agree": int(a.family == h["family"]),
            "agree_with_alt": int(a.family == h["family"] or a.family == h["alt_family"]),
            "task_text": texts[k],
        })
    with open(OUT / "rule_labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    rule = [r["rule_family"] for r in rows]
    hnd = [r["hand_family"] for r in rows]
    dis = [r for r in rows if not r["agree"]]
    amb = [r for r in rows if r["hand_ambiguous"] == "1"]
    confusion = {h: {p: sum(1 for r in rows if r["hand_family"] == h and r["rule_family"] == p)
                     for p in PRECEDENCE} for h in PRECEDENCE}
    report = {
        "rules_version": RULES_VERSION,
        "n_tasks": len(rows),
        "accuracy": sum(r["agree"] for r in rows) / len(rows),
        "cohen_kappa": cohen_kappa(hnd, rule, PRECEDENCE),
        "accuracy_counting_alt_family": sum(r["agree_with_alt"] for r in rows) / len(rows),
        "n_disagreements": len(dis),
        "n_disagreements_on_ambiguous": sum(r["hand_ambiguous"] == "1" for r in dis),
        "n_disagreements_matching_alt_family": sum(r["rule_family"] == r["hand_alt_family"] for r in dis),
        "n_ambiguous": len(amb),
        "accuracy_on_ambiguous": sum(r["agree"] for r in amb) / len(amb),
        "accuracy_on_unambiguous": sum(r["agree"] for r in rows if r["hand_ambiguous"] != "1") / (len(rows) - len(amb)),
        "family_counts_rule": dict(Counter(rule)),
        "family_counts_hand": dict(Counter(hnd)),
        "decided_by_default_INFO": sum(r["rule"] == "default" for r in rows),
        "confusion_rows_hand_cols_rule": confusion,
        "disagreements": [{k: r[k] for k in ("task_key", "hand_family", "hand_alt_family", "hand_ambiguous",
                                             "rule_family", "rule", "matched", "task_text")} for r in dis],
    }
    (OUT / "agreement.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: v for k, v in r.items() if k != "disagreements"}, indent=2))
