"""Propose the amendment-S3 split from the frozen manifest's pool metadata (no generations, labels or features).

    python scripts/split_s3/propose_split.py [--prompts data/manifest/prompts.parquet] [--out results/split_s3]
    python scripts/split_s3/propose_split.py --config configs/split_s3_no_units.yaml --out results/split_s3_no_units

Writes proposed_split.json and pair_coverage.csv. Does not modify the manifest or its split column.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from mife import split_v3 as V  # noqa: E402

COLS = ["prompt_id", "group_id", "split", "task_key", "task_family", "ordered_type_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(ROOT / "data" / "manifest" / "prompts.parquet"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "split_s3.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results" / "split_s3"))
    a = ap.parse_args()
    cfg_text = Path(a.config).read_text()
    cfg = yaml.safe_load(cfg_text)
    prompts = pd.read_parquet(a.prompts, columns=COLS)
    gm = V.group_meta_from_prompts(prompts)
    test, info = V.split_groups_pair_coverage(
        gm, cfg["n_test_groups"], cfg["seed"], min_pair_test_groups=cfg["min_pair_test_groups"],
        min_type_test_groups=cfg["min_type_test_groups"], min_type_train_groups=cfg["min_type_train_groups"],
        n_restarts=cfg["n_restarts"], max_steps=cfg["max_steps"],
        keep_related_units=cfg.get("keep_related_units", True))
    old_test = sorted(prompts.loc[prompts["split"] == "test", "group_id"].unique())
    per_group = prompts.groupby("group_id").size()
    by_split = {s: {"groups": len(g), "tasks": 3 * len(g), "prompts": int(per_group[list(g)].sum())}
                for s, g in (("test", sorted(test)), ("train", sorted(set(per_group.index) - test)))}
    old_pairs = V.pair_coverage(gm, set(old_test))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "amendment": V.AMENDMENT_ID, "status": "proposed; manifest split column unchanged",
        "config": cfg, "config_sha256": hashlib.sha256(cfg_text.encode()).hexdigest(),
        "manifest_prompts_sha256": hashlib.sha256(Path(a.prompts).read_bytes()).hexdigest(),
        "test_groups": sorted(test), "train_groups": sorted(set(per_group.index) - test), "counts": by_split,
        "vs_2026_11_14": {"old_test_groups": old_test, "kept_in_test": sorted(test & set(old_test)),
                          "old_test_to_train": sorted(set(old_test) - test),
                          "old_train_to_test": sorted(test - set(old_test)),
                          "old_min_pair_coverage": min(old_pairs.values()),
                          "old_n_pairs_below_2": sum(v < 2 for v in old_pairs.values())},
        **{k: v for k, v in info.items() if k not in ("pair_test_groups", "pair_all_groups", "pair_train_groups")},
    }
    (out / "proposed_split.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    rows = [{"type_a": p.split("|")[0], "type_b": p.split("|")[1], "test_groups_s3": c,
             "test_groups_2026_11_14": old_pairs[p], "train_groups_s3": info["pair_train_groups"][p],
             "all_groups": info["pair_all_groups"][p]}
            for p, c in info["pair_test_groups"].items()]
    pd.DataFrame(rows).sort_values(["test_groups_s3", "type_a", "type_b"]).to_csv(out / "pair_coverage.csv", index=False)
    print(json.dumps({k: summary[k] for k in ("test_groups", "counts", "score", "min_pair_coverage",
                                                "n_pairs_at_least_3", "n_pairs_absent_from_train",
                                                "related_units_split", "n_restarts_admissible")}, indent=1))


if __name__ == "__main__":
    main()
