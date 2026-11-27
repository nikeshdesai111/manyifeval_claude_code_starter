"""Scan one model's generations for the literal "<end_of_turn>" text (TRAIN rows only unless --split test is given).

    .venv/bin/python pilots/high_k/code/eot_scan.py gemma9b [--split train]
Writes pilots/high_k/data/verification/<model>_<split>_eot_scan.json.
"""
from __future__ import annotations

import argparse
import glob
import json
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

PILOT = Path(__file__).resolve().parents[1]
TOK = "<end_of_turn>"
_IDS: set = set()


def _init(split: str) -> None:
    global _IDS
    P = pd.read_parquet(PILOT / "data/inputs/prompts.parquet", columns=["prompt_id", "split"])
    _IDS = set(P.prompt_id[P.split == split])


def _one(f: str) -> dict:
    d = pd.read_parquet(f, columns=["response_id", "prompt_id", "response_text", "finish_reason"])
    d = d[d.prompt_id.isin(_IDS)]
    has = d.response_text.str.contains(TOK, regex=False)
    after = d.response_text[has].str.split(TOK, n=1).str[1]
    real = after.str.strip() != ""
    ex = [{"response_id": r, "around_cut": t[max(0, t.find(TOK) - 100): t.find(TOK) + 100]}
          for r, t in zip(d.response_id[has][real].head(3), d.response_text[has][real].head(3))]
    return {"n": len(d), "with_token": int(has.sum()), "real_after": int(real.sum()),
            "after_values": after[~real].value_counts().to_dict(), "examples": ex,
            "no_token_finish": d.finish_reason[~has].value_counts().to_dict()}


def main(model: str, split: str) -> dict:
    files = sorted(glob.glob(str(PILOT / f"data/generations/{model}/shard_*.parquet")))
    with Pool(8, initializer=_init, initargs=(split,)) as p:
        R = p.map(_one, files)
    agg = {"model": model, "split": split, "n": 0, "with_token": 0, "real_after": 0, "after_values": {}, "examples": [],
           "no_token_finish": {}}
    for r in R:
        for k in ("n", "with_token", "real_after"):
            agg[k] += r[k]
        for src, dst in (("after_values", "after_values"), ("no_token_finish", "no_token_finish")):
            for k, v in r[src].items():
                agg[dst][repr(k) if src == "after_values" else k] = agg[dst].get(repr(k) if src == "after_values" else k, 0) + v
        agg["examples"] += r["examples"]
    agg["examples"] = agg["examples"][:3]
    agg["no_token"] = agg["n"] - agg["with_token"]
    agg["real_after_pct"] = 100 * agg["real_after"] / max(agg["n"], 1)
    out = PILOT / f"data/verification/{model}_{split}_eot_scan.json"
    out.write_text(json.dumps(agg, indent=1))
    print(json.dumps(agg, indent=1))
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--split", default="train", choices=["train"])
    a = ap.parse_args()
    main(a.model, a.split)
