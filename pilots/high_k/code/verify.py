"""Official ManyIFEval verification of K10 generations (no outcome-based exclusions).

    PYTHONPATH=src python pilots/high_k/code/verify.py gemma2b              # TRAIN only (default), repo root
    PYTHONPATH=src python pilots/high_k/code/verify.py gemma2b --split test # only after a TEST prespec is frozen and
                                                                            # the owner has said "go"
Writes pilots/high_k/data/verification/<model>_<split>.parquet: one row per response x included rule. TEST responses
are never read unless --split test is given.
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import pandas as pd

from mife.config import ROOT
from mife.verifier import verify

PILOT = ROOT / "pilots" / "high_k"
INP = PILOT / "data" / "inputs"
_R = _T = None
_CUT = None
# Generation issue (PRESPEC.md, "Generation issues"): gemma9b ran with stop_token_ids [1] only, so token 107
# (<end_of_turn>) did not stop sampling and its text is in response_text. Verification uses the text before the first
# "<end_of_turn>", which is exactly what stopping at 107 would have produced. Raw shards are never modified.
CLEAN_EOT = {"gemma9b"}
EOT = "<end_of_turn>"


def _init(split: str, model: str = "") -> None:
    global _R, _T, _CUT
    _CUT = EOT if model in CLEAN_EOT else None
    P = pd.read_parquet(INP / "prompts.parquet", columns=["prompt_id", "split", "task_text", "is_original"])
    P = P[P.split == split]
    _T = dict(zip(P.prompt_id, P.task_text))
    R = pd.read_parquet(INP / "prompt_rules.parquet")
    R = R[R.prompt_id.isin(_T)]
    _R = {p: d.sort_values("position") for p, d in R.groupby("prompt_id")}


def _one(path: str) -> pd.DataFrame:
    G = pd.read_parquet(path, columns=["response_id", "prompt_id", "sample_index", "response_text",
                                       "n_output_tokens", "hit_max_tokens"])
    G = G[G.prompt_id.isin(_T)]
    if _CUT is not None:
        G = G.assign(eot_cut=G.response_text.str.contains(_CUT, regex=False),
                     response_text=G.response_text.str.split(_CUT, n=1).str[0])
    else:
        G = G.assign(eot_cut=False)
    rows = []
    for g in G.itertuples():
        rr = _R[g.prompt_id]
        v = verify(g.response_text, [(t, json.loads(k)) for t, k in zip(rr.type_id, rr.kwargs_json)],
                   task_text=_T[g.prompt_id])
        for r, x in zip(rr.itertuples(), v):
            rows.append({"response_id": g.response_id, "prompt_id": g.prompt_id, "sample_index": g.sample_index,
                         "group_id": r.group_id, "task_key": r.task_key, "k": r.k, "position": r.position,
                         "rule_id": r.rule_id, "type_id": r.type_id, "category": r.category, "origin": r.origin,
                         "violated": int(x), "n_output_tokens": g.n_output_tokens, "hit_max_tokens": bool(g.hit_max_tokens),
                         "eot_cut": bool(g.eot_cut)})
    return pd.DataFrame(rows)


def main(model: str, split: str, workers: int) -> None:
    files = sorted(str(f) for f in (PILOT / "data" / "generations" / model).glob("shard_*.parquet"))
    with Pool(workers, initializer=_init, initargs=(split, model)) as pool:
        V = pd.concat(pool.map(_one, files, chunksize=1), ignore_index=True)
    assert not V.duplicated(["response_id", "position"]).any()
    out = PILOT / "data" / "verification"
    out.mkdir(parents=True, exist_ok=True)
    V.to_parquet(out / f"{model}_{split}.parquet")
    print(json.dumps({"model": model, "split": split, "responses": V.response_id.nunique(), "checks": len(V),
                      "violation_rate": round(V.violated.mean(), 4),
                      "eot_cleaning": model in CLEAN_EOT,
                      "responses_cut_at_eot": int(V.drop_duplicates("response_id").eot_cut.sum())}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["gemma2b", "llama", "gemma9b"])
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    main(a.model, a.split, a.workers)
