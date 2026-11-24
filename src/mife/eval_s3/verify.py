"""Verify the S3 TEST expansion responses with the official verifier (same code and per-rule binary labels as run
`main`, mife.verify_stage), and assemble the S3 TEST Objective 1 label table (original prompts from run `main`).

Output: data/verification/<model>/expansion_s3_20261123/{verification_long,verification_responses,
labels_prompt_rule,labels_prompt}.parquet. Raw responses are not modified. Run: python -m mife.eval_s3.verify <model>
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .. import verifier
from ..config import DATA_DIR
from ..storage import write_parquet
from .paths import EXP_DIR, EXP_RUN

_RULES: dict = {}
_TASK: dict = {}


def _init(rules, task):
    _RULES.update(rules)
    _TASK.update(task)


def _verify(batch):
    out = []
    for rid, pid, si, text in batch:
        rl = _RULES[pid]
        v = verifier.verify(text, [(t, kw) for t, kw, _, _ in rl], _TASK[pid])
        out.append((rid, pid, si, v, verifier.diagnostics(text)))
    return out


def run(model: str, workers: int = 8) -> dict:
    gdir = DATA_DIR / "generations" / model / EXP_RUN
    gens = pd.concat([pd.read_parquet(s, columns=["response_id", "prompt_id", "sample_index", "response_text",
                                                  "rendered_sha256", "finish_reason", "hit_max_tokens",
                                                  "settings_hash", "backend"])
                      for s in sorted(gdir.glob("shard_*.parquet"))], ignore_index=True)
    P = pd.read_parquet(EXP_DIR / "prompts.parquet").set_index("prompt_id")
    tok = pd.read_parquet(EXP_DIR / f"tokenized_{model}.parquet", columns=["prompt_id", "rendered_sha256"]).set_index("prompt_id")
    new = P.index[~P.is_original]
    # completeness and integrity: exactly 20 distinct samples per expansion prompt, matching rendered hash, one setting
    assert gens.response_id.is_unique
    assert set(gens.prompt_id) == set(new), "expansion responses must cover exactly the expansion prompts"
    c = gens.groupby("prompt_id").sample_index.agg(["size", "nunique", "min", "max"])
    assert (c["size"] == 20).all() and (c["nunique"] == 20).all() and (c["min"] == 0).all() and (c["max"] == 19).all()
    assert (gens.rendered_sha256.values == tok.loc[gens.prompt_id, "rendered_sha256"].values).all()
    assert gens.settings_hash.nunique() == 1 and gens.backend.nunique() == 1
    PR = pd.read_parquet(EXP_DIR / "prompt_rules.parquet")
    PR = PR[PR.prompt_id.isin(set(new))].sort_values(["prompt_id", "position"])
    rules = {pid: list(zip(d.type_id, d.kwargs_json.map(json.loads), d.rule_id, d.position)) for pid, d in PR.groupby("prompt_id")}
    task = P.loc[new, "task_text"].to_dict()
    items = list(zip(gens.response_id, gens.prompt_id, gens.sample_index, gens.response_text))
    batches = [items[i:i + 2000] for i in range(0, len(items), 2000)]
    long_rows, resp_rows = [], []
    with Pool(workers, initializer=_init, initargs=(rules, task)) as pool:
        for n, res in enumerate(pool.imap(_verify, batches)):
            for rid, pid, si, v, diag in res:
                rl = rules[pid]
                for (t, _, ru, pos), vi in zip(rl, v):
                    long_rows.append((rid, pid, model, si, pos, ru, t, vi))
                resp_rows.append({"response_id": rid, "prompt_id": pid, "model_key": model, "violation_vector": v,
                                  "k": len(v), "fraction_violated": sum(v) / len(v),
                                  "verifier_version": verifier.VERIFIER_VERSION, **{f"diag_{a}": b for a, b in diag.items()}})
            if n % 10 == 0:
                print(f"[verify {model}] {min((n + 1) * 2000, len(items))}/{len(items)}", flush=True)
    long = pd.DataFrame(long_rows, columns=["response_id", "prompt_id", "model_key", "sample_index", "position", "rule_id",
                                            "type_id", "violated"])
    resp = pd.DataFrame(resp_rows).merge(gens[["response_id", "finish_reason", "hit_max_tokens"]], on="response_id")
    pr = long.groupby(["prompt_id", "position", "rule_id", "type_id"]).agg(n=("violated", "size"), v=("violated", "sum")).reset_index()
    pr["p_hat"] = pr["v"] / pr["n"]
    pr["model_key"] = model
    assert (pr["n"] == 20).all() and len(pr) == len(PR)
    pl = pr.groupby("prompt_id").agg(k=("p_hat", "size"), q_hat=("p_hat", "mean"), n=("n", "min")).reset_index()
    out = DATA_DIR / "verification" / model / EXP_RUN
    write_parquet(long, out / "verification_long.parquet")
    write_parquet(resp, out / "verification_responses.parquet")
    write_parquet(pr, out / "labels_prompt_rule.parquet")
    write_parquet(pl, out / "labels_prompt.parquet")
    return {"model": model, "n_responses": len(resp), "n_rule_checks": len(long), "n_prompt_rule_labels": len(pr),
            "verifier_version": verifier.VERIFIER_VERSION, "settings_hash": gens.settings_hash.iloc[0],
            "backend": gens.backend.iloc[0], "hit_max_tokens_share": float(gens.hit_max_tokens.mean())}


def labels(model: str) -> pd.DataFrame:
    """S3 TEST Objective 1 labels (v, n) for all 115,236 prompt-rule rows: originals from `main`, new from expansion."""
    PR = pd.read_parquet(EXP_DIR / "prompt_rules.parquet")[["prompt_id", "position", "rule_id", "type_id", "group_id", "k"]]
    P = pd.read_parquet(EXP_DIR / "prompts.parquet", columns=["prompt_id", "is_original", "prompt_sha256"])
    orig = set(P.prompt_id[P.is_original])
    import pyarrow.dataset as ds
    m = ds.dataset(DATA_DIR / "verification" / model / "main" / "labels_prompt_rule.parquet").to_table(
        columns=["prompt_id", "position", "rule_id", "v", "n", "prompt_sha256"],
        filter=ds.field("prompt_id").isin(sorted(orig))).to_pandas()
    sha = P.set_index("prompt_id").prompt_sha256
    assert (m.prompt_sha256.values == sha.loc[m.prompt_id].values).all(), "original prompt hash mismatch"
    e = pd.read_parquet(DATA_DIR / "verification" / model / EXP_RUN / "labels_prompt_rule.parquet",
                        columns=["prompt_id", "position", "rule_id", "v", "n"])
    L = pd.concat([m.drop(columns="prompt_sha256").assign(source="main"), e.assign(source=EXP_RUN)], ignore_index=True)
    out = PR.merge(L, on=["prompt_id", "position", "rule_id"], how="left", validate="one_to_one")
    assert out.n.notna().all() and (out.n == 20).all() and len(out) == 115236
    return out


if __name__ == "__main__":
    from .paths import EVAL_DIR
    (EVAL_DIR / "tables").mkdir(parents=True, exist_ok=True)
    for mk in sys.argv[1:]:
        r = run(mk)
        print(r)
        (EVAL_DIR / "tables" / f"verify_expansion_{mk}.json").write_text(json.dumps(r, indent=2, default=str))
