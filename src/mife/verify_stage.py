"""Verify every included rule for every stored response; build soft labels.

Outputs (per model/run):
  verification_long.parquet : one row per (response, included rule): binary `violated`.
  verification_responses.parquet : per response: ordered violation vector, fraction violated, diagnostics.
  labels_prompt_rule.parquet : per (prompt, rule): v violations out of n, p_hat = v / n.
  labels_prompt.parquet : per prompt: q_hat = mean_r p_hat (= sum violations / (n k)).
Raw responses are never modified; verification can be re-run if the verifier is updated.
"""
from __future__ import annotations

import json

import pandas as pd

from . import verifier
from .config import DATA_DIR, load_config
from .generate_mlx import load_generations
from .manifest import MANIFEST_DIR
from .storage import write_parquet


def expected_samples_for(run_name: str) -> int:
    """Samples per prompt: the executed pilot value for run "pilot", the design value (20, Amendment A2) otherwise."""
    cfg = load_config()
    return cfg["pilot"]["n_responses_per_prompt"] if run_name == "pilot" else cfg["design"]["n_responses_per_prompt"]


def run(model_key: str, run_name: str, expected_samples: int | None = None) -> dict:
    expected_samples = expected_samples or expected_samples_for(run_name)
    # only the columns verification needs (token-id lists for 708k responses would cost tens of GB of RAM)
    gens = load_generations(model_key, run_name, ["prompt_id", "sample_index", "response_text"])  # hash-checked
    prompts = pd.read_parquet(MANIFEST_DIR / "prompts.parquet").set_index("prompt_id")
    rules = pd.read_parquet(MANIFEST_DIR / "prompt_rules.parquet")
    rules = rules[rules["prompt_id"].isin(set(gens["prompt_id"]))].sort_values(["prompt_id", "position"])
    rule_lists = {pid: list(zip(d.type_id, d.kwargs_json.map(json.loads), d.rule_id, d.position))
                  for pid, d in rules.groupby("prompt_id")}
    long_rows, resp_rows = [], []
    for g in gens.itertuples():
        rl = rule_lists[g.prompt_id]
        task_text = prompts.loc[g.prompt_id, "task_text"]
        v = verifier.verify(g.response_text, [(t, kw) for t, kw, _, _ in rl], task_text)
        assert len(v) == prompts.loc[g.prompt_id, "k"]
        for (t, _, rid, pos), vi in zip(rl, v):
            long_rows.append({"response_id": g.response_id, "prompt_id": g.prompt_id, "model_key": model_key,
                              "sample_index": g.sample_index, "position": pos, "rule_id": rid, "type_id": t,
                              "violated": vi})
        resp_rows.append({"response_id": g.response_id, "prompt_id": g.prompt_id, "model_key": model_key,
                          "violation_vector": v, "k": len(v), "fraction_violated": sum(v) / len(v),
                          "verifier_version": verifier.VERIFIER_VERSION,
                          **{f"diag_{a}": b for a, b in verifier.diagnostics(g.response_text).items()}})
    long = pd.DataFrame(long_rows)
    resp = pd.DataFrame(resp_rows)
    pr = (long.groupby(["prompt_id", "position", "rule_id", "type_id"])
          .agg(n=("violated", "size"), v=("violated", "sum")).reset_index())
    pr["p_hat"] = pr["v"] / pr["n"]
    pr["model_key"] = model_key
    pl = pr.groupby("prompt_id").agg(k=("p_hat", "size"), q_hat=("p_hat", "mean"), n=("n", "min")).reset_index()
    pl["model_key"] = model_key
    # join / completeness checks
    assert (pr["n"] == expected_samples).all(), "every prompt-rule must have all samples verified"
    assert len(long) == sum(prompts.loc[p, "k"] for p in gens["prompt_id"])
    hashes = gens[["prompt_id", "rendered_sha256"]].drop_duplicates("prompt_id").set_index("prompt_id")
    hashes["prompt_sha256"] = prompts.loc[hashes.index, "prompt_sha256"]
    long, resp, pr, pl = (d.join(hashes, on="prompt_id") for d in (long, resp, pr, pl))
    out = DATA_DIR / "verification" / model_key / run_name
    write_parquet(long, out / "verification_long.parquet")
    write_parquet(resp, out / "verification_responses.parquet")
    write_parquet(pr, out / "labels_prompt_rule.parquet")
    write_parquet(pl, out / "labels_prompt.parquet")
    return {"n_responses": len(resp), "n_rule_checks": len(long), "n_prompt_rule_labels": len(pr),
            "n_prompt_labels": len(pl), "verifier_version": verifier.VERIFIER_VERSION}
