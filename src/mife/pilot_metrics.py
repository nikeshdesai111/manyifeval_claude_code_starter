"""Aggregate observed pilot measurements into data/pilot/pilot_metrics.json (inputs for the pilot report)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import DATA_DIR, load_config
from . import integrity
from .generate_mlx import load_generations
from .manifest import MANIFEST_DIR
from .storage import write_json

_D = load_config()["design"]
FULL_PROMPTS = _D["n_groups"] * _D["tasks_per_group"] * 246  # 53,136 manifest prompts


def _q(s, qs=(0, .1, .5, .9, .99, 1)):
    return {str(q): float(s.quantile(q)) for q in qs}


def model_metrics(model_key: str, run: str) -> dict:
    cfg = load_config()
    n_s = cfg["pilot"]["n_responses_per_prompt"]  # samples per prompt as executed in the pilot (20)
    n_main = cfg["design"]["n_responses_per_prompt"]  # main dataset (20, Amendment A2)
    pilot = pd.read_parquet(DATA_DIR / "pilot" / "pilot_prompts.parquet")
    gens = load_generations(model_key, run)
    gdir = DATA_DIR / "generations" / model_key / run
    sums = [json.loads(p.read_text()) for p in sorted(gdir.glob("run_summary_*.json"))]
    # throughput only from completed calls (crashed attempts wrote shards but no summary)
    gen_wall = sum(s["generation_wall_s"] for s in sums)
    meas_tok = sum(s["output_tokens"] for s in sums)
    out_tok = int(gens["n_output_tokens"].sum())
    vdir = DATA_DIR / "verification" / model_key / run
    resp = integrity.read_verification(model_key, run, "verification_responses.parquet")
    pr = integrity.read_verification(model_key, run, "labels_prompt_rule.parquet")
    pl = integrity.read_verification(model_key, run, "labels_prompt.parquet")
    k_of = pd.read_parquet(MANIFEST_DIR / "prompts.parquet").set_index("prompt_id")["k"]
    pr["k"] = pr["prompt_id"].map(k_of)
    fdir = DATA_DIR / "features" / model_key / run
    fmeta = json.loads((fdir / "features_meta.json").read_text()) if (fdir / "features_meta.json").exists() else None
    feat_bytes = sum(p.stat().st_size for p in fdir.glob("*") if not p.name.startswith("diag_")) if fmeta else None
    diag_bytes = sum(p.stat().st_size for p in fdir.glob("diag_*")) if fmeta else None

    tok_per_s = meas_tok / gen_wall
    full_out_tokens = FULL_PROMPTS * n_main * gens["n_output_tokens"].mean()
    m = {
        "model_key": model_key, "run": run,
        "records": {"expected_responses": len(pilot) * n_s, "responses": len(gens),
                    "unique_prompts": int(gens["prompt_id"].nunique()),
                    "complete_prompts": int((gens.groupby("prompt_id").size() == n_s).sum()),
                    "verified_responses": len(resp), "prompt_rule_labels": len(pr), "prompt_labels": len(pl),
                    "expected_prompt_rule_labels": int(pilot["k"].sum())},
        "throughput": {"measured_wall_s": gen_wall, "measured_output_tokens": meas_tok,
                       "total_output_tokens_all_shards": out_tok, "aggregate_output_tok_per_s": tok_per_s,
                       "backend": sums[-1].get("backend") or sums[-1]["settings"].get("backend"),
                       "peak_memory_gb": max(s.get("peak_mlx_memory_gb", s.get("peak_gpu_memory_used_gb", float("nan")))
                                             for s in sums),
                       "peak_memory_kind": ("MLX peak allocation" if "peak_mlx_memory_gb" in sums[-1]
                                            else "nvidia-smi memory.used (includes vLLM pre-reservation)"),
                       "gpu": (sums[-1].get("environment") or {}).get("nvidia_smi"),
                       "batch_config": [{k: x.get(k) for k in ("completion_batch_size", "n_prompts_generated_this_call")}
                                        for x in sums if x["n_prompts_generated_this_call"]],
                       "n_generation_calls": len(sums)},
        "lengths": {"input_tokens": _q(gens.drop_duplicates("prompt_id")["n_input_tokens"]),
                    "output_tokens": _q(gens["n_output_tokens"]),
                    "output_tokens_by_k": gens.assign(k=gens.prompt_id.map(k_of)).groupby("k")["n_output_tokens"].median().to_dict()},
        "truncation": {"input_truncation": 0, "hit_max_new_tokens": int(gens["hit_max_tokens"].sum()),
                       "hit_max_new_tokens_rate": float(gens["hit_max_tokens"].mean()),
                       "finish_reasons": gens["finish_reason"].value_counts().to_dict()},
        "verifier_health": {
            "empty_responses": int(resp["diag_empty"].sum()),
            "langdetect_non_en": int((resp["diag_langdetect"] != "en").sum()),
            "qwen_think_tags_in_output": int(gens["response_text"].str.contains("<think>").sum()),
            "violation_rate_overall": float(pr["v"].sum() / pr["n"].sum()),
            "violation_rate_by_type": (pr.groupby("type_id")["v"].sum() / pr.groupby("type_id")["n"].sum()).round(3).to_dict(),
            "violation_rate_by_k": (pr.groupby("k")["v"].sum() / pr.groupby("k")["n"].sum()).round(3).to_dict(),
            "prompt_rule_all_followed": int((pr["v"] == 0).sum()), "prompt_rule_all_violated": int((pr["v"] == n_s).sum()),
            "prompt_rule_mixed": int(((pr["v"] > 0) & (pr["v"] < n_s)).sum()),
            "q_hat_quantiles": _q(pl["q_hat"]),
        },
        "features": None if fmeta is None else {
            "n_prompts": fmeta["n_prompts"], "per_prompt_s": fmeta["per_prompt_s_quantiles"],
            "bytes_excluding_diag": feat_bytes, "bytes_per_prompt": feat_bytes / fmeta["n_prompts"],
            "diag_bytes": diag_bytes, "layers": fmeta["n_layers"], "heads": fmeta["n_heads"],
            "convention": fmeta["hidden_states_convention"],
            "projected_full_hours": FULL_PROMPTS * fmeta["per_prompt_s_quantiles"]["0.5"] / 3600,
            "projected_full_gb": FULL_PROMPTS * feat_bytes / fmeta["n_prompts"] / 1e9},
        "storage": _storage(gdir, vdir, len(gens), n_main, fmeta and feat_bytes / fmeta["n_prompts"]),
        "projection_full_run": {"samples_per_prompt": n_main, "est_gpu_hours_one_gpu": full_out_tokens / tok_per_s / 3600, "responses": FULL_PROMPTS * n_main,
                                "est_output_tokens": full_out_tokens,
                                "est_days_at_pilot_throughput": full_out_tokens / tok_per_s / 86400},
    }
    return m


def _storage(gdir, vdir, n_resp, n_main, feat_bytes_per_prompt) -> dict:
    gb = sum(p.stat().st_size for p in gdir.glob("shard_*.parquet"))
    vb = sum(p.stat().st_size for p in vdir.glob("*.parquet"))
    full = FULL_PROMPTS * n_main
    return {"generation_bytes_per_response": gb / n_resp, "verification_bytes_per_response": vb / n_resp,
            "projected_full_generation_gb": gb / n_resp * full / 1e9,
            "projected_full_verification_gb": vb / n_resp * full / 1e9,
            "projected_full_features_gb": (feat_bytes_per_prompt or 0) * FULL_PROMPTS / 1e9,
            "note": "parquet compression ratios at full scale may differ slightly from the 100-prompt pilot"}


def run(runs: dict[str, str], out_path=None) -> dict:
    out = {mk: model_metrics(mk, r) for mk, r in runs.items()}
    cpath = DATA_DIR / "consistency"
    for mk, r in runs.items():
        p = cpath / (f"{mk}.json" if r == "pilot" else f"{mk}_{r}_vllm_vs_hf.json")
        if p.exists():
            c = json.loads(p.read_text())
            c.pop("per_prompt", None)
            out[mk]["consistency"] = c
    out["_pilot_selection"] = json.loads((DATA_DIR / "pilot" / "pilot_selection.json").read_text())
    if out_path is None:  # never overwrite the Mac pilot metrics with another run's numbers
        tag = "" if set(runs.values()) == {"pilot"} else "_" + "_".join(sorted(set(runs.values())))
        out_path = DATA_DIR / "pilot" / f"pilot_metrics{tag}.json"
    write_json(out, out_path)
    return out
