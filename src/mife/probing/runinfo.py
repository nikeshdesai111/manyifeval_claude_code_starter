"""results/run_manifest.json: environment, seeds, data hashes, source-tree hash and stage wall times."""
from __future__ import annotations

import datetime as dt
import json
import platform
import sys

from ..config import DATA_DIR, ROOT, load_config
from ..storage import write_json
from .data import MODELS, RESULTS


def _v(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def write(extra: dict | None = None) -> dict:
    from .prespec import file_sha, source_tree_hash
    import torch
    p = RESULTS / "run_manifest.json"
    prev = json.loads(p.read_text()) if p.exists() else {}
    pc = load_config()["probing"]
    out = {
        "updated": dt.datetime.now().isoformat(timespec="seconds"),
        "git": "not a git repository; source_tree_sha256 instead",
        "source_tree_sha256": source_tree_hash(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "packages": {m: _v(m) for m in ("torch", "transformers", "tokenizers", "sklearn", "scipy", "numpy", "pandas",
                                        "pyarrow", "peft", "matplotlib", "nbformat", "nbclient", "ipywidgets")},
        "sentence_transformers / FlagEmbedding": "not used: BGE-M3 dense output computed with transformers "
                                                 "(L2-normalised [CLS] state, as BGEM3FlagModel's dense_vecs)",
        "mps_available": bool(torch.backends.mps.is_available()),
        "encoder": pc["encoder"],
        "encoder_weights_note": "transformers loaded the safetensors conversion BAAI/bge-m3 refs/pr/130 (9a0624b8...); "
                                "all 391 tensors are bit-identical to pytorch_model.bin at the pinned revision 5617a9f6...",
        "seeds": pc["seeds"],
        "config_sha256_now": file_sha(ROOT / "configs" / "experiment.yaml"),
        "config_sha256_at_manifest_build": json.loads((DATA_DIR / "manifest" / "manifest_summary.json").read_text())["config_sha256"],
        "data_sha256": {"manifest_prompts": file_sha(DATA_DIR / "manifest" / "prompts.parquet"),
                        "manifest_prompt_rules": file_sha(DATA_DIR / "manifest" / "prompt_rules.parquet"),
                        **{f"{m}_labels_prompt_rule": file_sha(DATA_DIR / "verification" / m / "main" / "labels_prompt_rule.parquet") for m in MODELS},
                        **{f"{m}_labels_prompt": file_sha(DATA_DIR / "verification" / m / "main" / "labels_prompt.parquet") for m in MODELS},
                        **{f"{m}_features_index": file_sha(DATA_DIR / "features" / m / "main" / "index.parquet") for m in MODELS}},
        "wall_times_s": {**prev.get("wall_times_s", {}), **((extra or {}).get("wall_times_s", {}))},
    }
    for k, v in (extra or {}).items():
        if k != "wall_times_s":
            out[k] = v
    write_json(out, p)
    return out


def runtime_table() -> "pd.DataFrame":
    """results/tables/runtime.csv from the wall times recorded in the output files themselves."""
    import glob

    import pandas as pd
    p = RESULTS / "run_manifest.json"
    prev = json.loads(p.read_text()).get("wall_times_s", {}) if p.exists() else {}
    rows = [{"stage": k, "where": "Mac (MPS/CPU)", "seconds": v, "note": ""} for k, v in prev.items()
            if k in ("features_train_bge_llm", "describe")]
    for m in MODELS:
        for o in (1, 2):
            w = [json.loads(open(f).read()).get("wall_s") or 0 for f in glob.glob(str(RESULTS / "cv" / m / f"obj{o}" / "grid" / "*.json"))]
            rows.append({"stage": f"cv_heads_{m}_obj{o}", "where": "Mac (CPU, 4 threads)", "seconds": sum(w),
                         "note": f"sum over {len(w)} grid files (all CV stages); the four model × objective processes ran in parallel"})
    for stage in ("grid", "final"):
        R = [json.loads(open(f).read()) for f in glob.glob(str(DATA_DIR / "probing_features" / "b2_runs" / stage / "*" / "result.json"))]
        n = {a: sum(r.get("adapt") == a for r in R) for a in ("lora", "full")}
        rows.append({"stage": f"b2_{stage}", "where": "Betty (dgx-b200, 1 GPU per task)",
                     "seconds": sum(r.get("wall_s") or 0 for r in R),
                     "note": f"sum over {n['lora']} LoRA and {n['full']} full fine-tuning runs (array tasks ran concurrently)"})
    for m in MODELS:
        for o in (1, 2):
            w = [json.loads(open(f).read()).get("wall_s") or 0 for f in glob.glob(str(RESULTS / "models" / m / f"obj{o}" / "*.json"))]
            rows.append({"stage": f"refit_{m}_obj{o}", "where": "Mac (CPU)", "seconds": sum(w), "note": "final heads on all 57 TRAIN groups"})
    d = pd.DataFrame(rows).assign(hours=lambda x: x.seconds / 3600)
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    d.to_csv(RESULTS / "tables" / "runtime.csv", index=False)
    return d
