"""Command-line entry points.  Usage: PYTHONPATH=src .venv/bin/python -m mife.cli <command> [...]

  manifest                      build the frozen prompt manifest (deterministic)
  tokenize  MODEL               tokenize all 53,136 prompts + context audit
  pilot-select                  predeclared train-only pilot selection
  generate  MODEL RUN [--pilot | --all-train-test-CONFIRMED] [--ppb N] [--cbs N]
  import-pilot MODEL [--from-run RUN]   copy a pilot run's samples into run "main" (all 20); refuses if
                                the pilot's backend/decoding settings differ from the current ones
  generate-vllm MODEL RUN [--pilot | --all-train-test-CONFIRMED] [--ppb N] [--gpu-mem F] [--max-num-seqs N]
                                CUDA/vLLM generation (RUN must not be "pilot", the Mac pilot)
  consistency-cuda MODEL RUN    vLLM vs HF-transformers final-token log-probs on CUDA
  xbackend  MODEL RUN [--mac-run pilot]   CUDA run vs Mac pilot: tokens, hidden states, verifier rates
  sampler-semantics MODEL       MLX vs vLLM sampler definitions, teacher-forced on Mac pilot responses
  env-report [OUT]              backend / precision / torch / vLLM / CUDA / driver versions (JSON)
  verify    MODEL RUN
  features  MODEL RUN [--pilot | --all] [--max-prompts N] [--shard-size N]   resumable; skips committed prompts
  consistency MODEL [N]
  probe-features | probe-describe | probe-b2-bench | probe-cv | probe-final | probe-eval | probe-report
                                probing phase (src/mife/probing/cli.py; README "Probing, evaluation and results")
Bulk generation over the full manifest requires the explicit flag --all-train-test-CONFIRMED,
which must only be used after human approval of the pilot report.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from .config import DATA_DIR
from .manifest import MANIFEST_DIR


def _ids(args):
    if getattr(args, "pilot", False):
        return list(pd.read_parquet(DATA_DIR / "pilot" / "pilot_prompts.parquet").prompt_id)
    if getattr(args, "all_train_test_CONFIRMED", False) or getattr(args, "all", False):
        return list(pd.read_parquet(MANIFEST_DIR / "prompts.parquet").prompt_id)
    raise SystemExit("choose --pilot (or the explicit full-run flag after approval)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest")
    t = sub.add_parser("tokenize"); t.add_argument("model")
    sub.add_parser("pilot-select")
    g = sub.add_parser("generate"); g.add_argument("model"); g.add_argument("run")
    g.add_argument("--pilot", action="store_true"); g.add_argument("--all-train-test-CONFIRMED", action="store_true")
    g.add_argument("--ppb", type=int, default=4, help="prompts per output shard")
    g.add_argument("--cbs", type=int, default=64, help="max concurrently decoding sequences")
    g.add_argument("--recycle", type=int, default=None, help="fresh BatchGenerator every N prompts (memory)")
    ip = sub.add_parser("import-pilot"); ip.add_argument("model")
    ip.add_argument("--from-run", default="pilot", help="pilot run to import (e.g. pilot_betty_dgx-b200)")
    gv = sub.add_parser("generate-vllm"); gv.add_argument("model"); gv.add_argument("run")
    gv.add_argument("--pilot", action="store_true"); gv.add_argument("--all-train-test-CONFIRMED", action="store_true")
    gv.add_argument("--ppb", type=int, default=50, help="prompts per output shard / per vLLM generate call")
    gv.add_argument("--gpu-mem", type=float, default=0.85, help="vLLM gpu_memory_utilization")
    gv.add_argument("--max-num-seqs", type=int, default=512)
    gv.add_argument("--max-model-len", type=int, default=4096)
    gv.add_argument("--max-prompts", type=int, default=None)
    gv.add_argument("--per-sample-seeds", action="store_true", help="seed every request (slower sampling path)")
    gv.add_argument("--chunk", type=int, default=None, help="bulk: index of this job's contiguous prompt chunk")
    gv.add_argument("--n-chunks", type=int, default=None)
    gv.add_argument("--n-probe", type=int, default=20, help="prompts for the vLLM log-prob probe (0 = skip)")
    cc = sub.add_parser("consistency-cuda"); cc.add_argument("model"); cc.add_argument("run")
    xb = sub.add_parser("xbackend"); xb.add_argument("model"); xb.add_argument("run"); xb.add_argument("--mac-run", default="pilot")
    ss = sub.add_parser("sampler-semantics"); ss.add_argument("model")
    er = sub.add_parser("env-report"); er.add_argument("out", nargs="?", default=None)
    v = sub.add_parser("verify"); v.add_argument("model"); v.add_argument("run")
    f = sub.add_parser("features"); f.add_argument("model"); f.add_argument("run")
    f.add_argument("--pilot", action="store_true"); f.add_argument("--all", action="store_true")
    f.add_argument("--max-prompts", type=int, default=None); f.add_argument("--shard-size", type=int, default=512)
    c = sub.add_parser("consistency"); c.add_argument("model"); c.add_argument("n", type=int, nargs="?", default=20)
    from .probing import cli as probing_cli
    probing_cli.add_parsers(sub)
    a = ap.parse_args()

    if a.cmd.startswith("probe-"):
        out = probing_cli.dispatch(a)
    elif a.cmd == "manifest":
        from . import manifest
        out = manifest.build()["audit"]
    elif a.cmd == "tokenize":
        from . import tokenize_audit
        out = tokenize_audit.run(a.model)
    elif a.cmd == "pilot-select":
        from . import pilot_select
        out = pilot_select.select()[["prompt_id", "k"]].to_dict("records")
    elif a.cmd == "generate":
        from . import generate_mlx
        out = generate_mlx.run(a.model, _ids(a), a.run, prompts_per_batch=a.ppb,
                                completion_batch_size=a.cbs, recycle_every_prompts=a.recycle)
        out.pop("batches", None)
    elif a.cmd == "generate-vllm":
        from . import generate_vllm
        ids, tag = _ids(a), None
        if a.chunk is not None:
            ids, tag = generate_vllm.chunk_ids(ids, a.chunk, a.n_chunks), f"c{a.chunk:02d}"
        out = generate_vllm.run(a.model, ids, a.run, prompts_per_shard=a.ppb, max_prompts=a.max_prompts,
                                gpu_memory_utilization=a.gpu_mem, max_num_seqs=a.max_num_seqs,
                                max_model_len=a.max_model_len, per_sample_seeds=a.per_sample_seeds,
                                n_probe=a.n_probe, chunk_tag=tag)
        out.pop("batches", None)
    elif a.cmd == "consistency-cuda":
        from . import consistency_cuda
        out = consistency_cuda.run(a.model, a.run)
        out.pop("per_prompt", None)
    elif a.cmd == "xbackend":
        from . import xbackend
        out = xbackend.run(a.model, a.run, a.mac_run)
        out = {"flags": out["flags"], "tokens": {k: v for k, v in out["tokens"].items() if k != "mismatches"},
               "hidden_worst_cos_min": out.get("hidden_states", {}).get("worst_cos_min"),
               "attention": out.get("attention")}
    elif a.cmd == "sampler-semantics":
        from . import sampler_semantics
        out = sampler_semantics.run(a.model)
        out.pop("per_response", None)
    elif a.cmd == "env-report":
        from .generate_vllm import cuda_env
        from .storage import write_json
        out = cuda_env()
        if a.out:
            write_json(out, a.out)
    elif a.cmd == "import-pilot":
        from . import generate_mlx
        out = generate_mlx.import_pilot(a.model, src_dir=DATA_DIR / "generations" / a.model / a.from_run)
    elif a.cmd == "verify":
        from . import verify_stage
        out = verify_stage.run(a.model, a.run)
    elif a.cmd == "features":
        from . import introspect
        out = introspect.run(a.model, a.run, _ids(a), shard_size=a.shard_size, max_prompts=a.max_prompts)
        out.pop("sessions", None)
    elif a.cmd == "consistency":
        from . import consistency
        ids = list(pd.read_parquet(DATA_DIR / "pilot" / "pilot_prompts.parquet").prompt_id)
        out = consistency.run(a.model, ids[:: max(1, len(ids) // a.n)][: a.n])
        out.pop("per_prompt", None)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
