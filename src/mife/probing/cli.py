"""Probing-phase subcommands for `python -m mife.cli`.

  probe-features [--split train] [--models llama qwen]   structure, BGE-M3 and LLM feature caches
  probe-describe                                           Phase 1 TRAIN-only descriptive tables
  probe-b2-bench                                           time B2 on this machine and project the protocol
  probe-cv MODEL OBJ [--configs ...] [--stage all]         training-only grouped CV (resumable)
  probe-final MODEL OBJ                                    refit selected settings on all 57 TRAIN groups
  probe-eval MODEL OBJ                                     TEST evaluation (refuses before unblinding)
  probe-report                                             tables, figures and results/RESULTS.md
"""
from __future__ import annotations


def add_parsers(sub) -> None:
    f = sub.add_parser("probe-features")
    f.add_argument("--split", default="train")
    f.add_argument("--models", nargs="*", default=["llama", "qwen"])
    f.add_argument("--blocks", nargs="*", default=["structure", "bge", "llm"])
    sub.add_parser("probe-describe")
    sub.add_parser("probe-b2-bench")
    for name in ("probe-cv", "probe-final", "probe-eval"):
        p = sub.add_parser(name)
        p.add_argument("model")
        p.add_argument("objective", type=int, choices=[1, 2])
        p.add_argument("--configs", nargs="*", default=None)
        p.add_argument("--stage", default="all")
    sub.add_parser("probe-report")


def dispatch(a):
    if a.cmd == "probe-features":
        from . import features
        if "structure" in a.blocks:
            features.build_structure(a.split)
        if "bge" in a.blocks:
            features.build_bge(a.split)
        if "llm" in a.blocks:
            for m in a.models:
                features.build_llm(m, a.split)
        return {"features": a.blocks, "split": a.split, "models": a.models}
    if a.cmd == "probe-describe":
        from . import describe
        return describe.run()
    if a.cmd == "probe-b2-bench":
        from . import b2
        return b2.run_benchmark()
    if a.cmd == "probe-cv":
        from . import cv
        return cv.run(a.model, a.objective, configs=a.configs, stage=a.stage)
    if a.cmd == "probe-final":
        from . import train
        return train.refit(a.model, a.objective, configs=a.configs)
    if a.cmd == "probe-eval":
        from . import evaluate
        return evaluate.run(a.model, a.objective)
    if a.cmd == "probe-report":
        from . import report
        return report.build()
    raise SystemExit(a.cmd)
