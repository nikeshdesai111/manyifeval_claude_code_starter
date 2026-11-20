"""Phase 5: TEST evaluation, run exactly once after the pre-specification is frozen and unblinding approved.

The hypothesis contrasts (docx §10.4) are fixed here, before any test data is read. Each contrast is a paired
difference metric(b) - metric(a) on identical test examples, with a percentile CI from the paired group bootstrap
(the same replicate weights for every configuration). Primary contrast metric: Objective 1 binomial NLL, Objective 2
RMSE (lower is better, so a negative difference favours b). Holm adjustment within each (generator, objective, family).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import configs as C
from . import data, metrics
from .data import RESULTS

PRIMARY = {1: "nll", 2: "rmse"}

# (family, a, b); "BETTER(x,y)" = whichever of x, y has the lower TRAINING CV selection loss (fixed before test)
CONTRASTS = (
    [("C1 prompt-only vs internal", a, b) for a in ("B1", "B2") for b in ("H1", "H2", "A1")]
    + [("C2 hidden-state increment", a, f"{a}+{h}") for a in ("B1", "B2") for h in ("H1", "H2")]
    + [("C3 attention increment", a, f"{a}+A1") for a in ("B1", "B2")]
    + [("C4 complementarity", "BETTER(H1,A1)", "H1+A1"), ("C4 complementarity", "BETTER(H2,A1)", "H2+A1")]
    + [("C4 complementarity", f"{a}+{h}", f"{a}+{h}+A1") for a in ("B1", "B2") for h in ("H1", "H2")]
    + [("C4 complementarity", f"{a}+A1", f"{a}+{h}+A1") for a in ("B1", "B2") for h in ("H1", "H2")]
    + [("C5 exact-rule conditioning", "H1", "H2"), ("C5 exact-rule conditioning", "B1+H1", "B1+H2"),
       ("C5 exact-rule conditioning", "B2+H1", "B2+H2"), ("C5 exact-rule conditioning", "H1+A1", "H2+A1")]
    + [("C6 encoder adaptation", f"B1{s}", f"B2{s}") for s in ("", "+H1", "+H2", "+A1")]
    + [("Probe-only vs structure-only reference", "REF_structure", c) for c in C.PROBE_ONLY]
)


def contrast_table() -> pd.DataFrame:
    return pd.DataFrame(CONTRASTS, columns=["family", "a", "b"])


def resolve_better(name: str, cv_loss: dict) -> str:
    if not name.startswith("BETTER("):
        return name
    x, y = name[7:-1].split(",")
    return x if cv_loss[x] <= cv_loss[y] else y


def run(model: str, objective: int) -> dict:
    """Refuses to run before unblinding. The full Phase 5 pipeline (test features, final B2 encoder, predictions,
    metrics, contrasts, Evaluation 2) is executed by `run_test_evaluation` once approved."""
    if not data.test_unblinded():
        raise data.TestSetLocked("Phase 5 needs the frozen pre-specification and the project owner's 'go'")
    raise NotImplementedError("Phase 5 runner is completed after unblinding is approved")
