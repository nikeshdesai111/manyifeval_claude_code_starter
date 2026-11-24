"""PRELIMINARY S3 TRAIN report: Evaluation 1 and 2, Objective 1, on grouped-CV OOF predictions (50 TRAIN pools).
Run: PYTHONPATH=src .venv/bin/python -m mife.prelim_train.run_all   (tables -> results/preliminary_train_report/tables)
"""
from __future__ import annotations

import json
import time

import pandas as pd

from . import eval1, eval2
from .common import GENS, data, type_order
from .paths import EVAL_DIR, N_POOLS
from .stats import Boot

T = EVAL_DIR / "tables"


def main():
    T.mkdir(parents=True, exist_ok=True)
    acc: dict[str, list] = {}
    qc = []
    for gen in GENS:
        t0 = time.time()
        d = data(gen)
        boot = Boot(sorted(d.group_id.unique()))
        assert boot.G == N_POOLS
        types = type_order(d.type_id.unique())
        e1 = eval1.run(gen, d, boot)
        for k in ("a1_metrics", "contrasts", "per_type", "fixed_vs_param", "calibration"):
            acc.setdefault(f"eval1_{k}", []).append(e1[k])
        qc.append(e1["qc"])
        print(gen, "eval1", round(time.time() - t0), "s", flush=True)
        pos = eval2.position(d, boot, gen, types)
        app = eval2.append(d, boot, gen)
        com = eval2.companion(d, boot, gen, types)
        tsk = eval2.task(d, boot, gen, types)
        for fac, o in (("position", pos), ("append", app), ("companion", com), ("task", tsk)):
            for k, v in o.items():
                if k in ("pairs", "rows"):
                    continue
                acc.setdefault(f"eval2_{fac}_{k}", []).append(v)
        acc.setdefault("observed_rate_by_type", []).append(
            d.groupby("type_id").apply(lambda x: pd.Series({"viol_rate": x.v.sum() / x.n.sum(), "n_pools": x.group_id.nunique(),
                                                             "n_rows": len(x)}), include_groups=False).reset_index().assign(generator=gen))
        print(gen, "eval2", round(time.time() - t0), "s", flush=True)
    for k, frames in acc.items():
        pd.concat(frames, ignore_index=True).to_csv(T / f"{k}.csv", index=False)
    (T / "eval1_qc.json").write_text(json.dumps(qc, indent=2, default=str))


if __name__ == "__main__":
    main()
