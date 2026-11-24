"""Run the whole S3 TEST Objective 1 evaluation and write every table to results/eval_s3/tables/.
Run: PYTHONPATH=src .venv/bin/python -m mife.eval_s3.run_all
"""
from __future__ import annotations

import json
import time

import pandas as pd

from . import eval1, eval2
from .common import GENS, data, type_order
from .paths import EVAL_DIR
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
        assert boot.G == 22
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
        # baseline target risk per type (context for floor/ceiling)
        acc.setdefault("observed_rate_by_type", []).append(
            d.groupby("type_id").apply(lambda x: pd.Series({"viol_rate": x.v.sum() / x.n.sum(), "n_pools": x.group_id.nunique(),
                                                             "n_rows": len(x)}), include_groups=False).reset_index().assign(generator=gen))
        print(gen, "eval2", round(time.time() - t0), "s", flush=True)
    for k, frames in acc.items():
        pd.concat(frames, ignore_index=True).to_csv(T / f"{k}.csv", index=False)
    (T / "eval1_qc.json").write_text(json.dumps(qc, indent=2, default=str))
    factor_e()


def factor_e():
    """Plan §5.9: S3 TRAIN k=1 violation rate per type -> easy/medium/hard terciles (5 each, ties by name)."""
    from ..probing import data as pdata
    rows = []
    for gen in GENS:
        L = pdata.labels_prompt_rule(gen, "train")
        P = pdata.prompts("train")[["prompt_id", "k"]]
        L = L.merge(P, on="prompt_id")
        L = L[L.k == 1]
        rate = (L.groupby("type_id").v.sum() / L.groupby("type_id").n.sum()).rename("train_k1_rate").reset_index()
        rate = rate.sort_values(["train_k1_rate", "type_id"]).reset_index(drop=True)
        rate["tercile"] = ["easy"] * 5 + ["medium"] * 5 + ["hard"] * 5
        rate["generator"] = gen
        obs = pd.read_csv(T / "observed_rate_by_type.csv")
        rate["test_rate"] = rate.type_id.map(obs[obs.generator == gen].set_index("type_id").viol_rate)
        for f in ("position", "companion", "task"):
            rr = pd.read_csv(T / f"eval2_{f}_rms_by_type.csv")
            rate[f"{f}_rms"] = rate.type_id.map(rr[rr.generator == gen].set_index("type_id").rms_pp)
        rows.append(rate)
    pd.concat(rows).to_csv(T / "factor_e_terciles.csv", index=False)


if __name__ == "__main__":
    main()
