"""Descriptive analysis of the saved out-of-fold (OOF) cross-validation predictions. TRAIN groups only.

Reads   results/tables/cv_selected.csv, results/cv/{model}/obj{o}/grid/*.json,
        results/predictions/{model}/obj{o}/cv/*.parquet and the TRAIN labels of run `main`.
Writes  report/tables/oof_metrics.csv, oof_contrasts.csv, oof_per_type.csv, oof_per_type_diff.csv,
        oof_per_k.csv, oof_reliability.csv.

Nothing is fitted and nothing is selected here. Every prediction was produced by a model that did not see the
row's group. Intervals are percentile intervals of a paired cluster bootstrap over the 57 TRAIN groups
(B = 2000, seed 20261116). They are descriptive: head settings were selected on the same folds, and the
confirmatory analysis is the pre-specified one on the held-out TEST groups.

Run: PYTHONPATH=src .venv/bin/python report/analysis_oof.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mife.probing import data
from mife.probing.data import MODELS, RESULTS
from mife.probing.metrics import Obj1Stats, Obj2Stats, bins_from

OUT = Path(__file__).resolve().parent / "tables"
B, SEED = 2000, 20261116
PRIMARY = {1: "nll", 2: "rmse"}
PROBE_ONLY = ["H1", "H2", "A1", "H1+H2", "H1+A1", "H2+A1", "H1+H2+A1"]

# The pre-specified contrasts of results/prespec/analysis_plan.md § 5 (difference = metric(b) - metric(a)).
CONTRASTS = (
    [("C1 prompt-only vs internal", a, b) for a in ("B1", "B2") for b in ("H1", "H2", "A1")]
    + [("C2 hidden-state increment", a, f"{a}+{h}") for a in ("B1", "B2") for h in ("H1", "H2")]
    + [("C3 attention increment", a, f"{a}+A1") for a in ("B1", "B2")]
    + [("C4 complementarity", "BETTER(H1,A1)", "H1+A1"), ("C4 complementarity", "BETTER(H2,A1)", "H2+A1")]
    + [("C4 complementarity", f"{b}+{h}", f"{b}+{h}+A1") for b in ("B1", "B2") for h in ("H1", "H2")]
    + [("C4 complementarity", f"{b}+A1", f"{b}+{h}+A1") for b in ("B1", "B2") for h in ("H1", "H2")]
    + [("C5 exact-rule conditioning", "H1", "H2"), ("C5 exact-rule conditioning", "B1+H1", "B1+H2"),
       ("C5 exact-rule conditioning", "B2+H1", "B2+H2"), ("C5 exact-rule conditioning", "H1+A1", "H2+A1")]
    + [("C6 encoder adaptation", "B1", "B2")]
    + [("C6 encoder adaptation", f"B1+{p}", f"B2+{p}") for p in ("H1", "H2", "A1")]
    + [("Probe-only vs structure-only reference", "REF_structure", p) for p in PROBE_ONLY]
)
assert len(CONTRASTS) == 37


def _ci(x: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def load_cell(model: str, o: int, sel: pd.DataFrame):
    """Labels and the selected OOF prediction of every row of cv_selected for one generator x objective."""
    P = data.prompts("train")[["prompt_id", "group_id", "k"]]
    if o == 1:
        U = data.labels_prompt_rule(model, "train")[["prompt_id", "position", "type_id", "v", "n"]].merge(P, on="prompt_id")
        keys = ["prompt_id", "position"]
    else:
        U = data.labels_prompt(model, "train")[["prompt_id", "q_hat"]].merge(P, on="prompt_id")
        keys = ["prompt_id"]
    U = U.sort_values(keys).reset_index(drop=True)
    preds, folds = {}, {}
    for r in sel.itertuples():
        d = pd.read_parquet(RESULTS / "predictions" / model / f"obj{o}" / "cv" / f"{r.key}.parquet")
        d = U[keys].merge(d[keys + [r.setting]], on=keys, how="left", validate="one_to_one")
        assert d[r.setting].notna().all(), (model, o, r.key)
        preds[r.config] = d[r.setting].to_numpy(float)
        g = json.loads((RESULTS / "cv" / model / f"obj{o}" / "grid" / f"{r.key}.json").read_text())
        row = next(x for x in g["grid"] if x["setting"] == r.setting)
        folds[r.config] = np.array(row["fold_losses"])
        assert abs(row["cv_loss"] - r.cv_loss) < 1e-9, (model, o, r.key)
    return U, preds, folds


def stats(o: int, U: pd.DataFrame, p: np.ndarray, rows=None):
    rows = np.arange(len(U)) if rows is None else rows
    u, q = U.iloc[rows], p[rows]
    if o == 1:
        return Obj1Stats(q, u.v.to_numpy(float), u.n.to_numpy(float), u.group_id.to_numpy(), bins_from(q))
    return Obj2Stats(q, u.q_hat.to_numpy(float), u.group_id.to_numpy(), bins_from(q))


def metrics(o: int, S, W: np.ndarray) -> dict:
    return S.metrics(W) if o == 1 else S.metrics(W, spearman=False)


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    SEL = pd.read_csv(RESULTS / "tables" / "cv_selected.csv")
    m_rows, c_rows, t_rows, td_rows, k_rows, r_rows = [], [], [], [], [], []
    for model in MODELS:
        for o in (1, 2):
            sel = SEL[(SEL.model == model) & (SEL.objective == o)]
            U, preds, folds = load_cell(model, o, sel)
            G = U.group_id.nunique()
            assert G == 57
            rng = np.random.default_rng(SEED)
            W = rng.multinomial(G, np.full(G, 1 / G), size=B).astype(float)
            one = np.ones((1, G))
            cv = dict(zip(sel.config, sel.cv_loss))
            fam = dict(zip(sel.config, sel.family))
            boot, point = {}, {}
            for c, p in preds.items():
                S = stats(o, U, p)
                pt = {k: float(v[0]) for k, v in metrics(o, S, one).items()}
                bt = metrics(o, S, W)
                if o == 2:
                    pt["spearman"] = float(spearmanr(p, U.q_hat).statistic)
                boot[c], point[c] = bt[PRIMARY[o]], pt[PRIMARY[o]]
                for k, v in pt.items():
                    lo, hi = _ci(bt[k]) if k in bt else (np.nan, np.nan)
                    m_rows.append({"model": model, "objective": o, "family": fam[c], "config": c, "metric": k,
                                   "est": v, "lo": lo, "hi": hi, "cv_loss": cv[c]})
            # contrasts (pre-specified list, plus the best probe-only configuration against the three comparators)
            best_po = min(PROBE_ONLY, key=cv.get)
            extra = [("Best probe-only vs comparator", a, best_po) for a in ("REF_structure", "B1", "B2")]
            for family, a, b in CONTRASTS + extra:
                if a.startswith("BETTER("):
                    x, y = a[7:-1].split(",")
                    a_res = x if cv[x] <= cv[y] else y
                else:
                    a_res = a
                d = boot[b] - boot[a_res]
                lo, hi = _ci(d)
                c_rows.append({"model": model, "objective": o, "family": family, "a": a, "a_resolved": a_res, "b": b,
                               "metric": PRIMARY[o], "metric_a": point[a_res], "metric_b": point[b],
                               "diff": point[b] - point[a_res], "lo": lo, "hi": hi,
                               "rel_diff_pct": 100 * (point[b] - point[a_res]) / point[a_res],
                               "share_boot_below_0": float((d < 0).mean()),
                               "folds_b_lower": int((folds[b] < folds[a_res]).sum()), "n_folds": 5,
                               "cv_loss_a": cv[a_res], "cv_loss_b": cv[b]})
            # breakdowns for a fixed set of rows
            best_all = min((c for c in cv if not c.startswith("REF_")), key=cv.get)
            show = list(dict.fromkeys(["REF_type_k", "REF_structure", "B1", "B2", "H1", "H2", "A1", best_po, best_all]))
            for kk in sorted(U.k.unique()):
                rows = np.nonzero(U.k.to_numpy() == kk)[0]
                for c in show:
                    S = stats(o, U, preds[c], rows)
                    pt = metrics(o, S, np.ones((1, len(S.groups))))
                    k_rows.append({"model": model, "objective": o, "config": c, "k": int(kk), "n_rows": len(rows),
                                   **{m: float(v[0]) for m, v in pt.items()}})
            if o == 1:
                for t in sorted(U.type_id.unique()):
                    rows = np.nonzero(U.type_id.to_numpy() == t)[0]
                    u = U.iloc[rows]
                    Gt = u.group_id.nunique()
                    Wt = np.random.default_rng(SEED).multinomial(Gt, np.full(Gt, 1 / Gt), size=B).astype(float)
                    bt = {}
                    for c in show:
                        S = stats(o, U, preds[c], rows)
                        pt = metrics(o, S, np.ones((1, Gt)))
                        bt[c] = metrics(o, S, Wt)["nll"]
                        t_rows.append({"model": model, "config": c, "type_id": t, "n_groups": Gt, "n_rows": len(rows),
                                       "violation_rate": float(u.v.sum() / u.n.sum()),
                                       "nll": float(pt["nll"][0]), "auroc": float(pt["auroc"][0]),
                                       "brier": float(pt["brier"][0])})
                    for a, b in (("REF_structure", "A1"), ("B2", "A1"), ("B1", "A1"), ("H1", "H2")):
                        d = bt[b] - bt[a]
                        lo, hi = _ci(d)
                        pa = next(r["nll"] for r in t_rows if r["model"] == model and r["config"] == a and r["type_id"] == t)
                        pb = next(r["nll"] for r in t_rows if r["model"] == model and r["config"] == b and r["type_id"] == t)
                        td_rows.append({"model": model, "type_id": t, "a": a, "b": b, "n_groups": Gt, "diff": pb - pa,
                                        "lo": lo, "hi": hi})
                for c in show:
                    S = stats(o, U, preds[c])
                    n, v, pp = S.bin_n.sum(0), S.bin_v.sum(0), S.bin_p.sum(0)
                    for i in range(len(n)):
                        if n[i] > 0:
                            r_rows.append({"model": model, "config": c, "bin": i, "n_responses": float(n[i]),
                                           "mean_predicted": float(pp[i] / n[i]), "observed_rate": float(v[i] / n[i])})
    out = {}
    for name, rows in (("oof_metrics", m_rows), ("oof_contrasts", c_rows), ("oof_per_type", t_rows),
                       ("oof_per_type_diff", td_rows), ("oof_per_k", k_rows), ("oof_reliability", r_rows)):
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
        out[name] = len(rows)
    return out


if __name__ == "__main__":
    print(json.dumps(run()))
