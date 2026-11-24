"""Evaluation 1, Objective 1 on S3 TRAIN OOF (PRELIMINARY) (plan §§4.2-4.5): Table 2, appendix A1-A4, per-type CSV, fixed vs parameterized.

Metrics are functions of pool-level sufficient statistics on a FIXED 50-pool index, so a bootstrap replicate is a
vector of pool multiplicities and every predictor/contrast/subgroup uses identical draws (paired).
  NLL   sum[-v log p - (n-v) log(1-p)] / sum n      (p clipped to [1e-6, 1 - 1e-6]; primary)
  Brier sum[v (1-p)^2 + (n-v) p^2] / sum n
  AUROC per-response binary outcomes (row: v positives, n-v negatives, all scored p; ties 1/2) via the pool
        concordance matrix
  ECE   bins = deciles of the predictor's S3 TRAIN out-of-fold predictions (fixed before TEST), responses weighted
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import COMPS, GENS, OOF_DIR, PREDS, type_order
from .matches import FIXED_TYPES
from .stats import Boot, summarize

EPS = 1e-6


class S1:
    def __init__(self, p, v, n, gi, G, edges=None):
        p = np.clip(p, EPS, 1 - EPS)
        bc = lambda x: np.bincount(gi, weights=x, minlength=G)
        self.N, self.V = bc(n), bc(v)
        self.nll = bc(-(v * np.log(p) + (n - v) * np.log1p(-p)))
        self.brier = bc(v * (1 - p) ** 2 + (n - v) * p ** 2)
        order = np.argsort(p, kind="mergesort")
        ps, vs, ns, gs = p[order], v[order], n[order], gi[order]
        neg = ns - vs
        uniq, start = np.unique(ps, return_index=True)
        grp_of = np.repeat(np.arange(len(uniq)), np.diff(np.append(start, len(ps))))
        tie = np.zeros((len(uniq), G))
        np.add.at(tie, (grp_of, gs), neg)
        cum = np.zeros((len(uniq) + 1, G))
        cum[1:] = np.cumsum(tie, axis=0)
        self.C = np.zeros((G, G))
        np.add.at(self.C, gs, vs[:, None] * (cum[grp_of] + 0.5 * tie[grp_of]))
        self.Nn = self.N - self.V
        if edges is not None:
            b = np.digitize(p, edges)
            nb = len(edges) + 1
            self.bn = np.zeros((G, nb)); self.bv = np.zeros((G, nb)); self.bp = np.zeros((G, nb))
            np.add.at(self.bn, (gi, b), n); np.add.at(self.bv, (gi, b), v); np.add.at(self.bp, (gi, b), n * p)

    def m(self, W):
        N = W @ self.N
        with np.errstate(invalid="ignore", divide="ignore"):
            out = {"nll": (W @ self.nll) / N, "brier": (W @ self.brier) / N,
                   "auroc": np.einsum("bg,gh,bh->b", W, self.C, W) / ((W @ self.V) * (W @ self.Nn)),
                   "rate": (W @ self.V) / N}
            if hasattr(self, "bn"):
                out["ece"] = np.abs(W @ self.bv - W @ self.bp).sum(1) / N
        return out


def edges_for(gen: str, pred: str) -> np.ndarray:
    oof = pd.read_parquet(OOF_DIR / gen / "obj1" / f"{pred}.parquet", columns=["selected"])["selected"].values
    return np.unique(np.quantile(oof, np.linspace(0, 1, 11)[1:-1]))


def run(gen: str, d: pd.DataFrame, boot: Boot) -> dict:
    gix = {g: i for i, g in enumerate(boot.groups)}
    G = boot.G
    gi = d.group_id.map(gix).values
    v, n = d.v.values.astype(float), d.n.values.astype(float)
    Wp, Wb, Wl = boot.one, boot.M, boot.loo
    out = {}
    # ---- overall stats per predictor
    st = {p: S1(d[p].values, v, n, gi, G, edges_for(gen, p)) for p in PREDS + ["REF_structure"]}
    a1 = []
    for p, s in st.items():
        pt, bt = s.m(Wp), s.m(Wb)
        row = {"generator": gen, "predictor": p}
        for k in ("nll", "brier", "auroc", "ece"):
            sm = summarize(pt[k][0], bt[k])
            row.update({k: sm["est"], f"{k}_lo": sm["lo"], f"{k}_hi": sm["hi"]})
        a1.append(row)
    out["a1_metrics"] = pd.DataFrame(a1)
    # ---- contrasts (Table 2 + A1 contrasts), with t(21)/leave-one-pool-out sensitivity
    rows = []
    for cid, c, k, lab, prim in COMPS:
        r = {"generator": gen, "comparison": cid, "label": lab, "primary": prim, "comparator": c, "candidate": k}
        for met, sign in (("nll", 1), ("brier", 1), ("auroc", -1)):
            f = lambda W: sign * (st[c].m(W)[met] - st[k].m(W)[met])
            sm = summarize(f(Wp)[0], f(Wb), f(Wl), G)
            name = {"nll": "nll_reduction", "brier": "brier_reduction", "auroc": "auroc_gain"}[met]
            r.update({name: sm["est"], f"{name}_lo": sm["lo"], f"{name}_hi": sm["hi"]})
            if met == "nll":
                r.update({"nll_t_lo": sm.get("t_lo"), "nll_t_hi": sm.get("t_hi"), "nll_loo_min": sm.get("loo_min"),
                          "nll_loo_max": sm.get("loo_max")})
            if met == "auroc":
                r.update({"auroc_loo_min": sm.get("loo_min"), "auroc_loo_max": sm.get("loo_max")})
        rows.append(r)
    con = pd.DataFrame(rows)
    # ---- per-type (210 rows / generator x 7) and types improved
    types = type_order(d.type_id.unique())
    pt_rows = []
    tstats = {}
    for t in types:
        m = (d.type_id == t).values
        tstats[t] = {p: S1(d[p].values[m], v[m], n[m], gi[m], G) for p in set(sum([[c, k] for _, c, k, _, _ in COMPS], []))}
        npools = int(len(set(d.group_id[m])))
        for cid, c, k, lab, prim in COMPS:
            f = lambda W: tstats[t][c].m(W)["nll"] - tstats[t][k].m(W)["nll"]
            b = f(Wb)
            sm = summarize(f(Wp)[0], b)
            nc = tstats[t][c].m(Wp)["nll"][0]
            nk = tstats[t][k].m(Wp)["nll"][0]
            pt_rows.append({"generator": gen, "rule_type": t, "rule_class": "fixed" if t in FIXED_TYPES else "parameterized",
                            "comparison": cid, "comparator": c, "candidate": k, "nll_comparator": nc, "nll_candidate": nk,
                            "nll_reduction": sm["est"], "ci_lo": sm["lo"], "ci_hi": sm["hi"],
                            "n_pools": npools, "n_rule_outcomes": int(n[m].sum()), "n_violations": int(v[m].sum()),
                            "n_boot_valid": sm["n_boot_valid"],
                            "status": "ok" if np.isfinite(sm["est"]) else "undefined",
                            "low_support_flag": npools < 8})
    per_type = pd.DataFrame(pt_rows)
    imp = per_type.groupby("comparison").apply(lambda x: int((x.nll_reduction > 0).sum()), include_groups=False)
    und = per_type.groupby("comparison").apply(lambda x: int((~np.isfinite(x.nll_reduction)).sum()), include_groups=False)
    con["types_improved"] = con.comparison.map(imp)
    con["types_undefined"] = con.comparison.map(und)
    out["contrasts"] = con
    out["per_type"] = per_type
    # ---- fixed vs parameterized (A3b), identical replicates for both classes and D
    cls = {"fixed": d.type_id.isin(FIXED_TYPES).values}
    cls["parameterized"] = ~cls["fixed"]
    cst = {c_: {p: S1(d[p].values[m], v[m], n[m], gi[m], G) for p in PREDS} for c_, m in cls.items()}
    fp = []
    for cid, c, k, lab, prim in COMPS:
        r = {"generator": gen, "comparison": cid, "label": lab, "primary": prim}
        fx = lambda W: cst["fixed"][c].m(W)["nll"] - cst["fixed"][k].m(W)["nll"]
        pa = lambda W: cst["parameterized"][c].m(W)["nll"] - cst["parameterized"][k].m(W)["nll"]
        for name, f in (("fixed", fx), ("parameterized", pa), ("diff", lambda W: fx(W) - pa(W))):
            sm = summarize(f(Wp)[0], f(Wb))
            r.update({f"{name}_dnll": sm["est"], f"{name}_lo": sm["lo"], f"{name}_hi": sm["hi"]})
        # type-equal-weight sensitivity: simple mean of the per-type reductions within each class
        sub = per_type[per_type.comparison == cid]
        r["fixed_typeeq_dnll"] = sub[sub.rule_class == "fixed"].nll_reduction.mean()
        r["param_typeeq_dnll"] = sub[sub.rule_class == "parameterized"].nll_reduction.mean()
        for c_, m in cls.items():
            r[f"{c_}_viol_rate"] = float(v[m].sum() / n[m].sum())
            r[f"{c_}_outcomes"] = int(n[m].sum())
            r[f"{c_}_pools"] = int(len(set(d.group_id[m])))
        fp.append(r)
    out["fixed_vs_param"] = pd.DataFrame(fp)
    # ---- calibration bins for reliability curves (A2)
    cal = []
    for p in PREDS:
        e = edges_for(gen, p)
        s = st[p]
        bn, bv, bp = s.bn.sum(0), s.bv.sum(0), s.bp.sum(0)
        lo = np.r_[0, e]; hi = np.r_[e, 1]
        for b in range(len(bn)):
            cal.append({"generator": gen, "predictor": p, "bin": b, "edge_lo": lo[b], "edge_hi": hi[b],
                        "n_outcomes": bn[b], "mean_pred": bp[b] / bn[b] if bn[b] else np.nan,
                        "obs_rate": bv[b] / bn[b] if bn[b] else np.nan, "n_edges": len(e)})
    out["calibration"] = pd.DataFrame(cal)
    out["qc"] = {"generator": gen, "rows": len(d), "outcomes": int(n.sum()), "violations": int(v.sum()),
                 "pools": int(d.group_id.nunique()), "tasks": int(d.prompt_id.str.split("-t").str[-1].nunique()),
                 "prompts": int(d.prompt_id.nunique()),
                 "clipped_rows": {p: int(((d[p] < EPS) | (d[p] > 1 - EPS)).sum()) for p in PREDS}}
    return out
