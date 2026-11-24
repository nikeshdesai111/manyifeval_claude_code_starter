"""Evaluation 2, Objective 1 on S3 TEST (plan §§5.2-5.8, 6): observed effects and predictive skill (Δ-MSE).

Weighting (the plan's proposed recipe, §§5.1/6.3; not separately signed off): equal split stratum -> pool -> task ->
context -> comparison; equal-k (or equal-transition) macro across strata for whole-factor summaries. Uncertainty: paired
22-pool bootstrap (B = 10,000, seed 20261203), percentile 95% intervals, pointwise/unadjusted.
Noise-adjusted RMS: weighted mean of (d^2 - V) aggregated first, square root of max(0, .) once at the end.
Δ-MSE reduction = MSE(comparator) - MSE(candidate) of (d_pred - d_obs); positive favours the candidate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import matches
from .common import COMPS, PREDS
from .stats import Boot, estimate, group_table

PP = 100.0


def _rates(d: pd.DataFrame):
    """(prompt_id, rule_id) -> observed rate, binomial variance of the rate, predictions."""
    x = d.set_index(["prompt_id", "rule_id"])
    var = x.p_obs * (1 - x.p_obs) / (x.n - 1)
    return x, var


def _skill_cols(df):
    """Per-comparison squared/absolute errors and paired differences; inputs dobs and d_<pred>."""
    for p in PREDS:
        df[f"se_{p}"] = (df[f"d_{p}"] - df.dobs) ** 2
        df[f"ae_{p}"] = (df[f"d_{p}"] - df.dobs).abs()
    df["se_zero"] = df.dobs ** 2
    df["ae_zero"] = df.dobs.abs()
    for cid, c, k, _, _ in COMPS:
        df[f"dmse_{cid}"] = df[f"se_{c}"] - df[f"se_{k}"]
        df[f"dmae_{cid}"] = df[f"ae_{c}"] - df[f"ae_{k}"]
    return df


SKILL_VALS = ([f"dmse_{c[0]}" for c in COMPS] + [f"dmae_{c[0]}" for c in COMPS] + [f"se_{p}" for p in PREDS]
              + [f"ae_{p}" for p in PREDS] + ["se_zero", "ae_zero"])


def skill(df, strata, levels, boot, factor, gen):
    """Table 4 column (macro over strata) and the by-stratum appendix table (Pos-B, K-B, C-B, D-C)."""
    tabs = group_table(df, SKILL_VALS, strata, levels)
    idx = list(tabs[SKILL_VALS[0]].index)
    rows = []
    for v in SKILL_VALS:
        one = {v: tabs[v]}
        mac = estimate(one, boot, lambda m: m[v])
        rows.append({"generator": gen, "factor": factor, "stratum": "macro", "quantity": v, **mac})
        for s, r in zip(idx, estimate(one, boot, lambda m: m[v], strata_mean=False, with_loo=False)):
            rows.append({"generator": gen, "factor": factor, "stratum": str(s), "quantity": v, **r})
    return pd.DataFrame(rows)


def _rms(df, val_sq, strata, levels, boot):
    tabs = group_table(df, [val_sq], strata, levels)
    return estimate(tabs, boot, lambda m: m[val_sq], post=lambda r: PP * np.sqrt(np.maximum(r, 0)))


def _csq(df, val_sq, strata, levels, boot):
    tabs = group_table(df, [val_sq], strata, levels)
    return estimate(tabs, boot, lambda m: m[val_sq], post=lambda r: 1e4 * r, with_loo=False)


def _signed(df, val, strata, levels, boot, per_stratum=True):
    tabs = group_table(df, [val], strata, levels)
    if per_stratum:
        res = estimate(tabs, boot, lambda m: PP * m[val], strata_mean=False, with_loo=False)
        return list(zip(tabs[val].index, res))
    return estimate(tabs, boot, lambda m: PP * m[val])


def by_type_rms(df, boot, levels, gen, factor, types, extra=None):
    """15 target types: noise-adjusted RMS (pp), macro over k within type; plus corrected squared, noise floor, raw."""
    rows = []
    for t in types:
        x = df[df.type_id == t]
        if x.empty:
            rows.append({"generator": gen, "factor": factor, "type_id": t, "status": "unsupported"})
            continue
        r = _rms(x, "csq", ["k"], levels, boot)
        cs = _csq(x, "csq", ["k"], levels, boot)
        raw = _csq(x, "dsq", ["k"], levels, boot)
        nf = _csq(x, "V", ["k"], levels, boot)
        rows.append({"generator": gen, "factor": factor, "type_id": t, "rms_pp": r["est"], "rms_lo": r["lo"],
                     "rms_hi": r["hi"], "corrected_sq_x1e4": cs["est"], "corrected_sq_lo": cs["lo"],
                     "corrected_sq_hi": cs["hi"], "raw_sq_x1e4": raw["est"], "noise_floor_x1e4": nf["est"],
                     "n_pools": int(x.group_id.nunique()), "n_comparisons": len(x),
                     "baseline_risk": float(x.base.mean()), "truncated_at_zero": cs["est"] <= 0,
                     "low_support_flag": x.group_id.nunique() < 8, "status": "ok"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------- position
def position(d, boot, gen, types):
    x, var = _rates(d)
    cells = matches.load("position_cells")
    key = pd.MultiIndex.from_arrays([cells.prompt_id, cells.rule_id])
    cells["p"] = x.p_obs.reindex(key).values
    cells["var"] = var.reindex(key).values
    for p in PREDS:
        cells[f"q_{p}"] = x[p].reindex(key).values
    assert cells.p.notna().all()
    cells["a2var"] = cells.a ** 2 * cells["var"]
    agg = {"p": "mean", "a2var": "sum", **{f"q_{p}": "mean" for p in PREDS}}
    C = cells.groupby(["ctx", "slot"]).agg(agg)
    pairs = matches.load("position_pairs")
    ci = C.reindex(pd.MultiIndex.from_arrays([pairs.ctx, pairs.i]))
    cj = C.reindex(pd.MultiIndex.from_arrays([pairs.ctx, pairs.j]))
    pairs["dobs"] = cj.p.values - ci.p.values
    pairs["V"] = cj.a2var.values + ci.a2var.values
    pairs["base"] = (cj.p.values + ci.p.values) / 2
    for p in PREDS:
        pairs[f"d_{p}"] = cj[f"q_{p}"].values - ci[f"q_{p}"].values
    pairs["dsq"] = pairs.dobs ** 2
    pairs["csq"] = pairs.dsq - pairs.V
    pairs = _skill_cols(pairs)
    lv = ["task_key", "ctx", "pair_id"]
    out = {"pairs": pairs}
    out["skill"] = skill(pairs, ["k"], lv, boot, "position", gen)
    out["rms_by_type"] = by_type_rms(pairs, boot, lv, gen, "position", types)
    # Panel B: signed last-first by type x k
    ends = pairs[(pairs.i == 1) & (pairs.j == pairs.k)]
    rows = []
    for (t, k), r in _signed(ends, "dobs", ["type_id", "k"], ["task_key", "ctx"], boot):
        sub = ends[(ends.type_id == t) & (ends.k == k)]
        rows.append({"generator": gen, "type_id": t, "k": k, "effect_pp": r["est"], "lo": r["lo"], "hi": r["hi"],
                     "n_pools": r["n_groups"], "n_contexts": len(sub)})
    out["signed_type_k"] = pd.DataFrame(rows)
    # Pos-A: slot risk curves and 20 pair strata
    cr = cells.groupby(["ctx", "slot", "group_id", "task_key", "k"]).p.mean().reset_index()
    rows = []
    for (k, j), r in _signed(cr, "p", ["k", "slot"], ["task_key", "ctx"], boot):
        rows.append({"generator": gen, "k": k, "slot": j, "risk_pct": r["est"], "lo": r["lo"], "hi": r["hi"],
                     "n_pools": r["n_groups"]})
    out["slot_risk"] = pd.DataFrame(rows)
    rows = []
    for (k, i, j), r in _signed(pairs, "dobs", ["k", "i", "j"], ["task_key", "ctx"], boot):
        rows.append({"generator": gen, "k": k, "i": i, "j": j, "effect_pp": r["est"], "lo": r["lo"], "hi": r["hi"],
                     "n_pools": r["n_groups"], "n_contexts": int(((pairs.k == k) & (pairs.i == i) & (pairs.j == j)).sum())})
    out["pair_strata"] = pd.DataFrame(rows)
    return out


# --------------------------------------------------------------------------------------------- strict append
def append(d, boot, gen):
    x, var = _rates(d)
    a = matches.load("append")
    kb = pd.MultiIndex.from_arrays([a.prompt_before, a.rule_id])
    ka = pd.MultiIndex.from_arrays([a.prompt_after, a.rule_id])
    a["dobs"] = x.p_obs.reindex(ka).values - x.p_obs.reindex(kb).values
    for p in PREDS:
        a[f"d_{p}"] = x[p].reindex(ka).values - x[p].reindex(kb).values
    assert a.dobs.notna().all()
    a = _skill_cols(a)
    lv = ["task_key", "pair_id"]  # retained targets averaged equally within a pair (last level)
    out = {"rows": a}
    out["skill"] = skill(a, ["transition"], lv, boot, "append", gen)
    rows = []
    for (tr,), r in [((s,), v) for s, v in _signed(a, "dobs", ["transition"], lv, boot)]:
        sub = a[a.transition == tr]
        rows.append({"generator": gen, "transition": tr, "effect_pp": r["est"], "lo": r["lo"], "hi": r["hi"],
                     "n_pools": r["n_groups"], "n_prompt_pairs": sub.pair_id.nunique(), "n_targets": len(sub)})
    out["by_transition"] = pd.DataFrame(rows)
    rows = []
    for (tr, j), r in _signed(a, "dobs", ["transition", "slot"], lv, boot):
        sub = a[(a.transition == tr) & (a.slot == j)]
        rows.append({"generator": gen, "transition": tr, "slot": j, "effect_pp": r["est"], "lo": r["lo"], "hi": r["hi"],
                     "n_pools": r["n_groups"], "n_prompt_pairs": sub.pair_id.nunique(), "n_targets": len(sub)})
    out["by_slot"] = pd.DataFrame(rows)
    return out


# ------------------------------------------------------------------------------------------------- companion
def companion(d, boot, gen, types):
    x, var = _rates(d)
    c = matches.load("companion_pairs")
    ka = pd.MultiIndex.from_arrays([c.prompt_a, c.rule_id])
    kb = pd.MultiIndex.from_arrays([c.prompt_b, c.rule_id])
    pa, pb = x.p_obs.reindex(ka).values, x.p_obs.reindex(kb).values
    c["dobs"] = pb - pa
    c["V"] = var.reindex(ka).values + var.reindex(kb).values
    c["base"] = (pa + pb) / 2
    for p in PREDS:
        c[f"d_{p}"] = x[p].reindex(kb).values - x[p].reindex(ka).values
    c["dsq"] = c.dobs ** 2
    c["csq"] = c.dsq - c.V
    c = _skill_cols(c)
    lv = ["task_key", "ctx", "cmp_id"]
    out = {"pairs": c}
    out["skill"] = skill(c, ["k"], lv, boot, "companion", gen)
    out["rms_by_type"] = by_type_rms(c, boot, lv, gen, "companion", types)
    # C-A heterogeneity: RMS by k x relative order (all types) and by type x relative order
    rows = []
    for k in sorted(c.k.unique()):
        for ro in ("earlier", "later", "both"):
            s = c[(c.k == k) & ((c.relative_order == ro) | (ro == "both"))]
            r = _rms(s, "csq", ["k"], lv, boot)
            rows.append({"generator": gen, "k": k, "relative_order": ro, "rms_pp": r["est"], "lo": r["lo"], "hi": r["hi"],
                         "n_comparisons": len(s), "n_pools": s.group_id.nunique()})
    for ro in ("earlier", "later"):
        s = c[c.relative_order == ro]
        r = _rms(s, "csq", ["k"], lv, boot)
        rows.append({"generator": gen, "k": "macro", "relative_order": ro, "rms_pp": r["est"], "lo": r["lo"],
                     "hi": r["hi"], "n_comparisons": len(s), "n_pools": s.group_id.nunique()})
    out["rms_k_order"] = pd.DataFrame(rows)
    # Panel B: named companion vs the mean of the other available companions in the same fixed context
    h = matches.load("companion_ctx")
    kk = pd.MultiIndex.from_arrays([h.prompt_id, h.rule_id])
    h["p"] = x.p_obs.reindex(kk).values
    s_ = h.groupby("ctx").p.transform("sum")
    n_ = h.groupby("ctx").p.transform("size")
    h = h[n_ >= 2].copy()
    h["dsig"] = h.p - (s_[n_ >= 2] - h.p) / (n_[n_ >= 2] - 1)
    rows = []
    for (tt, ct), r in _signed(h, "dsig", ["type_id", "comp_type_id"], ["task_key", "ctx"], boot):
        sub = h[(h.type_id == tt) & (h.comp_type_id == ct)]
        rows.append({"generator": gen, "target_type": tt, "companion_type": ct, "effect_pp": r["est"], "lo": r["lo"],
                     "hi": r["hi"], "n_pools": r["n_groups"], "n_tasks": sub.task_key.nunique(),
                     "n_contexts": sub.ctx.nunique(), "mean_alternatives": float(sub.n_alternatives.mean())})
    out["signed_matrix"] = pd.DataFrame(rows)
    return out


# ------------------------------------------------------------------------------------------------------- task
def task(d, boot, gen, types):
    x, var = _rates(d)
    t = matches.load("task_pairs")
    ka = pd.MultiIndex.from_arrays([t.prompt_a, t.rule_id])
    kb = pd.MultiIndex.from_arrays([t.prompt_b, t.rule_id])
    pa, pb = x.p_obs.reindex(ka).values, x.p_obs.reindex(kb).values
    t["dobs"] = pb - pa
    t["V"] = var.reindex(ka).values + var.reindex(kb).values
    t["base"] = (pa + pb) / 2
    for p in PREDS:
        t[f"d_{p}"] = x[p].reindex(kb).values - x[p].reindex(ka).values
    t["dsq"] = t.dobs ** 2
    t["csq"] = t.dsq - t.V
    t["cfg_target"] = t.config_id + "|" + t.rule_id
    t = _skill_cols(t)
    lv = ["cfg_target", "cmp_id"]
    out = {"pairs": t}
    out["skill"] = skill(t, ["k"], lv, boot, "task", gen)
    out["rms_by_type"] = by_type_rms(t, boot, lv, gen, "task", types)
    rows = []
    for ty in types:
        for k in sorted(t.k.unique()):
            s = t[(t.type_id == ty) & (t.k == k)]
            if s.empty:
                rows.append({"generator": gen, "type_id": ty, "k": k, "status": "unsupported"})
                continue
            r = _rms(s, "csq", ["k"], lv, boot)
            cs = _csq(s, "csq", ["k"], lv, boot)
            rows.append({"generator": gen, "type_id": ty, "k": k, "rms_pp": r["est"], "lo": r["lo"], "hi": r["hi"],
                         "corrected_sq_x1e4": cs["est"], "baseline_risk": float(s.base.mean()),
                         "n_pools": s.group_id.nunique(), "n_task_pairs": len(s), "status": "ok"})
    out["rms_type_k"] = pd.DataFrame(rows)
    f = matches.load("task_family")
    pf = x.p_obs.reindex(pd.MultiIndex.from_arrays([f.prompt_f, f.rule_id])).values
    o1 = x.p_obs.reindex(pd.MultiIndex.from_arrays([f.prompt_o1, f.rule_id])).values
    o2 = x.p_obs.reindex(pd.MultiIndex.from_arrays([f.prompt_o2, f.rule_id])).values
    f["dfam"] = pf - (o1 + o2) / 2
    f["cfg_target"] = f.config_id + "|" + f.rule_id
    rows = []
    for (fam,), r in [((s,), v) for s, v in _signed(f, "dfam", ["family"], ["k", "cfg_target"], boot)]:
        rows.append({"generator": gen, "family": fam, "effect_pp": r["est"], "lo": r["lo"], "hi": r["hi"],
                     "n_pools": r["n_groups"], "low_support_flag": r["n_groups"] < 5})
    out["family"] = pd.DataFrame(rows)
    rows = []
    for (ty, fam), r in _signed(f, "dfam", ["type_id", "family"], ["k", "cfg_target"], boot):
        rows.append({"generator": gen, "type_id": ty, "family": fam, "effect_pp": r["est"], "lo": r["lo"],
                     "hi": r["hi"], "n_pools": r["n_groups"], "low_support_flag": r["n_groups"] < 3})
    out["type_family"] = pd.DataFrame(rows)
    # directed family pairs (task of family A minus task of family B in the same pool), where >= 3 pools support it
    tp = t.copy()
    a2 = tp.rename(columns={"family_a": "F1", "family_b": "F2"}).assign(dd=tp.dobs * -1)  # p_a - p_b
    b2 = tp.rename(columns={"family_b": "F1", "family_a": "F2"}).assign(dd=tp.dobs)       # p_b - p_a
    fpairs = pd.concat([a2, b2])[["group_id", "k", "cfg_target", "F1", "F2", "dd"]]
    rows = []
    for (f1, f2), r in _signed(fpairs, "dd", ["F1", "F2"], ["k", "cfg_target"], boot):
        if r["n_groups"] >= 3 and f1 < f2:
            rows.append({"generator": gen, "family_1": f1, "family_2": f2, "effect_pp": r["est"], "lo": r["lo"],
                         "hi": r["hi"], "n_pools": r["n_groups"]})
    out["family_pairs"] = pd.DataFrame(rows)
    return out
