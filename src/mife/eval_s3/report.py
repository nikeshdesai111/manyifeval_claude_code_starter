"""Write results/eval_s3/RESULTS_objective1.md from the saved tables and figures (every number from a saved CSV)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import figures
from .common import COMPS, GEN_LABEL, GENS, PRED_LABEL, short_type
from .matches import FIXED_TYPES
from .paths import EVAL_DIR

T = EVAL_DIR / "tables"
OUT = EVAL_DIR / "RESULTS_objective1.md"
TRS = ["1->2", "2->3", "3->4", "4->5"]


def r(n):
    return pd.read_csv(T / f"{n}.csv")


def ci(e, lo, hi, d=2, s=1.0, sign=True):
    if not np.isfinite(e):
        return "NA"
    f = f"{{:{'+' if sign else ''}.{d}f}}"
    return f"{f.format(e * s)} [{f.format(lo * s)}, {f.format(hi * s)}]"


def bold_if(txt, cond):
    return f"**{txt}**" if cond else txt


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|"]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(str(x) for x in row.values) + " |")
    return "\n".join(out)


def _types(df, col="type_id"):
    ts = sorted(df[col].dropna().unique())
    return [t for t in ts if t in FIXED_TYPES] + [t for t in ts if t not in FIXED_TYPES]


def tname(t):
    return f"{short_type(t)}{' (fixed)' if t in FIXED_TYPES else ''}"


# ----------------------------------------------------------------------------------------------- sections
def table2():
    c = r("eval1_contrasts")
    s = []
    for k, gen in enumerate(GENS):
        x = c[c.generator == gen]
        rows = []
        for _, z in x.iterrows():
            rows.append({"Method comparison (comparator → candidate)": bold_if(z.label, z.primary),
                         "NLL reduction [95% CI] ↑": bold_if(ci(z.nll_reduction, z.nll_reduction_lo, z.nll_reduction_hi, 4), z.primary),
                         "AUROC gain [95% CI] ↑": ci(z.auroc_gain, z.auroc_gain_lo, z.auroc_gain_hi, 3),
                         "Types improved (of 15) ↑": f"{int(z.types_improved)}/15"
                         + (f" ({int(z.types_undefined)} undefined)" if z.types_undefined else "")})
        s.append(f"**Panel {'AB'[k]}: {GEN_LABEL[gen]}**\n\n" + md(pd.DataFrame(rows)))
    return "\n\n".join(s)


def a1():
    m = r("eval1_a1_metrics")
    c = r("eval1_contrasts")
    out = []
    for gen in GENS:
        x = m[m.generator == gen]
        rows = [{"Predictor": PRED_LABEL[z.predictor], "NLL [95% CI]": ci(z.nll, z.nll_lo, z.nll_hi, 4, sign=False),
                 "Brier [95% CI]": ci(z.brier, z.brier_lo, z.brier_hi, 4, sign=False),
                 "AUROC [95% CI]": ci(z.auroc, z.auroc_lo, z.auroc_hi, 3, sign=False),
                 "ECE [95% CI]": ci(z.ece, z.ece_lo, z.ece_hi, 4, sign=False)} for _, z in x.iterrows()]
        out.append(f"**{GEN_LABEL[gen]}: raw pooled metrics** (unit: one response × included-rule outcome)\n\n" + md(pd.DataFrame(rows)))
        y = c[c.generator == gen]
        rows = [{"Comparison": bold_if(z.label, z.primary),
                 "NLL(comparator)": f"{m[(m.generator == gen) & (m.predictor == z.comparator)].nll.iloc[0]:.4f}",
                 "NLL(candidate)": f"{m[(m.generator == gen) & (m.predictor == z.candidate)].nll.iloc[0]:.4f}",
                 "NLL reduction": ci(z.nll_reduction, z.nll_reduction_lo, z.nll_reduction_hi, 4),
                 "Brier reduction": ci(z.brier_reduction, z.brier_reduction_lo, z.brier_reduction_hi, 4),
                 "AUROC gain": ci(z.auroc_gain, z.auroc_gain_lo, z.auroc_gain_hi, 3)} for _, z in y.iterrows()]
        out.append(f"**{GEN_LABEL[gen]}: all seven contrasts with raw scores**\n\n" + md(pd.DataFrame(rows)))
    return "\n\n".join(out)


def a3b():
    f = r("eval1_fixed_vs_param")
    out = []
    for k, gen in enumerate(GENS):
        x = f[f.generator == gen]
        rows = [{"Comparator → candidate": bold_if(z.label, z.primary),
                 "Fixed (5 types): ΔNLL [95% CI] ↑": ci(z.fixed_dnll, z.fixed_lo, z.fixed_hi, 4),
                 "Parameterized (10 types): ΔNLL [95% CI] ↑": ci(z.parameterized_dnll, z.parameterized_lo, z.parameterized_hi, 4),
                 "Fixed − parameterized improvement [95% CI]": ci(z.diff_dnll, z.diff_lo, z.diff_hi, 4),
                 "Type-equal-weight sensitivity (fixed / param.)": f"{z.fixed_typeeq_dnll:+.4f} / {z.param_typeeq_dnll:+.4f}"}
                for _, z in x.iterrows()]
        z = x.iloc[0]
        ctx = (f"Observed violation rate: fixed {100 * z.fixed_viol_rate:.1f}% ({int(z.fixed_outcomes):,} response×rule outcomes, "
               f"{int(z.fixed_pools)} pools); parameterized {100 * z.parameterized_viol_rate:.1f}% "
               f"({int(z.parameterized_outcomes):,} outcomes, {int(z.parameterized_pools)} pools).")
        out.append(f"**Panel {'AB'[k]}: {GEN_LABEL[gen]}.** {ctx}\n\n" + md(pd.DataFrame(rows)))
    return "\n\n".join(out)


def a4():
    c = r("eval1_contrasts")
    qc = json.loads((T / "eval1_qc.json").read_text())
    rows = []
    for _, z in c.iterrows():
        rows.append({"Generator": z.generator, "Comparison": z.comparison,
                     "NLL reduction": f"{z.nll_reduction:+.4f}",
                     "bootstrap 95% CI": f"[{z.nll_reduction_lo:+.4f}, {z.nll_reduction_hi:+.4f}]",
                     "t(21) jackknife CI": f"[{z.nll_t_lo:+.4f}, {z.nll_t_hi:+.4f}]",
                     "leave-one-pool-out range": f"[{z.nll_loo_min:+.4f}, {z.nll_loo_max:+.4f}]"})
    q = "\n".join(f"- {d['generator']}: {d['rows']:,} prompt–rule rows, {d['outcomes']:,} response×rule outcomes "
                  f"({d['violations']:,} violations), {d['prompts']:,} prompts, {d['pools']} pools, {d['tasks']} tasks; "
                  f"rows clipped at 1e-6: {sum(d['clipped_rows'].values())}." for d in qc)
    pt = r("eval1_per_type")
    low = pt[pt.low_support_flag].drop_duplicates(["generator", "rule_type"])
    lows = ", ".join(f"{g}/{short_type(t)} ({n} pools)" for g, t, n in zip(low.generator, low.rule_type, low.n_pools)) or "none"
    return (q + "\n" + f"- Probabilities clipped to [1e-6, 1 − 1e-6] before the log. Paired cluster bootstrap over the 22 TEST pools, "
            "B = 10,000, seed 20261203, percentile intervals (pointwise, unadjusted). No sparse-type fallback head fired in "
            "the S3 refits (every type has its own ridge-logistic head).\n"
            f"- Rule types with fewer than 8 contributing TEST pools (per-type estimates less stable): {lows}.\n\n"
            "Sensitivity of the Table 2 NLL reductions to the interval method:\n\n" + md(pd.DataFrame(rows)))


def pos_b_like(factor, strata_order, strata_label):
    s = r(f"eval2_{factor}_skill")
    out = []
    for gen in GENS:
        x = s[s.generator == gen]
        rows = []
        for cid, c, k, lab, prim in COMPS:
            row = {"Comparison": bold_if(lab, prim)}
            for st in strata_order:
                z = x[(x.quantity == f"dmse_{cid}") & (x.stratum == str(st))].iloc[0]
                row[strata_label(st)] = ci(z.est, z.lo, z.hi, 2, 1e4)
            z = x[(x.quantity == f"dmse_{cid}") & (x.stratum == "macro")].iloc[0]
            row["Macro (Table 4)"] = ci(z.est, z.lo, z.hi, 2, 1e4)
            rows.append(row)
        out.append(f"**{GEN_LABEL[gen]}: Δ-MSE reduction (pp²) [95% CI]**\n\n" + md(pd.DataFrame(rows)))
        rows = []
        for cid, c, k, lab, prim in COMPS:
            z = x[(x.quantity == f"dmae_{cid}") & (x.stratum == "macro")].iloc[0]
            rows.append({"Comparison": lab, "Δ-MAE reduction, macro (pp) [95% CI]": ci(z.est, z.lo, z.hi, 3, 100)})
        out.append(md(pd.DataFrame(rows)))
        rows = []
        for p in ["zero"] + [c for c in PRED_LABEL if c != "REF_structure"]:
            row = {"Change predictor": "zero change (d=0)" if p == "zero" else PRED_LABEL[p]}
            for st in strata_order + ["macro"]:
                z = x[(x.quantity == f"se_{p}") & (x.stratum == str(st))].iloc[0]
                row[strata_label(st) if st != "macro" else "Macro"] = f"{1e4 * z.est:.1f}"
            rows.append(row)
        out.append(f"Raw Δ-MSE (pp²) of each predictor and of the zero-change reference ({GEN_LABEL[gen]}):\n\n"
                   + md(pd.DataFrame(rows)))
    return "\n\n".join(out)


def table4():
    out = []
    facs = [("position", "Position"), ("append", "Strict append"), ("companion", "Companion replacement"),
            ("task", "Task context")]
    sk = {f: r(f"eval2_{f}_skill") for f, _ in facs}
    for k, gen in enumerate(GENS):
        rows = []
        for cid, c, kk, lab, prim in COMPS:
            row = {"V2 comparison": bold_if(lab, prim)}
            for f, name in facs:
                z = sk[f][(sk[f].generator == gen) & (sk[f].quantity == f"dmse_{cid}") & (sk[f].stratum == "macro")].iloc[0]
                row[f"{name}: Δ-MSE reduction [95% CI] ↑"] = ci(z.est, z.lo, z.hi, 2, 1e4)
            rows.append(row)
        out.append(f"**Panel {'AB'[k]}: {GEN_LABEL[gen]}**\n\n" + md(pd.DataFrame(rows)))
    return "\n\n".join(out)


def rms_table(name, extra_cols=()):
    a = r(name)
    types = _types(a)
    rows = []
    for t in types:
        row = {"Target type": tname(t)}
        for gen in GENS:
            z = a[(a.generator == gen) & (a.type_id == t)].iloc[0]
            g = gen.capitalize()
            row[f"{g} RMS pp [95% CI]"] = ci(z.rms_pp, z.rms_lo, z.rms_hi, 1, sign=False)
            row[f"{g} corrected sq. / noise floor (pp²)"] = f"{z.corrected_sq_x1e4:.1f} / {z.noise_floor_x1e4:.1f}"
            row[f"{g} base risk"] = f"{100 * z.baseline_risk:.0f}%"
            row[f"{g} pools"] = int(z.n_pools)
        rows.append(row)
    return md(pd.DataFrame(rows))


def signed_type_k():
    b = r("eval2_position_signed_type_k")
    types = _types(b)
    rows = []
    for t in types:
        row = {"Target type": tname(t)}
        for gen in GENS:
            for k in [2, 3, 4, 5]:
                z = b[(b.generator == gen) & (b.type_id == t) & (b.k == k)]
                row[f"{gen[0].upper()} k={k}"] = "NA" if z.empty else ci(z.effect_pp.iloc[0], z.lo.iloc[0], z.hi.iloc[0], 1) + f" ({int(z.n_pools.iloc[0])})"
        rows.append(row)
    return md(pd.DataFrame(rows))


def pair_strata():
    p = r("eval2_position_pair_strata")
    rows = []
    for (k, i, j), x in p.groupby(["k", "i", "j"]):
        row = {"k": k, "slot i → j": f"{i} → {j}"}
        for gen in GENS:
            z = x[x.generator == gen].iloc[0]
            row[f"{GEN_LABEL[gen].split(' ')[0]} effect pp [95% CI]"] = ci(z.effect_pp, z.lo, z.hi, 2)
        row["contexts / pools"] = f"{int(z.n_contexts):,} / {int(z.n_pools)}"
        rows.append(row)
    return md(pd.DataFrame(rows))


def append_tables():
    a = r("eval2_append_by_transition")
    b = r("eval2_append_by_slot")
    rows = []
    for tr in TRS:
        row = {"Transition": tr.replace("->", "→")}
        for gen in GENS:
            z = a[(a.generator == gen) & (a.transition == tr)].iloc[0]
            row[f"{GEN_LABEL[gen].split(' ')[0]} Δ risk pp [95% CI]"] = ci(z.effect_pp, z.lo, z.hi, 2)
        row["prompt pairs / retained targets / pools"] = f"{int(z.n_prompt_pairs):,} / {int(z.n_targets):,} / {int(z.n_pools)}"
        rows.append(row)
    fig_tab = md(pd.DataFrame(rows))
    rows = []
    for tr in TRS:
        for j in sorted(b[b.transition == tr].slot.unique()):
            row = {"Transition": tr.replace("->", "→"), "Retained slot j": j}
            for gen in GENS:
                z = b[(b.generator == gen) & (b.transition == tr) & (b.slot == j)].iloc[0]
                row[f"{GEN_LABEL[gen].split(' ')[0]} Δ risk pp [95% CI]"] = ci(z.effect_pp, z.lo, z.hi, 2)
            row["prompt pairs / pools"] = f"{int(z.n_prompt_pairs):,} / {int(z.n_pools)}"
            rows.append(row)
    return fig_tab, md(pd.DataFrame(rows))


def companion_tables():
    k = r("eval2_companion_rms_k_order")
    rows = []
    for kk in ["2", "3", "4", "5", "macro"]:
        for ro in ["earlier", "later", "both"]:
            x = k[(k.k.astype(str) == kk) & (k.relative_order == ro)]
            if x.empty:
                continue
            row = {"k": kk, "companion placement": ro}
            for gen in GENS:
                z = x[x.generator == gen].iloc[0]
                row[f"{GEN_LABEL[gen].split(' ')[0]} RMS pp [95% CI]"] = ci(z.rms_pp, z.lo, z.hi, 1, sign=False)
            row["comparisons"] = f"{int(z.n_comparisons):,}"
            rows.append(row)
    m = r("eval2_companion_signed_matrix")
    sup = m.groupby("generator").agg(cells=("effect_pp", "size"), low=("n_pools", lambda s: int((s < 3).sum())),
                                     median_pools=("n_pools", "median")).reset_index()
    top = []
    for gen in GENS:
        x = m[(m.generator == gen) & (m.n_pools >= 3)].copy()
        x["excl0"] = (x.lo > 0) | (x.hi < 0)
        top.append(f"- {GEN_LABEL[gen]}: {len(m[m.generator == gen])} supported directed cells (of 210 off-diagonal), "
                   f"{int((m[m.generator == gen].n_pools < 3).sum())} with fewer than 3 pools; "
                   f"{int(x.excl0.sum())} of the {len(x)} cells with ≥ 3 pools have a pointwise 95% CI excluding zero "
                   f"(about {0.05 * len(x):.0f} expected by chance if all were null).")
    return md(pd.DataFrame(rows)), "\n".join(top)


def task_tables():
    a = r("eval2_task_rms_type_k")
    types = _types(a)
    rows = []
    for t in types:
        row = {"Target type": tname(t)}
        for gen in GENS:
            for kk in [1, 2, 3, 4, 5]:
                z = a[(a.generator == gen) & (a.type_id == t) & (a.k == kk)]
                row[f"{gen[0].upper()} k={kk}"] = ("NA" if z.empty or z.status.iloc[0] != "ok" else
                                                   f"{z.rms_pp.iloc[0]:.1f} [{z.lo.iloc[0]:.1f}, {z.hi.iloc[0]:.1f}]")
        rows.append(row)
    da = md(pd.DataFrame(rows))
    f = r("eval2_task_family")
    rows = []
    for fam in sorted(f.family.unique()):
        row = {"Family": fam}
        for gen in GENS:
            z = f[(f.generator == gen) & (f.family == fam)].iloc[0]
            row[f"{GEN_LABEL[gen].split(' ')[0]} pp [95% CI]"] = ci(z.effect_pp, z.lo, z.hi, 2)
        row["pools"] = int(z.n_pools)
        rows.append(row)
    fam_t = md(pd.DataFrame(rows))
    fp = r("eval2_task_family_pairs")
    rows = []
    for (f1, f2), x in fp.groupby(["family_1", "family_2"]):
        row = {"Family pair (A − B)": f"{f1} − {f2}"}
        for gen in GENS:
            z = x[x.generator == gen]
            row[f"{GEN_LABEL[gen].split(' ')[0]} pp [95% CI]"] = "NA" if z.empty else ci(z.effect_pp.iloc[0], z.lo.iloc[0], z.hi.iloc[0], 2)
        row["pools"] = int(x.n_pools.iloc[0])
        rows.append(row)
    return da, fam_t, md(pd.DataFrame(rows)) if rows else "_No family pair is supported by at least 3 pools._"


def difficulty():
    e = r("factor_e_terciles")
    rows = []
    for gen in GENS:
        x = e[e.generator == gen]
        for terc in ["easy", "medium", "hard"]:
            z = x[x.tercile == terc]
            row = {"Generator": GEN_LABEL[gen].split(" ")[0], "TRAIN tercile": terc,
                   "types": ", ".join(short_type(t) for t in z.type_id)}
            for f in ["position", "companion", "task"]:
                row[f"mean {f} RMS (pp)"] = f"{z[f'{f}_rms'].mean():.1f}"
            row["TEST violation rate"] = f"{100 * z.test_rate.mean():.1f}%"
            rows.append(row)
    return md(pd.DataFrame(rows))


def build(findings: str = "") -> str:
    figs = figures.all_figures()
    fc = pd.concat([pd.read_csv(p) for p in sorted(T.glob("feature_consistency_*.csv"))])
    ver = [json.loads((T / f"verify_expansion_{g}.json").read_text()) for g in GENS]
    nv = pd.read_csv(T / "noise_validation_synthetic.csv")
    cov = json.loads((T / "match_coverage.json").read_text())
    ft, kat = append_tables()
    ctab, ctop = companion_tables()
    da, famt, fpt = task_tables()
    parts = [f"""# ManyIFEval S3 TEST results — Objective 1 (rule-level violation risk)

Generated from `results/eval_s3/tables/*.csv` by `mife.eval_s3.report`. Every number below comes from a saved table.
Scope: **Objective 1 only** (Evaluation 1 and Evaluation 2), following the agreed layout of
`manyifeval_s3_complete_evaluation_analysis_and_reporting_plan.md` (revised reporting draft, 2026-11-23). Objective 2 is
not analysed here.

{findings}

## 0. What was evaluated, and how

- **Design (S3 TEST):** 22 six-rule pools, 66 tasks, 456 ordered configurations per task, **30,096 prompts and 115,236
  prompt–rule rows per generator**, 20 verified responses per prompt (16,236 original prompts from run `main`, 13,860
  expansion prompts from run `expansion_s3_20261123`, vLLM on Betty, per-sample seeds; same decoding settings).
- **Predictors:** the frozen V2 refits on the 50 S3 TRAIN pools (`results/v2_s3/models/`), applied once to TEST. No
  model, layer, penalty, calibration edge or weight was chosen on TEST. Hidden row: Llama layer 12, Qwen layer 18.
- **Comparisons (positive always favours the candidate, right of the arrow):** primary #1 B1 → H1, #2 B2 → H2,
  #3 B2 → rule-conditioned attention, #4 H2 → rule-conditioned attention; secondary #5 B1 → B1+H1, #6 B2 → B2+H1, #7 B2 → B2+A1.
- **Uncertainty:** paired cluster bootstrap over the 22 TEST pools (B = 10,000, seed 20261203), percentile 95%
  intervals, pointwise and unadjusted for multiplicity. All predictors and contrasts use identical pool draws.
- **Evaluation 2 weighting:** the plan's *proposed* recipe (§§5.1, 6.3), which was not separately signed off before
  scoring: equal weight per pool, then per task, per matched context, per comparison; equal-k (or equal-transition)
  macro across strata. Δ-MSE is in pp² (squared percentage points); observed effects are in percentage points.
- **Noise-adjusted RMS** (position, companion and task Panel A): weighted mean of (d̂² − V̂) over matched comparisons,
  aggregated first, then √max(0, ·) once. Synthetic validation (Appendix V) shows the squared estimate is
  unbiased, intervals cover 92–100%, and RMS values under ~1.5 pp cannot be told apart from zero.
- **Disclosure:** 16 of the 22 S3 TEST pools were TRAIN pools under the 2026-11-14 split, so their original-prompt
  outcomes informed earlier V1/V2 development. The S3 refits themselves never saw them.
- **Authorization:** TEST unblinding for Objective 1 was approved by the project owner in the project thread on
  2026-11-24.

**Table 1. TEST inventory and controlled coverage (per generator; identical for Llama and Qwen).**

{md(pd.DataFrame([
        {"Item": "Pools / tasks / prompts / prompt–rule rows", "Count": "22 / 66 / 30,096 / 115,236"},
        {"Item": "Response × rule outcomes", "Count": "2,304,720"},
        {"Item": "Position: exact contexts / all-pair target contrasts (k = 2–5)", "Count": f"{cov['position_contexts']:,} / {cov['position_pairs']:,}"},
        {"Item": "Strict append: prompt pairs / retained-rule contrasts", "Count": f"{cov['append_prompt_pairs']:,} / {cov['append_targets']:,}"},
        {"Item": "Companion replacement: prompt pairs / retained-target contrasts", "Count": f"{sum(cov['companion_prompt_pairs_by_k'].values()):,} / {cov['companion_targets']:,}"},
        {"Item": "Task context: task-prompt pairs / exact-rule contrasts", "Count": f"{cov['task_prompt_pairs']:,} / {cov['task_targets']:,}"},
    ]))}

All counts equal the plan's design counts exactly (Appendix QC). Every original and expansion prompt has 20 distinct
verified responses (expansion: Llama {ver[0]['n_responses']:,}, Qwen {ver[1]['n_responses']:,}; one settings hash per
model; {100 * ver[0]['hit_max_tokens_share']:.1f}% / {100 * ver[1]['hit_max_tokens_share']:.1f}% hit the token limit).

---

## 1. Evaluation 1, Objective 1: rule-level violation-risk prediction

**Table 2. Rule-level violation-risk prediction on S3 TEST.** Each method predicts the probability that an included
rule is violated in one generated response. Binomial NLL is scored against the 20 observed binary verification outcomes
per prompt–rule pair and pooled over the same 115,236 TEST rows per generator (unit: one response × rule outcome).
NLL reduction = NLL(comparator) − NLL(candidate); AUROC gain = AUROC(candidate) − AUROC(comparator); positive favours
the candidate. Types improved counts rule types with strictly positive per-type NLL reduction (descriptive, not 15
significance tests). Paired pointwise 95% CIs over the 22 TEST pools. Primary comparisons in bold.

{table2()}

### Appendix A1. Compact overall metrics

{a1()}

### Appendix A2. Calibration (reliability curves)

Bins are the deciles of each predictor's S3 TRAIN out-of-fold predictions (fixed before TEST); unit = response × rule
outcome. Numerical ECE for every predictor is in A1.

![Reliability curves]({figs['e1_cal']})

### Appendix A3. Per-rule-type NLL reduction (15 types × 7 comparisons)

Cells show NLL(comparator) − NLL(candidate) in nats on that type's TEST outcomes; positive (red) favours the candidate.
The five fixed types are above the line, the ten parameterized types below. Per-cell CIs, support and NLLs are in
`tables/eval1_per_type.csv` (210 rows).

![Per-type NLL heatmap]({figs['e1_heat']})

### Appendix A3b. Fixed vs parameterized rule types (secondary, exploratory)

*NLL reductions of the same seven predictor comparisons, calculated separately on rule-level outcomes from the five fixed
and ten parameterized ManyIFEval rule types. Positive improvement favours the candidate (right of the arrow). The
fixed-minus-parameterized column compares improvements between subgroups, not raw violation rates. Intervals are paired
pointwise 95% confidence intervals from resampling the 22 TEST rule pools; all seven subgroup differences are
exploratory.*

{a3b()}

### Appendix A4. Technical uncertainty and QC

{a4()}

---

## 2. Evaluation 2, Objective 1: matched prompt changes

### 2.1 Position (Figure E2-Pos)

**Figure E2-Pos. Target-type heterogeneity in observed rule-position effects.** All comparisons hold the exact target
rule, task, included companions and k fixed; target-position risks are COA-balanced averages over the k−1 orderings
that place the target in each slot. **(A)** Noise-adjusted RMS change over all position pairs for each of the 15 target
types (equal-k macro, k = 2–5; 95% pool-cluster intervals). **(B)** Signed last − first risk change (pp) by target type
and k; positive = higher violation risk when last; hatched = unsupported. Moving a target also changes the placement of
other rules and the distance to the prompt end, so this is not a pure recency effect.

![Figure E2-Pos]({figs['pos']})

### 2.2 Strict append (Figure E2-Append)

**Figure E2-Append.** Mean observed change in the violation probability of the already-included (retained) rules
when exactly one rule is appended at the end (strict prefix matches; the added rule itself is not evaluated).
Retained targets are averaged within each prompt pair, then pairs → tasks → pools. 95% pool-cluster CIs. Only part of the
4- and 5-rule COA orders have an included exact prefix (3→4 and 4→5 coverage is partial by design).

![Figure E2-Append]({figs['append']})

{ft}

### 2.3 Companion replacement (Figure E2-Companion)

**Figure E2-Companion.** Strict one-companion replacement with the exact target, its slot, k, task and every other rule
fixed; earlier and later companions are pooled. **(A)** Noise-adjusted RMS change of the target's risk across companion
substitutions, by target type (equal-k macro; 95% pool-cluster CIs). **(B)** Directed signed effects: risk of the target
(row) with the named companion type (column) minus the mean over the other companions available in the same fixed
context (pp). Hatched = no valid strict match (including the diagonal); black outline = fewer than 3 supporting pools.
Not symmetric by construction; exploratory across 210 directed cells, pointwise only.

![Figure E2-Companion]({figs['companion']})

### 2.4 Task context (Figure E2-Task)

**Figure E2-Task.** The same exact ordered rules under the group's three different tasks (three different
researcher-defined task families). **(A)** Noise-adjusted RMS task-change sensitivity by target type (k = 1–5, equal-k
macro; 95% pool-cluster CIs). **(B)** Risk under the family-F task minus the mean of the other two matched tasks in the
same pool, pooled over targets, configurations and k; pool counts shown at right. The reference differs by pool, and a
family label is confounded with the whole task text, so these are associations, not causal family effects.

![Figure E2-Task]({figs['task']})

{famt}

### 2.5 Table 4: predicting matched changes (shared across factors)

**Table 4. Evaluation 2, Objective 1 predictor skill.** Δ-MSE = weighted mean of (predicted change − observed change)²
over the same matched comparisons for both methods; entries are Δ-MSE(comparator) − Δ-MSE(candidate) in pp²
(positive favours the candidate). Position: 57,420 all-pair contrasts, equal-k macro; strict append: 69,300 retained-rule
contrasts, equal-transition macro; companion: 177,408 retained-target contrasts, equal-k macro; task: 115,236 exact-rule
task-pair contrasts, equal-k macro (k = 1–5). Paired 95% pool-cluster CIs.

{table4()}

---

## Appendix Pos-A. Position profiles and all 20 pair strata

Observed target violation risk at every slot (COA-balanced, 95% pool CIs) with the paired last − first difference.

![Position curves]({figs['pos_curves']})

{pair_strata()}

## Appendix Pos-B. Predicting position changes by k

{pos_b_like('position', [2, 3, 4, 5], lambda s: f'k={s}')}

## Appendix Pos-C. Numerical type-level position effects

Noise-adjusted RMS sensitivity (Panel A), corrected squared component and estimated sampling-noise floor (pp²), mean
baseline risk of the compared cells, and supporting pools:

{rms_table('eval2_position_rms_by_type')}

Signed last − first effect (pp) [95% CI] (supporting pools), by target type and k (Panel B numbers):

{signed_type_k()}

## Appendix K-A. Strict append effect by retained-rule slot

{kat}

## Appendix K-B. Predicting strict-append changes by transition

{pos_b_like('append', TRS, lambda s: s.replace('->', '→'))}

Coverage: expected and realized strict prefix pairs are equal for every transition (1,980 / 7,920 / 11,880 / 3,960
prompt pairs; 1,980 / 15,840 / 35,640 / 15,840 retained targets per generator). Weights: retained targets equal within a
pair, pairs equal within task, tasks equal within pool, pools equal within transition, transitions equal in the macro.

## Appendix C-A. Companion effects: numbers and heterogeneity

{rms_table('eval2_companion_rms_by_type')}

RMS sensitivity by k and companion placement (all target types):

{ctab}

Directed signed matrix (Panel B): full numbers in `tables/eval2_companion_signed_matrix.csv`.

{ctop}

## Appendix C-B. Predicting companion-replacement changes by k

{pos_b_like('companion', [2, 3, 4, 5], lambda s: f'k={s}')}

## Appendix C-C. Companion implementation checks

Strict one-companion replacements: same pool, task, k, target rule and slot, and every other rule and slot; exactly one
slot differs. Counts by k equal the plan (prompt pairs 7,920 / 35,640 / 22,176 / 7,920; retained-target comparisons
7,920 / 71,280 / 66,528 / 31,680). Each unordered pair is oriented by prompt id and counted once per retained target.
Panel B alternatives are the other companions actually present in the same fixed context (no prompt is manufactured).

## Appendix D-A. Task sensitivity by target type and k

Noise-adjusted RMS (pp) [95% CI]:

{da}

Type-level numbers (Panel A):

{rms_table('eval2_task_rms_by_type')}

## Appendix D-B. Target type × task family

![Type x family heatmap]({figs['task_heat']})

Directed family pairs within pools (task of family A minus task of family B, same exact rules), where ≥ 3 pools support
the pair:

{fpt}

## Appendix D-C. Predicting task-context changes by k

{pos_b_like('task', [1, 2, 3, 4, 5], lambda s: f'k={s}')}

## Appendix D-D. Task matching and families

Families come from `configs/task_families_v1.csv` (researcher-defined rubric, `docs/task_family_rubric.md`). Every TEST
pool has three tasks from three different families, so no within-family matched comparison exists. Each identical ordered
configuration yields exactly three unordered task pairs (oriented by task key); 30,096 task-prompt pairs and 115,236
exact-rule contrasts per generator.

## Appendix E. TRAIN-defined rule difficulty (descriptive)

Terciles from S3 TRAIN k = 1 violation rates per generator (frozen before TEST scoring; ties by name). Mean of the
type-level RMS sensitivities within each tercile:

{difficulty()}

## Appendix V. Validation and QC

**Noise-adjusted RMS estimator, synthetic data** (22 pools × 40 contexts, 20 responses per prompt; `tables/noise_validation_synthetic.csv`):

{md(nv.round(3))}

**Feature consistency** (new TEST extraction vs frozen V1 caches on original prompts):

{md(fc)}

**Supplementary machine-readable files** (all in `results/eval_s3/tables/`): `eval1_per_type.csv` (210 per-type rows),
`eval1_fixed_vs_param.csv` (14 rows), `eval1_a1_metrics.csv`, `eval1_contrasts.csv`, `eval1_calibration.csv`,
`eval2_*_skill.csv` (all Δ-MSE, Δ-MAE, raw and zero-change values by stratum), `eval2_position_*`, `eval2_append_*`,
`eval2_companion_*`, `eval2_task_*`, `observed_rate_by_type.csv`, `match_coverage.json`.
"""]
    text = "\n".join(parts)
    OUT.write_text(text)
    return text


if __name__ == "__main__":
    build()
