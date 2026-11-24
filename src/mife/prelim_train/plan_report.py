"""PRELIMINARY S3 TRAIN results laid out exhibit by exhibit in the order of the evaluation plan
(manyifeval_s3_complete_evaluation_analysis_and_reporting_plan.md, §§4.4-4.6, 5.2-5.9, 8.1-8.2), with TRAIN
grouped-CV out-of-fold results in place of TEST. Built only from results/preliminary_train_report/tables/*.csv
(plus TRAIN k=1 labels for Factor E).
Run: PYTHONPATH=src .venv/bin/python -m mife.prelim_train.plan_report
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import COMPS
from .matches import FIXED_TYPES
from .paths import EVAL_DIR

T = EVAL_DIR / "tables"
OUT = EVAL_DIR / "PRELIMINARY_RESULTS_FOLLOWING_PLAN.md"
GEN = {"llama": "Llama-3.1-8B-Instruct", "qwen": "Qwen3-8B (non-thinking)"}
CL = {"#1": "#1 B1 → H1 + structure", "#2": "#2 B2 → H2 + structure", "#3": "#3 B2 → A2 (rule-conditioned attention)",
      "#4": "#4 H2 → A2", "#5": "#5 B1 → B1 + H", "#6": "#6 B2 → B2 + H", "#7": "#7 B2 → B2 + A", "#8": "#8 B2 + H → B2 + H + A"}
ORDER = ["#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8"]
PRIM = {"#1", "#2", "#3", "#4"}
PL = {"B1": "B1 (S+P)", "B2": "B2 (S+P+R)", "H1": "H1 + structure", "H2": "H2 + structure", "RCAP": "A2: rule-cond. attention",
      "B1+H1": "B1 + H", "B2+H1": "B2 + H", "B2+A1": "B2 + A", "B2+H1+A1": "B2 + H + A", "REF_structure": "Structure only (ref.)",
      "REF_trainmean": "Training-fold mean (ref.)"}


def r(n):
    return pd.read_csv(T / f"{n}.csv")


def f(x, d=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    v = round(float(x), d)
    return f"{0.0 if v == 0 else v:.{d}f}"


def ci(e, lo, hi, s=1.0, d=1):
    return f"{f(e * s, d)} [{f(lo * s, d)}, {f(hi * s, d)}]"


def tn(t):
    return t.split(":")[1]


def types_order(ts):
    return [t for t in sorted(ts) if t in FIXED_TYPES] + [t for t in sorted(ts) if t not in FIXED_TYPES]


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def bold(cid, s):
    return f"**{s}**" if cid in PRIM else s


def main():
    L = []
    w = L.append
    cov = json.loads((T / "match_coverage.json").read_text()) if (T / "match_coverage.json").exists() else {}
    qc = json.loads((T / "eval1_qc.json").read_text())
    w("# ManyIFEval — PRELIMINARY results following the evaluation plan (S3 TRAIN, grouped-CV out-of-fold)\n")
    w("> **Interim exhibit set for the 24 Nov 2026 course submission.** Every exhibit below follows the structure, order, "
      "metrics, matching and weighting of `manyifeval_s3_complete_evaluation_analysis_and_reporting_plan.md` (unchanged), "
      "but is computed on the **50 S3 TRAIN rule pools** with **grouped 5-fold out-of-fold (OOF) predictions** instead of "
      "the 22 S3 TEST pools with frozen refits. These are **not held-out TEST results**: the same folds selected settings, "
      "penalties and layers, so predictive numbers may be optimistic. Descriptive (observed-outcome) results are exploratory "
      "TRAIN findings. Final TEST results will replace every number here. Provenance: `README.md`.\n")
    w("**Global substitutions versus the plan.** 22 TEST pools → 50 TRAIN pools (paired cluster bootstrap, B = 10,000, seed "
      "20261203; Evaluation 1 Objective 2 uses B = 2,000). TEST design (456 configurations per task, 30,096 prompts) → TRAIN "
      "design (246 configurations per task, 36,900 prompts). Frozen TEST predictions → V2 S3 OOF predictions at the "
      "CV-selected setting (`results/v2_s3/predictions/`). ECE/calibration bins: deciles of the same OOF predictions "
      "(the plan fixes TEST bins from TRAIN OOF; on TRAIN they coincide). All intervals are paired, pointwise, unadjusted "
      "95% percentile intervals. Positive differences favour the candidate (right of the arrow).\n")

    # ------------------------------------------------------------------ Table 1
    w("## Table 1 — Inventory and controlled coverage (TRAIN, per generator)\n")
    rows = []
    for q in qc:
        rows.append([GEN[q["generator"]], q["pools"], q["tasks"], f"{q['prompts']:,}", f"{q['rows']:,}",
                     f"{q['prompts'] * 20:,}", f"{q['outcomes']:,}", f"{q['violations']:,}"])
    w(table(["Generator", "Rule pools", "Tasks", "Prompts", "Prompt–rule rows", "Responses (20/prompt, all present)",
             "Rule outcomes", "Violations"], rows))
    w("")
    if cov:
        w(table(["Matched design (metadata only; identical for both generators)", "Count"], [
            ["A Position: contexts / target position-pairs (k2–5)", f"{cov['position_contexts']:,} / {cov['position_pairs']:,} "
             f"(by k: {cov['position_pairs_by_k']})"],
            ["B1 Strict append: prompt pairs / retained-rule targets", f"{cov['append_prompt_pairs']:,} / {cov['append_targets']:,} "
             f"(targets by transition: {cov['append_targets_by_transition']})"],
            ["C2 Strict companion replacement: target contrasts", f"{cov['companion_targets']:,} (by k: {cov['companion_targets_by_k']}; "
             "**no strict replacements exist at k = 4 in the TRAIN design**)"],
            ["D Task: prompt pairs / exact-rule target contrasts", f"{cov['task_prompt_pairs']:,} / {cov['task_targets']:,}"]]))
        w("\nPlan's TEST-design counts for reference: position 57,420; append 69,300; companion 177,408; task 115,236.\n")

    # ------------------------------------------------------------------ Table 2
    con = r("eval1_contrasts")
    w("## Evaluation 1, Objective 1\n")
    w("### Table 2 — Rule-level violation-risk prediction (§4.4), TRAIN OOF\n")
    w("NLL reduction in nats per response × rule outcome, pooled over 153,900 TRAIN prompt–rule rows per generator; "
      "AUROC gain ×100 on response-level binary outcomes; types improved = strictly positive per-type NLL reduction "
      "(descriptive, not 15 significance tests). 95% CIs: paired 50-pool cluster bootstrap.\n")
    for gen in ("llama", "qwen"):
        w(f"**Panel {'A' if gen == 'llama' else 'B'} — {GEN[gen]}**\n")
        rows = []
        for cid in ORDER:
            x = con[(con.generator == gen) & (con.comparison == cid)].iloc[0]
            rows.append([bold(cid, CL[cid]), bold(cid, ci(x.nll_reduction, x.nll_reduction_lo, x.nll_reduction_hi, 1, 4)),
                         ci(x.auroc_gain, x.auroc_gain_lo, x.auroc_gain_hi, 100),
                         f"{int(x.types_improved)}/15" + (f" ({int(x.types_undefined)} undefined)" if x.types_undefined else "")])
        w(table(["Method comparison (comparator → candidate)", "NLL reduction [95% CI] ↑", "AUROC gain [95% CI] ↑",
                 "Types improved (of 15) ↑"], rows))
        w("")
    w("Sensitivity (A4): NLL reduction t(49) interval and leave-one-pool-out range.\n")
    rows = []
    for gen in ("llama", "qwen"):
        for cid in ORDER:
            x = con[(con.generator == gen) & (con.comparison == cid)].iloc[0]
            rows.append([gen, CL[cid], f"[{f(x.nll_t_lo, 4)}, {f(x.nll_t_hi, 4)}]",
                         f"{f(x.nll_loo_min, 4)} to {f(x.nll_loo_max, 4)}"])
    w(table(["Generator", "Comparison", "t(49) 95% interval (nats)", "Leave-one-pool-out range (nats)"], rows))
    w("")

    # A1
    a1 = r("eval1_a1_metrics")
    w("### Appendix A1 — Complete overall metrics (Objective 1)\n")
    rows = []
    for gen in ("llama", "qwen"):
        for p in ["REF_structure", "B1", "B2", "H1", "H2", "RCAP", "B1+H1", "B2+H1", "B2+A1", "B2+H1+A1"]:
            x = a1[(a1.generator == gen) & (a1.predictor == p)].iloc[0]
            rows.append([gen, PL[p], ci(x.nll, x.nll_lo, x.nll_hi, 1, 4), ci(x.brier, x.brier_lo, x.brier_hi, 1, 4),
                         ci(x.auroc, x.auroc_lo, x.auroc_hi, 100), ci(x.ece, x.ece_lo, x.ece_hi, 100, 2)])
    w(table(["Generator", "Predictor", "NLL", "Brier", "AUROC ×100", "ECE ×100"], rows))
    w("\nContrasts for all seven comparisons (Brier reduction):\n")
    rows = []
    for gen in ("llama", "qwen"):
        for cid in ORDER:
            x = con[(con.generator == gen) & (con.comparison == cid)].iloc[0]
            rows.append([gen, CL[cid], ci(x.brier_reduction, x.brier_reduction_lo, x.brier_reduction_hi, 1, 4)])
    w(table(["Generator", "Comparison", "Brier reduction [95% CI]"], rows))
    w("")
    # A2
    w("### Appendix A2 — Calibration (Objective 1)\n")
    w("![A2 reliability](figures/e1_A2_reliability.png)\n")
    w("Per-bin support, mean prediction and observed rate for every predictor: `tables/eval1_calibration.csv`. ECE for "
      "every predictor is in A1.\n")
    # A3
    w("### Appendix A3 — Per-rule-type NLL reduction heatmap (5 fixed above 10 parameterised types)\n")
    w("![A3 heatmap](figures/e1_A3_per_type_nll_heatmap.png)\n")
    pt = r("eval1_per_type")
    for gen in ("llama", "qwen"):
        w(f"**{GEN[gen]}** — NLL reduction (nats) by type (columns = comparisons)\n")
        rows = []
        for t in types_order(pt.rule_type.unique()):
            row = [tn(t) + (" (fixed)" if t in FIXED_TYPES else "")]
            for cid in ORDER:
                x = pt[(pt.generator == gen) & (pt.rule_type == t) & (pt.comparison == cid)].iloc[0]
                row.append(f(x.nll_reduction, 4))
            n = pt[(pt.generator == gen) & (pt.rule_type == t)].iloc[0]
            row.append(int(n.n_pools))
            rows.append(row)
        w(table(["Rule type"] + ORDER + ["Pools"], rows))
        w("")
    w("Supplementary CSV with all 210 rows (per-type NLLs, reductions, paired 95% CIs, pools, outcomes, flags): "
      "`tables/eval1_per_type.csv`.\n")
    # A3b
    fp = r("eval1_fixed_vs_param")
    w("### Appendix A3b — Fixed versus parameterised NLL improvement (Objective 1 only)\n")
    w("*NLL reductions (nats) of the same seven comparisons, calculated separately on rule-level outcomes from the five fixed "
      "and ten parameterised rule types. Fixed-minus-parameterised compares improvements, not raw violation rates. Paired "
      "pointwise 95% intervals from resampling the 50 TRAIN pools; all subgroup differences are exploratory.*\n")
    for gen in ("llama", "qwen"):
        x0 = fp[fp.generator == gen].iloc[0]
        w(f"**Panel {'A' if gen == 'llama' else 'B'} — {GEN[gen]}** (observed violation rate: fixed "
          f"{100 * x0.fixed_viol_rate:.1f}% of {int(x0.fixed_outcomes):,} outcomes; parameterised "
          f"{100 * x0.parameterized_viol_rate:.1f}% of {int(x0.parameterized_outcomes):,})\n")
        rows = []
        for cid in ORDER:
            x = fp[(fp.generator == gen) & (fp.comparison == cid)].iloc[0]
            rows.append([bold(cid, CL[cid]), ci(x.fixed_dnll, x.fixed_lo, x.fixed_hi, 1, 4),
                         ci(x.parameterized_dnll, x.parameterized_lo, x.parameterized_hi, 1, 4),
                         ci(x["diff_dnll"], x.diff_lo, x.diff_hi, 1, 4),
                         f"{f(x.fixed_typeeq_dnll, 4)} / {f(x.param_typeeq_dnll, 4)}"])
        w(table(["Comparator → candidate", "Fixed (5 types): ΔNLL ↑", "Parameterised (10 types): ΔNLL ↑",
                 "Fixed − parameterised", "Type-equal-weight sensitivity (fixed / param.)"], rows))
        w("")
    # A4
    w("### Appendix A4 — Technical uncertainty / QC\n")
    rows = [[q["generator"], q["rows"], q["outcomes"], q["pools"], q["prompts"],
             ", ".join(f"{k}: {v}" for k, v in q["clipped_rows"].items())] for q in qc]
    w(table(["Generator", "Rows", "Outcomes", "Pools", "Prompts", "Rows clipped to [1e-6, 1−1e-6] per predictor"], rows))
    w("\nEvery prompt has 20 verified responses (checked). Each OOF prediction's stored fold equals the fold of its "
      "pool in `results/cv_s3/folds.json` (checked for every row and predictor). Low-support per-type flags (< 8 pools): "
      f"{int(pt.low_support_flag.sum())} of {len(pt)} per-type rows. Sparse-head fallbacks fired during fitting are listed "
      "in `results/v2_s3/tables/fallbacks_fired.csv`.\n")

    # ------------------------------------------------------------------ Table 3 / A5-A7
    c2, s2, cal2 = r("obj2_eval1_contrasts"), r("obj2_eval1_scores"), r("obj2_eval1_calibration")
    w("## Evaluation 1, Objective 2\n")
    w("### Table 3 — Prompt-level expected violated-rule fraction (§4.6), TRAIN OOF\n")
    w("One observation per prompt (36,900 TRAIN prompts per generator). RMSE reduction in percentage points; Spearman gain "
      "×100. Paired 50-pool cluster bootstrap, B = 2,000.\n")
    for gen in ("llama", "qwen"):
        w(f"**Panel {'A' if gen == 'llama' else 'B'} — {GEN[gen]}**\n")
        rows = []
        for cid in ORDER:
            x = c2[(c2.generator == gen) & (c2.comparison == cid)].iloc[0]
            rows.append([bold(cid, CL[cid]), bold(cid, ci(x.rmse_reduction, x.rmse_reduction_lo, x.rmse_reduction_hi, 100, 2)),
                         ci(x.spearman_gain, x.spearman_gain_lo, x.spearman_gain_hi, 100)])
        w(table(["Method comparison (comparator → candidate)", "RMSE reduction [95% CI] (pp) ↑", "Spearman gain [95% CI] ↑"], rows))
        w("")
    w("### Appendix A5 — Complete overall metric scores (Objective 2)\n")
    rows = []
    for gen in ("llama", "qwen"):
        for p in ["B1", "B2", "H1", "H2", "RCAP", "B1+H1", "B2+H1", "B2+A1", "B2+H1+A1", "REF_structure"]:
            x = s2[(s2.generator == gen) & (s2.predictor == p)].iloc[0]
            rows.append([gen, PL[p], ci(x.rmse, x.rmse_lo, x.rmse_hi, 100, 2), ci(x.mae, x.mae_lo, x.mae_hi, 100, 2),
                         ci(x.mse, x.mse_lo, x.mse_hi, 1e4, 1), ci(x.pearson, x.pearson_lo, x.pearson_hi, 100),
                         ci(x.spearman, x.spearman_lo, x.spearman_hi, 100), ci(x.ece, x.ece_lo, x.ece_hi, 100, 2)])
    w(table(["Generator", "Predictor", "RMSE (pp)", "MAE (pp)", "MSE ×10⁴", "Pearson ×100", "Spearman ×100", "ECE (pp)"], rows))
    w("\nAll seven contrasts, secondary metrics:\n")
    rows = []
    for gen in ("llama", "qwen"):
        for cid in ORDER:
            x = c2[(c2.generator == gen) & (c2.comparison == cid)].iloc[0]
            rows.append([gen, CL[cid], ci(x.mae_reduction, x.mae_reduction_lo, x.mae_reduction_hi, 100, 2),
                         ci(x.mse_reduction, x.mse_reduction_lo, x.mse_reduction_hi, 1e4, 1),
                         ci(x.pearson_gain, x.pearson_gain_lo, x.pearson_gain_hi, 100)])
    w(table(["Generator", "Comparison", "MAE reduction (pp)", "MSE reduction ×10⁴", "Pearson gain ×100"], rows))
    w("")
    w("### Appendix A6 — Fraction calibration (Objective 2; key predictors B2, H2, A2)\n")
    rows = []
    for gen in ("llama", "qwen"):
        for p in ["B2", "H2", "RCAP"]:
            x = cal2[(cal2.generator == gen) & (cal2.predictor == p)]
            rows.append([gen, PL[p], " · ".join(f"{100 * a:.1f}→{100 * b:.1f}" for a, b in zip(x.mean_pred, x.mean_obs)),
                         int(x.n_prompts.min())])
    w(table(["Generator", "Predictor", "Decile bins: mean predicted → mean observed fraction (pp)", "Min prompts per bin"], rows))
    w("\nAll predictors: `tables/obj2_eval1_calibration.csv` (no separate figure was drawn tonight).\n")
    w("### Appendix A7 — Trivial and derived reference predictions\n")
    rows = []
    for gen in ("llama", "qwen"):
        for p, lab in [("REF_trainmean", "Training-fold mean (constant within fold)"), ("AVG1_H2", "Mean of Objective 1 H2 p̂"),
                       ("AVG1_B2", "Mean of Objective 1 B2 p̂"), ("AVG1_RCAP", "Mean of Objective 1 A2 p̂"), ("H2", "Objective 2 H2 head")]:
            x = s2[(s2.generator == gen) & (s2.predictor == p)].iloc[0]
            sp = "undefined within fold (constant)" if p == "REF_trainmean" else f(x.spearman * 100)
            rows.append([gen, lab, f(x.rmse * 100, 2), sp])
    w(table(["Generator", "Reference", "RMSE (pp)", "Spearman ×100"], rows))
    w("\nThe training-fold mean is constant within each fold; across the pooled OOF set it takes five values, so its pooled "
      "Spearman (negative) is an artefact and is not interpreted. Paired contrasts: `tables/obj2_eval1_contrasts.csv` rows "
      "refA (training mean → H2) and refB (mean of Objective 1 H2 → Objective 2 H2).\n")

    # ------------------------------------------------------------------ Evaluation 2
    w("## Evaluation 2, Objective 1 — controlled factors (observed effects first, then shared Table 4)\n")
    w("Noise-adjusted RMS = √max(0, weighted mean of (d² − V)), V the binomial variance of the observed difference; the "
      "estimator and its 50-pool interval were validated by simulation (`tables/noise_validation_synthetic.csv`: ≤0.1 pp "
      "bias when true RMS ≥ 3 pp, +0.5–1 pp upward bias at an exact null from truncation, 91–100% coverage). Weights: equal "
      "per k (or transition), then pool → task → context → pair, as in the plan's §6.3 proposal.\n")
    # E2-Pos
    pr = r("eval2_position_rms_by_type")
    w("### Figure E2-Pos — Position (§5.2)\n")
    w("![E2-Pos](figures/e2_pos_figure.png)\n")
    w("**Panel A — noise-adjusted RMS positional sensitivity over all position pairs, by target type (pp)**\n")
    w(rms_tbl(pr))
    st = r("eval2_position_signed_type_k")
    w("**Panel B — signed first→last change by type × k (pp; positive = last slot riskier)**\n")
    for gen in ("llama", "qwen"):
        w(f"*{GEN[gen]}*\n")
        rows = []
        for t in types_order(st.type_id.unique()):
            row = [tn(t)]
            for k in (2, 3, 4, 5):
                x = st[(st.generator == gen) & (st.type_id == t) & (st.k == k)]
                row.append(ci(x.effect_pp.iloc[0], x.lo.iloc[0], x.hi.iloc[0]) + f" ({int(x.n_pools.iloc[0])})" if len(x) else "unsupported")
            rows.append(row)
        w(table(["Target type", "k=2", "k=3", "k=4", "k=5"], rows))
        w("\n(pools in parentheses)\n")
    # E2-Append
    a = r("eval2_append_by_transition")
    w("### Figure E2-Append — Strict append (§5.3)\n")
    w("![E2-Append](figures/e2_append_figure.png)\n")
    rows = [[GEN[x.generator], x.transition, ci(x.effect_pp, x.lo, x.hi, 1, 2), int(x.n_pools), f"{int(x.n_prompt_pairs):,}",
             f"{int(x.n_targets):,}"] for x in a.itertuples()]
    w(table(["Generator", "Transition", "Mean retained-rule change (pp) [95% CI]", "Pools", "Prompt pairs", "Retained targets"], rows))
    w("")
    # E2-Companion
    cr = r("eval2_companion_rms_by_type")
    w("### Figure E2-Companion — Strict companion replacement (§5.5)\n")
    w("![E2-Companion](figures/e2_companion_figure.png)\n")
    w("**Panel A — noise-adjusted RMS by target type (pp), earlier and later companions, k = 2, 3, 5 (no TRAIN k = 4 pairs)**\n")
    w(rms_tbl(cr))
    sm = r("eval2_companion_signed_matrix")
    w(f"**Panel B — signed target × companion effects** (companion minus mean of the other eligible companions in the same "
      f"fixed context): {len(sm)} supported directed cells across both generators; unsupported cells are absent, not zero. "
      "Full numerical matrix with CIs, pools, tasks, contexts and mean alternatives: `tables/eval2_companion_signed_matrix.csv` "
      f"(Appendix C-A). Cells with ≥ 3 pools whose interval excludes zero: "
      f"{int(((sm.n_pools >= 3) & ((sm.lo > 0) | (sm.hi < 0))).sum())} (pointwise, unadjusted; exploratory).\n")
    # E2-Task
    tr = r("eval2_task_rms_by_type")
    fam = r("eval2_task_family")
    w("### Figure E2-Task — Task context (§5.8)\n")
    w("![E2-Task](figures/e2_task_figure.png)\n")
    w("**Panel A — noise-adjusted RMS task sensitivity by target type (pp)**\n")
    w(rms_tbl(tr))
    w("**Panel B — signed family-vs-other-two matched-task differences (pp; descriptive associations, not causal)**\n")
    rows = [[GEN[x.generator], x.family, ci(x.effect_pp, x.lo, x.hi, 1, 2), int(x.n_pools)] for x in fam.itertuples()]
    w(table(["Generator", "Task family", "Signed difference [95% CI]", "Supporting pools"], rows))
    w("")
    # Table 4
    w("### Table 4 — Evaluation 2 predictor skill: paired Δ-MSE reduction ×10⁴ [95% CI], all seven comparisons\n")
    sk = {fac: r(f"eval2_{fac}_skill") for fac in ("position", "append", "companion", "task")}
    for gen in ("llama", "qwen"):
        w(f"**{GEN[gen]}**\n")
        rows = []
        for cid in ORDER:
            row = [bold(cid, CL[cid])]
            for fac in sk:
                s = sk[fac]
                x = s[(s.generator == gen) & (s.stratum == "macro") & (s.quantity == f"dmse_{cid}")].iloc[0]
                row.append(ci(x.est, x.lo, x.hi, 1e4))
            rows.append(row)
        rows.append(["Zero-change reference MSE (×10⁴)"] + [
            f(sk[fac][(sk[fac].generator == gen) & (sk[fac].stratum == "macro") & (sk[fac].quantity == "se_zero")].est.iloc[0] * 1e4)
            for fac in sk])
        w(table(["Comparison", "Position (equal-k, k2–5)", "Strict append (4 transitions)", "Companion (k 2, 3, 5)",
                 "Task (k1–5)"], rows))
        w("")
    # Pos-A, Pos-B, Pos-C
    w("### Appendix Pos-A — Position profiles at every slot and all 20 pair strata\n")
    w("![Pos-A](figures/e2_posA_slot_curves.png)\n")
    ps = r("eval2_position_pair_strata")
    rows = [[x.generator, x.k, f"{x.i}→{x.j}", ci(x.effect_pp, x.lo, x.hi, 1, 2), int(x.n_pools), int(x.n_contexts)]
            for x in ps.itertuples()]
    w(table(["Generator", "k", "Slot pair", "Signed change (pp)", "Pools", "Contexts"], rows))
    w("\nSlot risk curves: `tables/eval2_position_slot_risk.csv`.\n")
    for fac, name, strata_label in (("position", "Pos-B", "k"), ("append", "K-B", "transition"),
                                     ("companion", "C-B", "k"), ("task", "D-C", "k")):
        w(f"### Appendix {name} — {fac} predictor skill by {strata_label} (Δ-MSE ×10⁴ [95% CI]; Δ-MAE ×10² point estimates)\n")
        s = sk[fac]
        strata = sorted(x for x in s.stratum.unique() if x != "macro")
        rows = []
        for gen in ("llama", "qwen"):
            for cid in ORDER:
                row = [gen, CL[cid]]
                for st_ in strata:
                    x = s[(s.generator == gen) & (s.stratum == st_) & (s.quantity == f"dmse_{cid}")].iloc[0]
                    m = s[(s.generator == gen) & (s.stratum == st_) & (s.quantity == f"dmae_{cid}")].iloc[0]
                    row.append(ci(x.est, x.lo, x.hi, 1e4) + f"; ΔMAE {f(m.est * 100, 2)}")
                rows.append(row)
            row = [gen, "zero-change MSE ×10⁴"]
            for st_ in strata:
                row.append(f(s[(s.generator == gen) & (s.stratum == st_) & (s.quantity == "se_zero")].est.iloc[0] * 1e4))
            rows.append(row)
        w(table(["Generator", "Comparison"] + [f"{strata_label} {x}" for x in strata], rows))
        w("")
    w("### Appendix Pos-C — Numerical RMS components, noise floors and support\n")
    w(rms_detail(pr))
    w("### Appendix K-A — Observed append effect by retained slot × transition\n")
    bs = r("eval2_append_by_slot")
    rows = [[x.generator, x.transition, x.slot, ci(x.effect_pp, x.lo, x.hi, 1, 2), int(x.n_pools), int(x.n_targets)]
            for x in bs.itertuples()]
    w(table(["Generator", "Transition", "Retained slot", "Change (pp) [95% CI]", "Pools", "Targets"], rows))
    w("\nCoverage note: every TRAIN chain prefix has an exact strict append for 1→2 and 2→3; 3→4 and 4→5 exist only for COA "
      "orders whose first k rules equal the shorter prompt's order (counts in Table 1).\n")
    w("### Appendix C-A — Companion observed effects, heterogeneity and coverage\n")
    w(rms_detail(cr))
    ko = r("eval2_companion_rms_k_order")
    rows = [[x.generator, x.k, x.relative_order, ci(x.rms_pp, x.lo, x.hi), int(x.n_comparisons), int(x.n_pools)] for x in ko.itertuples()]
    w(table(["Generator", "k", "Companion placement", "Noise-adj. RMS (pp)", "Comparisons", "Pools"], rows))
    w("")
    w("### Appendix C-C — Companion implementation checks\n")
    w("Strict replacement: same pool, task, k, target rule and slot, every other slot identical, exactly one companion "
      "differs; both earlier and later replacements included. TRAIN design yields replacements only at k = 2, 3 and 5. "
      "Pairs are oriented by prompt_id (unordered); RMS is orientation-free. Signed matrix: companion versus mean of the other "
      f"available companions in the same fixed context (mean alternatives per context: "
      f"{f(sm.mean_alternatives.mean(), 2)}).\n")
    w("### Appendix D-A — Task sensitivity by type × k (noise-adjusted RMS, pp)\n")
    tk = r("eval2_task_rms_type_k")
    rows = []
    for gen in ("llama", "qwen"):
        for t in types_order(tk.type_id.unique()):
            row = [gen, tn(t)]
            for k in (1, 2, 3, 4, 5):
                x = tk[(tk.generator == gen) & (tk.type_id == t) & (tk.k == k)]
                row.append(ci(x.rms_pp.iloc[0], x.lo.iloc[0], x.hi.iloc[0]) if len(x) and x.status.iloc[0] == "ok" else "unsupported")
            rows.append(row)
    w(table(["Generator", "Target type", "k=1", "k=2", "k=3", "k=4", "k=5"], rows))
    w("")
    w("### Appendix D-B — Rule type × task family signed effects (pp)\n")
    w("![D-B](figures/e2_taskDB_type_family_heatmap.png)\n")
    w("Numerical cells with CIs and pool support: `tables/eval2_task_type_family.csv`; supported family-pair contrasts "
      "(≥ 3 pools): `tables/eval2_task_family_pairs.csv`. Family is confounded with task text.\n")
    w("### Appendix D-D — Task matching and limitations\n")
    w("Each pool's 246 configurations appear with its three tasks (one per family); every target in every configuration "
      "gives three unordered task pairs. Families: `configs/task_families_v1.csv` (our proposed taxonomy). No within-family "
      "matched comparison exists.\n")
    w(rms_detail(tr))

    # Factor E
    w("### §5.9 Factor E — TRAIN-defined difficulty terciles (descriptive stratification)\n")
    try:
        from ..probing import data as pdata
        rows = []
        for gen in ("llama", "qwen"):
            Lb = pdata.labels_prompt_rule(gen, "train").merge(pdata.prompts("train")[["prompt_id", "k"]], on="prompt_id")
            Lb = Lb[Lb.k == 1]
            rate = (Lb.groupby("type_id").v.sum() / Lb.groupby("type_id").n.sum()).rename("rate").reset_index()
            rate = rate.sort_values(["rate", "type_id"]).reset_index(drop=True)
            rate["tercile"] = ["easy"] * 5 + ["medium"] * 5 + ["hard"] * 5
            for fac, df in (("position", pr), ("companion", cr), ("task", tr)):
                rate[fac] = rate.type_id.map(df[df.generator == gen].set_index("type_id").rms_pp)
            for x in rate.itertuples():
                rows.append([gen, tn(x.type_id), f"{100 * x.rate:.1f}", x.tercile, f(x.position), f(x.companion), f(x.task)])
        w(table(["Generator", "Type", "TRAIN k=1 violation rate (%)", "Tercile", "Position RMS", "Companion RMS", "Task RMS"], rows))
        w("\nOn TRAIN the terciles are defined on the same data they stratify (on TEST they would be defined out of sample).\n")
    except Exception as e:  # pragma: no cover
        w(f"Not computed: {e}\n")

    # Objective 2 Evaluation 2
    w("## Evaluation 2, Objective 2 — PRELIMINARY design (not decided in the plan, §8.2 \"remain undecided\")\n")
    e = r("obj2_eval2")
    sup = r("obj2_eval2_support")
    w("Prompt-level pairs from the same metadata manifests; outcome = whole-prompt fraction violated q̂. Position: every pair "
      "of COA orders of the same rule set and task. Append: k→k+1 strict prefix, new rule included (denominator changes). "
      "Companion: strict one-rule replacement. Task: same configuration, other task. Noise-adjusted RMS uses the "
      "response-level variance of the violated fraction / 20.\n")
    rows = []
    for gen in ("llama", "qwen"):
        for fac in ("position", "append", "companion", "task"):
            g = e[(e.generator == gen) & (e.factor == fac) & (e.stratum == "macro")].set_index("quantity")
            sp = sup[(sup.generator == gen) & (sup.factor == fac)].iloc[0]
            q = lambda n, s=1e4, d=1: ci(g.loc[n, "est"], g.loc[n, "lo"], g.loc[n, "hi"], s, d)
            rows.append([gen, fac, f"{int(sp.n_prompt_pairs):,}", q("rms_pp_noise_adjusted", 1), q("dobs", 100, 2),
                         f(g.loc["se_zero", "est"] * 1e4), q("dmse_#1"), q("dmse_#2"), q("dmse_#3"), q("dmse_#4")])
    w(table(["Generator", "Factor", "Prompt pairs", "Observed RMS (pp)", "Signed mean (pp)", "Zero-change MSE ×10⁴",
             "Δ-MSE #1", "Δ-MSE #2", "Δ-MSE #3", "Δ-MSE #4"], rows))
    w("\nSecondary comparisons and per-stratum values: `tables/obj2_eval2.csv`.\n")

    w("## Not available tonight / deviations\n")
    w("- Anything on the 22 S3 TEST pools (sealed by design); independent evaluation of frozen refits.\n"
      "- Companion strict replacements at k = 4 (absent from the TRAIN design); the TEST expansion adds them.\n"
      "- Objective 2 fraction-calibration figure (numbers in A6 only); Evaluation 2 leave-one-pool-out sensitivity "
      "(t/LOO columns exist in the `*_skill.csv` macro rows only).\n"
      "- Response-quality sensitivity (truncation, repetition, english_capital verifier) not rerun for this report.\n"
      "- Model-selection optimism: settings, penalties and layers were chosen on the same grouped folds.\n")
    OUT.write_text("\n".join(L) + "\n")
    print(OUT, len(L))


def rms_tbl(df):
    rows = []
    for t in types_order(df.type_id.unique()):
        row = [tn(t) + (" (fixed)" if t in FIXED_TYPES else "")]
        for gen in ("llama", "qwen"):
            x = df[(df.generator == gen) & (df.type_id == t)]
            if len(x) and x.status.iloc[0] == "ok":
                x = x.iloc[0]
                row.append(ci(x.rms_pp, x.rms_lo, x.rms_hi) + f" ({int(x.n_pools)})" + (" †" if x.truncated_at_zero else ""))
            else:
                row.append("unsupported")
        rows.append(row)
    return table(["Target type", "Llama RMS [95% CI] (pools)", "Qwen RMS [95% CI] (pools)"], rows) + \
        "\n\n† corrected squared estimate ≤ 0, truncated to 0.\n"


def rms_detail(df):
    rows = []
    for gen in ("llama", "qwen"):
        for t in types_order(df.type_id.unique()):
            x = df[(df.generator == gen) & (df.type_id == t)]
            if not len(x) or x.status.iloc[0] != "ok":
                continue
            x = x.iloc[0]
            rows.append([gen, tn(t), f(x.corrected_sq_x1e4), f(x.raw_sq_x1e4), f(x.noise_floor_x1e4),
                         f(100 * x.baseline_risk), int(x.n_pools), f"{int(x.n_comparisons):,}"])
    return table(["Generator", "Type", "Corrected sq. ×10⁴", "Raw sq. ×10⁴", "Noise floor ×10⁴", "Baseline risk (%)", "Pools",
                  "Comparisons"], rows) + "\n"


if __name__ == "__main__":
    main()
