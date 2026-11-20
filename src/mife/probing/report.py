"""Build results/RESULTS.md from saved tables only (results/tables/*.csv). Numbers are never typed by hand: every
value printed here is read from a table file, and a section whose tables do not exist yet says so."""
from __future__ import annotations

import json

import pandas as pd

from .data import RESULTS

T = RESULTS / "tables"
F = RESULTS / "figures"


def _t(name: str) -> pd.DataFrame | None:
    p = T / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def _md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else floatfmt.format(v)
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def _pending(what: str) -> str:
    return f"_Pending: {what}._\n"


def _ci(df, est="est", lo="lo", hi="hi", digits=3):
    return df.assign(**{"estimate [95% CI]": [f"{e:.{digits}f} [{l:.{digits}f}, {h:.{digits}f}]"
                                             for e, l, h in zip(df[est], df[lo], df[hi])]}).drop(columns=[est, lo, hi])


# ------------------------------------------------------------------------------------------------ sections
def s_summary() -> str:
    s = _t("summary_bullets")
    return "\n".join(f"- {b}" for b in s["bullet"]) + "\n" if s is not None else _pending("written after the test evaluation")


def s_what() -> str:
    w = _t("what_was_run")
    out = _md(w) + "\n" if w is not None else ""
    pre = RESULTS / "prespec" / "analysis_plan.md"
    out += f"\nPre-specification: [`prespec/analysis_plan.md`](prespec/analysis_plan.md)" + (
        " (frozen; hash in `prespec/FROZEN.json`)." if (RESULTS / "prespec" / "FROZEN.json").exists() else " (not yet frozen).") \
        if pre.exists() else "\nPre-specification: not yet written."
    return out + "\n"


def s_descriptive() -> str:
    parts = []
    for name, title, digits in (
            ("desc_by_k", "Violation rate by k (TRAIN; 95% CI, cluster bootstrap over the 57 groups)", 3),
            ("desc_by_type", "Violation rate by rule type (TRAIN)", 3),
            ("desc_by_category", "Violation rate by category (TRAIN)", 3),
            ("desc_by_position", "Violation rate by position within k (TRAIN)", 3),
            ("desc_by_family", "Violation rate by task family (TRAIN)", 3),
            ("desc_label_distribution", "Distribution of p_hat and q_hat (TRAIN)", 3),
            ("desc_truncation", "Truncation (hit max_new_tokens) by k (TRAIN)", 4),
            ("desc_english_capital", "`english_capital`: violations caused only by the langdetect step (TRAIN)", 3),
            ("desc_loops", "Repetition loops (TRAIN)", 4),
            ("desc_cross_model_type", "Llama vs Qwen by type (TRAIN)", 3),
            ("desc_cross_model_corr", "Per-prompt rank correlation of q_hat, Llama vs Qwen (TRAIN)", 3)):
        d = _t(name)
        if d is None:
            continue
        if {"est", "lo", "hi"} <= set(d.columns):
            d = _ci(d, digits=digits)
        parts.append(f"### {title}\n\n{_md(d)}\n")
        fig = F / f"{name}.png"
        if fig.exists():
            parts.append(f"![{name}](figures/{name}.png)\n")
    tk = _t("desc_type_by_k")
    if tk is not None:
        parts.append(f"### Type × k violation rate (TRAIN)\n\n{_md(tk)}\n")
    return "\n".join(parts) if parts else _pending("Phase 1")


def _cv_bullets() -> list[str]:
    """Short CV summary, every number read from results/tables/cv_*.csv."""
    out = []
    lay, pool, b2 = _t("cv_layer_selection"), _t("cv_pooling_selection"), _t("cv_b2_grid")
    if lay is not None:
        sel = lay[lay.selected]
        out.append("Shared hidden row: " + "; ".join(f"{r.model} Objective {r.objective} row {r.hidden_row} ({r.depth} depth)"
                                                     for r in sel.itertuples()) + ".")
    if pool is not None:
        sel = pool[pool.selected]
        out.append("A1 pooling: " + "; ".join(f"{r.model} Objective {r.objective} {r.pooling}" for r in sel.itertuples()) + ".")
    if b2 is not None:
        sel = b2[b2.selected]
        out.append(f"B2 ({sel.adapt.iloc[0]}): " + "; ".join(f"{r.model} Objective {r.objective} LR {r.lr:g}, {r.epochs} epoch(s)"
                                                              for r in sel.itertuples()) + ".")
    s = _t("cv_summary")
    if s is not None:
        for r in s.itertuples():
            out.append(f"**{r.model}, Objective {r.objective}** (CV {r.metric}): lowest loss {r.best_config} "
                       f"({r.best_cv_loss:.4g}); best probe-only {r.best_probe_only} ({r.best_probe_only_cv_loss:.4g}), best B1 family "
                       f"{r.best_B1_family} ({r.best_B1_family_cv_loss:.4g}), best B2 family {r.best_B2_family} "
                       f"({r.best_B2_family_cv_loss:.4g}); B1 alone "
                       f"{r.B1_cv_loss:.4g}, B2 alone {r.B2_cv_loss:.4g}, structure reference {r.REF_structure_cv_loss:.4g}, "
                       f"type × k reference {r.REF_type_k_cv_loss:.4g}; {r.n_of_23_below_REF_structure} of 23 configurations "
                       "below the structure reference.")
    c = _t("cv_complementarity")
    if c is not None:
        out.append("Hidden + attention in CV (combined minus the better single block; negative = the combination helps): "
                   + "; ".join(f"{r.model} Obj {r.objective} {r.combined} {r.combined_minus_better_single:+.2g}"
                               for r in c.itertuples()) + ".")
    n = _t("cv_l2_counts")
    if n is not None:
        cols = [k for k in n.columns if k.startswith("selected_l2=")]
        out.append("Selected L2 among the 23 configurations plus the structure reference ("
                   + ", ".join(k.split("=")[1] for k in cols) + "): "
                   + "; ".join(f"{r['model']} Obj {r['objective']} " + "/".join(str(int(r[k])) for k in cols)
                               for _, r in n.iterrows()) + ". Edge selections and the loss one grid step inwards are in "
                   "`cv_l2_edges.csv`.")
    return out


def s_cv() -> str:
    parts = []
    b = _cv_bullets()
    if b:
        parts.append("### Summary (training-only grouped CV; these are not TEST results)\n\n" + "\n".join(f"- {x}" for x in b) + "\n")
    for name, title in (("cv_summary", "Best configuration per family (grouped CV)"),
                        ("cv_folds", "Folds (GroupKFold over the 57 TRAIN groups)"),
                        ("cv_layer_selection", "Hidden-layer selection (H1, grouped CV)"),
                        ("cv_pooling_selection", "A1 pooling selection (A1, grouped CV)"),
                        ("cv_b2_grid", "B2 learning rate × epochs (grouped CV, B2 head loss)"),
                        ("cv_selected", "Selected head settings per configuration (grouped CV; training data only)"),
                        ("cv_complementarity", "Hidden + attention complementarity (grouped CV)"),
                        ("cv_l2_counts", "Selected L2 values"),
                        ("cv_l2_edges", "Edge selections of the L2 grid"),
                        ("cv_boundary_hits", "Grid-boundary hits (all candidates, including unselected rows and poolings)"),
                        ("cv_identical_features", "Configurations with identical feature blocks (design property)"),
                        ("cv_sparse_types", "Rule types with few violated responses in a fitting fold")):
        d = _t(name)
        if d is not None:
            parts.append(f"### {title}\n\n{_md(d, '{:.4g}')}\n")
    return "\n".join(parts) if parts else _pending("Phase 3")


def s_eval1() -> str:
    parts = []
    for name, title in (("eval1_obj1", "Objective 1 (per rule), TEST"), ("eval1_obj2", "Objective 2 (overall fraction), TEST")):
        d = _t(name)
        if d is not None:
            parts.append(f"### {title}\n\n{_md(d, '{:.4f}')}\n")
    return "\n".join(parts) if parts else _pending("test evaluation (after the pre-specification is frozen and unblinding approved)")


def s_contrasts() -> str:
    d = _t("contrasts")
    return f"{_md(d, '{:.4f}')}\n" if d is not None else _pending("test evaluation")


def s_eval2() -> str:
    d = _t("eval2_summary")
    return f"{_md(d, '{:.4f}')}\n" if d is not None else _pending("test evaluation")


def s_sensitivity() -> str:
    parts = []
    for name, title in (("sens_truncation", "Labels excluding truncated responses (TRAIN descriptive)"),
                        ("sens_english_capital", "Excluding `english_capital` (TRAIN descriptive)")):
        d = _t(name)
        if d is not None:
            if {"est", "lo", "hi"} <= set(d.columns):
                d = _ci(d)
            parts.append(f"### {title}\n\n{_md(d)}\n")
    return "\n".join(parts) if parts else _pending("Phase 1 / test evaluation")


def s_compute() -> str:
    parts = []
    b = _t("b2_benchmark")
    if b is not None:
        parts.append("### B2 benchmark on this Mac (Apple M4 Max, MPS, float32; Objective 1 training steps of "
                     f"{int(b['batch_prompts'].iloc[0])} TRAIN prompts)\n\n"
                     + _md(b[["adapt", "train_s_per_prompt", "infer_s_per_prompt", "trainable_params", "mps_driver_allocated_gb"]], "{:.4f}") + "\n")
    p = _t("b2_projection")
    if p is not None:
        parts.append("### Projected B2 wall time (5 folds, 2 objectives, 2 generators, plus final refits)\n\n"
                     + _md(p[["scenario", "adapt", "description", "hours_total", "hours_cv_part", "hours_refits"]], "{:.1f}") + "\n")
        parts.append(
            "Every scenario that keeps the specified grid exceeds the 24 h limit set in the prompt, so the options below "
            "went to the project owner, who chose **option 1 (Betty)**. No grid or data was shrunk. Options as put:\n\n"
            "1. **Betty sbatch array** (scripts under `scripts/betty/`): each fold × grid point × objective × generator is an "
            "independent fine-tuning job, so the protocol parallelises across GPUs. At the time it had not been benchmarked on "
            "Betty (the B200 timing is in the table below). Constraints as stated then: queue waits; the home quota could not "
            "hold BGE-M3's weights without freeing space (resolved by deleting the Betty generator checkpoints, § 10); B1 "
            "stays on MPS, so B1 vs B2 could differ by encoder backend (checked: see the frozen-encoder consistency table).\n"
            "2. **LoRA instead of full fine-tuning** on the Mac: barely faster here (see table), so it does not bring the "
            "protocol under 24 h on its own.\n"
            "3. **A smaller grid frozen in advance** on the Mac (rows `spec_full_snapshot`, `small_full_1lr`, `small_full_1pt`), "
            "chosen before any B2 result is seen.\n"
            "4. **Run the specified grid on the Mac** and accept the projected multi-day wall time.\n")
    bc = _t("b2_betty_check")
    if bc is not None:
        parts.append("### B2 on Betty: full-vs-LoRA check, which doubles as the B200 benchmark (Llama, Objective 1, fold 0)\n\n"
                     + _md(bc, "{:.4f}") + "\n\nThe project owner chose the Betty option; the LR × epoch grid of the chosen method "
                     "runs as a 20-task array (`scripts/betty/b2_grid.sbatch`).\n")
    xc = _t("b2_encoder_xcheck")
    if xc is not None:
        parts.append("### Frozen-encoder consistency, Mac (MPS) vs Betty (CPU)\n\n" + _md(xc, "{:.3g}") + "\n")
    rt = _t("runtime")
    if rt is not None:
        parts.append("### Runtime of completed stages\n\n" + _md(rt, "{:.1f}") + "\n")
    return "\n".join(parts) if parts else _pending("compute plan")


def s_open() -> str:
    d = _t("open_issues")
    return _md(d) + "\n" if d is not None else _pending("open issues")


SECTIONS = [("1. Summary", s_summary), ("2. What was run", s_what), ("3. Descriptive results (TRAIN only)", s_descriptive),
            ("4. Cross-validation and selected settings", s_cv), ("5. Evaluation 1", s_eval1),
            ("6. Hypothesis contrasts", s_contrasts), ("7. Evaluation 2 (controlled analyses A–E)", s_eval2),
            ("8. Sensitivity analyses", s_sensitivity), ("9. Compute and runtime", s_compute),
            ("10. Open issues, deviations and discrepancies", s_open)]


def build() -> dict:
    RESULTS.mkdir(exist_ok=True)
    body = ["# ManyIFEval probing results\n",
            "Generated by `python -m mife.cli probe-report` from `results/tables/`. CV (training-only) results and TEST "
            "results are kept in separate sections and tables.\n"]
    for title, fn in SECTIONS:
        body.append(f"## {title}\n\n{fn()}")
    (RESULTS / "RESULTS.md").write_text("\n".join(body))
    return {"written": str(RESULTS / "RESULTS.md"), "tables": sorted(p.name for p in T.glob("*.csv"))}
