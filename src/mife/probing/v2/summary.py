"""results/v2/SUMMARY.md and its tables, built only from saved V2 artifacts (training-only milestone).

Every number in SUMMARY.md comes from a file under results/v2/ (selection JSONs, extraction checks, tables).
Training-CV losses are the selection criterion itself: they are reported to document the selection and are NOT
held-out estimates of performance.
"""
from __future__ import annotations

import json

import pandas as pd

from .versioning import FEAT_V2, RESULTS_V2, SPLIT, read_versioned_json

# Priority split decided by the project owner on 2026-11-22 (amendment section 9): primary = 1, 2, 3, 7.
COMPARISONS = [("Primary 1", "H1", "B1"), ("Primary 2", "H2", "B2"), ("Primary 3", "RCAP", "B2"),
               ("Primary 7", "H2", "RCAP"),
               ("Secondary 4", "B1+H1", "B1"), ("Secondary 5", "B2+H1", "B2"), ("Secondary 6", "B2+A1", "B2")]
LABEL = {"H1": "H1 + structure", "H2": "H2 + structure", "RCAP": "Rule-conditioned attention probe",
         "B1": "B1", "B2": "B2", "B1+H1": "B1 + H1", "B2+H1": "B2 + H1", "B2+A1": "B2 + A1",
         "REF_structure": "Structure-only reference", "XH1": "Other generator's H1 + structure (reference)"}
MODELS = ("llama", "qwen")


def _cell(x):
    if isinstance(x, float):
        return f"{x:.6g}" if abs(x) >= 1e-3 or x == 0 else f"{x:.3e}"
    return "" if x is None else str(x).replace("|", "\\|")


def _md(df: pd.DataFrame) -> str:
    """Markdown table without optional dependencies."""
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def build() -> str:
    from ... import splits
    from .. import data
    N_TRAIN = data.prompts("train")["group_id"].nunique()
    N_TEST = len(splits.test_groups())
    T = RESULTS_V2 / "tables"
    T.mkdir(parents=True, exist_ok=True)
    lines = ["# V2 training-only milestone: summary", "",
             f"Split: `{SPLIT}`. Training-only. The {N_TEST} TEST groups were not read: no TEST features, labels or scores. "
             "Every loss below is a "
             f"**training-CV** value (5 grouped folds over the {N_TRAIN} TRAIN groups). These values were used to select settings, "
             "so they document the selection and are not held-out performance estimates. "
             "Amendment: `docs/amendment_v2_2026-11-20.md`; recipe: `configs/v2_recipe.yaml` "
             f"(recorded in `results/{RESULTS_V2.name}/recipe_recorded.json`).", ""]
    rec = read_versioned_json(RESULTS_V2 / "recipe_recorded.json")
    lines += [f"Recipe sha256 `{rec['sha256'][:16]}…`, recorded {rec['recorded_at']}, before any selection reported here.", "",
              "## What changed from V1", "",
              "- **Predictors:** 7 planned comparisons (primary: 1, 2, 3 and 7; secondary: 4, 5 and 6) plus references, "
              "instead of 23 configurations. "
              "B2 is now B1 plus a *frozen* exact-rule embedding (no fine-tuning). A1 is the per-layer×head **max** "
              "attention over the target-rule span (new training-only extraction). The rule-conditioned attention probe "
              "is A1 + exact-rule embedding + structure.",
              "- **Architecture:** every Objective 1 predictor uses 15 per-type ridge-logistic heads with the same structure "
              "block. Every Objective 2 predictor uses one shared ridge fractional-logit head on the existing q_hat target. "
              "There are no MLP heads.",
              "- **Fitting:** summed binomial NLL (Objective 1) or fractional logit (Objective 2), with penalty λ/N. There are "
              "three penalty slots (fixed structure, tuned prompt representation, tuned rule embedding) on identical grids, a "
              "deterministic tie rule, a one-time ×10 boundary expansion, a block-off option for the incremental comparisons, "
              "and a sparse-head fallback that also uses group coverage.",
              "- **Layer search:** widened to 5 rows per generator, selected by H1 + structure, separately per objective.",
              "- **Naming (B1 / B2):** in V2, B1 = S + P (structure + frozen BGE-M3 prompt embedding) and B2 = S + P + R "
              "(B1 + frozen BGE-M3 exact-rule embedding). In V1 (`results/RESULTS.md`, `results/tables/`, `results/cv/`, "
              "`docs/probing_schema.md`), **B2 means the LoRA-adapted BGE-M3 baseline**, which V2 retired. The two are "
              "different models and are never compared or mixed.",
              "- **Kept:** the 2026-11-14 freeze, every V1 file, the folds, labels and frozen features. Everything V2 is "
              "versioned (`v2-2026-11-20`) under `data/probing_features_v2/` and `results/v2/`.", ""]

    # ---- extraction
    lines += ["## 1. Max-attention extraction (TRAIN prompts only)", ""]
    ext = []
    for m in MODELS:
        meta = json.loads((FEAT_V2 / m / "maxattn" / "maxattn_meta.json").read_text())
        wall = sum(s["wall_s"] for s in meta["sessions"])
        idx = sorted((FEAT_V2 / m / "maxattn").glob("index_[0-9]*.parquet"))
        mt = [f.stat().st_mtime for f in idx]
        per_shard = (mt[-1] - mt[0]) / max(len(mt) - 1, 1)
        span_h = (mt[-1] - mt[0] + per_shard) / 3600  # first shard start (approx.) to last commit
        row = {"model": m, "status": meta["status"], "n_prompts": meta["n_prompts"], "n_train_prompts": meta["n_train_prompts"],
               "wall_h_recorded_sessions": round(wall / 3600, 2), "wall_h_shard_span": round(span_h, 2), "storage_gb": round(meta["storage_bytes"] / 1e9, 3),
               "backend": f"{meta['backend']}, {meta['attn_implementation']}, {meta['dtype']}",
               "torch": meta["torch_version"], "transformers": meta["transformers_version"]}
        for tag in ("pre", "post"):
            p = RESULTS_V2 / "extraction" / f"{m}_checks_{tag}.json"
            if p.exists():
                c = read_versioned_json(p)
                row[f"{tag}_prompts/spans"] = f"{c['n_prompts']}/{c['n_spans']}"
                row[f"{tag}_alignment_ok"] = c["alignment_all_ok"]
                row[f"{tag}_max_equals_direct"] = c["max_equals_direct_all"]
                if tag == "post":
                    row["post_stored_equals_direct"] = c["stored_equals_direct_all"]
                row[f"{tag}_sum_max_abs_diff_vs_v1"] = c["sum_max_abs_diff_vs_v1"]
                row["sum_tolerance"] = c["sum_tolerance_abs"]
                row["straddling_spans_whitespace"] = c.get("n_straddling_spans_whitespace")
        ext.append(row)
    E = pd.DataFrame(ext)
    E.to_csv(T / "extraction_summary.csv", index=False)
    lines += [_md(E), "",
              "Checks: (1) each indexed span, rebuilt from character offsets, equals the frozen span and contains the exact "
              "rule text. The frozen tokenization flags no TRAIN span with a boundary token covering NON-whitespace text "
              "outside the rule; every sampled span has a boundary token that covers whitespace outside it. (2) For every "
              "layer and head, the extracted max equals the direct max over the span from the full attention row (exact "
              "equality); after the run, the stored value equals it too. (3) The recomputed span SUM reproduces the cached "
              "V1 sum within the documented absolute tolerance.", "",
              "Wall time: `wall_h_recorded_sessions` sums the sessions written to the metadata. The Qwen run was restarted "
              "once, and the killed session wrote no record, so `wall_h_shard_span` (from the shard commit times) is the "
              "elapsed time. Both runs shared the Mac with CV jobs and other applications under memory pressure; the "
              "unloaded rate was about 0.25–0.30 s per prompt.", ""]

    # ---- selections
    sel_rows, grid_notes, fb_rows, layer_rows = [], [], [], []
    for m in MODELS:
        for o in (1, 2):
            p = RESULTS_V2 / "cv" / m / f"obj{o}" / "selection.json"
            if not p.exists():
                continue
            S = read_versioned_json(p)
            for name, s in S["selected"].items():
                sel_rows.append({"model": m, "objective": o, "predictor": name, "label": LABEL.get(name, name),
                                 "selected_setting": s["setting"], "frozen_baseline": json.dumps(s["fixed"]) if s["fixed"] else "",
                                 "hidden_row": s["hidden_row"], "n_features": s["n_features"], "cv_loss": s["cv_loss"],
                                 "fold_losses": json.dumps([round(x, 6) for x in s["fold_losses"]]), "note": s["note"] or ""})
            grid_notes.append({"model": m, "objective": o, "final_grids": json.dumps(S["final_grids"]),
                               "expansions": json.dumps(S["expansions"]),
                               "base_round_edge_hits": json.dumps(S["boundary_hits_base_round"]),
                               "edges_after_expansion": json.dumps(S["edges_after_expansion"])})
            for r, v in S["hidden_row_scores"].items():
                layer_rows.append({"model": m, "objective": o, "which": "own (H1 + structure)", "row": int(r),
                                   "meaning": S["row_convention"][r]["meaning"], "depth": S["row_convention"][r]["depth_fraction"],
                                   "cv_loss_best_setting": v, "selected": int(r) == S["hidden_row"]})
            for r, v in S["other_generator_row_scores"].items():
                layer_rows.append({"model": m, "objective": o, "which": "other generator (reference)", "row": int(r),
                                   "meaning": S["other_row_convention"][r]["meaning"],
                                   "depth": S["other_row_convention"][r]["depth_fraction"],
                                   "cv_loss_best_setting": v, "selected": int(r) == S["other_generator_row"]})
            for f in S.get("fallbacks_fired", []):
                fb_rows.append(f)
    SEL, G, LY = pd.DataFrame(sel_rows), pd.DataFrame(grid_notes), pd.DataFrame(layer_rows)
    SEL.to_csv(T / "selected_settings.csv", index=False)
    G.to_csv(T / "grids_and_expansions.csv", index=False)
    LY.to_csv(T / "layer_selection.csv", index=False)
    FB = pd.DataFrame(fb_rows)
    FB.to_csv(T / "fallbacks_fired.csv", index=False)

    lines += ["## 2. Hidden-layer selection (H1 + structure, training CV)", "",
              "Row convention: 0 = embeddings; r < L = output of decoder layer r; L = layer L after the final norm; L+1 = "
              "layer L before the final norm (Llama L = 32, Qwen L = 36). Disclosure: hidden-state methods get this 5-row "
              "search; the text and attention methods do not.", "",
              _md(LY) if len(LY) else "(not run)", ""]
    lines += ["## 3. Grids, boundary expansions and edges", "", _md(G) if len(G) else "(not run)", "",
              "Penalties are on the summed-loss scale. An expansion adds one ×10 step, once. A setting still on an edge after "
              "expansion is reported as best among the tested values.", ""]
    lines += ["## 4. Selected settings per generator, objective and predictor", "",
              _md(SEL.drop(columns=["fold_losses"])) if len(SEL) else "(not run)", "",
              "Objective 1 loss is the pooled mean log loss per completion; Objective 2 loss is the pooled squared error of "
              "the predicted fraction. `P` = prompt-representation penalty, `R` = rule-embedding penalty, `X` = added "
              "block in comparisons 4–6 (\"off\" = CV did not select the block, so the gain is 0 by construction; this is "
              "not evidence that the block carries no information).", ""]
    lines += ["## 5. Sparse-head fallbacks that fired (Objective 1)", "",
              _md(FB[["model", "fold", "type_id", "violated", "compliant", "groups_with_violation", "reason"]])
              if len(FB) else "None fired.", "",
              "The decision depends only on the fitting rows' labels, so a fallback type contributes the same intercept-only "
              "prediction (v + 0.5) / (n + 1) to every method.", ""]

    # ---- comparisons on training CV (descriptive)
    comp = []
    for m in MODELS:
        for o in (1, 2):
            sub = SEL[(SEL.model == m) & (SEL.objective == o)].set_index("predictor") if len(SEL) else None
            if sub is None or not len(sub):
                continue
            for pr, left, right in COMPARISONS:
                if left in sub.index and right in sub.index:
                    comp.append({"model": m, "objective": o, "comparison": pr, "left": LABEL[left], "right": LABEL[right],
                                 "cv_left": sub.loc[left, "cv_loss"], "cv_right": sub.loc[right, "cv_loss"],
                                 "right_minus_left": sub.loc[right, "cv_loss"] - sub.loc[left, "cv_loss"]})
    C = pd.DataFrame(comp)
    C.to_csv(T / "training_cv_comparisons.csv", index=False)
    lines += ["## 6. The seven comparisons on training CV (descriptive only)", "",
              "Positive `right_minus_left` = lower training-CV loss for the left-hand predictor. These are the selection "
              "criteria themselves, with no interval and no held-out data. The planned test evaluation (Part 1 and 2 "
              "reporting plans) has not been run.", "", _md(C) if len(C) else "(not run)", ""]

    # ---- Objective 2 extras
    extra = []
    for m in MODELS:
        p = RESULTS_V2 / "cv" / m / "obj2" / "selection_ordered.json"
        if p.exists():
            S = read_versioned_json(p)
            for name, s in S["selected"].items():
                extra.append({"model": m, "analysis": "ordered-slot sensitivity", "predictor": name, "setting": s["setting"],
                              "cv_mse": s["cv_loss"], "n_features": s["n_features"], "note": s["note"] or ""})
        p = RESULTS_V2 / "cv" / m / "obj2" / "avg_obj1_reference.json"
        if p.exists():
            for name, r in read_versioned_json(p)["results"].items():
                extra.append({"model": m, "analysis": "averaged Objective 1 OOF predictions", "predictor": name,
                              "setting": "Objective 1 selection", "cv_mse": r["cv_mse"], "n_features": None, "note": ""})
    X = pd.DataFrame(extra)
    X.to_csv(T / "obj2_sensitivity_and_avg_obj1.csv", index=False)
    lines += ["## 7. Objective 2: ordered-slot sensitivity and averaged Objective 1 predictions", "",
              _md(X) if len(X) else "(not run)", ""]
    lines += ["## 8. Where the outputs are", "",
              f"- OOF predictions (every tested setting plus `selected`): `results/{RESULTS_V2.name}/predictions/<model>/obj<o>/<predictor>.parquet`; "
              "Objective 2 ordered-slot: `.../obj2/ordered/`; averaged Objective 1: `.../obj2/avg_obj1/`.",
              f"- Selections: `results/{RESULTS_V2.name}/cv/<model>/obj<o>/selection.json` (+ `selection_ordered.json`, `avg_obj1_reference.json`).",
              f"- Refitted heads (all {N_TRAIN} TRAIN groups): `{RESULTS_V2.relative_to(RESULTS_V2.parents[1])}/models/<model>/obj<o>/<predictor>.npz/.json`.",
              f"- Tables: `results/{RESULTS_V2.name}/tables/`. Caches: `data/probing_features_v2/` (split-dependent ones under `split_s3/` for S3).",
              "- Parquet, NumPy and npz files are git-ignored and stay on the Mac; JSON, CSV and Markdown are committed.", ""]
    md = "\n".join(lines)
    (RESULTS_V2 / "SUMMARY.md").write_text(md)
    return md
