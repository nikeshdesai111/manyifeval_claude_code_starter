"""Phase 4: write results/prespec/analysis_plan.md from the training-only CV outputs and the fixed definitions, then
freeze it (results/prespec/FROZEN.json: SHA-256 of the plan, source-tree hash, data hashes). Unblinding is a separate,
explicit step (`approve_unblinding`) that only the project owner's "go" may trigger."""
from __future__ import annotations

import datetime as dt
import hashlib
import json

import pandas as pd

from ..config import DATA_DIR, ROOT, load_config
from ..storage import write_json
from . import configs as C
from . import controlled, data, evaluate
from .cv import CV_DIR
from .data import MODELS, RESULTS

PRE = RESULTS / "prespec"


def source_tree_hash() -> str:
    h = hashlib.sha256()
    for d in ("src", "configs", "scripts", "tests"):
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc",):
                h.update(str(p.relative_to(ROOT)).encode())
                h.update(p.read_bytes())
    return h.hexdigest()


def file_sha(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join("" if (isinstance(v, float) and pd.isna(v)) else (f"{v:.5g}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def write_plan() -> str:
    pc = load_config()["probing"]
    sel = pd.read_csv(RESULTS / "tables" / "cv_selected.csv").astype({"epoch": "Int64", "n_features": "Int64"})
    lay = pd.read_csv(RESULTS / "tables" / "cv_layer_selection.csv")
    pool = pd.read_csv(RESULTS / "tables" / "cv_pooling_selection.csv")
    b2p = RESULTS / "tables" / "cv_b2_grid.csv"
    P, PR = data.prompts("train"), data.prompt_rules("train")
    counts = controlled.pair_counts(P, PR)[["analysis", "objective", "pairs_per_group"]]
    counts["test_pairs (15 groups)"] = (counts["pairs_per_group"] * 15).astype(int)
    terc = {m: controlled.type_terciles(data.labels_prompt_rule(m, "train"), P).assign(model=m) for m in MODELS}
    terc_df = pd.concat(terc.values())[["model", "type_id", "train_k1_rate", "tercile"]]
    terc_df.to_csv(RESULTS / "tables" / "eval2_E_terciles.csv", index=False)
    s = []
    s.append("# Pre-specified analysis plan (frozen before the TEST set is read)\n")
    s.append(f"Written {dt.datetime.now().isoformat(timespec='seconds')}. Every choice below comes from TRAINING data only "
             "(5-fold GroupKFold over the 57 TRAIN groups, `results/cv/folds.json`). The 15 TEST groups (11,070 prompts per "
             "generator) have not been read: no labels, outputs, descriptive statistics or feature scaling.\n")
    s.append("## 1. Selected settings\n")
    s.append("### Shared hidden row and attention pooling (per generator and objective)\n")
    s.append(_md(lay[lay.selected][["model", "objective", "hidden_row", "depth", "best_cv_loss"]]) + "\n")
    s.append(_md(pool[pool.selected][["model", "objective", "pooling", "best_cv_loss"]]) + "\n")
    if b2p.exists():
        b2 = pd.read_csv(b2p)
        meth = json.loads((DATA_DIR / "probing_features" / "b2_runs" / "method.json").read_text())
        s.append(f"### B2 adaptation\n\nFull vs LoRA check: {meth['rule']}. Chosen: **{meth['method']}**.\n")
        s.append(_md(b2[b2.selected][["model", "objective", "adapt", "lr", "epochs", "cv_loss"]]) + "\n")
    s.append("### Head setting per configuration (selected by mean held-out-fold loss; MLP epoch chosen inside CV)\n")
    s.append(_md(sel[["model", "objective", "family", "config", "key", "setting", "epoch", "cv_loss", "n_features", "per_type_heads"]]) + "\n")
    s.append("## 2. Models and fitting\n")
    s.append("- Objective 1 heads: one per instruction TYPE (15) for every configuration containing H1, H2 or A1; one shared "
             "rule-conditioned head for B1-only, B2-only and the structure reference. Objective 2: one shared head per configuration.\n"
             "- Loss: Objective 1 aggregated binomial NLL with n = 20 (mean per completion); Objective 2 sigmoid-bounded MSE.\n"
             f"- Penalty: (l2/2)·||W||² on weights after fold-wise standardisation; grid l2 ∈ {pc['heads']['l2']}, linear vs "
             f"MLP({pc['heads']['mlp_hidden']}), MLP Adam lr {pc['heads']['mlp_lr']}, batch {pc['heads']['mlp_batch']}, up to "
             f"{pc['heads']['mlp_epochs']} epochs.\n"
             "- Final models: each selected setting refitted on all 57 TRAIN groups (MLP: the CV-selected epoch count), then "
             "applied once to the 15 TEST groups. B2: the selected LR × epochs refitted on all 57 TRAIN groups; its frozen "
             "embeddings feed B2 and every B2+probe head. TEST prompts and rules are embedded with the saved final adapters "
             "on Betty (CUDA), the backend that produced the TRAIN embeddings the final B2 heads were fitted on.\n"
             "- Reference rows at test time (as in CV, with all 57 TRAIN groups as the fitting set): REF_constant = TRAIN mean "
             "label; REF_type and REF_type_k = TRAIN violation rates per type (per type × k), averaged over the prompt's rules "
             "for Objective 2; REF_structure = its refitted head. Standardisation of every head uses TRAIN statistics only.\n"
             "- Features: H = final-prompt-token hidden state at the selected row; A = final-token attention to the target "
             "rule span over all layers × heads (selected pooling); Objective 2 uses 5 ordered slots with presence masks. "
             "Exact rule IDs are never model inputs.\n")
    s.append("## 3. Metrics (definitions in `src/mife/probing/metrics.py`)\n")
    s.append("- Objective 1 (prompt–rule rows, 20 responses each): binomial NLL per completion (primary); Brier on per-response "
             "binary outcomes; ECE; AUROC on per-response binary outcomes (ties = 1/2); RiskTop / RiskBottom = observed violation "
             "rate in the top / bottom decile of predicted risk (decile thresholds from the full test set, fixed within the "
             "bootstrap). Reported overall, per type (plus the macro mean over the 15 types) and per k.\n"
             "- Objective 2 (prompts): MAE, RMSE (primary), ECE, Pearson, Spearman; overall and per k.\n"
             "- Calibration binning: 10 bins whose edges are the deciles of the configuration's TRAINING out-of-fold CV "
             "predictions at its selected setting; ECE weights bins by responses (Objective 1) or prompts (Objective 2). "
             "Reliability curves use the same bins.\n")
    s.append("## 4. Uncertainty\n")
    s.append(f"- Paired cluster bootstrap over the 15 TEST groups: B = {pc['bootstrap']['test_B']}, seed "
             f"{pc['seeds']['bootstrap']}; every configuration and contrast uses the SAME replicate weights. All metrics are "
             "computed exactly from group-level sufficient statistics (AUROC via the cross-group concordance matrix; Spearman "
             "recomputed per replicate). 95% percentile intervals.\n")
    s.append("## 5. Hypothesis contrasts (difference = metric(b) − metric(a); negative favours b)\n")
    s.append("Primary metric: Objective 1 NLL, Objective 2 RMSE; the other metrics are reported descriptively. "
             "`BETTER(x,y)` is whichever of x, y has the lower TRAINING CV loss. Each contrast is run for both objectives and "
             "both generators.\n")
    s.append(_md(evaluate.contrast_table()) + "\n")
    s.append("## 6. Multiple comparisons\n")
    s.append("- Two-sided bootstrap p-value per contrast (2·min share of replicates on either side of 0); Holm adjustment "
             "within each (generator, objective, contrast family). Intervals are reported unadjusted.\n"
             "- Per-type and per-k breakdowns and Evaluation 2 are descriptive (no adjustment, no claims of significance).\n")
    s.append("## 7. Evaluation 2 (controlled analyses on TEST prompts; algorithms in `src/mife/probing/controlled.py`)\n")
    s.append(controlled.__doc__.strip() + "\n")
    s.append("Pair counts are fixed by the design (identical in every group):\n\n" + _md(counts) + "\n")
    s.append("E (difficulty) terciles, frozen from TRAIN k = 1 violation rates per type:\n\n" + _md(terc_df) + "\n")
    s.append("Every configuration is scored on the same pairs: mean |Δpred − Δobs| (group-bootstrap CI) and sign accuracy on "
             "reliable pairs (with the number of reliable pairs). E reports the Evaluation 1 metrics by tercile.\n")
    s.append("## 8. Sensitivity analyses (TEST)\n")
    s.append("Evaluation 1 primary metrics recomputed (a) with labels that exclude truncated responses (`hit_max_tokens`), "
             "(b) without `change_case:english_capital` (Objective 1: its rows; Objective 2: prompts containing it), and "
             "(c) without prompts that have any looping response (loop definition in `describe.py`). Primary labels are unchanged.\n")
    s.append("## 9. Known properties and deviations\n")
    ident = pd.read_csv(RESULTS / "tables" / "cv_identical_features.csv")
    s.append("- With one shared hidden row and deduplicated blocks, these configurations have identical feature blocks "
             "(linear heads give identical predictions; MLP heads differ only through their seeds):\n\n" + _md(ident) + "\n")
    s.append("- Consequently some contrasts in § 5 compare identical feature sets (C5 B1+H1 vs B1+H2) or repeat another "
             "contrast (C4 B1+H1 → B1+H1+A1 and B1+H2 → B1+H2+A1; B1+A1 → B1+H1+A1 and B1+A1 → B1+H2+A1; probe-only "
             "H2 and H1+H2, H2+A1 and H1+H2+A1). They are kept as listed and counted in the Holm family, which makes the "
             "adjustment slightly conservative.\n")
    cnt = RESULTS / "tables" / "cv_l2_counts.csv"
    if cnt.exists():
        s.append("- L2 grid: number of the 23 configurations plus the structure reference selecting each value (edge selections "
                 "and the loss one grid step inwards in `results/tables/cv_l2_edges.csv`). The grid is used as specified:\n\n"
                 + _md(pd.read_csv(cnt)) + "\n")
    s.append("- See `results/RESULTS.md` § 10 for every open issue.\n")
    s.append("## 10. Execution\n")
    s.append("- The TEST evaluation runs exactly once, after the project owner's explicit approval is recorded in "
             "`FROZEN.json` (`unblind_approved`); each run is appended to `FROZEN.json` `test_runs` with its time and the "
             "plan hash it used. Nothing in this plan is changed after the TEST set is read.\n")
    PRE.mkdir(parents=True, exist_ok=True)
    p = PRE / "analysis_plan.md"
    p.write_text("\n".join(s))
    return str(p)


def freeze() -> dict:
    p = PRE / "analysis_plan.md"
    out = {"analysis_plan_sha256": file_sha(p), "frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
           "source_tree_sha256": source_tree_hash(), "git": "not a git repository (source-tree hash recorded instead)",
           "folds_sha256": file_sha(CV_DIR / "folds.json"), "config_sha256": file_sha(ROOT / "configs" / "experiment.yaml"),
           "manifest_sha256": file_sha(DATA_DIR / "manifest" / "prompts.parquet"),
           "unblind_approved": False, "test_runs": []}
    write_json(out, PRE / "FROZEN.json")
    return out


def approve_unblinding(note: str) -> dict:
    """Only after the project owner's explicit 'go'. Refuses if the plan changed since it was frozen."""
    f = json.loads((PRE / "FROZEN.json").read_text())
    if file_sha(PRE / "analysis_plan.md") != f["analysis_plan_sha256"]:
        raise RuntimeError("analysis_plan.md changed after freezing; refusing to unblind")
    f.update({"unblind_approved": True, "unblind_approved_at": dt.datetime.now().isoformat(timespec="seconds"), "approval_note": note})
    write_json(f, PRE / "FROZEN.json")
    return f
