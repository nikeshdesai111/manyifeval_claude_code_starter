"""Training-only CV summary tables (results/tables/cv_*.csv) from results/cv/*/obj*/grid/*.json."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import load_config
from . import configs as C
from .cv import CV_DIR
from .data import MODELS, RESULTS

T = RESULTS / "tables"
DEPTH = {"llama": {16: "50%", 24: "75%", 33: "100% (pre-norm)"}, "qwen": {18: "50%", 27: "75%", 37: "100% (pre-norm)"}}


def _grids():
    for m in MODELS:
        for o in (1, 2):
            for p in sorted((CV_DIR / m / f"obj{o}" / "grid").glob("*.json")):
                yield m, o, json.loads(p.read_text())


def build() -> dict:
    T.mkdir(parents=True, exist_ok=True)
    l2 = load_config()["probing"]["heads"]["l2"]
    ep_max = load_config()["probing"]["heads"]["mlp_epochs"]
    folds = json.loads((CV_DIR / "folds.json").read_text())
    pd.DataFrame([{"fold": i, "groups": len(f), "prompts": folds["n_prompts_per_fold"][i],
                   "min_fitting_rows_per_type": min(folds["fitting_rows_per_type"][str(i)].values()),
                   "min_fitting_groups_per_type": min(folds["fitting_groups_per_type"][str(i)].values())}
                  for i, f in enumerate(folds["folds"])]).to_csv(T / "cv_folds.csv", index=False)
    grid_rows, sel_rows, bound, sparse = [], [], [], []
    for m, o, g in _grids():
        for r in g["grid"]:
            grid_rows.append({"model": m, "objective": o, "key": g["key"], "config": g["config"], "setting": r["setting"],
                              "epoch": r.get("epoch"), "cv_loss": r["cv_loss"], "fold_sd": float(np.std(r["fold_losses"], ddof=1))})
        best = next(r for r in g["grid"] if r["setting"] == g["selected"])
        sel_rows.append({"model": m, "objective": o, "family": C.family(g["config"]), "config": g["config"], "key": g["key"],
                         "setting": g["selected"], "epoch": g.get("selected_epoch"), "cv_loss": g["selected_cv_loss"],
                         "fold_sd": float(np.std(best["fold_losses"], ddof=1)), "n_features": g.get("n_features"),
                         "per_type_heads": g.get("per_type")})
        s = g["selected"]
        if "_l2=" in s:
            lam = float(s.split("_l2=")[1])
            if lam in (min(l2), max(l2)):
                bound.append({"model": m, "objective": o, "key": g["key"], "setting": s, "edge": "l2 at grid edge"})
            if s.startswith("mlp") and g.get("selected_epoch") in (1, ep_max):
                bound.append({"model": m, "objective": o, "key": g["key"], "setting": s,
                              "edge": f"MLP epoch {g.get('selected_epoch')} (schedule edge)"})
        for sp in g.get("sparse_types", []):
            sparse.append({"model": m, "objective": o, "key": g["key"], **sp})
    G, S = pd.DataFrame(grid_rows), pd.DataFrame(sel_rows)
    G.to_csv(T / "cv_grid_all.csv", index=False)
    lay, pool = [], []
    for m in MODELS:
        for o in (1, 2):
            sp = CV_DIR / m / f"obj{o}" / "selection.json"
            if not sp.exists():
                continue
            sel = json.loads(sp.read_text())
            for r, v in sel.get("layer_scores", {}).items():
                lay.append({"model": m, "objective": o, "hidden_row": int(r), "depth": DEPTH[m][int(r)], "best_cv_loss": v,
                            "selected": int(r) == sel.get("hidden_row")})
            for pl, v in sel.get("pooling_scores", {}).items():
                pool.append({"model": m, "objective": o, "pooling": pl, "best_cv_loss": v, "selected": pl == sel.get("pooling")})
    pd.DataFrame(lay).to_csv(T / "cv_layer_selection.csv", index=False)
    pd.DataFrame(pool).to_csv(T / "cv_pooling_selection.csv", index=False)
    # selected settings of the configurations actually used (the selected row / pooling only)
    keep = []
    for m in MODELS:
        for o in (1, 2):
            sp = CV_DIR / m / f"obj{o}" / "selection.json"
            if not sp.exists() or S.empty:
                continue
            sel = json.loads(sp.read_text())
            want = {c + (f"__h{sel.get('hidden_row')}" if C.uses_hidden(c) else "") +
                    (f"__a{sel.get('pooling')}" if C.uses_attention(c) else "") for c in C.CONFIGS + C.REFERENCES}
            keep.append(S[(S.model == m) & (S.objective == o) & S.key.isin(want)])
    used = pd.concat(keep) if keep else S
    if not used.empty:
        ref = used[used.config == "REF_structure"].set_index(["model", "objective"]).cv_loss
        used = used.assign(minus_ref_structure=[r.cv_loss - ref.get((r.model, r.objective), np.nan) for r in used.itertuples()])
        order = {c: i for i, c in enumerate(C.REFERENCES + C.CONFIGS)}
        used = used.sort_values(["model", "objective", "config"], key=lambda s: s.map(order) if s.name == "config" else s)
    used.to_csv(T / "cv_selected.csv", index=False)
    pd.DataFrame(bound).to_csv(T / "cv_boundary_hits.csv", index=False)
    pd.DataFrame(sparse if sparse else [{"note": "no type had fewer than 50 violated responses in any fitting fold"}]).to_csv(
        T / "cv_sparse_types.csv", index=False)
    b2 = []
    for m in MODELS:
        for o in (1, 2):
            p = CV_DIR / m / f"obj{o}" / "b2_selection.json"
            if p.exists():
                s = json.loads(p.read_text())
                for r in s["grid"]:
                    b2.append({"model": m, "objective": o, "adapt": r["adapt"], "lr": r["lr"], "epochs": r["epochs"],
                               "cv_loss": r["cv_loss"], "selected": {k: r[k] for k in ("adapt", "lr", "epochs")} == s["selected"]})
    if b2:
        pd.DataFrame(b2).to_csv(T / "cv_b2_grid.csv", index=False)
    if not used.empty:
        _edges(G, used, l2)
        _summary(used)
    _identical()
    return {"selected_rows": len(used), "grid_rows": len(G), "boundary_hits": len(bound)}


def _l2(setting: str) -> tuple[str, float] | None:
    if "_l2=" not in setting:
        return None
    k, v = setting.split("_l2=")
    return k, float(v)


def _edges(G: pd.DataFrame, used: pd.DataFrame, l2: list[float]) -> None:
    """For the configurations in use: how many selected each l2 value, and, for edge selections, how much worse the
    nearest interior value is (a steep gap suggests the optimum may lie outside the grid)."""
    U = used[used.config.isin(C.CONFIGS + ["REF_structure"])]
    lam = [(r.model, r.objective, r.config, *(_l2(r.setting) or (None, None))) for r in U.itertuples()]
    L = pd.DataFrame(lam, columns=["model", "objective", "config", "head", "l2"]).dropna()
    cnt = L.groupby(["model", "objective", "l2"]).size().unstack("l2", fill_value=0)
    cnt.columns = [f"selected_l2={c:g}" for c in cnt.columns]
    cnt.reset_index().to_csv(T / "cv_l2_counts.csv", index=False)
    lo, hi, mid = min(l2), max(l2), sorted(l2)[len(l2) // 2]
    rows = []
    for r in U.itertuples():
        kl = _l2(r.setting)
        if kl is None or kl[1] not in (lo, hi):
            continue
        g = G[(G.model == r.model) & (G.objective == r.objective) & (G.key == r.key)]
        nxt = g[g.setting == f"{kl[0]}_l2={mid:g}"].cv_loss.min()
        rows.append({"model": r.model, "objective": r.objective, "config": r.config, "selected": r.setting,
                     "edge": "lowest l2" if kl[1] == lo else "highest l2", "cv_loss": r.cv_loss,
                     f"cv_loss_at_l2={mid:g}": nxt, "gap_pct": 100 * (nxt - r.cv_loss) / r.cv_loss, "fold_sd": r.fold_sd})
    pd.DataFrame(rows).to_csv(T / "cv_l2_edges.csv", index=False)


def _summary(used: pd.DataFrame) -> None:
    """One row per generator × objective: best configuration overall and per family, the reference rows, how many of
    the 23 fall below the structure reference, and the CV complementarity of hidden + attention."""
    rows, comp = [], []
    for (m, o), d in used.groupby(["model", "objective"]):
        d = d.set_index("config")
        cfg = d.loc[[c for c in C.CONFIGS if c in d.index]]
        ref = d.cv_loss.get("REF_structure", np.nan)
        row = {"model": m, "objective": o, "metric": "binomial NLL" if o == 1 else "MSE",
               "best_config": cfg.cv_loss.idxmin(), "best_cv_loss": cfg.cv_loss.min()}
        for fam, tag in (("probe-only", "probe_only"), ("B1 family", "B1_family"), ("B2 family", "B2_family")):
            f = cfg[cfg.family == fam]
            row[f"best_{tag}"] = f.cv_loss.idxmin()
            row[f"best_{tag}_cv_loss"] = f.cv_loss.min()
        row.update({"B1_cv_loss": d.cv_loss.get("B1"), "B2_cv_loss": d.cv_loss.get("B2"), "REF_structure_cv_loss": ref,
                    "REF_type_k_cv_loss": d.cv_loss.get("REF_type_k"), "REF_constant_cv_loss": d.cv_loss.get("REF_constant"),
                    "n_of_23_below_REF_structure": int((cfg.cv_loss < ref).sum())})
        rows.append(row)
        for h in ("H1", "H2"):
            a, b, ab = d.cv_loss.get(h), d.cv_loss.get("A1"), d.cv_loss.get(f"{h}+A1")
            comp.append({"model": m, "objective": o, "hidden": h, "combined": f"{h}+A1", "cv_loss_hidden": a,
                         "cv_loss_A1": b, "cv_loss_combined": ab, "combined_minus_better_single": ab - min(a, b)})
    pd.DataFrame(rows).to_csv(T / "cv_summary.csv", index=False)
    pd.DataFrame(comp).to_csv(T / "cv_complementarity.csv", index=False)


def _identical() -> None:
    """Configurations whose deduplicated feature blocks coincide (a design property, not a result)."""
    rows = []
    for o in (1, 2):
        cls: dict[tuple, list[str]] = {}
        for c in C.CONFIGS:
            cls.setdefault(tuple(sorted(C.blocks(c, o))), []).append(c)
        rows += [{"objective": o, "configurations": " = ".join(v), "blocks": " + ".join(C.blocks(v[0], o))}
                 for v in cls.values() if len(v) > 1]
    pd.DataFrame(rows).to_csv(T / "cv_identical_features.csv", index=False)
