"""Training-only 5-fold GroupKFold over the 57 TRAIN groups, the staged grid, and out-of-fold predictions.

Folds: the 57 TRAIN group_ids are permuted with the fixed seed (configs/experiment.yaml probing.seeds.folds) and cut
into 12/12/11/11/11 groups. Saved to results/cv/folds.json.

Stages (per generator and objective, tracker §3A / docx §9.1.1):
  layer   H1 at each candidate hidden row x 6 head settings; the row with the lowest best-setting CV loss is the
          ONE shared row for H1/H2 and every combination.
  pool    A1 with SUM vs MEAN x 6 head settings; the lower best-setting CV loss is used by every A1 configuration.
  configs every non-B2 configuration and learned reference with the selected row/pooling x 6 head settings, plus
          the non-learned references.
Head settings: {linear, mlp} x l2 {0.01, 0.1, 1}; for an MLP setting the epoch count (1..30) is chosen by the mean
held-out-fold loss across folds. Selection score: mean over the 5 folds (equal fold weights) of the held-out-fold
mean loss (Objective 1 binomial NLL per completion, Objective 2 MSE). Out-of-fold predictions for every grid point
are saved to results/predictions/{model}/obj{o}/cv/. Resumable: finished (config, row, pooling) keys are skipped.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from .. import splits
from ..config import derive_seed, load_config
from ..storage import write_json
from . import configs as C
from . import data, features, heads
from .data import RESULTS

CV_DIR = splits.cv_dir()  # results/cv (V1, 2026-11-14 split) or the active split's folder (results/cv_s3)
MIN_VIOLATED = 50


def pcfg() -> dict:
    return load_config()["probing"]


# ----------------------------------------------------------------------------------------------------- folds
def make_folds(write: bool = True) -> dict:
    P = data.prompts("train")
    groups = np.sort(P["group_id"].unique())
    counts = splits.fold_group_counts() or pcfg()["cv"]["fold_group_counts"]
    assert len(groups) == sum(counts)
    rng = np.random.default_rng(pcfg()["seeds"]["folds"])
    perm = list(rng.permutation(groups))
    folds, s = [], 0
    for c in counts:
        folds.append(sorted(perm[s:s + c]))
        s += c
    fog = {g: i for i, f in enumerate(folds) for g in f}
    PR = data.prompt_rules("train")
    PR["fold"] = PR["group_id"].map(fog)
    cov = {str(f): {t: int(((PR.fold != f) & (PR.type_id == t)).sum()) for t in features.TYPES} for f in range(5)}
    gcov = {str(f): {t: int(PR[(PR.fold != f) & (PR.type_id == t)].group_id.nunique()) for t in features.TYPES} for f in range(5)}
    out = {"split_version": splits.version(), "seed": pcfg()["seeds"]["folds"], "n_groups": len(groups), "fold_group_counts": counts, "folds": folds,
           "fold_of_group": fog, "fitting_rows_per_type": cov, "fitting_groups_per_type": gcov,
           "n_prompts_per_fold": [int(P["group_id"].isin(f).sum()) for f in folds]}
    if write:
        CV_DIR.mkdir(parents=True, exist_ok=True)
        write_json(out, CV_DIR / "folds.json")
    return out


def route(types: np.ndarray | None, fold: np.ndarray, f: int, t: str | None) -> tuple[np.ndarray, np.ndarray]:
    """(fitting rows, held-out rows) of one head: per-type heads see ONLY rows of their own type t."""
    sel = np.ones(len(fold), bool) if t is None else (types == t)
    return np.nonzero(sel & (fold != f))[0], np.nonzero(sel & (fold == f))[0]


def load_folds() -> dict:
    p = CV_DIR / "folds.json"
    return json.loads(p.read_text()) if p.exists() else make_folds()


# ------------------------------------------------------------------------------------------------ feature store
class Store:
    """Row-aligned labels, groups and lazily loaded feature blocks for one generator, objective and split."""

    def __init__(self, model: str, objective: int, split: str = "train"):
        self.model, self.objective, self.split = model, objective, split
        self.P = data.prompts(split)
        self.pid = {x: i for i, x in enumerate(self.P["prompt_id"])}
        self._b: dict[str, np.ndarray] = {}
        if objective == 1:
            self.U = data.prompt_rules(split)[["prompt_id", "position", "rule_id", "type_id", "group_id"]]
            L = data.labels_prompt_rule(model, split)[["prompt_id", "position", "rule_id", "v", "n", "p_hat"]]
            self.U = self.U.merge(L, on=["prompt_id", "position", "rule_id"], how="left", validate="one_to_one")
            assert self.U["n"].notna().all() and len(self.U) == len(data.prompt_rules(split))
            self.y, self.n = self.U["v"].values.astype(np.float64), self.U["n"].values.astype(np.float64)
            self.pix = self.U["prompt_id"].map(self.pid).values
            rk = pd.read_parquet(features.cache_dir() / f"ER_rules_{split}_keys.parquet")
            self.rix = self.U["rule_id"].map({r: i for i, r in enumerate(rk["rule_id"])}).values
            self.types = self.U["type_id"].values
        else:
            L = data.labels_prompt(model, split)[["prompt_id", "q_hat", "k"]]
            self.U = self.P[["prompt_id", "group_id", "k"]].merge(L[["prompt_id", "q_hat"]], on="prompt_id", how="left",
                                                                 validate="one_to_one")
            assert self.U["q_hat"].notna().all()
            self.y, self.n = self.U["q_hat"].values.astype(np.float64), None
            self.types = None
        self.groups = self.U["group_id"].values
        self.cur_fold = None  # B2 blocks are fold-specific: the encoder adapted on that fold's fitting groups

    def _b2(self, name: str) -> np.ndarray:
        """EP_B2@f / ER_B2@f / ER5_B2@f from the selected B2 run of fold f ('final' = refit on all 57 groups)."""
        from . import b2
        base, f = name.split("@")
        z = np.load(b2.selected_run_dir(self.model, self.objective, f) / "embeddings_fp16.npz")
        if base == "EP_B2":
            return z["EP"].astype(np.float32)
        ER = z["ER"].astype(np.float32)
        if base == "ER_B2":
            return ER
        pr = data.prompt_rules(self.split)
        rk = pd.read_parquet(features.cache_dir() / f"ER_rules_{self.split}_keys.parquet")
        rix = {r: i for i, r in enumerate(rk["rule_id"])}
        out = np.zeros((len(self.P), 5, ER.shape[1]), dtype=np.float32)
        out[pr["prompt_id"].map(self.pid).values, pr["position"].values - 1] = ER[pr["rule_id"].map(rix).values]
        return out.reshape(len(self.P), -1)

    # raw blocks
    def _raw(self, name: str) -> np.ndarray:
        if name in self._b:
            return self._b[name]
        o, s, m = self.objective, self.split, self.model
        if "@" in name:
            for k in [k for k in self._b if "@" in k and k.split("@")[1] != name.split("@")[1]]:
                del self._b[k]  # keep one fold's B2 blocks in memory
            self._b[name] = self._b2(name)
            return self._b[name]
        if o == 1:
            if name == "S1":
                a = features.load_rule_block("S1", s)
            elif name == "ER":
                a = np.load(features.cache_dir() / f"ER_rules_{s}.npy")
            elif name == "EP" or name.startswith("H"):
                a = np.asarray(features.load_prompt_block(name, s, m))
            elif name in ("Asum", "Amean"):
                a = features.load_rule_block(name, s, m)
            else:
                raise KeyError(name)
        else:
            if name == "KM":
                a = np.asarray(features.load_prompt_block("M2", s))[:, :6]
            elif name in ("S2", "M2", "EP", "ER5", "A5sum", "A5mean") or name.startswith("H"):
                a = np.asarray(features.load_prompt_block(name, s, m))
            else:
                raise KeyError(name)
        self._b[name] = np.ascontiguousarray(a, dtype=np.float32)
        return self._b[name]

    def drop(self, keep: set[str]) -> None:
        for k in list(self._b):
            if k not in keep:
                del self._b[k]

    def resolve(self, block: str, hrow: int | None, pool: str | None, objective: int) -> str:
        if block == "H":
            return f"H{hrow}"
        if block == "A":
            return f"A{pool}"
        if block == "A5":
            return f"A5{pool}"
        if block.endswith("_B2"):
            if self.cur_fold is None:
                raise RuntimeError("B2 blocks need the current fold (Store.cur_fold)")
            return f"{block}@{self.cur_fold}"
        return block

    def X(self, blocks: list[str], rows: np.ndarray, hrow=None, pool=None) -> np.ndarray:
        parts = []
        for b in blocks:
            name = self.resolve(b, hrow, pool, self.objective)
            a = self._raw(name)
            if self.objective == 1:
                if name.split("@")[0] in ("ER", "ER_B2"):
                    parts.append(a[self.rix[rows]])
                elif name.split("@")[0] in ("EP", "EP_B2") or name.startswith("H"):
                    parts.append(a[self.pix[rows]])
                else:
                    parts.append(a[rows])
            else:
                parts.append(a[rows])
        return np.concatenate(parts, axis=1)


# ----------------------------------------------------------------------------------------------------- settings
def settings() -> list[tuple[str, float]]:
    h = pcfg()["heads"]
    return [(k, float(l)) for k in h["kinds"] for l in sorted(h["l2"], reverse=True)]


def skey(kind: str, l2: float) -> str:
    return f"{kind}_l2={l2:g}"


def _fit_unit(st: Store, blocks, fit, held, hrow, pool, seed_parts, objective):
    """Fit all 6 settings on rows `fit`; return {setting: held predictions} (MLP: (epochs, n_held))."""
    Xf, Xh = st.X(blocks, fit, hrow, pool), st.X(blocks, held, hrow, pool)
    yf, nf = st.y[fit], None if st.n is None else st.n[fit]
    sc = heads.Scaler.fit(Xf)
    out, init = {}, None
    ep = pcfg()["heads"]["mlp_epochs"]
    for kind, l2 in settings():
        if kind == "linear":
            h = heads.fit_linear(Xf, yf, nf, objective, l2, init=init, scaler=sc)
            init = h.state  # warm start along the l2 path (strong -> weak)
            out[skey(kind, l2)] = h.predict(Xh)
        else:
            seed = derive_seed(pcfg()["seeds"]["heads"], *seed_parts, kind, l2) % (1 << 31)
            _, preds = heads.fit_mlp(Xf, yf, nf, objective, l2, ep, seed, lr=pcfg()["heads"]["mlp_lr"],
                                     batch=pcfg()["heads"]["mlp_batch"], X_eval=Xh, scaler=sc)
            out[skey(kind, l2)] = np.stack(preds)
    return out


def run_config(st: Store, config: str, hrow=None, pool=None) -> dict:
    """Grouped CV of one learned configuration over all head settings; writes grid results + OOF predictions."""
    o = st.objective
    key = config + (f"__h{hrow}" if C.uses_hidden(config) else "") + (f"__a{pool}" if C.uses_attention(config) else "")
    gdir = CV_DIR / st.model / f"obj{o}" / "grid"
    pdir = RESULTS / "predictions" / st.model / f"obj{o}" / "cv"
    if (gdir / f"{key}.json").exists():
        return json.loads((gdir / f"{key}.json").read_text())
    t0 = time.time()
    blocks = C.blocks(config, o)
    folds = load_folds()
    fog = folds["fold_of_group"]
    fold = np.array([fog[g] for g in st.groups])
    n_units = len(st.y)
    ep = pcfg()["heads"]["mlp_epochs"]
    oof = {skey(k, l): (np.full(n_units, np.nan) if k == "linear" else np.full((ep, n_units), np.nan)) for k, l in settings()}
    sparse = []
    for f in range(5):
        st.cur_fold = f
        units = [None] if not C.per_type(config, o) else sorted(set(st.types))
        for t in units:
            fit, held = route(st.types, fold, f, t)
            if t is not None and st.y[fit].sum() < MIN_VIOLATED:
                sparse.append({"fold": f, "type_id": t, "violated_responses_fitting": float(st.y[fit].sum())})
            res = _fit_unit(st, blocks, fit, held, hrow, pool, (st.model, o, key, f, t), o)
            for s, p in res.items():
                if p.ndim == 1:
                    oof[s][held] = p
                else:
                    oof[s][:, held] = p
    # fold-wise scores
    rows = []
    for s, p in oof.items():
        if p.ndim == 1:
            fl = [heads.score(p[fold == f], st.y[fold == f], None if st.n is None else st.n[fold == f], o) for f in range(5)]
            rows.append({"setting": s, "epoch": None, "fold_losses": fl, "cv_loss": float(np.mean(fl))})
        else:
            per_ep = np.array([[heads.score(p[e, fold == f], st.y[fold == f], None if st.n is None else st.n[fold == f], o)
                                for f in range(5)] for e in range(ep)])
            e_best = int(np.argmin(per_ep.mean(1)))
            rows.append({"setting": s, "epoch": e_best + 1, "fold_losses": per_ep[e_best].tolist(),
                         "cv_loss": float(per_ep[e_best].mean()), "cv_loss_by_epoch": per_ep.mean(1).tolist()})
            oof[s] = p[e_best]
    best = min(rows, key=lambda r: r["cv_loss"])
    n_feat = int(st.X(blocks, np.arange(2), hrow, pool).shape[1])
    st.cur_fold = None
    out = {"model": st.model, "objective": o, "config": config, "key": key, "hidden_row": hrow if C.uses_hidden(config) else None,
           "pooling": pool if C.uses_attention(config) else None, "blocks": blocks, "per_type": C.per_type(config, o),
           "n_features": n_feat, "grid": rows, "selected": best["setting"],
           "selected_epoch": best["epoch"], "selected_cv_loss": best["cv_loss"], "sparse_types": sparse,
           "wall_s": time.time() - t0}
    _save_oof(st, pdir / f"{key}.parquet", fold, {s: oof[s] for s in oof})
    gdir.mkdir(parents=True, exist_ok=True)
    write_json(out, gdir / f"{key}.json")
    return out


def _save_oof(st: Store, path, fold, cols: dict) -> None:
    keys = ["prompt_id", "position", "rule_id", "type_id"] if st.objective == 1 else ["prompt_id"]
    df = st.U[keys].copy()
    df["fold"] = fold
    for s, p in cols.items():
        df[s] = p
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def run_references(st: Store) -> dict:
    """Non-learned references from fitting-fold rates: constant, type, type x k (Objective 2: mean over the
    prompt's rules of the per-type / per-type-x-k rates estimated from the fitting fold's Objective 1 labels)."""
    o = st.objective
    fog = load_folds()["fold_of_group"]
    fold = np.array([fog[g] for g in st.groups])
    L1 = st.U if o == 1 else data.prompt_rules("train")[["prompt_id", "position", "type_id", "group_id"]].merge(
        data.labels_prompt_rule(st.model, "train")[["prompt_id", "position", "v", "n"]], on=["prompt_id", "position"])
    L1 = L1.merge(st.P[["prompt_id", "k"]], on="prompt_id") if "k" not in L1 else L1
    f1 = np.array([fog[g] for g in L1["group_id"]])
    preds = {r: np.full(len(st.y), np.nan) for r in ("REF_constant", "REF_type", "REF_type_k")}
    for f in range(5):
        fit = L1[f1 != f]
        held = fold == f
        if o == 1:
            preds["REF_constant"][held] = fit.v.sum() / fit.n.sum()
            g1 = fit.groupby("type_id")[["v", "n"]].sum()
            preds["REF_type"][held] = (g1.v / g1.n).reindex(st.U.type_id[held]).values
            g2 = fit.groupby(["type_id", "k"])[["v", "n"]].sum()
            idx = pd.MultiIndex.from_arrays([st.U.type_id[held], st.P.k.values[st.pix[held]]])
            preds["REF_type_k"][held] = (g2.v / g2.n).reindex(idx).values
        else:
            Qfit = st.U[fold != f]
            preds["REF_constant"][held] = Qfit.q_hat.mean()
            g1 = fit.groupby("type_id")[["v", "n"]].sum()
            g2 = fit.groupby(["type_id", "k"])[["v", "n"]].sum()
            Lh = L1[f1 == f].copy()
            Lh["r1"] = (g1.v / g1.n).reindex(Lh.type_id).values
            Lh["r2"] = (g2.v / g2.n).reindex(pd.MultiIndex.from_arrays([Lh.type_id, Lh.k])).values
            m = Lh.groupby("prompt_id")[["r1", "r2"]].mean()
            ids = st.U.prompt_id[held]
            preds["REF_type"][held] = m.r1.reindex(ids).values
            preds["REF_type_k"][held] = m.r2.reindex(ids).values
    out = {}
    for r, p in preds.items():
        assert np.isfinite(p).all(), r
        fl = [heads.score(p[fold == f], st.y[fold == f], None if st.n is None else st.n[fold == f], o) for f in range(5)]
        out[r] = {"model": st.model, "objective": o, "config": r, "key": r, "grid": [{"setting": "fitting-fold rates",
                  "fold_losses": fl, "cv_loss": float(np.mean(fl))}], "selected": "fitting-fold rates",
                  "selected_cv_loss": float(np.mean(fl))}
        _save_oof(st, RESULTS / "predictions" / st.model / f"obj{o}" / "cv" / f"{r}.parquet", fold, {"fitting-fold rates": p})
        write_json(out[r], CV_DIR / st.model / f"obj{o}" / "grid" / f"{r}.json")
    return out


def run(model: str, objective: int, configs: list[str] | None = None, stage: str = "all") -> dict:
    torch.set_num_threads(int(pcfg().get("torch_threads", 4)))
    st = Store(model, objective)
    load_folds()
    sel_path = CV_DIR / model / f"obj{objective}" / "selection.json"
    sel = json.loads(sel_path.read_text()) if sel_path.exists() else {}
    rows = pcfg()["hidden_rows"][model]
    log = lambda r: print(json.dumps({k: r[k] for k in ("config", "key", "selected", "selected_cv_loss", "wall_s") if k in r}), flush=True)
    if stage in ("layer", "all") and "hidden_row" not in sel:
        res = {}
        for r in rows:
            res[r] = run_config(st, "H1", hrow=r)
            log(res[r])
            st.drop(set())
        sel["hidden_row"] = int(min(res, key=lambda r: res[r]["selected_cv_loss"]))
        sel["layer_scores"] = {str(r): res[r]["selected_cv_loss"] for r in rows}
        write_json(sel, sel_path)
    if stage in ("pool", "all") and "pooling" not in sel:
        res = {}
        for p in ("sum", "mean"):
            res[p] = run_config(st, "A1", pool=p)
            log(res[p])
        sel["pooling"] = min(res, key=lambda p: res[p]["selected_cv_loss"])
        sel["pooling_scores"] = {p: res[p]["selected_cv_loss"] for p in res}
        write_json(sel, sel_path)
    if stage in ("configs", "all"):
        hrow, pool = sel["hidden_row"], sel["pooling"]
        keep = {f"H{hrow}", f"A{pool}", f"A5{pool}"}
        st.drop(keep | {"S1", "S2", "M2", "KM", "EP", "ER", "ER5"})
        todo = configs or [c for c in C.CONFIGS if not C.uses_b2(c)] + sorted(C.LEARNED_REFS)
        for c in todo:
            if C.uses_b2(c):
                continue
            log(run_config(st, c, hrow=hrow, pool=pool))
        for r in run_references(st).values():
            log(r)
    if stage in ("b2", "all") and (CV_DIR / model / f"obj{objective}" / "b2_selection.json").exists():
        hrow, pool = sel["hidden_row"], sel["pooling"]
        for c in configs or C.B2_FAMILY:
            log(run_config(st, c, hrow=hrow, pool=pool))
    return sel
