"""K10 snapshot of mife.probing.v2.cv (TRAIN only, Objective 1), re-pointed at K10 TRAIN data.

Differences from the V2 original, all mechanical: data from mife.k10paper.kdata/features (50 S3 TRAIN groups, S3
folds results/cv_s3/folds.json); 5 completions per prompt (n = 5); the attention block A1 is stored rule-row-aligned
(A1R); the other-generator reference pairs llama <-> gemma2b; outputs under results/k10_paper/v2/; the recipe is
src/mife/k10paper/recipe_k10.yaml (identical to configs/v2_recipe.yaml except hidden_rows). `prepare` builds the same
standardised matrix and rule-block bases as the original with less memory (the rule block's distinct rows are its
distinct rules, so the basis is computed from those rows directly). Refits are skipped (only OOF predictions are used).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

from ..probing.v2 import ridge
from ..storage import sha256_file, write_json
from . import features as F
from . import kdata
from .paths import FOLDS, HIDDEN_ROWS, N_SAMPLES, OTHER, OUT

RESULTS_V2 = OUT / "v2"
VERSION = "k10-v2-2026-11-27"
RECIPE = __import__("pathlib").Path(__file__).with_name("recipe_k10.yaml")


def recipe() -> dict:
    return yaml.safe_load(RECIPE.read_text())


def record_recipe() -> dict:
    out = RESULTS_V2 / "recipe_recorded.json"
    RESULTS_V2.mkdir(parents=True, exist_ok=True)
    sha = sha256_file(RECIPE)
    if out.exists():
        prev = json.loads(out.read_text())
        if prev["sha256"] != sha:
            raise RuntimeError("recipe_k10.yaml changed after it was recorded")
        return prev
    rec = {"sha256": sha, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "recipe": recipe(), "version": VERSION}
    write_json(rec, out)
    return rec


def _require_recorded():
    out = RESULTS_V2 / "recipe_recorded.json"
    if not out.exists() or json.loads(out.read_text())["sha256"] != sha256_file(RECIPE):
        raise RuntimeError("record the K10 recipe (cv.record_recipe()) before running selection")


class _SafeGram(ridge.Gram):
    """ridge.Gram with a CPU fallback: if an MPS command buffer fails (seen under GPU contention), this call and the
    rest of this head use the CPU (Accelerate) float32 assembly, which ridge.Gram documents as equivalent."""

    def __call__(self, s):
        if self.dev == "mps":
            try:
                return super().__call__(s)
            except Exception as e:  # torch.AcceleratorError
                print(f"MPS Hessian failed ({type(e).__name__}); falling back to CPU for this head", flush=True)
                self.X = self.X.cpu().numpy()
                self.dev = "cpu"
        return super().__call__(s)


ridge.Gram = _SafeGram


def write_versioned_json(obj, path):
    write_json({**obj, "version": VERSION}, path)


def read_versioned_json(path):
    return json.loads(path.read_text())


# ------------------------------------------------------------------------------------------------- data
class Data:
    """Row-aligned K10 TRAIN labels, folds and feature blocks for one label generator (Objective 1)."""

    def __init__(self, model: str, objective: int):
        assert objective == 1
        self.model, self.o = model, objective
        self.P = kdata.prompts()
        pid = {x: i for i, x in enumerate(self.P["prompt_id"])}
        fog = kdata.fold_of_group()
        PR = kdata.prompt_rules()[["prompt_id", "position", "rule_id", "type_id", "group_id"]]
        L = kdata.labels_prompt_rule(model)[["prompt_id", "position", "rule_id", "v", "n"]]
        self.U = PR.merge(L, on=["prompt_id", "position", "rule_id"], how="left", validate="one_to_one")
        assert self.U["n"].notna().all() and (self.U["n"] == N_SAMPLES).all()
        self.v, self.n = self.U["v"].values.astype(np.float64), self.U["n"].values.astype(np.float64)
        self.pix = self.U["prompt_id"].map(pid).values
        self.rix = self.U["rule_id"].map(F.rule_index()).values
        self.slot = self.U["position"].values - 1
        self.types = self.U["type_id"].values
        self.groups = self.U["group_id"].values
        self.fold = np.array([fog[g] for g in self.groups])
        self._b: dict[str, np.ndarray] = {}

    # ---- blocks (Objective 1: row-level views are built lazily per head with `rows`)
    def _prompt_block(self, name: str) -> np.ndarray:
        if name in self._b:
            return self._b[name]
        m = self.model
        if name == "EP":
            a = F.load_shared("EP")
        elif name.startswith("XH"):
            a = F.load_model_block(OTHER[m], name[1:])
        elif name.startswith("H"):
            a = F.load_model_block(m, name)
        elif name == "A1":
            a = F.load_model_block(m, "A1R")  # rule-row aligned
        elif name == "S":
            a = F.load_model_block(m, "S1")
        elif name == "ER":
            a = F.load_shared("ER")
        else:
            raise KeyError(name)
        self._b[name] = a
        return a

    def block_rows(self, b: str, rows: np.ndarray, out=None) -> np.ndarray:
        a = self._prompt_block(b)
        if b in ("S", "A1"):
            x = a[rows]
        elif b == "ER":
            x = a[self.rix[rows]]
        else:
            x = a[self.pix[rows]]
        if out is None:
            return np.asarray(x, dtype=np.float64)
        out[...] = x
        return out

    def X(self, blocks: list[str], rows: np.ndarray) -> np.ndarray:
        return np.concatenate([self.block_rows(b, rows) for b in blocks], axis=1)

    def widths(self, blocks: list[str]) -> list[int]:
        return [self.block_rows(b, np.arange(1)).shape[1] for b in blocks]

    def drop(self, keep=()):
        for k in list(self._b):
            if k not in keep:
                del self._b[k]


# --------------------------------------------------------------------------------------------- predictors
@dataclass(frozen=True)
class Spec:
    name: str
    blocks: tuple            # ((block, slot), ...), slot in S (fixed structure), P, R (tuned), F_P / F_R (frozen), X (added)
    fixed: tuple = ()        # (("F_P", lam), ("F_R", lam)) for incremental baselines
    baseline: str | None = None

    @property
    def key(self) -> str:
        b = "+".join(f"{bl}:{sl}" for bl, sl in self.blocks)
        fx = ",".join(f"{s}={l:g}" for s, l in self.fixed)
        return f"{self.name}__" + hashlib.sha1(f"{b}|{fx}".encode()).hexdigest()[:10]

    @property
    def tuned(self) -> tuple:
        return tuple(s for s in ("P", "R", "X") if any(sl == s for _, sl in self.blocks))


def setting_key(st: dict) -> str:
    return ",".join(f"{k}={st[k]:g}" for k in sorted(st)) or "none"


def make_specs(model: str, o: int, row: int | None, xrow: int | None, sel: dict | None = None, ordered=False) -> dict:
    """Head-to-head predictors (+ references). Incremental ones are built by `incremental_spec`."""
    ER = "ER" if o == 1 else ("ER5" if ordered else "ERmean")
    A = "A1" if o == 1 else ("A5" if ordered else "A1mean")
    s = {"REF_structure": Spec("REF_structure", (("S", "S"),))}
    sfx = "o" if ordered else ""
    if not ordered:
        s["B1"] = Spec("B1", (("EP", "P"), ("S", "S")))
    s["B2" + sfx] = Spec("B2" + sfx, (("EP", "P"), (ER, "R"), ("S", "S")))
    if row is not None:
        if not ordered:
            s["H1"] = Spec("H1", ((f"H{row}", "P"), ("S", "S")))
        s["H2" + sfx] = Spec("H2" + sfx, ((f"H{row}", "P"), (ER, "R"), ("S", "S")))
    s["RCAP" + sfx] = Spec("RCAP" + sfx, ((A, "P"), (ER, "R"), ("S", "S")))
    if xrow is not None and not ordered:
        s["XH1"] = Spec("XH1", ((f"XH{xrow}", "P"), ("S", "S")))
    return s


def incremental_spec(name: str, base: Spec, base_setting: dict, added_block: str) -> Spec:
    blocks = tuple((b, {"P": "F_P", "R": "F_R"}.get(sl, sl)) for b, sl in base.blocks) + ((added_block, "X"),)
    fixed = tuple((f"F_{k}", float(v)) for k, v in sorted(base_setting.items()))
    return Spec(name, blocks, fixed, baseline=base.name)


# ------------------------------------------------------------------------------------------------ engine
class Engine:
    def __init__(self, model: str, o: int):
        _require_recorded()
        self.model, self.o = model, o
        self.R = recipe()[f"objective{o}"]
        self.D = Data(model, o)
        self.root = RESULTS_V2 / "cv" / model / f"obj{o}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.prec = int(round(-np.log10(self.R["tie_precision"])))
        self.fallback = self._fallbacks() if o == 1 else {}

    # ---- sparse-head fallback (Objective 1): decided from labels of the fitting rows, identical for every method
    def _fallbacks(self) -> dict:
        fb = self.R["fallback"]
        out, rows = {}, []
        D = self.D
        for f in list(range(5)) + ["final"]:
            for t in F.TYPES:
                m = (D.types == t) & ((D.fold != f) if f != "final" else True)
                v, n = D.v[m].sum(), D.n[m].sum()
                gv = len(set(D.groups[m & (D.v > 0)]))
                fire = bool(v < fb["min_violated"] or (n - v) < fb["min_compliant"] or gv < fb["min_groups_with_violation"])
                out[(f, t)] = (fire, float(v), float(n))
                rows.append({"model": self.model, "fold": f, "type_id": t, "violated": v, "compliant": n - v,
                             "groups_with_violation": gv, "fallback": fire,
                             "reason": ";".join(r for r, c in (("violated<50", v < fb["min_violated"]),
                                                              ("compliant<50", (n - v) < fb["min_compliant"]),
                                                              ("groups_with_violation<5", gv < fb["min_groups_with_violation"])) if c)})
        self.fallback_table = pd.DataFrame(rows)
        return out

    def units(self):
        return [None] if self.o == 2 else F.TYPES

    def rows(self, f, t):
        D = self.D
        sel = np.ones(len(D.v), bool) if t is None else (D.types == t)
        if f == "final":
            return np.nonzero(sel)[0], np.zeros(0, int)
        return np.nonzero(sel & (D.fold != f))[0], np.nonzero(sel & (D.fold == f))[0]

    RULE_BLOCKS = ("ER", "ERmean", "ER5")

    def prepare(self, blocks, widths, fit, held, cache_key):
        """Same result as the V2 original (standardise on the fitting rows; each rule-embedding block replaced by its
        coordinates in an orthonormal basis of its distinct standardised fitting rows), built in place: non-rule
        blocks are written straight into one float64 matrix and standardised there; the rule block's distinct rows
        are its distinct rules (row = ER[rule]), so its basis comes from those rows."""
        D = self.D
        ranks, bases, mus, sds = [], [], [], []
        for b, w in zip(blocks, widths):
            if b in self.RULE_BLOCKS:
                ER = np.asarray(D._prompt_block(b), dtype=np.float64)
                rf = D.rix[fit]
                cnt = np.bincount(rf, minlength=len(ER)).astype(np.float64)
                mu = (cnt[:, None] * ER).sum(0) / cnt.sum()
                sd = np.sqrt((cnt[:, None] * (ER - mu) ** 2).sum(0) / cnt.sum())
                sd[sd < 1e-8] = 1.0
                Ustd = (ER[np.unique(rf)] - mu) / sd
                U = np.unique(Ustd, axis=0)
                _, sv, Vt = np.linalg.svd(U, full_matrices=False)
                V = Vt[sv > sv[0] * 1e-10].T
                bases.append(V)
                ranks.append(V.shape[1])
                mus.append(mu)
                sds.append(sd)
            else:
                bases.append(None)
                ranks.append(w)
                mus.append(None)
                sds.append(None)
        tot = sum(ranks)
        Xf = np.empty((len(fit), tot))
        Xh = np.empty((len(held), tot))
        c = 0
        for i, (b, w) in enumerate(zip(blocks, widths)):
            r = ranks[i]
            if bases[i] is None:
                for X_, rows in ((Xf, fit), (Xh, held)):
                    if len(rows):
                        D.block_rows(b, rows, out=X_[:, c:c + r])
                blk = Xf[:, c:c + r]
                mu = blk.mean(0)
                sd = blk.std(0)
                sd[sd < 1e-8] = 1.0
                blk -= mu
                blk /= sd
                if len(held):
                    Xh[:, c:c + r] -= mu
                    Xh[:, c:c + r] /= sd
                mus[i], sds[i] = mu, sd
            else:
                ER = np.asarray(D._prompt_block(b), dtype=np.float64)
                proj = ((ER - mus[i]) / sds[i]) @ bases[i]
                Xf[:, c:c + r] = proj[D.rix[fit]]
                if len(held):
                    Xh[:, c:c + r] = proj[D.rix[held]]
            c += r
        return Xf, Xh, np.concatenate(mus), np.concatenate(sds), ranks, bases

    def lam_vector(self, spec: Spec, st: dict, widths) -> np.ndarray:
        fixed = dict(spec.fixed)
        lam = []
        for (b, sl), w in zip(spec.blocks, widths):
            v = self.R["structure_lambda"] if sl == "S" else fixed[sl] if sl.startswith("F_") else st[sl]
            lam.append(np.full(w, float(v)))
        return np.concatenate(lam)

    # ---- point cache
    def pdir(self, spec: Spec):
        d = self.root / "points" / spec.key
        d.mkdir(parents=True, exist_ok=True)
        return d

    def cached(self, spec: Spec, st: dict) -> bool:
        return (self.pdir(spec) / f"{setting_key(st)}.npy").exists()

    def oof(self, spec: Spec, st: dict | None) -> np.ndarray:
        if st is None:  # "off": the frozen baseline's predictions
            raise RuntimeError("off is resolved by the caller")
        return np.load(self.pdir(spec) / f"{setting_key(st)}.npy")

    def evaluate(self, spec: Spec, settings: list[dict]) -> None:
        todo = [s for s in settings if not self.cached(spec, s)]
        if not todo:
            return
        t0 = time.time()
        order = sorted(todo, key=lambda s: tuple(-s.get(k, 0) for k in ("R", "X", "P")))  # strong -> weak
        blocks = [b for b, _ in spec.blocks]
        widths = self.D.widths(blocks)
        out = {setting_key(s): np.full(len(self.D.v), np.nan) for s in order}
        diag = []
        for f in range(5):
            for t in self.units():
                fit, held = self.rows(f, t)
                if self.o == 1 and self.fallback[(f, t)][0]:
                    _, v, n = self.fallback[(f, t)]
                    for s in order:
                        out[setting_key(s)][held] = (v + 0.5) / (n + 1)
                    continue
                Xf, Xh, mu, sd, weff, _ = self.prepare(blocks, widths, fit, held, (f, t))
                gram = ridge.Gram(Xf)
                init = None
                for s in order:
                    h = ridge.fit(Xf, self.D.v[fit], self.D.n[fit], self.lam_vector(spec, s, weff), mu, sd, init, gram)
                    init = np.r_[h.w, h.b]
                    out[setting_key(s)][held] = ridge.expit(Xh @ h.w + h.b)
                    diag.append({"fold": f, "type_id": t, "setting": setting_key(s), "n_iter": h.n_iter, "converged": h.converged,
                                 "hessian_jitter_steps": len(ridge.JITTER_EVENTS)})
                    ridge.JITTER_EVENTS.clear()
                del Xf, Xh, gram
        for k, p in out.items():
            assert np.isfinite(p).all(), (spec.name, k)
            np.save(self.pdir(spec) / f"{k}.npy", p)
        dj = self.pdir(spec) / "fits.jsonl"
        with open(dj, "a") as fh:
            for r in diag:
                fh.write(json.dumps(r) + "\n")
        write_json({"spec": spec.name, "blocks": spec.blocks, "fixed": spec.fixed, "widths": widths,
                    "version": VERSION}, self.pdir(spec) / "spec.json")
        print(json.dumps({"model": self.model, "obj": self.o, "spec": spec.name, "n_settings": len(order),
                          "wall_s": round(time.time() - t0, 1),
                          "not_converged": sum(not r["converged"] for r in diag)}), flush=True)

    # ---- scoring and selection
    def score(self, p: np.ndarray) -> float:
        if self.o == 1:
            s, n = ridge.log_loss(p, self.D.v, self.D.n)
            return s / n
        return float(((p - self.D.v) ** 2).mean())

    def fold_scores(self, p):
        out = []
        for f in range(5):
            m = self.D.fold == f
            if self.o == 1:
                s, n = ridge.log_loss(p[m], self.D.v[m], self.D.n[m])
                out.append(s / n)
            else:
                out.append(float(((p[m] - self.D.v[m]) ** 2).mean()))
        return out

    def select(self, spec: Spec, settings: list[dict], off: np.ndarray | None = None) -> dict:
        """Tie rule: rounded loss, then largest sum of log10 tuned penalties (off = +inf), then larger P (or X)."""
        cands = []
        for s in settings:
            p = self.oof(spec, s)
            loss = self.score(p)
            cands.append({"setting": s, "loss": loss, "rounded": round(loss, self.prec),
                          "logsum": sum(np.log10(v) for v in s.values()), "pen": s.get("P", s.get("X", 0.0))})
        if off is not None:
            loss = self.score(off)
            cands.append({"setting": None, "loss": loss, "rounded": round(loss, self.prec), "logsum": np.inf, "pen": np.inf})
        best = min(cands, key=lambda c: (c["rounded"], -c["logsum"], -c["pen"]))
        return {"setting": best["setting"], "loss": best["loss"],
                "grid": [{"setting": c["setting"], "loss": c["loss"]} for c in cands]}


# ---------------------------------------------------------------------------------------------- orchestration
def _grid(values, slots):
    if not slots:
        return [{}]
    out = [{}]
    for s in slots:
        out = [{**o, s: float(v)} for o in out for v in values[s]]
    return out


def _edge(sel_setting: dict | None, grids: dict, slot: str) -> str | None:
    if not sel_setting or slot not in sel_setting:
        return None
    v = sel_setting[slot]
    g = grids["X" if slot == "X" else slot]
    if v == min(g):
        return "low"
    if v == max(g) and slot != "X":
        return "high"
    return None


def run(model: str, o: int = 1, with_attention: bool = True, final: bool = False, with_xh: bool = False) -> dict:
    E = Engine(model, o)
    R = E.R
    base = {"P": list(R["prompt_lambda_grid"]), "R": list(R["rule_lambda_grid"])}
    state_path = E.root / "selection.json"
    log = {"model": model, "objective": o, "version": VERSION, "base_grids": base, "with_attention": with_attention}

    def layer_select(grids, prefix, rows_):
        res = {}
        for r in rows_:
            sp = Spec(prefix, ((f"{'X' if prefix == 'XH1' else ''}H{r}", "P"), ("S", "S")))
            E.evaluate(sp, _grid(grids, ("P",)))
            res[r] = E.select(sp, _grid(grids, ("P",)))
        best = min(rows_, key=lambda r: (round(res[r]["loss"], E.prec), rows_.index(r)))  # tie -> earlier (shallower) row
        return best, res

    def stage(grids):
        rows_ = R["hidden_rows"][model]
        xrows = R["hidden_rows"][OTHER[model]]
        row, lres = layer_select(grids, "H1", rows_)
        xrow, xres = layer_select(grids, "XH1", xrows) if with_xh else (None, {})
        E.D.drop(keep={"S", "EP", "ER", f"H{row}", f"XH{xrow}"})
        specs = make_specs(model, o, row, xrow)
        if not with_attention:
            specs.pop("RCAP")
        sel = {}
        for name, sp in specs.items():
            g = _grid(grids, sp.tuned)
            E.evaluate(sp, g)
            sel[name] = {"spec": sp, **E.select(sp, g)}
        return row, xrow, lres, xres, specs, sel

    def incremental(grids, specs, sel, row):
        out = {}
        inc = [("B1+H1", "B1", f"H{row}"), ("B2+H1", "B2", f"H{row}")]
        if with_attention:
            inc.append(("B2+A1", "B2", "A1" if o == 1 else "A1mean"))
        for name, bname, blk in inc:
            sp = incremental_spec(name, specs[bname], sel[bname]["setting"], blk)
            g = [{"X": float(v)} for v in grids["P"]]
            E.evaluate(sp, g)
            out[name] = {"spec": sp, **E.select(sp, g, off=E.oof(specs[bname], sel[bname]["setting"]))}
        return out

    # (1) + (2) base round
    grids = {k: list(v) for k, v in base.items()}
    row, xrow, lres, xres, specs, sel = stage(grids)
    inc = incremental(grids, specs, sel, row)
    # (3) boundary expansion, once
    hits = []
    for name, s in {**sel, **inc}.items():
        if name == "REF_structure":
            continue
        for slot in ("P", "R", "X"):
            e = _edge(s["setting"], {**grids, "X": grids["P"]}, slot)
            if e:
                hits.append({"predictor": name, "slot": "P" if slot == "X" else slot, "direction": e,
                             "selected": s["setting"][slot]})
    expansions = sorted({(h["slot"], h["direction"]) for h in hits})
    for slot, d in expansions:
        g = grids[slot]
        grids[slot] = sorted(g + [min(g) / 10 if d == "low" else max(g) * 10])
    if expansions:
        # (4) evaluate extra values, reselect; (5) incremental against the reselected baselines
        row, xrow, lres, xres, specs, sel = stage(grids)
        inc = incremental(grids, specs, sel, row)
    if not with_attention:
        return {"warmup": True, "hidden_row": row, "expansions": expansions}  # cache filling only; nothing written
    log.update({"final_grids": grids, "boundary_hits_base_round": hits,
                "expansions": [{"slot": s, "direction": d, "factor": 10} for s, d in expansions],
                "hidden_row": row, "hidden_row_scores": {str(r): v["loss"] for r, v in lres.items()},
                "other_generator_row": xrow, "other_generator_row_scores": {str(r): v["loss"] for r, v in xres.items()},
                "candidate_rows": HIDDEN_ROWS[model], "other_candidate_rows": HIDDEN_ROWS[OTHER[model]]})
    # edges after expansion (reported, never expanded again)
    post = []
    for name, s in {**sel, **inc}.items():
        for slot in ("P", "R", "X"):
            e = _edge(s["setting"], {**grids, "X": grids["P"]}, slot)
            if e:
                post.append({"predictor": name, "slot": slot, "direction": e, "note": "best among the tested values"})
    log["edges_after_expansion"] = post
    # selections, OOF parquet, tables
    pred_dir = RESULTS_V2 / "predictions" / model / f"obj{o}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    keys = ["prompt_id", "position", "rule_id", "type_id"] if o == 1 else ["prompt_id"]
    selected, grid_rows = {}, []
    for name, s in {**sel, **inc}.items():
        sp = s["spec"]
        df = E.D.U[keys].copy()
        df["fold"] = E.D.fold
        for gpt in s["grid"]:
            col = "off" if gpt["setting"] is None else setting_key(gpt["setting"])
            df[col] = E.oof(specs[sp.baseline], sel[sp.baseline]["setting"]) if gpt["setting"] is None else E.oof(sp, gpt["setting"])
            grid_rows.append({"model": model, "objective": o, "predictor": name, "setting": col, "cv_loss": gpt["loss"]})
        scol = "off" if s["setting"] is None else setting_key(s["setting"])
        df["selected"] = df[scol]
        df.to_parquet(pred_dir / f"{name}.parquet")
        selected[name] = {"setting": scol, "cv_loss": s["loss"], "fold_losses": E.fold_scores(df["selected"].values),
                          "blocks": [list(b) for b in sp.blocks], "fixed": dict(sp.fixed), "baseline": sp.baseline,
                          "n_features": int(sum(E.D.widths([b for b, _ in sp.blocks]))),
                          "hidden_row": row if name in ("H1", "H2", "B1+H1", "B2+H1") else xrow if name == "XH1" else None,
                          "note": ("CV did not select the block, so the gain is 0 by construction." if s["setting"] is None else None)}
    log["selected"] = selected
    if o == 1:
        fbt = E.fallback_table
        log["fallbacks_fired"] = fbt[fbt.fallback].to_dict("records")
    write_versioned_json(log, state_path)
    t = RESULTS_V2 / "tables"
    t.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(grid_rows).to_csv(t / f"cv_grid_{model}_obj{o}.csv", index=False)
    if o == 1:
        E.fallback_table.to_csv(t / f"fallbacks_{model}_obj1.csv", index=False)
    if final:
        refit(E, {**sel, **inc}, specs, sel)
    return log


def refit(E: Engine, allsel: dict, specs: dict, sel: dict) -> None:
    """Refit each selected predictor on all 50 TRAIN groups (fallback per type decided on all 50)."""
    mdir = RESULTS_V2 / "models" / E.model / f"obj{E.o}"
    mdir.mkdir(parents=True, exist_ok=True)
    for name, s in allsel.items():
        sp = s["spec"]
        st = s["setting"]
        if st is None:
            sp, st = specs[sp.baseline], sel[sp.baseline]["setting"]
        blocks = [b for b, _ in sp.blocks]
        widths = E.D.widths(blocks)
        heads = {}
        for t in E.units():
            fit, _ = E.rows("final", t)
            key = t or "shared"
            if E.o == 1 and E.fallback[("final", t)][0]:
                _, v, n = E.fallback[("final", t)]
                heads[key] = {"fallback": True, "p": (v + 0.5) / (n + 1)}
                continue
            Xf, _, mu, sd, weff, bases = E.prepare(blocks, widths, fit, np.zeros(0, int), ("final", t))
            h = ridge.fit(Xf, E.D.v[fit], E.D.n[fit], E.lam_vector(sp, st, weff), mu, sd)
            w, c = [], 0
            for V, we in zip(bases, weff):  # back to coefficients on the standardised original columns
                a = h.w[c:c + we]
                w.append(a if V is None else V @ a)
                c += we
            heads[key] = {"fallback": False, "mean": mu, "scale": sd, "w": np.concatenate(w), "b": h.b,
                          "converged": h.converged}
        np.savez(mdir / f"{name}.npz", **{f"{k}__{a}": np.asarray(v) for k, d in heads.items() for a, v in d.items()})
        write_versioned_json({"predictor": name, "setting": setting_key(st) if s["setting"] is not None else "off",
                              "fitted_as": sp.name, "blocks": [list(b) for b in sp.blocks], "fixed": dict(sp.fixed),
                              "widths": widths, "n_fitting_groups": 50,
                              "fallback_types": [k for k, d in heads.items() if d["fallback"]]}, mdir / f"{name}.json")


def refit_all(model: str) -> None:
    """Refit every selected predictor (incl. comparison 8) on all 50 TRAIN groups at its CV-selected setting."""
    from .extra import _parse
    E = Engine(model, 1)
    sel = json.loads((E.root / "selection.json").read_text())
    row = sel["hidden_row"]
    specs = make_specs(model, 1, row, None)
    st = {n: _parse(v["setting"]) for n, v in sel["selected"].items()}
    allsel = {n: {"spec": specs[n], "setting": st[n]} for n in specs if n in sel["selected"]}
    for name, bname, blk in (("B1+H1", "B1", f"H{row}"), ("B2+H1", "B2", f"H{row}"), ("B2+A1", "B2", "A1")):
        sp = incremental_spec(name, specs[bname], st[bname], blk)
        allsel[name] = {"spec": sp, "setting": st[name] or None}
    basesel = {n: {"setting": st[n]} for n in specs if n in st}
    refit(E, allsel, specs, basesel)


def freeze(model: str) -> dict:
    """Refit every selected predictor on all 50 TRAIN groups and record sha256 hashes (TEST prespec step 1)."""
    import hashlib
    refit_all(model)
    mdir = RESULTS_V2 / "models" / model / "obj1"
    h = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(mdir.glob("*.npz"))}
    write_json({**h, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, mdir / "HASHES.json")
    return h
