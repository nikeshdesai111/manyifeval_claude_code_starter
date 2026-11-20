"""Evaluation metrics, written so every one can be bootstrapped EXACTLY over groups.

Each metric is a function of group-level sufficient statistics; a bootstrap replicate is a vector of group weights
w (multinomial counts), so statistics are weighted sums and nothing is approximated by resampling prompts.

Objective 1 (prompt-rule rows with v violations out of n responses, prediction p):
  NLL      sum[-v log p - (n - v) log(1 - p)] / sum n                      (per completion; primary)
  Brier    sum[v (1 - p)^2 + (n - v) p^2] / sum n                           (per-response binary outcomes)
  ECE      sum_b (N_b / N) |obs_b - pred_b|, responses weighted, bins = deciles of the configuration's TRAIN
           out-of-fold CV predictions (fixed before the test set is read)
  AUROC    on per-response binary outcomes (each row: v positives, n - v negatives, all scored p; ties count 1/2),
           via the cross-group concordance matrix C[g, h] (positives in g vs negatives in h)
  RiskTop / RiskBottom   observed violation rate among rows in the top / bottom decile of predicted risk; decile
           thresholds are taken from the full evaluation set and held fixed within the bootstrap
Objective 2 (prompts, label q, prediction p): MAE, RMSE, ECE (bins as above, prompts weighted), Pearson (weighted
  moments) and Spearman (recomputed on each replicate's expanded sample).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def gindex(groups: np.ndarray):
    g, inv = np.unique(groups, return_inverse=True)
    return g, inv


def _gsum(x, inv, G):
    return np.bincount(inv, weights=x, minlength=G)


def bins_from(train_oof: np.ndarray, n_bins: int = 10) -> np.ndarray:
    e = np.unique(np.quantile(train_oof, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return e


# ------------------------------------------------------------------------------------------------ Objective 1
class Obj1Stats:
    """Group-level sufficient statistics for one configuration's predictions on one row set."""

    def __init__(self, p, v, n, groups, edges):
        self.groups, inv = gindex(groups)
        G = len(self.groups)
        p = np.clip(p, EPS, 1 - EPS)
        self.N = _gsum(n, inv, G)
        self.nll = _gsum(-(v * np.log(p) + (n - v) * np.log1p(-p)), inv, G)
        self.brier = _gsum(v * (1 - p) ** 2 + (n - v) * p ** 2, inv, G)
        b = np.digitize(p, edges)
        nb = len(edges) + 1
        self.bin_n = np.zeros((G, nb)); self.bin_v = np.zeros((G, nb)); self.bin_p = np.zeros((G, nb))
        np.add.at(self.bin_n, (inv, b), n); np.add.at(self.bin_v, (inv, b), v); np.add.at(self.bin_p, (inv, b), n * p)
        # AUROC concordance: positives of group g vs negatives of group h
        order = np.argsort(p, kind="mergesort")
        ps, vs, ns, gs = p[order], v[order], n[order], inv[order]
        neg = ns - vs
        uniq, start = np.unique(ps, return_index=True)
        # cumulative negatives per group strictly below each unique score, and at equal score
        cum = np.zeros((len(uniq) + 1, G))
        tie = np.zeros((len(uniq), G))
        grp_of = np.repeat(np.arange(len(uniq)), np.diff(np.append(start, len(ps))))
        np.add.at(tie, (grp_of, gs), neg)
        cum[1:] = np.cumsum(tie, axis=0)
        below = cum[grp_of]                         # negatives strictly below, per group, for each row
        eq = tie[grp_of]
        self.C = np.zeros((G, G))
        np.add.at(self.C, gs, vs[:, None] * (below + 0.5 * eq))
        self.P = _gsum(v, inv, G)
        self.Nn = self.N - self.P
        # risk deciles (thresholds on the full set)
        lo_t, hi_t = np.quantile(p, 0.1), np.quantile(p, 0.9)
        top, bot = p >= hi_t, p <= lo_t
        self.top_v, self.top_n = _gsum(v * top, inv, G), _gsum(n * top, inv, G)
        self.bot_v, self.bot_n = _gsum(v * bot, inv, G), _gsum(n * bot, inv, G)

    def metrics(self, w: np.ndarray) -> dict:
        """w: (B, G) or (G,) group weights."""
        w = np.atleast_2d(w)
        N = w @ self.N
        bn, bv, bp = np.einsum("bg,gk->bk", w, self.bin_n), np.einsum("bg,gk->bk", w, self.bin_v), np.einsum("bg,gk->bk", w, self.bin_p)
        with np.errstate(invalid="ignore", divide="ignore"):
            ece = np.nansum(np.abs(bv - bp), axis=1) / N
            auc = np.einsum("bg,gh,bh->b", w, self.C, w) / ((w @ self.P) * (w @ self.Nn))
            out = {"nll": (w @ self.nll) / N, "brier": (w @ self.brier) / N, "ece": ece, "auroc": auc,
                   "risk_top": (w @ self.top_v) / (w @ self.top_n), "risk_bottom": (w @ self.bot_v) / (w @ self.bot_n)}
        return out


# ------------------------------------------------------------------------------------------------ Objective 2
class Obj2Stats:
    def __init__(self, p, q, groups, edges):
        self.groups, inv = gindex(groups)
        self.inv, G = inv, len(self.groups)
        self.p, self.q = p, q
        self.cnt = _gsum(np.ones_like(q), inv, G)
        self.ae, self.se = _gsum(np.abs(p - q), inv, G), _gsum((p - q) ** 2, inv, G)
        self.sx, self.sy, self.sxx, self.syy, self.sxy = (_gsum(a, inv, G) for a in (p, q, p * p, q * q, p * q))
        b = np.digitize(p, edges)
        nb = len(edges) + 1
        self.bin_c = np.zeros((G, nb)); self.bin_q = np.zeros((G, nb)); self.bin_p = np.zeros((G, nb))
        np.add.at(self.bin_c, (inv, b), 1.0); np.add.at(self.bin_q, (inv, b), q); np.add.at(self.bin_p, (inv, b), p)
        self.rows_of = [np.nonzero(inv == g)[0] for g in range(G)]

    def metrics(self, w: np.ndarray, spearman: bool = True) -> dict:
        from scipy.stats import rankdata
        w = np.atleast_2d(w)
        n = w @ self.cnt
        mx, my = (w @ self.sx) / n, (w @ self.sy) / n
        cov = (w @ self.sxy) / n - mx * my
        vx, vy = (w @ self.sxx) / n - mx ** 2, (w @ self.syy) / n - my ** 2
        bq, bp = np.einsum("bg,gk->bk", w, self.bin_q), np.einsum("bg,gk->bk", w, self.bin_p)
        out = {"mae": (w @ self.ae) / n, "rmse": np.sqrt((w @ self.se) / n), "mse": (w @ self.se) / n,
               "ece": np.abs(bq - bp).sum(1) / n, "pearson": cov / np.sqrt(vx * vy)}
        if spearman:
            rho = np.empty(len(w))
            for i, wi in enumerate(w):
                idx = np.concatenate([np.tile(self.rows_of[g], int(c)) for g, c in enumerate(wi) if c > 0])
                rx, ry = rankdata(self.p[idx]), rankdata(self.q[idx])
                rho[i] = np.corrcoef(rx, ry)[0, 1]
            out["spearman"] = rho
        return out


def summarize(boot: np.ndarray, point: float) -> dict:
    return {"est": float(point), "lo": float(np.nanpercentile(boot, 2.5)), "hi": float(np.nanpercentile(boot, 97.5))}


def boot_p(diff_boot: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: difference = 0 (percentile method)."""
    d = diff_boot[np.isfinite(diff_boot)]
    if len(d) == 0:
        return float("nan")
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def holm(pvals: pd.Series) -> pd.Series:
    order = pvals.sort_values().index
    m = len(order)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return pd.Series(adj).reindex(pvals.index)
