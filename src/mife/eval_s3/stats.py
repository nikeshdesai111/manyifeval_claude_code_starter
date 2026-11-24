"""Hierarchical weights and the paired 22-pool cluster bootstrap (plan §§5.1, 6.3, 7).

Every Evaluation 2 estimate is a weighted mean whose weights split equally at each level of a declared hierarchy:
    stratum -> group (pool) -> level_1 -> ... -> comparison row
`group_table` returns the group-level means G[stratum, group] (equal split below the group). A summary over strata
(e.g. equal-k macro) is the equal-weight mean over strata of the equal-weight mean over groups that support the
stratum. In a bootstrap replicate, group g carries its multinomial count m_g, so the group mean for stratum s is
sum_g m_g G[s,g] / sum_g m_g 1[g supports s]: whole pools are resampled jointly for every stratum, predictor and
quantity, so paired contrasts use identical draws. A stratum with no supporting group in a replicate is skipped in that
replicate's macro mean (counted in `n_degenerate`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

B = 10_000
SEED = 20261203


class Boot:
    def __init__(self, groups: list[str], b: int = B, seed: int = SEED):
        self.groups = sorted(groups)
        self.G = len(self.groups)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, self.G, size=(b, self.G))
        self.M = np.zeros((b, self.G))
        np.add.at(self.M, (np.repeat(np.arange(b), self.G), idx.ravel()), 1.0)
        self.loo = 1.0 - np.eye(self.G)  # leave-one-group-out weights
        self.one = np.ones((1, self.G))

    def weights(self, kind: str):
        return {"point": self.one, "boot": self.M, "loo": self.loo}[kind]


def group_table(df: pd.DataFrame, values: list[str], strata: list[str], levels: list[str]) -> dict:
    """Equal-split means. df rows are comparisons; levels are the columns below the group, outermost first.
    Returns {value: DataFrame indexed by stratum tuple, columns = group ids (NaN = unsupported)}."""
    keys = strata + ["group_id"] + levels
    m = df.groupby(keys, observed=True, sort=False)[values].mean()
    for lv in range(len(levels), 0, -1):
        m = m.groupby(level=list(range(len(strata) + lv)), observed=True, sort=False).mean()
    out = {}
    for v in values:
        s = m[v]
        t = s.unstack("group_id") if strata else s.to_frame().T
        out[v] = t
    return out


def macro(tabs: dict, boot: Boot, expr, kind: str = "boot", strata_mean: bool = True, post=None) -> np.ndarray:
    """Apply `expr(dict_of_stratum_means)` per replicate after the group step; returns (b,) (macro over strata) or
    (b, n_strata). `tabs` from group_table; all values share stratum/group support."""
    W = boot.weights(kind)
    means = {}
    for v, t in tabs.items():
        t = t.reindex(columns=boot.groups)
        X = t.to_numpy(dtype=float)
        S = ~np.isnan(X)
        num = W @ np.nan_to_num(X).T
        den = W @ S.T.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            means[v] = np.where(den > 0, num / den, np.nan)
    r = expr(means)
    if strata_mean and r.ndim == 2:
        with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            r = np.nanmean(r, axis=1)
    return post(r) if post is not None else r


def summarize(point: float, boot: np.ndarray, loo: np.ndarray | None = None, n_groups: int | None = None) -> dict:
    b = boot[np.isfinite(boot)]
    out = {"est": float(point), "lo": float(np.quantile(b, 0.025)) if len(b) else np.nan,
           "hi": float(np.quantile(b, 0.975)) if len(b) else np.nan, "n_boot_valid": int(len(b))}
    if loo is not None:
        L = loo[np.isfinite(loo)]
        g = len(L)
        if g > 2:
            jk_se = np.sqrt((g - 1) / g * ((L - L.mean()) ** 2).sum())
            tq = sps.t.ppf(0.975, g - 1)
            out.update({"t_lo": float(point - tq * jk_se), "t_hi": float(point + tq * jk_se), "t_df": g - 1,
                        "loo_min": float(L.min()), "loo_max": float(L.max())})
    if n_groups is not None:
        out["n_groups"] = int(n_groups)
    return out


def estimate(tabs: dict, boot: Boot, expr, strata_mean: bool = True, with_loo: bool = True, post=None):
    pt = macro(tabs, boot, expr, "point", strata_mean, post)
    bt = macro(tabs, boot, expr, "boot", strata_mean, post)
    lo = macro(tabs, boot, expr, "loo", strata_mean, post) if with_loo else None
    first = next(iter(tabs.values()))
    support = (~first.reindex(columns=boot.groups).isna()).sum(1).to_numpy()
    if strata_mean:
        return summarize(pt[0], bt, lo, int((~first.reindex(columns=boot.groups).isna()).any(axis=0).sum()))
    res = []
    for s in range(pt.shape[1]):
        res.append(summarize(pt[0, s], bt[:, s], None if lo is None else lo[:, s], int(support[s])))
    return res


def strata_index(tabs: dict) -> list:
    return list(next(iter(tabs.values())).index)
