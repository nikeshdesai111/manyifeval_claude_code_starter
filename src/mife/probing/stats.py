"""Cluster (group) bootstrap. The resampling unit is always the three-task GROUP, never a prompt or response:
each replicate draws groups with replacement (multinomial weights over the groups present), and every statistic is
recomputed from group-level sums under those weights."""
from __future__ import annotations

import numpy as np
import pandas as pd


def group_weights(n_groups: int, B: int, seed: int) -> np.ndarray:
    """(B, n_groups) multinomial resampling counts; each row sums to n_groups."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=B).astype(np.float64)


def _pivot(df: pd.DataFrame, group: str, by: list[str], cols: list[str]):
    g = df.groupby([group] + by, observed=True)[cols].sum()
    cells = g.reset_index()[by].drop_duplicates().reset_index(drop=True) if by else pd.DataFrame(index=[0])
    groups = np.sort(df[group].unique())
    mats = {}
    for c in cols:
        s = g[c].unstack(by) if by else g[c].to_frame()
        s = s.reindex(index=groups).fillna(0.0)
        if by:
            s = s.reindex(columns=pd.MultiIndex.from_frame(cells) if len(by) > 1 else cells[by[0]]).fillna(0.0)
        mats[c] = s.values.astype(np.float64)
    return groups, cells, mats


def ratio_ci(df: pd.DataFrame, num: str, den: str, by: list[str], group: str = "group_id", B: int = 2000,
             seed: int = 0, W: np.ndarray | None = None) -> pd.DataFrame:
    """Ratio sum(num)/sum(den) per cell of `by`, with a 95% percentile CI from the group bootstrap."""
    groups, cells, m = _pivot(df, group, by, [num, den])
    V, N = m[num], m[den]
    W = group_weights(len(groups), B, seed) if W is None else W
    with np.errstate(invalid="ignore", divide="ignore"):
        est = V.sum(0) / N.sum(0)
        boot = (W @ V) / (W @ N)
    out = cells.copy()
    out["est"] = est
    out["lo"], out["hi"] = np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)
    out["n_groups"] = (N > 0).sum(0)
    out[f"sum_{den}"] = N.sum(0)
    return out


def paired_ratio_diff_ci(a: pd.DataFrame, b: pd.DataFrame, num: str, den: str, by: list[str], group: str = "group_id",
                         B: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Difference of two ratios (a - b) over the SAME groups, resampled jointly (paired group bootstrap)."""
    ga, ca, ma = _pivot(a, group, by, [num, den])
    gb, cb, mb = _pivot(b, group, by, [num, den])
    assert (ga == gb).all() and ca.equals(cb)
    W = group_weights(len(ga), B, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        est = ma[num].sum(0) / ma[den].sum(0) - mb[num].sum(0) / mb[den].sum(0)
        boot = (W @ ma[num]) / (W @ ma[den]) - (W @ mb[num]) / (W @ mb[den])
    out = ca.copy()
    out["est"], out["lo"], out["hi"] = est, np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)
    return out


def statistic_ci(df: pd.DataFrame, fn, group: str = "group_id", B: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Any statistic fn(df) with a group-bootstrap 95% CI (replicates rebuild df from resampled groups)."""
    groups = np.sort(df[group].unique())
    parts = {g: d for g, d in df.groupby(group)}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        pick = rng.choice(groups, len(groups), replace=True)
        vals.append(fn(pd.concat([parts[g] for g in pick], ignore_index=True)))
    return float(fn(df)), float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))
