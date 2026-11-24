"""Synthetic validation of the noise-adjusted RMS estimator and its 22-pool bootstrap interval (plan §§5.2, 5.5, 5.8,
6.2). No TEST data are read.

Design per simulated dataset: 22 pools x C contexts; each comparison e is a difference of two cell rates; a cell
averages m prompts (m = 1 for companion/task pairs, m = k-1 for position cells), each with 20 binomial responses.
Latent per-prompt risks: base ~ Beta(0.6, 1.6) per context (many near 0, as in the data), prompt-level jitter, and a
true pair effect Delta_e ~ N(0, tau^2) on the logit scale, so tau = 0 is the exact null.
Reported: mean of the corrected squared estimate vs the true weighted mean of Delta_e^2 (bias), mean RMS estimate vs
true RMS, share of zero-truncated estimates, and coverage of the 95% percentile pool-bootstrap interval for the true RMS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.special import expit, logit


def _sim(rng, tau, m, jit=0.3, n_pools=22, n_ctx=40, n=20):
    base = rng.beta(0.6, 1.6, size=(n_pools, n_ctx))
    base = np.clip(base, 0.01, 0.99)
    eff = rng.normal(0, tau, size=(n_pools, n_ctx))
    pa = expit(logit(base)[..., None] + rng.normal(0, jit, size=(n_pools, n_ctx, m)))
    pb = expit(logit(base)[..., None] + eff[..., None] + rng.normal(0, jit, size=(n_pools, n_ctx, m)))
    true_d = pb.mean(-1) - pa.mean(-1)
    va, vb = rng.binomial(n, pa) / n, rng.binomial(n, pb) / n
    d = vb.mean(-1) - va.mean(-1)
    V = ((va * (1 - va) / (n - 1)).sum(-1) + (vb * (1 - vb) / (n - 1)).sum(-1)) / m ** 2
    return true_d, d, V


def run(n_sims: int = 300, b: int = 2000, seed: int = 20261203) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for m, tau, jit in [(m, t, j) for m in (1, 2, 4) for t, j in ((0.0, 0.0), (0.05, 0.0), (0.0, 0.3), (0.4, 0.3), (0.8, 0.3))]:
        if True:
            est_sq, true_sq, est_rms, true_rms, cover, trunc, cover_sq = [], [], [], [], [], [], []
            for _ in range(n_sims):
                td, d, V = _sim(rng, tau, m, jit)
                gq = (d ** 2 - V).mean(1)            # per-pool corrected squared mean (equal weights)
                tq = (td ** 2).mean()
                q = gq.mean()
                est_sq.append(q); true_sq.append(tq)
                r = np.sqrt(max(q, 0)); est_rms.append(r); true_rms.append(np.sqrt(tq)); trunc.append(q <= 0)
                idx = rng.integers(0, 22, size=(b, 22))
                bq = gq[idx].mean(1)
                br = np.sqrt(np.maximum(bq, 0))
                lo, hi = np.quantile(br, [0.025, 0.975])
                cover.append(lo - 1e-12 <= np.sqrt(tq) <= hi + 1e-12)
                lq, hq = np.quantile(bq, [0.025, 0.975])
                cover_sq.append(lq <= tq <= hq)
            rows.append({"cell_prompts_m": m, "tau_logit": tau, "prompt_jitter_logit": jit, "true_rms_pp": 100 * np.mean(true_rms),
                         "mean_rms_est_pp": 100 * np.mean(est_rms), "true_sq_x1e4": 1e4 * np.mean(true_sq),
                         "mean_corrected_sq_x1e4": 1e4 * np.mean(est_sq),
                         "naive_sq_bias_x1e4": None, "share_truncated_at_0": np.mean(trunc),
                         "coverage_rms_95": np.mean(cover), "coverage_sq_95": np.mean(cover_sq), "n_sims": n_sims})
    return pd.DataFrame(rows).drop(columns="naive_sq_bias_x1e4")


if __name__ == "__main__":
    from .paths import EVAL_DIR
    out = EVAL_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df = run()
    df.to_csv(out / "noise_validation_synthetic.csv", index=False)
    print(df.to_string())
