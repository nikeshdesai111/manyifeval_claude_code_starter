"""V2 ridge-logistic heads with block-specific penalties (CPU, float64, deterministic).

Solver: damped Newton with the exact Hessian, warm-started along the penalty path.

Objective 1 (per rule type head), summed binomial NLL with counts v (violated) and n (completions):
    J(w, b) = sum_i [n_i softplus(z_i) - v_i z_i] + 1/2 sum_j lam_j w_j^2,   z = X w + b
implemented, equivalently, as the mean per completion with penalty lam / N (N = sum_i n_i of the head's fitting
rows):  J / N = mean NLL per completion + 1/(2N) sum_j lam_j w_j^2.
Objective 2 (one shared head), fractional logit: soft label q_i, every prompt weight 1 (n_i = 1, v_i = q_i), N = number
of fitting prompts.  The intercept is never penalised.  Features are standardised on the fitting rows only
(constant columns -> scale 1).  Fits are warm-started along the penalty path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12
MAX_NEWTON = 50
NEWTON_GTOL = 1e-9
DEC_TOL = 1e-14


@dataclass
class Fitted:
    mean: np.ndarray
    scale: np.ndarray
    w: np.ndarray
    b: float
    lam: np.ndarray | None = None
    n_iter: int = 0
    converged: bool = True

    def logit(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.scale) @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return expit(self.logit(X))


def expit(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def softplus(z):
    return np.logaddexp(0.0, z)


def standardise(X: np.ndarray):
    mu = X.mean(0, dtype=np.float64)
    sd = X.std(0, dtype=np.float64)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def fit(Xs: np.ndarray, v: np.ndarray, n: np.ndarray, lam: np.ndarray, mean, scale, init=None,
        Xs32: np.ndarray | None = None) -> Fitted:
    """Xs: ALREADY standardised fitting matrix (float64). lam: per-column penalty (summed-loss scale).
    Damped Newton with the exact Hessian (assembled in float32 on Accelerate, solved by float64 Cholesky);
    gradient and objective in float64, so the fixed point is the exact float64 optimum."""
    N = float(n.sum())
    lam_n = lam / N
    d = Xs.shape[1]
    Xs32 = Xs.astype(np.float32) if Xs32 is None else Xs32
    w_, y_ = n / N, v / N
    x = np.zeros(d + 1)
    if init is None:
        rate = np.clip(v.sum() / N, 1e-6, 1 - 1e-6)
        x[-1] = np.log(rate / (1 - rate))
    else:
        x[:] = init

    def f(x):
        z = Xs @ x[:-1] + x[-1]
        loss = float((w_ * softplus(z)).sum() - (y_ * z).sum() + 0.5 * (lam_n * x[:-1] ** 2).sum())
        return loss, z
    loss, z = f(x)
    ok, k = False, 0
    for k in range(1, MAX_NEWTON + 1):
        p = expit(z)
        r = w_ * p - y_
        g = np.r_[Xs.T @ r + lam_n * x[:-1], r.sum()]
        if np.abs(g).max() < NEWTON_GTOL:
            ok, k = True, k - 1
            break
        s = (w_ * p * (1 - p))
        Xw = Xs32 * np.sqrt(s).astype(np.float32)[:, None]
        H = np.empty((d + 1, d + 1))
        H[:d, :d] = Xw.T @ Xw
        H[:d, :d][np.diag_indices(d)] += lam_n
        H[:d, d] = H[d, :d] = Xs.T @ s
        H[d, d] = s.sum()
        step = _solve(H, g)
        dec = float(g @ step)
        t = 1.0
        while True:
            xn = x - t * step
            ln, zn = f(xn)
            if ln <= loss - 1e-4 * t * dec or t < 1e-10:
                break
            t *= 0.5
        x, loss, z = xn, ln, zn
        if 0.5 * dec < DEC_TOL:
            ok = True
            break
    return Fitted(mean, scale, x[:-1].copy(), float(x[-1]), lam, k, ok)


def _solve(H, g):
    """Newton direction H^-1 g. If float32 rounding in the assembled Hessian makes it numerically indefinite (nearly
    collinear structure columns under a weak penalty), add the smallest diagonal jitter (1e-10 x mean diagonal,
    x10 steps) that allows a Cholesky factorisation. The jittered direction is still a descent direction and
    convergence is judged on the exact float64 gradient, so the optimum is unchanged."""
    from numpy.linalg import LinAlgError
    from scipy.linalg import cho_factor, cho_solve
    jit, base = 0.0, float(np.mean(np.diag(H)))
    while True:
        try:
            Hj = H if jit == 0 else H + jit * np.eye(len(H))
            return cho_solve(cho_factor(Hj, check_finite=False), g, check_finite=False)
        except LinAlgError:
            jit = base * 1e-10 if jit == 0 else jit * 10
            JITTER_EVENTS.append(jit / base)
            if jit > base * 1e-2:
                raise


JITTER_EVENTS: list = []


def log_loss(p, v, n) -> tuple[float, float]:
    """(summed NLL, summed n): mean per completion = ratio. Objective 2: v = q, n = 1 -> cross-entropy per prompt."""
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(v * np.log(p) + (n - v) * np.log1p(-p)).sum()), float(n.sum())
