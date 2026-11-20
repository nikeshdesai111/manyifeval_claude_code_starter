"""Prediction heads and losses (CPU, float32, deterministic seeds).

Objective 1 loss: aggregated binomial NLL with n = 20, as the mean per completion:
    L = sum_i [n_i softplus(z_i) - v_i z_i] / sum_i n_i   (= -sum[v log p + (n - v) log(1 - p)] / sum n)
Objective 2 loss: sigmoid-bounded MSE, mean over prompts: mean_i (sigmoid(z_i) - q_i)^2.
Penalty: (l2 / 2) * ||W||^2 over weight matrices (not biases), added to the mean loss.
Features are standardised per head on its own fitting rows (constant columns -> scale 1).
Linear heads: full-batch L-BFGS (strong Wolfe) to convergence. MLP heads: 1 hidden layer (128, ReLU), Adam
lr 1e-3, mini-batches of 1,024 rows, 30 epochs; per-epoch held-out losses are returned so CV can choose the epoch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

LBFGS_MAX_ITER = 500
EPS = 1e-6


def binom_nll_np(p: np.ndarray, v: np.ndarray, n: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(v * np.log(p) + (n - v) * np.log1p(-p)).sum() / n.sum())


def mse_np(p: np.ndarray, q: np.ndarray) -> float:
    return float(((p - q) ** 2).mean())


def _loss_t(z: torch.Tensor, y: torch.Tensor, n: torch.Tensor | None, objective: int) -> torch.Tensor:
    if objective == 1:
        return (n * nn.functional.softplus(z) - y * z).sum() / n.sum()
    return ((torch.sigmoid(z) - y) ** 2).mean()


@dataclass
class Scaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Scaler":
        mu = X.mean(0, dtype=np.float64).astype(np.float32)
        sd = X.std(0, dtype=np.float64).astype(np.float32)
        sd[sd < 1e-8] = 1.0
        return cls(mu, sd)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale


@dataclass
class Head:
    kind: str
    l2: float
    scaler: Scaler
    state: dict
    epochs: int | None = None
    history: list = field(default_factory=list)

    def module(self) -> nn.Module:
        d = self.scaler.mean.shape[0]
        m = _make(self.kind, d)
        m.load_state_dict(self.state)
        return m

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        m = self.module().eval()
        z = m(torch.from_numpy(np.ascontiguousarray(self.scaler(X), dtype=np.float32))).squeeze(-1)
        return torch.sigmoid(z).numpy().astype(np.float64)


def _make(kind: str, d: int) -> nn.Module:
    if kind == "linear":
        return nn.Linear(d, 1)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 1))
    raise ValueError(kind)


def _penalty(m: nn.Module, l2: float) -> torch.Tensor:
    return 0.5 * l2 * sum((p ** 2).sum() for name, p in m.named_parameters() if name.endswith("weight"))


def _init_bias(m: nn.Module, y: np.ndarray, n: np.ndarray | None, objective: int) -> None:
    rate = float(y.sum() / n.sum()) if objective == 1 else float(y.mean())
    rate = min(max(rate, 1e-4), 1 - 1e-4)
    last = m if isinstance(m, nn.Linear) else m[-1]
    with torch.no_grad():
        last.bias.fill_(float(np.log(rate / (1 - rate))))


def fit_linear(X, y, n, objective: int, l2: float, init: dict | None = None, scaler: Scaler | None = None) -> Head:
    sc = scaler or Scaler.fit(X)
    Xt = torch.from_numpy(np.ascontiguousarray(sc(X), dtype=np.float32))
    yt = torch.from_numpy(y.astype(np.float32))
    nt = None if n is None else torch.from_numpy(n.astype(np.float32))
    m = nn.Linear(X.shape[1], 1)
    with torch.no_grad():
        m.weight.zero_()
    _init_bias(m, y, n, objective)
    if init is not None:
        m.load_state_dict(init)
    opt = torch.optim.LBFGS(m.parameters(), lr=1.0, max_iter=LBFGS_MAX_ITER, tolerance_grad=1e-7,
                            tolerance_change=1e-10, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = _loss_t(m(Xt).squeeze(-1), yt, nt, objective) + _penalty(m, l2)
        loss.backward()
        return loss
    opt.step(closure)
    return Head("linear", l2, sc, {k: v.detach().clone() for k, v in m.state_dict().items()})


def fit_mlp(X, y, n, objective: int, l2: float, epochs: int, seed: int, lr: float = 1e-3, batch: int = 1024,
            X_eval=None, y_eval=None, n_eval=None, scaler: Scaler | None = None):
    """Returns (head after `epochs`, per-epoch held-out predictions list or None)."""
    sc = scaler or Scaler.fit(X)
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    m = _make("mlp", X.shape[1])
    _init_bias(m, y, n, objective)
    Xt = torch.from_numpy(np.ascontiguousarray(sc(X), dtype=np.float32))
    yt = torch.from_numpy(y.astype(np.float32))
    nt = None if n is None else torch.from_numpy(n.astype(np.float32))
    Xe = None if X_eval is None else torch.from_numpy(np.ascontiguousarray(sc(X_eval), dtype=np.float32))
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    preds = [] if Xe is not None else None
    N = Xt.shape[0]
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(N, generator=g)
        for s in range(0, N, batch):
            ix = perm[s:s + batch]
            opt.zero_grad()
            loss = _loss_t(m(Xt[ix]).squeeze(-1), yt[ix], None if nt is None else nt[ix], objective) + _penalty(m, l2)
            loss.backward()
            opt.step()
        if Xe is not None:
            m.eval()
            with torch.no_grad():
                preds.append(torch.sigmoid(m(Xe).squeeze(-1)).numpy().astype(np.float64))
    head = Head("mlp", l2, sc, {k: v.detach().clone() for k, v in m.state_dict().items()}, epochs=epochs)
    return head, preds


def score(p: np.ndarray, y: np.ndarray, n: np.ndarray | None, objective: int) -> float:
    return binom_nll_np(p, y, n) if objective == 1 else mse_np(p, y)
