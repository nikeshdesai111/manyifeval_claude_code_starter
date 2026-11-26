"""Search the frozen relabelings for the k = 6 and k = 10 orderings (run once; output in orderings_k6_k10.json).

    python pilots/high_k/code/search_orderings.py 6 6000 6 0.999
    python pilots/high_k/code/search_orderings.py 10 25000 2 0.9997
k - 1 Williams squares of order k (k = 6, 10 are even, so one square already balances position and immediate
neighbours exactly). Square 1 uses the identity labelling; squares 2..k-1 get a symbol relabelling found by simulated
annealing (row-wise swaps, seeds 0..n-1) that minimises the squared deviation of every (position pair, ordered rule
pair) count from 1, i.e. moves the k(k-1) rows as close to a strength-two index-1 COA (impossible at k = 6, 10) as
possible. Relabelling a Williams square keeps it a Williams square, so position and neighbour balance stay exact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def williams(n: int) -> np.ndarray:
    first, lo, hi = [0], 1, n - 1
    while len(first) < n:
        first.append(lo)
        lo += 1
        if len(first) < n:
            first.append(hi)
            hi -= 1
    return np.array([[(x + i) % n for x in first] for i in range(n)])


def pair_dev(R: np.ndarray, k: int) -> tuple[float, np.ndarray]:
    pair = np.zeros((k, k, k, k), int)
    for r in R:
        pair[np.arange(k)[:, None], np.arange(k)[None, :], r[:, None], r[None, :]] += 1
    m = ~np.eye(k, dtype=bool)
    P = pair[m][:, m]
    return float(((P - len(R) / (k * (k - 1))) ** 2).sum()), P


def build(k: int, labs) -> np.ndarray:
    W = williams(k)
    return np.vstack([np.asarray(l)[W] for l in labs])


def search(k: int, iters: int, seeds: int, decay: float) -> dict:
    best = None
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        labs = [np.arange(k)] + [rng.permutation(k) for _ in range(k - 2)]
        cur, _ = pair_dev(build(k, labs), k)
        T, bs = 5.0, (cur, [l.copy() for l in labs])
        for _ in range(iters):
            j = rng.integers(1, k - 1)
            l2 = [l.copy() for l in labs]
            x, y = rng.choice(k, 2, replace=False)
            l2[j][[x, y]] = l2[j][[y, x]]
            new, _ = pair_dev(build(k, l2), k)
            if new <= cur or rng.random() < np.exp((cur - new) / T):
                labs, cur = l2, new
                if cur < bs[0]:
                    bs = (cur, [l.copy() for l in labs])
            T = max(0.02, T * decay)
        if best is None or bs[0] < best[0]:
            best = (bs[0], bs[1], seed)
    _, P = pair_dev(build(k, best[1]), k)
    return {"k": k, "n_rows": k * (k - 1), "labels": [l.tolist() for l in best[1]], "seed": best[2],
            "iters": iters, "seeds": seeds, "decay": decay, "pair_sq_dev": best[0],
            "pair_count_histogram": np.bincount(P.ravel()).tolist()}


if __name__ == "__main__":
    k, iters, seeds, decay = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    out = Path(__file__).with_name("orderings_k6_k10.json")
    d = json.loads(out.read_text()) if out.exists() else {}
    d[str(k)] = search(k, iters, seeds, decay)
    out.write_text(json.dumps(d, indent=1))
    print(json.dumps({x: d[str(k)][x] for x in ("k", "seed", "pair_sq_dev", "pair_count_histogram")}))
