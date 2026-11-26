"""Exact strength-two COA orderings for k = 1..10 (pilots/high_k/PRESPEC.md).

k <= 5: the main design's sharply 2-transitive templates (identity, S2, S3, A4, affine mod 5), index 1.
k = 7, 8, 9: AGL(1, q) over GF(q) (sharply 2-transitive), index 1, k(k-1) rows.
k = 6, 10: no index-1 COA exists (it would give a projective plane of order 6 or 10). To keep the row count at
k(k-1) (30 and 90, so it grows with k), these use k - 1 stacked Williams squares with frozen relabelings
(orderings_k6_k10.json, from search_orderings.py): position balance and immediate-neighbour balance are exact, and
strength-two balance is approximate (every position-pair x rule-pair count is within 0..2 at k = 6, 0..3 at k = 10).
At k <= 5 and 7..9 each ordered pair of rules occupies each ordered pair of positions exactly once.
"""
from itertools import permutations, combinations, product

def _parity(p):
    return sum(1 for i, j in combinations(range(len(p)), 2) if p[i] > p[j]) % 2

def gf(q):
    """GF(q) for q in {2,3,4,5,7,8,9}: elements 0..q-1, returns (add, mul) tables."""
    polys = {4: (2, [1, 1]), 8: (2, [1, 1, 0]), 9: (3, [1, 0])}  # x^2=x+1 ; x^3=x+1 ; x^2=-1 (low->high coeffs of reduction)
    if q in (2, 3, 5, 7):
        return [[(a + b) % q for b in range(q)] for a in range(q)], [[(a * b) % q for b in range(q)] for a in range(q)]
    p, red = polys[q]
    n = {4: 2, 8: 3, 9: 2}[q]
    def vec(a): return [(a // p ** i) % p for i in range(n)]
    def num(v): return sum(c * p ** i for i, c in enumerate(v))
    def mul(a, b):
        va, vb = vec(a), vec(b)
        r = [0] * (2 * n - 1)
        for i, x in enumerate(va):
            for j, y in enumerate(vb):
                r[i + j] = (r[i + j] + x * y) % p
        for d in range(2 * n - 2, n - 1, -1):  # x^d = x^(d-n) * x^n, x^n = red
            c, r[d] = r[d], 0
            for i, rc in enumerate(red if q != 9 else [p - 1, 0]):
                r[d - n + i] = (r[d - n + i] + c * rc) % p
        return num(r[:n])
    add = [[num([(x + y) % p for x, y in zip(vec(a), vec(b))]) for b in range(q)] for a in range(q)]
    return add, [[mul(a, b) for b in range(q)] for a in range(q)]

def affine_rows(q):
    add, mul = gf(q)
    return sorted({tuple(add[mul[a][i]][b] for i in range(q)) for a in range(1, q) for b in range(q)})

def coa_rows(k):
    """k(k-1) ordering rows for k symbols (1 at k = 1): exact COA at k in EXACT, stacked Williams squares at 6 and 10."""
    if k == 1:
        return [(0,)]
    if k in (2, 3):
        return sorted(permutations(range(k)))
    if k == 4:
        return sorted(p for p in permutations(range(4)) if _parity(p) == 0)
    if k == 5:
        return sorted(tuple((a * i + b) % 5 for i in range(5)) for a in range(1, 5) for b in range(5))
    if k in (7, 8, 9):
        return affine_rows(k)
    if k in (6, 10):
        return stacked_williams(k)
    raise ValueError(k)


EXACT = {1, 2, 3, 4, 5, 7, 8, 9}     # strength-two index-1 COA
MAX_PAIR = {6: 2, 10: 3}             # observed bound for the frozen near-COA designs


def williams(n):
    first, lo, hi = [0], 1, n - 1
    while len(first) < n:
        first.append(lo)
        lo += 1
        if len(first) < n:
            first.append(hi)
            hi -= 1
    return [[(x + i) % n for x in first] for i in range(n)]


def stacked_williams(k):
    import json
    from pathlib import Path
    labs = json.loads((Path(__file__).with_name("orderings_k6_k10.json")).read_text())[str(k)]["labels"]
    assert len(labs) == k - 1
    return [tuple(lab[x] for x in row) for lab in labs for row in williams(k)]


def check(rows, k):
    from collections import Counter
    n = 1 if k == 1 else k * (k - 1)
    assert len(rows) == n and len(set(rows)) == n, (k, len(rows))
    for r in rows:
        assert sorted(r) == list(range(k))
    for p in range(k):  # position balance (exact at every k)
        assert set(Counter(r[p] for r in rows).values()) == {n // k}, (k, p)
    if k > 1:  # immediate-neighbour balance (exact at every k)
        adj = Counter((r[i], r[i + 1]) for r in rows for i in range(k - 1))
        assert len(adj) == k * (k - 1) and set(adj.values()) == {k - 1}, (k, adj)
    for p1 in range(k):  # strength two
        for p2 in range(k):
            if p1 != p2:
                c = Counter((r[p1], r[p2]) for r in rows)
                if k in EXACT:
                    assert len(c) == k * (k - 1) and set(c.values()) == {1}, (k, p1, p2)
                else:
                    assert max(c.values()) <= MAX_PAIR[k], (k, p1, p2)
