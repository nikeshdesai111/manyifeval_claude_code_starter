"""Six cyclic nested chains (k = 1..5) and index-one strength-two COA orderings.

COA construction (plan v3.4 Section 5.5), as sharply 2-transitive permutation sets:
  k=1: identity; k=2: S2; k=3: S3 (6); k=4: A4, the 12 even permutations;
  k=5: the 20 affine maps i -> (a*i + b) mod 5, a in {1..4}, b in {0..4}.
Row r of the template for size k is a permutation sigma; the ordered prompt places the
subset's chain-order element sigma[p] at position p.
"""
from __future__ import annotations

from itertools import combinations, permutations

COA_VERSION = "coa/v1-sharply-2-transitive"


def _parity(p) -> int:
    inv = sum(1 for i, j in combinations(range(len(p)), 2) if p[i] > p[j])
    return inv % 2


def coa_template(k: int) -> list[tuple[int, ...]]:
    if k == 1:
        rows = [(0,)]
    elif k in (2, 3):
        rows = list(permutations(range(k)))
    elif k == 4:
        rows = [p for p in permutations(range(4)) if _parity(p) == 0]
    elif k == 5:
        rows = [tuple((a * i + b) % 5 for i in range(5)) for a in range(1, 5) for b in range(5)]
    else:
        raise ValueError(k)
    return sorted(rows)


def check_coa(rows: list[tuple[int, ...]], k: int) -> None:
    """Assert uniqueness, permutation rows, and strength-two index-one balance."""
    assert len(rows) == (1 if k == 1 else k * (k - 1)), (k, len(rows))
    assert len(set(rows)) == len(rows)
    for r in rows:
        assert sorted(r) == list(range(k))
    if k == 1:
        return
    for p1 in range(k):
        for p2 in range(k):
            if p1 == p2:
                continue
            seen = {(r[p1], r[p2]) for r in rows}
            assert len(seen) == k * (k - 1), "ordered pair repeated at a position pair"
    for sym in range(k):
        for pos in range(k):
            assert sum(1 for r in rows if r[pos] == sym) == k - 1


def chains(order: list[str]) -> list[list[str]]:
    """Six cyclic rotations of a 6-rule ordering, truncated to five elements."""
    n = len(order)
    assert n == 6
    return [[order[(c + i) % n] for i in range(5)] for c in range(n)]


def configurations(order: list[str], k_max: int = 5) -> list[dict]:
    """All ordered rule configurations for one group (246 for six rules, k_max=5)."""
    out = []
    for c_idx, chain in enumerate(chains(order)):
        for k in range(1, k_max + 1):
            subset = chain[:k]
            for row_idx, sigma in enumerate(coa_template(k)):
                out.append({
                    "chain_index": c_idx + 1,
                    "k": k,
                    "subset_chain_order": list(subset),
                    "coa_row": row_idx,
                    "coa_perm": list(sigma),
                    "ordered_rule_labels": [subset[s] for s in sigma],
                })
    return out
