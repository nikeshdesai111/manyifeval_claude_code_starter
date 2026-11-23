"""Stage-3 proofs: library facts the arguments rely on, and hand witnesses that pass every check."""
import pytest

from mife_ifb import inventory, stage3_proofs as P, witness


@pytest.mark.parametrize("name", sorted(P.FACTS))
def test_fact(name):
    assert P.FACTS[name]()


def test_pairs_are_candidates_and_disjoint():
    cand = set(inventory.candidates())
    pr, hw, lo = P.proven(), P.hand_witnesses(), P.loophole_only()
    for d in (pr, hw, lo):
        for a, b in d:
            assert a in cand and b in cand and a < b
    assert not (set(pr) & set(hw)) and not (set(pr) & set(lo)) and not (set(hw) & set(lo))
    assert all(basis in ("official", "plain") for basis, _ in pr.values())


@pytest.mark.parametrize("pair", sorted(P.hand_witnesses()))
def test_hand_witness_passes(pair):
    rules, text = P.hand_witnesses()[pair]
    assert set(rules) == set(pair)
    assert witness.verify(rules, text, None) == []
