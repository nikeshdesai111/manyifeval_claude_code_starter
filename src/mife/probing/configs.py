"""The 23 configurations per objective (tracker §6) and the reference rows, as feature-block lists.

Objective 1 (prompt-rule row):  H1 = [H]; H2 = [H, ER]; A1 = [A]; B1 = [EP, ER, S1]; B2 = [EP_B2, ER_B2, S1].
Objective 2 (prompt):           H1 = [H, M2]; H2 = [H, M2, ER5]; A1 = [A5, KM]; B1 = [EP, ER5, S2];
                                B2 = [EP_B2, ER5_B2, S2].
Combinations concatenate the union of their blocks. Identical blocks appear once (H2's frozen rule embedding is
B1's ER). Column subsets are dropped: KM ⊂ M2 ⊂ S2. Adapted (B2) and frozen embeddings are different blocks.
Objective 1 configurations containing H1, H2 or A1 use one head per instruction TYPE (15); B1/B2-only
configurations and the structure reference use one shared rule-conditioned head. Objective 2 uses one shared head.
"""
from __future__ import annotations

PROBES = ("H1", "H2", "A1")
PROBE_ONLY = ["H1", "H2", "A1", "H1+H2", "H1+A1", "H2+A1", "H1+H2+A1"]
B1_FAMILY = ["B1"] + [f"B1+{p}" for p in ["H1", "H2", "A1", "H1+H2", "H1+A1", "H2+A1", "H1+H2+A1"]]
B2_FAMILY = ["B2"] + [f"B2+{p}" for p in ["H1", "H2", "A1", "H1+H2", "H1+A1", "H2+A1", "H1+H2+A1"]]
CONFIGS = PROBE_ONLY + B1_FAMILY + B2_FAMILY
assert len(CONFIGS) == 23
# reference rows (labelled as references, not among the 23)
REFERENCES = ["REF_constant", "REF_type", "REF_type_k", "REF_structure"]
LEARNED_REFS = {"REF_structure"}

_BLOCKS = {
    1: {"H1": ["H"], "H2": ["H", "ER"], "A1": ["A"], "B1": ["EP", "ER", "S1"], "B2": ["EP_B2", "ER_B2", "S1"],
        "REF_structure": ["S1"]},
    2: {"H1": ["H", "M2"], "H2": ["H", "M2", "ER5"], "A1": ["A5", "KM"], "B1": ["EP", "ER5", "S2"],
        "B2": ["EP_B2", "ER5_B2", "S2"], "REF_structure": ["S2"]},
}
_SUBSETS = {"KM": ("M2", "S2"), "M2": ("S2",)}


def components(config: str) -> list[str]:
    return config.split("+")


def blocks(config: str, objective: int) -> list[str]:
    out: list[str] = []
    for c in components(config):
        for b in _BLOCKS[objective][c]:
            if b not in out:
                out.append(b)
    return [b for b in out if not any(s in out for s in _SUBSETS.get(b, ()))]


def per_type(config: str, objective: int) -> bool:
    return objective == 1 and any(c in PROBES for c in components(config))


def uses_b2(config: str) -> bool:
    return "B2" in components(config)


def uses_hidden(config: str) -> bool:
    return any(c in ("H1", "H2") for c in components(config))


def uses_attention(config: str) -> bool:
    return "A1" in components(config)


def family(config: str) -> str:
    if config in REFERENCES:
        return "reference"
    c = components(config)
    return "B2 family" if "B2" in c else "B1 family" if "B1" in c else "probe-only"
