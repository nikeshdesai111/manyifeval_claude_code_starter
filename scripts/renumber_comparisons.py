"""Renumber the seven V2 comparisons (owner's request, 2026-11-24): old 7 (RCAP -> H2) becomes 4, and the secondary
comparisons shift down one: old 4 -> 5, old 5 -> 6, old 6 -> 7. Comparisons 1-3 are unchanged. The mapping is applied
simultaneously, only inside patterns that denote a comparison number (never PR numbers, colours or counts).

Usage: python scripts/renumber_comparisons.py [--write] FILE...
"""
from __future__ import annotations

import re
import sys

M = {"4": "5", "5": "6", "6": "7", "7": "4"}
m = lambda d: M.get(d, d)

PATTERNS = [
    # "#7", "dmse_#7", "#4/#5/#6" (not PR numbers, not hex colours such as #6250d6)
    (re.compile(r"(?<!PR )(?<!PRs )(?<!PR~)(?<!pull )(?<!issue )#([4-7])(?![0-9A-Za-z])"), lambda x: "#" + m(x.group(1))),
    # LaTeX / table row labels "7: RCAP \cto{} H2", "4: B1 \cto{}"
    (re.compile(r"(?<![0-9.])([4-7]): (B1|B2|RCAP)(\s*(?:\\cto|→|->|\$\\to\$|\\to))"),
     lambda x: m(x.group(1)) + ": " + x.group(2) + x.group(3)),
    # ranges and lists that only ever denote the secondary comparisons or the primary set
    (re.compile(r"\b4([–-]{1,2}|--|–)6\b"), lambda x: "5" + x.group(1) + "7"),
    (re.compile(r"\b1, 2, 3,? and 7\b"), lambda x: "1, 2, 3 and 4"),
    (re.compile(r"\b1, 2, 3, 7\b"), lambda x: "1, 2, 3, 4"),
    (re.compile(r"\b4, 5,? and 6\b"), lambda x: "5, 6 and 7"),
    (re.compile(r"\b4, 5, 6\b"), lambda x: "5, 6, 7"),
    # "comparison~7", "comparison 7", "comparisons~4", "Comparison 5"
    (re.compile(r"([Cc]omparisons?)([~ ])([4-7])(?![0-9.,]\d)"), lambda x: x.group(1) + x.group(2) + m(x.group(3))),
]


def apply(text: str) -> tuple[str, int]:
    """All patterns are matched on the ORIGINAL text and replaced in one pass, so no number is remapped twice."""
    spans = []
    for pat, rep in PATTERNS:
        for x in pat.finditer(text):
            if not any(a < x.end() and x.start() < b for a, b, _ in spans):
                spans.append((x.start(), x.end(), rep(x)))
    out, pos = [], 0
    for a, b, r in sorted(spans):
        out.append(text[pos:a]); out.append(r); pos = b
    out.append(text[pos:])
    return "".join(out), len(spans)


if __name__ == "__main__":
    write = "--write" in sys.argv
    for p in [a for a in sys.argv[1:] if a != "--write"]:
        s = open(p, encoding="utf-8").read()
        t, n = apply(s)
        if n:
            print(f"{n:5d}  {p}")
            if write:
                open(p, "w", encoding="utf-8").write(t)
