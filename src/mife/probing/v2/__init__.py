"""V2 (2026-11-20 amendment) of the Objective 1 / Objective 2 analysis: training-only milestone.

Everything here writes to versioned paths (`data/probing_features_v2/`, `results/v2/`) with a `version` field in
every sidecar; loaders refuse a mismatching version. V1 code, caches, results and the 2026-11-14 freeze are untouched.
See docs/amendment_v2_2026-11-20.md.
"""
