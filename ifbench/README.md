# IFBench adaptation (ManyIFEval-style design)

Separate from the ManyIFEval experiment: nothing outside `ifbench/` is read for writing or changed,
and `tests/test_ifb_isolation.py` checks that. The plan and Nicky's decisions are in the plan doc
"Adapting IFBench to the ManyIFEval-style design". Work runs in seven stages with a review stop after each.

| Stage | Status |
|---|---|
| 1. Load IFBench, freeze plain-meaning checks | done, reviewed (`docs/stage1_report.md`) |
| 2. Witness builder | done, reviewed (`docs/stage2_report.md`, `docs/stage2_builder_limits.md`) |
| 3. 50-pair timing, 861-pair audit, six-rule pilot | done, awaiting review (`docs/stage3_report.md`) |
| 4. Tasks, families, groups | not started |
| 5. Pool sampling with D3, split | not started |
| 6. 100-prompt pilot | not started |
| 7. Full generation | not started |

## Setup

```bash
cd ifbench
python -m venv .venv && .venv/bin/pip install -r requirements.txt
scripts/fetch_source.sh             # pinned IFBench checkout + checksum-verified NLTK data (no NLTK downloader)
.venv/bin/python -m pytest -q       # unit tests
.venv/bin/python scripts/stage1_report.py
.venv/bin/python scripts/stage3_audit.py final --workers 4   # resumable; data/stage3/ is git-ignored
.venv/bin/python scripts/stage3_six_pilot.py 300 --workers 4
.venv/bin/python scripts/stage3_summary.py && .venv/bin/python scripts/stage3_report.py
```

## Layout

- `configs/ifbench.yaml`: pins (commit, test-file hash, NLTK zip hashes, package versions) and the new seed family.
- `src/mife_ifb/source.py`: the only door to IFBench code: wording, default parameter draws, strict and loose official checks.
- `src/mife_ifb/inventory.py`: 15 exclusions with reasons, 42 candidates (15 parameterized, 27 fixed); count:words_japanese dropped after stage-1 review.
- `src/mife_ifb/plain_meaning.py`: one frozen plain-meaning check per candidate (witnesses only, never labels).
- `src/mife_ifb/witness.py`, `lexicon.py`: the programmatic witness builder (no LLM) and its vocabulary.
- `freeze/`: sha256 freeze record, inventory, stage-1 audit. `docs/`: generated check table and stage report.
- `third_party/` (git-ignored): the IFBench checkout and NLTK data, recreated by `fetch_source.sh`.
