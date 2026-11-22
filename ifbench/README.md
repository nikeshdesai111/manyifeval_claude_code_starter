# IFBench adaptation (ManyIFEval-style design)

Separate from the ManyIFEval experiment: nothing outside `ifbench/` is read for writing or changed,
and `tests/test_ifb_isolation.py` checks that. The plan and Nicky's decisions are in the plan doc
"Adapting IFBench to the ManyIFEval-style design". Work runs in seven stages with a review stop after each.

| Stage | Status |
|---|---|
| 1. Load IFBench, freeze plain-meaning checks | done, awaiting review (`docs/stage1_report.md`) |
| 2. Witness builder | not started |
| 3. 50-pair timing, 903-pair audit, six-rule pilot | not started |
| 4. Tasks, families, groups | not started |
| 5. Pool sampling with D3, split | not started |
| 6. 100-prompt pilot | not started |
| 7. Full generation | not started |

## Setup

```bash
cd ifbench
python -m venv .venv && .venv/bin/pip install -r requirements.txt
scripts/fetch_source.sh             # pinned IFBench checkout + checksum-verified NLTK data (no NLTK downloader)
.venv/bin/python -m pytest -q       # 28 tests
.venv/bin/python scripts/stage1_report.py
```

## Layout

- `configs/ifbench.yaml`: pins (commit, test-file hash, NLTK zip hashes, package versions) and the new seed family.
- `src/mife_ifb/source.py`: the only door to IFBench code: wording, default parameter draws, strict and loose official checks.
- `src/mife_ifb/inventory.py`: 15 exclusions with reasons, 42 candidates (15 parameterized, 27 fixed); count:words_japanese dropped after stage-1 review.
- `src/mife_ifb/plain_meaning.py`: one frozen plain-meaning check per candidate (witnesses only, never labels).
- `freeze/`: sha256 freeze record, inventory, stage-1 audit. `docs/`: generated check table and stage report.
- `third_party/` (git-ignored): the IFBench checkout and NLTK data, recreated by `fetch_source.sh`.
