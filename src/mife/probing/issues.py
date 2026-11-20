"""Open issues, deviations and discrepancies for results/RESULTS.md § 10 (written to results/tables/open_issues.csv).
Text only; any result number belongs in a computed table, not here."""
from __future__ import annotations

import pandas as pd

from .data import RESULTS

ISSUES = [
    ("Tracker §7 / §9 said 'ten' held-out groups", "Discrepancy (fixed)",
     "The test split has 15 groups (tracker §1, §3A). Fixed both lines in docs/decision_tracker.md; logged in v1.16."),
    ("Hidden-row count in the prompt ('33 / 37 hidden rows')", "Discrepancy (documentation)",
     "Stored arrays have L+2 rows (34 Llama / 38 Qwen): row 0 embeddings, rows 1..L hidden_states, row L+1 pre-final-norm "
     "output (docs/storage_schema.md, features_meta.json). The prompt's own Phase 2 refers to rows L and L+1, so the data "
     "match the schema; followed the stored schema. Not treated as a count mismatch."),
    ("'100% depth' hidden row", "Decision (pre-CV)",
     "100% = row L+1, the pre-final-norm output of the last layer (Llama 33, Qwen 37), so all three candidates "
     "(50%, 75%, 100%) are un-normalised residual-stream states. Row L is the same state after the final RMSNorm "
     "(a per-prompt rescaling plus a fixed per-dimension weight, largely absorbed by fold-wise standardisation)."),
    ("No git repository", "Deviation",
     "The project folder is not a git repository, so no git hash exists. run_manifest.json and FROZEN.json record a "
     "SHA-256 over the source tree (src/, configs/, scripts/, tests/) instead."),
    ("configs/experiment.yaml hash", "Note",
     "Adding the `probing:` section changed the file hash. The manifest records the hash before the change "
     "(manifest_summary.json config_sha256); nothing reads it back, and no generation-phase value changed."),
    ("research_plan.md vs docx", "Note",
     "research_plan.md is the v3.6 summary; method detail (§8–11) was taken from docs/research_plan_original_v3.4.docx "
     "with its 144-task / 48-group / 10-test-group counts replaced by the tracker's 216 / 72 / 15 (as instructed). Docx §9.1 "
     "fold sizes (8/8/8/7/7 for 38 groups) are replaced by the tracker's 12/12/11/11/11 for 57."),
    ("Test-set statistic seen before this phase", "Disclosure",
     "During the bulk-run integrity check (2026-11-13, before this phase) truncation counts (hit_max_tokens) were computed "
     "over all prompts, TEST included, and reported in chat. No probing choice uses them; every probing choice comes from "
     "TRAIN-only grouped CV."),
    ("B2 compute", "Stop point (resolved)",
     "The projected B2 wall time on the Mac exceeded 24 h (§9). The project owner chose the Betty option: B2 runs as a "
     "GPU job array on Betty (dgx-b200) from a TRAIN-only input bundle; B1 stays on the Mac (MPS). Frozen BGE-M3 "
     "embeddings computed on Betty's CPU match the Mac's MPS embeddings (`results/tables/b2_encoder_xcheck.csv`), "
     "so both baselines start from the same encoder; the B2 adaptation itself runs in fp32 with TF32 matmuls on CUDA."),
    ("B2 full vs LoRA", "Decision (training-only)",
     "Chosen by the rule fixed in `b2.py` before any B2 result was seen (lower held-out NLL on Llama, Objective 1, CV fold 0, "
     "one epoch, each method's lower LR; tie -> full). Result in `data/probing_features/b2_runs/method.json` and "
     "the pre-specification. The chosen method is used for every B2 run."),
    ("Betty storage", "Note",
     "To fit BGE-M3 and the B2 outputs in the 50 GB Betty home quota, the generator checkpoints on Betty (Llama-3.1-8B, "
     "Qwen3-8B) were deleted with the project owner's approval; generation is finished and the Mac keeps its own copies. "
     "/vast/projects/zives exists but is not writable by this account."),
    ("BGE-M3 weight file", "Provenance",
     "transformers loaded the safetensors conversion of BAAI/bge-m3 (refs/pr/130). All 391 tensors are bit-identical to "
     "pytorch_model.bin at the pinned revision, so the embeddings are those of the pinned weights."),
    ("Configurations with identical features", "Design property",
     "With one shared hidden row and deduplicated blocks, several of the 23 configurations have exactly the same feature "
     "blocks: H1+H2 = H2, H1+H2+A1 = H2+A1, B1+H1 = B1+H2 = B1+H1+H2 (B1 already contains the frozen rule embedding that "
     "H2 adds), B1+H1+A1 = B1+H2+A1 = B1+H1+H2+A1, B2+H1+H2 = B2+H2 and B2+H1+H2+A1 = B2+H2+A1, in both objectives "
     "(`cv_identical_features.csv`). Linear heads give identical results within such a set; MLP heads differ only through "
     "their seeds. All 23 are kept, as the tracker lists them."),
    ("L2 grid edge", "Flag (raised at the pre-specification stop)",
     "CV selects an edge of the fixed grid {0.01, 0.1, 1} for part of the configurations in use (`cv_l2_counts.csv`, "
     "`cv_l2_edges.csv`). Objective 1 selects mostly the interior value; the edge cases are the structure reference and "
     "Llama A1 at the lowest value, and Qwen H1 and Qwen B2-family heads at the highest. Objective 2 selects the lowest "
     "value for the structure reference, H1, A1, H1+A1, B1 and several B1 combinations in both generators, and the "
     "highest value for every Llama B2 combination. `cv_l2_edges.csv` gives the CV loss one grid step inwards: for "
     "several lowest-edge Objective 2 selections it is markedly worse, so the CV optimum may lie outside the grid. One "
     "likely cause for Objective 2 is the loss scale: the same penalty is added to a mean squared error that is far "
     "smaller than the Objective 1 NLL, so a given l2 regularises Objective 2 heads much more strongly. The grid was "
     "not extended (tracker §3A)."),
    ("MLP epoch edge", "Flag",
     "A few MLP selections sit at the first or last epoch of the schedule (`cv_boundary_hits.csv`, kind 'schedule edge'). "
     "The epoch count is chosen inside CV, so this is reported rather than changed."),
    ("Hidden + attention fusion", "Observation (training-only)",
     "In Objective 1 CV, H1+A1 has a higher loss than A1 alone for both generators, and H2+A1 is slightly worse than "
     "the better single block (`cv_complementarity.csv`); in Objective 2 both combinations are slightly better. All blocks share one "
     "penalty after standardisation, so a high-dimensional block can dominate a smaller one. The plan's complementarity "
     "contrast tests this on TEST; no block-specific penalty was added."),
    ("L2 penalty scale", "Definition (pre-CV)",
     "The grid {0.01, 0.1, 1} is applied as (l2/2)·||W||² added to the mean per-completion binomial NLL (Objective 1) or "
     "mean per-prompt squared error (Objective 2), after fold-wise standardisation; biases are not penalised. The plan "
     "does not fix the scale; this convention is recorded in the pre-specification."),
]


def write() -> pd.DataFrame:
    d = pd.DataFrame(ISSUES, columns=["issue", "kind", "detail / what was followed"])
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    d.to_csv(RESULTS / "tables" / "open_issues.csv", index=False)
    return d
