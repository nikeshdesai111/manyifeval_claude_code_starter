# Preliminary S3 TRAIN report (24 Nov 2026 course submission)

**Interim only.** The S3 TEST generation and verification were not finished by the deadline, so these results use the
50 S3 TRAIN groups and grouped-CV out-of-fold (OOF) predictions. They do **not** replace the evaluation plan
(`manyifeval_s3_complete_evaluation_analysis_and_reporting_plan.md`, unchanged), the TEST split, the TEST access
controls or the final pipeline (`src/mife/eval_s3/`, untouched). Final TEST results will replace every number here.

## Disclosure (as used in the paper)

Because generation and verification for the designated held-out test set were incomplete at the submission deadline,
we report preliminary analyses on the 50 training groups of the S3 split. Descriptive results characterise observed
violation behaviour under controlled prompt changes and are exploratory. Predictive results use the out-of-fold
predictions of the selected models from 5-fold cross-validation by group: every prediction comes from a model fitted
without the rule pool that contains it. Settings, penalties and layers were selected with the same folds, so these
estimates may be optimistic; they are not held-out test results. Independent evaluation on the 22 S3 test groups
remains pending.

## Provenance audit

| Item | What was used | Check |
|---|---|---|
| Split | `split-s3-2026-11-23` via `mife.splits` (`configs/active_split.yaml`); 50 TRAIN / 22 TEST groups | `common.data` asserts 50 groups, 36,900 prompts, no overlap with `splits.test_groups()` |
| Manifest | `data/manifest/prompts.parquet` (frozen, `manifest-2026-11-14`) with the S3 overlay; manifest `split` column never read directly | via `mife.probing.data.prompts("train")` |
| Configurations | TRAIN uses the original 246 configurations per task (k=1..5 COA design). The 456-configuration expansion exists only for the 22 TEST groups and is not used | 36,900 prompts = 50 x 3 x 246 |
| Labels | Official verifier labels, run `main` (`data/verification/<model>/main/`), 20 responses per prompt (checked: 738,000 responses, 3,078,000 rule outcomes per generator) | read through `mife.probing.data` (TRAIN filter) |
| Predictions | V2 S3 rerun OOF predictions, `results/v2_s3/predictions/<gen>/obj{1,2}/<pred>.parquet`, column `selected` (CV-selected setting); version `v2-2026-11-20` | for every row and predictor the stored fold equals the fold of the row's group in `results/cv_s3/folds.json` (so each prediction is from a model fitted on the other 40 groups) |
| Reproduction | Pooled OOF NLL reproduces `results/v2_s3/tables/selected_settings.csv` cv_loss (e.g. Llama B2 0.401253 vs 0.401254) | |
| Not used | Historical 57/15 split, V1 results (`results/tables/`, `results/cv/`), V1 "B2" (LoRA), pilot or CUDA data, any TEST file | |

Seven comparisons (unchanged from the plan): primary 1 B1→H1, 2 B2→H2, 3 B2→RCAP, 7 RCAP→H2; secondary 4 B1→B1+H1,
5 B2→B2+H1, 6 B2→B2+A1. All seven have valid OOF predictions for both generators and both objectives.

## Completed / partial / unavailable

| Analysis | Status |
|---|---|
| Eval 1, Obj 1: NLL, AUROC, Brier, ECE, types improved, per-type, fixed vs parameterised, calibration | Completed (`eval1_*.csv`) |
| Eval 1, Obj 2: RMSE, MSE, MAE, Pearson, Spearman, ECE, training-mean and averaged-Obj1 references, calibration | Completed (`obj2_eval1_*.csv`; B = 2,000) |
| Eval 2, Obj 1, position (RMS by type, signed first-to-last by type x k, slot curves, 20 pair strata, Δ-MSE by k) | Completed |
| Eval 2, Obj 1, strict append (by transition, by retained slot, Δ-MSE by transition) | Completed (fewer pairs than the TEST design) |
| Eval 2, Obj 1, companion (strict replacement RMS by type, by k x relative order, target x companion signed matrix, Δ-MSE) | Partial: the TRAIN design has strict replacements only at k = 2, 3, 5 (none at k = 4) |
| Eval 2, Obj 1, task (RMS by type and by type x k, family and type x family signed effects, Δ-MSE) | Completed (family effects are descriptive associations) |
| Noise-adjusted RMS validation | Completed by simulation with 50 pools (`noise_validation_synthetic.csv`): ≤0.1 pp bias when true RMS ≥ 3 pp; +0.5–1 pp at an exact null (truncation); 91–100% coverage |
| Eval 2, Obj 2 (position, append, companion, task: observed RMS / signed mean, zero-change MSE, Δ-MSE) | Completed, but its design choices are **preliminary and not approved in the plan** |
| Factor E (difficulty terciles), leave-one-pool-out / t-interval sensitivity for Eval 2 | Not reported in the paper (t / LOO columns exist for Eval 1 contrasts in `eval1_contrasts.csv`) |
| Anything on TEST | Unavailable by design (sealed) |

## Deviations from the TEST plan

Resampling units are the 50 TRAIN groups (paired cluster bootstrap, B = 10,000, seed 20261203; Obj 2 Eval 1 B = 2,000);
ECE bins are deciles of the same OOF predictions; the TRAIN design has fewer append and companion contrasts (counts in
`match_coverage.json`); Obj 2 Eval 2 is a preliminary design. Intervals are pointwise and unadjusted.

## Plan-structured results

`PRELIMINARY_RESULTS_FOLLOWING_PLAN.md` lays out every exhibit of the evaluation plan in its order (Tables 1–4, A1–A7, E2 figures, Pos-A/B/C, K-A/B, C-A/B/C, D-A/B/C/D, Factor E, Objective 2 Evaluation 2) with TRAIN OOF numbers (`python -m mife.prelim_train.plan_report`).

## Main numbers (all from `tables/`)

See `RESULTS_PRELIMINARY.md`. Paper: `report/neurips_draft/paper.pdf` (Table 3, Table 4, Figure 1, Appendix H).

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python -m mife.prelim_train.matches        # metadata-only matched pairs -> matches/
PYTHONPATH=src .venv/bin/python -m mife.prelim_train.run_all        # Eval 1 + 2, Objective 1 -> tables/
PYTHONPATH=src .venv/bin/python -m mife.prelim_train.obj2           # Eval 1 + 2, Objective 2 -> tables/
PYTHONPATH=src .venv/bin/python -m mife.prelim_train.validate       # noise-adjusted RMS simulation -> tables/
PYTHONPATH=src .venv/bin/python -c "from mife.prelim_train import figures; figures.all_figures()"   # appendix PNGs
PYTHONPATH=src .venv/bin/python -m mife.prelim_train.paper_assets   # paper figure + LaTeX rows -> paper/
```

`src/mife/prelim_train/` is a snapshot of `src/mife/eval_s3/` (copied 2026-11-24) re-pointed at TRAIN OOF data; to run
the final TEST evaluation use `mife.eval_s3`, not this module.
