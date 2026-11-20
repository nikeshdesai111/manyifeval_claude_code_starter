# Claude Code prompt — probing, evaluation and results phase

Paste this whole file into Claude Code from the repo root, or tell it: "Read `PROMPT_probing_and_results.md` and carry it out."

---

## 0. Scope change (read this first)

The generation phase is finished. **This prompt lifts the "stop at the generation milestone" scope in `CLAUDE.md` / `AGENTS.md`.** You are now authorised to build features, train and tune the probe and baseline models, evaluate them, and produce the results. Update the scope sections of `CLAUDE.md`, `AGENTS.md` and `README.md` to say so (keep the old text under a "Phase 1 (complete)" heading). Do **not** regenerate responses, re-verify, re-extract LLM features, or change the manifest; those outputs are frozen.

Goal: carry out the rest of the research plan, and put **every result** into
1. `results/RESULTS.md`: one written results file, and
2. `notebooks/results_explorer.ipynb`: a notebook I can open and use to dig through the results in depth.

## 1. Read before writing code

1. `docs/decision_tracker.md`: authoritative for counts, splits, models, configurations, metrics and status.
2. `research_plan.md` (v3.6 + Amendment R1).
3. `docs/research_plan_original_v3.4.docx`, **sections 8–11** (model families, feature specs, the 23 configurations, losses, CV and tuning protocol, metrics, results-table layout, hypothesis comparisons, controlled analyses). Its counts (144 tasks, 48 groups, 10 test groups) are superseded by the tracker (216 / 72 / 15); use the docx only for method detail.
4. `docs/storage_schema.md`, `docs/pilot_report_betty.md`, `docs/audit_findings.md`, `data/features/*/main/features_meta.json`.

Where the tracker, plan and docx disagree, **flag the discrepancy in `results/RESULTS.md` § Open issues** and say which you followed. Known ones: tracker §7 "clustered by the ten held-out 3-task pools" and §9 "Ten independent groups" should be **15**. Fix those two lines in the tracker and log them.

## 2. Frozen inputs (both models: `llama`, `qwen`; run id `main`)

| What | Where |
|---|---|
| Manifest, split, task families, rule pools | `data/manifest/` |
| Token IDs, rule-token spans | `data/tokenized/{model}.parquet` |
| Raw responses (20/prompt) | `data/generations/{model}/main/` |
| Per-response, per-rule verifier outcomes | `data/verification/{model}/main/verification_long.parquet` |
| Objective 1 labels (n=20, v, p_hat) | `data/verification/{model}/main/labels_prompt_rule.parquet` (221,616 rows) |
| Objective 2 labels (q_hat) | `data/verification/{model}/main/labels_prompt.parquet` (53,136 rows) |
| Hidden states (all layers) + rule-span attention (sum, mean) | `data/features/{model}/main/` (Apple MPS, HF eager BF16) |

- Read everything through the existing integrity guard (`src/mife/integrity.py`); every join must check `prompt_sha256` / `rendered_sha256`.
- **Only `main` features (MPS) may be used. Never mix in `pilot*` or CUDA features.**
- Load hidden states lazily (memmap, one layer at a time; decode with `introspect.bits_to_float32`). Never load all layers into RAM.

## 3. Test-set discipline (non-negotiable)

- The 15 TEST groups (11,070 prompts) are **not read** (not labels, not descriptive statistics, not feature scaling) until Phase 4's gate is passed.
- All choices (layer, attention pooling, head type, L2, B2 settings, epochs, calibration bins, matching rules for Evaluation 2, reliability thresholds) come from **5-fold GroupKFold over the 57 TRAIN groups** (12/12/11/11/11 groups, fixed seed, folds saved to `results/cv/folds.json`).
- Preprocessing (scalers, PCA if any, category encodings) is fit within each fitting fold, then refit on all 57 TRAIN groups for the final model.
- Add a test (`tests/test_no_test_leakage.py`) that fails if any CV/tuning code path touches a TEST prompt_id.

## 4. Work plan

Create a task list and work through these phases. Keep jobs resumable and idempotent: cache features and fold predictions to disk, and skip finished work on rerun.

### Phase 0: Inventory, environment and compute plan
- Confirm counts against the tracker: 53,136 prompts, 221,616 prompt–rule rows, 1,062,720 responses per model, 42,066 / 11,070 train/test prompts, and 33 (Llama) / 37 (Qwen) hidden rows. Any mismatch → stop and report.
- Record the environment (Python, torch, transformers, sklearn, sentence-transformers / FlagEmbedding versions, MPS availability, git hash) in `results/run_manifest.json`. Add any new dependencies to `requirements.lock`.
- Pin `BAAI/bge-m3` to a revision in `configs/experiment.yaml` (new `probing:` section). Verify every full prompt fits its 8,192-token limit; refuse silent truncation.
- **Benchmark B2 (fine-tuned BGE-M3) on MPS** with a short timed run, then project the wall time for 5 folds × grid × 2 objectives × 2 generators, plus final refits. If the projection exceeds **24 h**, do not shrink the grid or the data silently. Write the options (Betty sbatch job with the scripts under `scripts/betty/`; LoRA instead of full fine-tuning; a smaller grid frozen in advance) into `results/RESULTS.md` § Compute, **ask me**, and continue with every non-B2 configuration in the meantime.

### Phase 1: Descriptive results (TRAIN only at this stage)
Per model, with 95% CIs from a cluster bootstrap over **groups**:
- Violation rate by k, by rule type, by category, by position within k, by task family, and type × k.
- Distribution of p_hat (share of rows with v = 0 and v = 20) and q_hat.
- Truncation: share of responses with `hit_max_tokens`, by type and k. **Primary labels keep truncated responses as they are (current pipeline).** Add a sensitivity table that recomputes labels excluding truncated samples.
- `change_case:english_capital`: share of its violations caused only by the langdetect step (see `docs/pilot_report_betty.md`). Do not change the checker; add a sensitivity row that excludes this type.
- Qwen repetition loops: count and share; report them, do not drop them.
- Cross-model: Llama vs Qwen per type, and per-prompt rank correlation of q_hat.

### Phase 2: Feature blocks (cached under `data/probing_features/{model}/`)
- **Structure features**, shared by B1/B2: k, target type, target position, typed parameters (numeric values, relation), companion types (multi-hot), task family, prompt length in tokens and characters. Exact rule IDs are kept for audit only, never as model inputs.
- **BGE-M3** (frozen, dense 1,024-d): the full prompt, plus each exact rule text. Compute once; it is model-independent, so share it across generators.
- **H1**: final-prompt-token hidden state at a candidate layer. Candidates are 50%, 75% and 100% depth (Llama rows 16 / 24 / 32, Qwen 18 / 27 / 36). Before CV, write down whether "100%" means the post-final-norm row L or the pre-norm row L+1, and why.
- **H2**: H1 plus the frozen BGE-M3 embedding of the exact target rule.
- **A1**: final-token attention to the target rule span, across all layers × heads, as SUM or token-normalised MEAN (CV chooses one). Objective 2 uses ordered, masked, 5-slot per-rule attention vectors plus a presence mask.
- **Objective 2 versions** follow docx §8.9: ordered ≤5 slots with padding and masks (rule types for H1; exact-rule embeddings for H2 and B1/B2).

### Phase 3: Models and training-only CV (per generator, separately)
- Objective 1 (prompt–rule unit): **one head per instruction TYPE (15)** for H1/H2/A1 and every combination containing them, never per exact rule. B1/B2-only configurations use their shared rule-conditioned head. Loss: aggregated **binomial NLL with n = 20**.
- Objective 2 (prompt unit): one shared head per configuration; loss sigmoid-bounded MSE on q_hat.
- **All 23 configurations per objective**, exactly as listed in the tracker §6 (7 probe-only, 8 B1-family, 8 B2-family), combined by explicit feature concatenation followed by an objective-specific head.
- Grid (tracker §3A): linear-sigmoid vs 1-hidden-layer MLP (128); L2 ∈ {0.01, 0.1, 1}; MLP 30 epochs, early stopping chosen inside CV; one shared H1/H2 layer; A1 SUM vs MEAN. B2: choose full vs LoRA in a training-only check, then hold it fixed; full LR {1e-5, 3e-5} or LoRA LR {1e-4, 2e-4} (rank 16, dropout 0.05), epochs {1, 3}, weight decay 0.01. Fusion models reuse the selected extraction and B2 adaptation; fit B2 only on each fold's fitting groups.
- Selection: mean held-out-fold NLL (Objective 1) or MSE (Objective 2). Save **out-of-fold predictions for every configuration and every grid point** to `results/predictions/{model}/obj{1,2}/cv/`.
- **Reference rows** (labelled as references, not counted among the 23): constant (train mean); type-only rate; type × k rate; structure-features-only logistic/MLP head (no embeddings, no LLM features). Earlier versions of this project found probes no better than a metadata logistic regression, so this row has to be in every table.
- Refit each selected configuration on all 57 TRAIN groups and save the fitted heads (small files only; never copy LLM or BGE weights).

### Phase 4: Pre-specification freeze → STOP
Before any TEST data is read:
1. Write `results/prespec/analysis_plan.md` containing: the selected hyperparameters per configuration; metric definitions; calibration binning; bootstrap settings (cluster = test group, B = 10,000, seed); the exact list of hypothesis contrasts (§4 Phase 5 below); Evaluation 2 matching algorithms and the "reliable change" definition (see below); and the multiple-comparison handling.
2. Save the file's SHA-256 and the git hash to `results/prespec/FROZEN.json`.
3. Put a short CV summary into `results/RESULTS.md` § Cross-validation.
4. **STOP and ask me to approve unblinding the test set.** Do not continue without an explicit "go".

### Phase 5: Test evaluation (run exactly once; log the run in `results/prespec/FROZEN.json`)
**Evaluation 1, all 11,070 test prompts per generator**
- Objective 1: binomial NLL (primary), Brier on per-response binary outcomes, calibration (ECE + reliability curve), AUROC on per-response binary outcomes, RiskTop/RiskBottom (observed violation rate in the top/bottom predicted-risk decile). Report overall, per type (plus the macro mean across types) and per k.
- Objective 2: MAE, RMSE, calibration of the predicted proportion, Pearson and Spearman. Overall and per k.
- Uncertainty: paired cluster bootstrap over the 15 test groups.
- The 23-row tables per objective per model, laid out as in docx §10.3, plus the reference rows.

**Hypothesis contrasts (docx §10.4), as paired differences with group-bootstrap CIs:** B1 vs B2; each baseline vs baseline + {H1, H2, A1}; H1 vs H2; hidden + attention complementarity (H1+A1 vs the better of H1 and A1, likewise for H2); probe-only vs structure-only reference. Report each for both objectives and both generators, and say where Llama and Qwen agree or disagree.

**Evaluation 2, controlled analyses on test prompts (docx §11)**
- **A, position:** fix task, exact rule set, k and target; compare target positions using COA-balanced mean rates.
- **B, count:** nested +1-rule comparisons along each chain, k = 1→2→…→5.
- **C, composition:** fix task, k, target and position; replace exactly one companion. Report how many strict matches exist.
- **D, task:** identical ordered rule configuration (`config_id`) across a group's three tasks.
- **E, difficulty:** TRAIN-only k = 1 violation rates per type define type terciles; report Evaluation 1 metrics by tercile.
- For A–D, score each configuration on the **absolute error between predicted and observed differences**, and on sign accuracy for **reliable changes only**. Define "reliable" in the prespec (e.g. a two-proportion difference whose 95% CI over the 2 × 20 samples excludes 0) and keep that definition fixed.

### Phase 6: Results file, notebook, docs
**`results/` layout**
```
results/
  RESULTS.md                 # the written report (below)
  run_manifest.json          # versions, seeds, data hashes, git hash, wall times
  tables/*.csv               # every table in RESULTS.md, machine-readable
  figures/*.png              # every figure in RESULTS.md
  cv/                        # folds, grid results, selected settings
  predictions/{model}/obj{1,2}/{cv,test}/*.parquet   # keyed by prompt_id (+ rule_id), one column per config
  prespec/                   # analysis_plan.md, FROZEN.json
```

**`results/RESULTS.md`**, in this order:
1. Summary: 5–8 bullets with the headline numbers and CIs.
2. What was run: data, models, splits, configurations, what is frozen; links to the prespec.
3. Descriptive results (Phase 1).
4. Cross-validation and selected settings.
5. Evaluation 1 tables and calibration plots, Objective 1 then Objective 2, per model.
6. Hypothesis contrasts.
7. Evaluation 2 (A–E).
8. Sensitivity analyses (truncation, english_capital, Qwen loops).
9. Compute and runtime.
10. Open issues, deviations from the plan, discrepancies found.

Every number in the file must come from a saved table in `results/tables/`; never type numbers by hand.

**`notebooks/results_explorer.ipynb`** reads only from `results/` and `data/` (it loads no LLM and runs no training), runs top to bottom in a few minutes, and is saved **executed, with outputs**. Sections:
1. Setup and integrity check (hashes match `run_manifest.json`).
2. Dataset descriptives, with interactive filters by model / type / k / position / task family.
3. CV results: grid heatmaps, the selected layer and pooling, fold-to-fold variation.
4. Evaluation 1 leaderboard per objective and model (sortable), per-type and per-k breakdowns, reliability diagrams.
5. Hypothesis contrasts as forest plots.
6. Evaluation 2 A–E plots: predicted vs observed differences.
7. **Error analysis:** the prompt–rule pairs with the largest residuals for a chosen configuration, and where configurations disagree most.
8. **Prompt drill-down:** pick a `prompt_id` and see the prompt text, its rules (type, parameters, position), p_hat / q_hat for both models, every configuration's prediction, and 2–3 sampled responses with the verifier outcome for each rule.
9. Llama vs Qwen comparison.

Use pandas + matplotlib/plotly; use ipywidgets only if it is already installed or added to the lock file.

**Docs:** add a v1.16 tracker entry (scope change, settings selected, prespec hash, the "ten"→"15" fix); update `README.md` with the commands for each phase; add `docs/probing_schema.md` for the new files.

## 5. Code organisation and tests
- New modules under `src/mife/probing/`: `data.py` (loaders, joins, integrity), `features.py`, `heads.py`, `cv.py`, `train.py`, `evaluate.py`, `controlled.py` (Evaluation 2), `report.py` (builds tables, figures and RESULTS.md). Add CLI subcommands to `mife.cli` (`probe-features`, `probe-cv`, `probe-final`, `probe-eval`, `probe-report`).
- Tests: the binomial NLL equals a hand computation; per-type head routing (a rule of type t only reaches head t); fold assignment is grouped and deterministic; no test leakage; Objective 2 slot ordering and masks; the bootstrap resamples groups, not prompts; a tiny synthetic end-to-end run. Synthetic data is for tests only and must never appear in `results/`.
- Seed everything; record the seeds in `run_manifest.json`.

## 6. Honesty rules
- Never fabricate or hand-edit a number. If something fails, say so in RESULTS.md § Open issues.
- Clearly separate CV (training) results from test results in every table and figure.
- If a type has too few violations for a stable head (e.g. fewer than 50 violated responses in a fitting fold), keep it, flag it, and report its per-type metrics with that caveat. Do not drop it silently.
- Probes beating or failing to beat the baselines are both results; report whichever happens plainly.

## 7. Stop points (summary)
1. Phase 0: count mismatch, or B2 projected above 24 h → ask (continue non-B2 work meanwhile).
2. **Phase 4: always stop before touching the test set** and wait for my "go".
3. After Phase 6: report what was produced, the headline results, and anything unresolved.
