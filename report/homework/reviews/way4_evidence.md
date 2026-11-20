# Evidence check of `report/paper.tex` (draft of 16 Nov 2026)

How I checked: I read the paper, every `\input` table in `report/generated/`, `results/tables/*.csv`, `report/tables/*.csv`, `results/RESULTS.md`, `results/prespec/analysis_plan.md`, `results/prespec/FROZEN.json` and the code in `report/*.py` and `src/mife/probing/`. I recomputed the quantitative claims from the CSVs with pandas and also ran `report/check_numbers.py`, which reads only the CSVs and the paper. It reported "prose numbers checked: 466; unmatched: []" and all its assertions passed. Nothing under `data/` was read and no test-split data was loaded. Figures were not inspected pixel by pixel. Each figure claim is checked against the CSV its caption names.

Status labels: **S** = supported, **U** = unsupported, **C** = can't tell. When a claim appears more than once (abstract, introduction, body, conclusion), it is listed once at its main location and the repeats are noted.

Abbreviations: `cs` = `results/tables/cv_selected.csv`, `oc` = `report/tables/oof_contrasts.csv`, `om` = `report/tables/oof_metrics.csv`, `opt` = `report/tables/oof_per_type.csv`, `optd` = `report/tables/oof_per_type_diff.csv`.

## A. Status, abstract and introduction

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | Generation, verification, features, model selection and final fits are complete | S | `runtime.csv` (cv_heads_*, b2_grid, b2_final, refit_* rows) |
| 2 | Analysis plan frozen; 15 test groups not read; no test results in the draft | S | `FROZEN.json`: plan hash recorded, `test_runs: []`, `unblind_approved: false`; RESULTS.md §5–7 "Pending" |
| 3 | Plan hash recorded with hashes of folds, manifest and config | S | `FROZEN.json`; sha256 of `analysis_plan.md` recomputed = 5070f05b…5d7, which matches |
| 4 | All 216 tasks, 72 groups of 3, 6-rule pools, k=1..5, 53,136 prompts | S | Design arithmetic: 72×3 = 216; 246×216 = 53,136; analysis_plan, RESULTS.md |
| 5 | 20 responses per prompt, about 2.1 M responses in total | S | 53,136×20×2 = 2,125,440; train sum_n in `desc_by_k.csv` (841,320 responses per model) is consistent |
| 6 | 23 predictors per generator and objective, compared with structure-only references | S | `cs` (23 configurations + 4 references per cell) |
| 7 | Attention probe is the best single per-rule predictor for both generators | S | `cs`: A1 has the lowest Obj. 1 single-component CV loss (0.3316 / 0.2361). Repeated in the contributions and the conclusion |
| 8 | OOF AUROC 0.85 / 0.84 (A1) vs 0.79 / 0.77 (B2) | S | `om` auroc: A1 0.8517 / 0.8444; B2 0.7933 / 0.7740 |
| 9 | Probes lower the loss of the strongest non-internal comparator by 8–14% in 3 of 4 cells | S | `cs`: best probe-only vs min(structure, B1, B2) = −10.4% (Llama O1), −8.4% (Qwen O1), −13.5% (Qwen O2) |
| 10 | In Llama Objective 2, B2 matches the probes | S | `cs` −1.7%; `oc` best-probe vs B2 −0.0013 [−0.0095, +0.0068], 3/5 folds |
| 11 | Uppercase-English verifier artefact accounts for 43% of Qwen's violations of that rule | S | `audit_ec_causes.csv` share 0.4315 |
| 12 | The artefact affects 6.1% of Qwen's checks of the rule | S | `audit_ec_causes.csv` share_of_checks 0.0612 |
| 13 | Contributions: 37 contrasts per cell, plan frozen and hashed before test read | S | analysis_plan §5 (37 rows), `FROZEN.json`; `oc` has the 37 contrasts per cell plus 3 summary rows |
| 14 | Contributions: probes beat prompt-only comparators in 3 of 4 cells; B2 matches them in the fourth | S | As #9, #10 |
| 15 | Introduction literature claims (ManyIFEval compliance falls with instruction count; IFScale; probing of correctness, instruction following and attention to constraints) | C | External citations; cannot be checked against project files |
| 16 | Example: exact-three-paragraphs fails far more often than lowercase | S | `desc_by_type.csv`: paragraphs 0.511 / 0.059 vs lowercase 0.016 / 0.007 |

## B. Related work

| # | Claim | Status | Evidence |
|---|---|---|---|
| 17 | Characterisations of IFEval, ManyIFEval, FollowBench, InFoBench, IFScale | C | External literature |
| 18 | Characterisations of probing work (control tasks, truthfulness, KEEN, etc.) | C | External literature |
| 19 | Characterisations of Heo et al. and Stolfo et al.; how this study differs | C | External literature (the "differs" half matches the design) |
| 20 | Characterisations of SAT Probe, lost-in-the-middle, PASTA, attention-as-explanation debate | C | External literature |

## C. Setup and dataset

| # | Claim | Status | Evidence |
|---|---|---|---|
| 21 | Definitions of p, q, p̂ = v/20, q̂ = mean p̂ (q is the expected fraction) | S | analysis_plan §2–3; `desc_label_distribution.csv` (mean p̂, mean q̂ reported separately) |
| 22 | Generators are separate experiments; nothing pooled | S | All tables keyed by model; heads are fitted per model in `cs` |
| 23 | Pinned ManyIFEval commit; 15 types; official verifiers from IFEval | C | Generation-phase provenance not in the provided files |
| 24 | Serialiser reproduces all 2,160 released ManyIFEval prompts byte for byte | C | No table or log in the provided sources |
| 25 | Six family labels from a written rubric; each group spans three families | C | `desc_by_family.csv` confirms 6 families exist; rubric and per-group constraint not in sources |
| 26 | Six distinct types per pool; mutually exclusive sets; the rule reproduces all released combinations | C | Not in sources (the incompatibility of the 3 case types and keyword pairs is consistent with `audit_ec_cooccurring_types.csv` n=0 for case types) |
| 27 | Parameters from default generators; one draw rejected; witness passes all verifiers | C | Not in sources (the parameter ranges in `tab_types` are not backed by a CSV) |
| 28 | 432 rule instances | S | 72×6 |
| 29 | Six nested chains; every rule in exactly k subsets; COA orders 1, 2, 6, 12, 20 with stated balance properties | C | Design property; 1+2+6+12+20 = 41, ×6 = 246 is consistent, but the balance is not verified in the provided files |
| 30 | 246 ordered lists per task, reused across the 3 tasks | S | Arithmetic; analysis_plan D-pairs (3 task pairs per config, 738 per group) |
| 31 | Table 1 counts (246 / 738 / 42,066 / 11,070; 1,026 / 3,078 / 175,446 / 46,170 / 221,616; responses 4,920 … 1,062,720) | S | Arithmetic is consistent; `cv_folds.csv` train prompts sum to 42,066; analysis_plan D-task 46,170 / 11,070 |
| 32 | 15 test / 57 train groups; no validation set | S | `cv_folds.csv` (57 groups); analysis_plan |
| 33 | Split chosen from 20,000 candidates subject to coverage constraints | C | Selection procedure not in sources (coverage itself is consistent with `tab_types`: ≥13 train and ≥3 test groups per type) |
| 34 | Eight near-duplicate task clusters kept on the same side | C | Not in sources |
| 35 | 50 of 90 test rule instances coincide with a training rule | C | Not in sources (90 = 15×6 is right) |
| 36 | Generators, pinned revisions, bf16, decoding settings, 2,048 tokens, vLLM on B200 | C | Not in sources (truncation at the token limit is consistent with `desc_truncation.csv`) |
| 37 | No prompt truncated; longest prompt 216 tokens | C | Not in sources |
| 38 | A rule counts as followed only if the response is non-empty and passes the check; verifier randomness seeded/pinned | C | Not in sources (`audit_ec_consistency.csv` shows 0 diagnostic disagreements, which is partial support) |
| 39 | Nothing dropped; truncated and looping responses kept in labels | S | `sens_truncation.csv` primary labels include all responses; `desc_loops.csv` |
| 40 | Features: hidden state at the final token for every layer; attention per layer×head (32×32, 36×32), sum and mean | S | Widths in `cs` (A1 1,024 / 1,152; H 4,096); `cv_pooling_selection.csv`; `open_issues.csv` (L+2 stored rows) |
| 41 | Features computed with HF Transformers on Apple silicon; pilot cosine ≥ 0.998 vs generating backend (repeated in Limitations) | C | No pilot table in sources |

## D. Predictors and protocol

| # | Claim | Status | Evidence |
|---|---|---|---|
| 42 | Component and block definitions (H1, H2, A1, B1, B2; Table 2) | S | `cv_identical_features.csv`; widths in `cs`; `src/mife/probing/features.py` |
| 43 | Structured features: 75 columns (O1) and 175 (O2); no exact rule, group or task IDs | S | `cs` n_features 75 / 175; analysis_plan §2 "Exact rule IDs are never model inputs" |
| 44 | 23 = 7 + 8 + 8; 15 distinct feature sets; duplicates listed in the appendix | S | `cv_identical_features.csv` (the 12 rows collapse 23 to 15 per objective) |
| 45 | O1 per-type heads for configurations with probes; shared head for B1, B2 and structure; O2 always shared | S | `cs` per_type_heads column; analysis_plan §2 |
| 46 | Four references (constant, type, type×k, structure head) | S | `cs` |
| 47 | Linear and MLP(128) heads × λ ∈ {0.01, 0.1, 1}; sigmoid; binomial NLL / squared error; standardisation on fitting rows | S | analysis_plan §2; `src/mife/probing/heads.py` |
| 48 | Folds of 12, 12, 11, 11, 11 groups | S | `cv_folds.csv` |
| 49 | MLP epochs up to 30, chosen in CV; layer chosen on H1, pooling chosen on A1 | S | `cv_boundary_hits.csv` (epoch 30 edge); `cv_layer_selection.csv`; `cv_pooling_selection.csv` |
| 50 | B2: LoRA rank 16 with a linear head, grid of 2 LRs × {1, 3} epochs, inside each fold, 80 runs | S | `cv_b2_grid.csv`; `runtime.csv` "80 LoRA … runs"; `b2.py` |
| 51 | LoRA over full fine-tuning chosen by a rule fixed in advance on one fold | S | `b2_betty_check.csv` (fold-0 NLL 0.4141 vs 0.4173, LoRA chosen); analysis_plan §1 |
| 52 | Final models refitted on all 57 groups | S | `runtime.csv` refit_* rows |
| 53 | Loaders refuse test rows until unblinding; an automated test guards this | S | `src/mife/probing/data.py` (`test_unblinded`, guard); `tests/test_no_test_leakage.py` exists (not run) |
| 54 | B2 fine-tuning jobs received a train-only bundle | C | Stated in RESULTS.md §10, but the bundle itself is not checkable from sources |
| 55 | Disclosed exception: truncation share computed over all prompts before probing; unused | S | `open_issues.csv` "Test-set statistic seen before this phase" (disclosure; non-use cannot be proven) |
| 56 | OOF analysis: paired group bootstrap, 2,000 replicates; not part of the frozen plan | C | Intervals exist in `om`/`oc`; replicate count only in `report/analysis_oof.py` (not checked line by line). Absence from the plan is confirmed |

## E. Results: violation rates

| # | Claim | Status | Evidence |
|---|---|---|---|
| 57 | Llama 17.7% (15.5–20.0); Qwen 9.9% (8.0–11.9) | S | `desc_by_k.csv` "all" rows |
| 58 | Qwen rises 6.2% → 10.6% with k; Llama 16.5% → 18.2% with a dip at k=2 | S | `desc_by_k.csv` |
| 59 | Rule type matters far more than k; paragraph 51.1 / 5.9; letter 36.5 / 34.1; lowercase 1.6 / 0.7 | S | `desc_by_type.csv` |
| 60 | Qwen word count 1.1% → 23.1%; Llama all-uppercase 22.3% → 14.4% | S | `desc_type_by_k.csv` |
| 61 | Llama k=5 position 1 → 5: 17.7% → 18.9%; Qwen flat 10.4% / 10.5% | S | `desc_by_position.csv` |
| 62 | Cross-model Spearman of q̂ 0.43 (0.36–0.51) | S | `desc_cross_model_corr.csv` |
| 63 | v=0 in 53.2% / 74.6% of rows; v=20 in 5.4% / 3.9% | S | `desc_label_distribution.csv` |
| 64 | Fig. 1 and Fig. 6 bands are group-bootstrap 95% intervals | S | lo/hi columns in `desc_by_k.csv`, `desc_type_by_k_long.csv` |

## F. Results: cross-validation

| # | Claim | Status | Evidence |
|---|---|---|---|
| 65 | Table 3 values (all cells) | S | `cv_summary.csv`, `cs` (each value matches) |
| 66 | Type×k vs constant −11.8% / −15.3%; structure head −19.7% / −21.0% | S | `cs` recomputed |
| 67 | B1 no better than the structure head: higher for Qwen in both objectives and Llama O1; 0.5% lower for Llama O2 | S | `cs`: +2.8%, +6.7%, +2.4%, −0.45% |
| 68 | B2 helps for Llama in both objectives | S | `cs`: B2 vs B1 −4.3% (O1), −12.3% (O2); `oc` C6 intervals exclude 0 |
| 69 | B2 "does not help for Qwen" | C | CV loss (the subject of this paragraph) is 1.5% *lower* for Qwen O1 (0.2606 vs 0.2646) and 1.4% higher for O2. The OOF interval for O1 includes 0 (−0.0040 [−0.0183, +0.0120]). "Not detectably" would be accurate; "does not help" overstates |
| 70 | A1 lowest single component O1 (0.3316 / 0.2361); H2 next (0.3390 / 0.2396) | S | `cs` |
| 71 | H1 (0.3761 / 0.2613) no better than the structure head | S | `cs` 0.3761 vs 0.3779, 0.2613 vs 0.2573; `oc` Ref H1 intervals include 0 |
| 72 | Best probe-only lowers structure by 12.3% / 8.4% and B2 by 10.4% / 9.5% | S | `cs` |
| 73 | Best B2+probe 0.3341 Llama (above A1); 0.2324 Qwen (lowest, 1.4% below best probe-only) | S | `cs` |
| 74 | Qwen O2: H1 1.130, H1+A1 1.103 vs structure 1.275, B1 1.360, B2 1.379; B2+probe best 1.342 | S | `cs` (×100) |
| 75 | Llama O2: B2 1.613, best probe-only 1.585, lowest 1.562 (B2+H2+A1) | S | `cs` (tie with B2+H1+H2+A1, same inputs) |
| 76 | 22, 23, 20, 13 of 23 beat the structure head; Qwen O2 = 7 probe-only + 6 B1-family, 0 B2-family | S | `cs` recount |

## G. Results: out-of-fold effect sizes

| # | Claim | Status | Evidence |
|---|---|---|---|
| 77 | Table 4 (best probe-only vs comparators: ΔCV, ΔOOF, intervals, folds) | S | `cs`, `oc` "Best probe-only vs comparator" rows |
| 78 | A1 vs B2 O1: −0.039 (0.023–0.054) Llama, −0.024 (0.010–0.041) Qwen; 5/5 folds both | S | `oc` |
| 79 | H2 vs B2 O1 lower for both, by slightly less, intervals exclude 0 | S | `oc` −0.0315 [−0.0488, −0.0129]; −0.0199 [−0.0397, −0.0024] |
| 80 | H1 vs structure and vs B2 in O1: all four intervals include 0 | S | `oc` |
| 81 | Qwen O2: H1 vs B1 −0.010 (0.005–0.016); vs B2 −0.011 (0.004–0.018) | S | `oc` |
| 82 | Llama O2: no probe differs from B2; best probe-only −0.001 (−0.010 to +0.007) | S | `oc` (all C1 B2 intervals include 0) |
| 83 | AUROC A1 0.852 / 0.844, B2 0.793 / 0.774, structure 0.809 / 0.817 | S | `om` |
| 84 | Llama top decile 70.9% vs 57.2%; bottom 0.7% vs 2.6% | S | `om` risk_top / risk_bottom |
| 85 | O2 Spearman best probe-only 0.55 / 0.50 vs structure 0.47 / 0.42 | S | `om` |
| 86 | ECE < 0.03 for every O1 predictor in Table 5 | S | `om` ece: max is 0.0281 (Llama type×k) |
| 87 | Tables 5, 9 (OOF metrics O1 / O2) values | S | `om` |

## H. Results: where the signal helps

| # | Claim | Status | Evidence |
|---|---|---|---|
| 88 | Llama: A1 < B2 on 13/15 types (8 exclude 0); < structure on 11 (8). Qwen 12 (5) and 14 (5) | S | `optd` recount |
| 89 | Largest Llama gains over structure: letter count, paragraph count, bullets, title, quotation | S | `opt`: Δ 0.144, 0.120, 0.086, 0.072, 0.069 (next is word count at 0.045) |
| 90 | These are "mostly rules about the layout of the response" | S | 4 of 5 are layout/format types (letter count is the exception), so "mostly" holds |
| 91 | Letter count: B2 lowest for both generators (Llama 0.281 vs A1 0.347 vs structure 0.491); B2–A1 interval includes 0 | S | `opt`; `optd` B2→A1 +0.066 [−0.012, 0.159] (Llama), +0.036 [−0.102, 0.146] (Qwen) |
| 92 | Letter-count difficulty depends on the named letter, which only the rule text carries | C | Letter absent from structured features is supported (`features.py`, appendix). That the letter drives difficulty is not tested in any table |
| 93 | Keyword inclusion, keyword count, all-lowercase: no predictor improves materially on type×k | C | `opt`: mostly ≤1–2% gains, but Qwen keyword count H1+H2+A1 0.1687 vs type×k 0.1766 (−4.5%). "Materially" is undefined and there are no intervals |
| 94 | Mean within-type AUROC: Llama A1 0.73, structure 0.63, B2 0.59; Qwen 0.56, 0.54, 0.58 | S | `tab_per_type_*` / `opt` |
| 95 | Three Qwen types violated in <1% of checks | S | `desc_by_type.csv`: lowercase 0.007, no commas 0.001, title 0.000 |
| 96 | Much of the overall AUROC comes from knowing the rule type; internal features add mainly for Llama | S | Overall AUROC 0.85 vs within-type means 0.73 / 0.56 (inference, consistent) |
| 97 | A1's advantage over structure grows with k; k=1 0.332 vs 0.341, 0.154 vs 0.155; k=5 0.338 vs 0.387, 0.249 vs 0.274 | S | `tab_per_k` / `oof_per_k.csv`: gap increases monotonically for both |
| 98 | Interpretation of the k trend (model weighs rules) | C | Explicitly flagged as untested in the paper |

## I. Design choices and contrasts

| # | Claim | Status | Evidence |
|---|---|---|---|
| 99 | 50% depth selected in 3 cells, 75% for Qwen O2; last layer worst everywhere | S | `cv_layer_selection.csv` |
| 100 | Sum pooling selected in every cell by small margins | S | `cv_pooling_selection.csv` |
| 101 | RQ4: H1→H2 lowers O1 NLL 0.037 (0.018–0.056) Llama, 0.021 (0.0004–0.039) Qwen; O2 intervals include 0 | S | `oc` C5 |
| 102 | Per-type heads tell H1 the type but not parameters or keywords; the embedding supplies these | S | Design: `features.py`, analysis_plan §2 (explanation of the H2 gain is not separately tested) |
| 103 | RQ3: in O1 every single probe added to B1 or B2 lowers NLL, all intervals exclude 0 | S | `oc` C2/C3 (12 of 12 exclude 0) |
| 104 | This comparison is confounded with the shared→per-type head switch | S | `cs` per_type_heads False for B1/B2, True for B+probe |
| 105 | O2: adding H1 or H2 to B1 lowers RMSE for both; adding a probe to B2 makes no detectable difference | S | `oc` (B1+H: intervals exclude 0; all B2+probe include 0) |
| 106 | H1+A1 is 8.1% / 8.3% worse than A1 in O1; O2 combinations 2.3–4.3% better | S | `cv_complementarity.csv` recomputed |
| 107 | Attention features roughly 1,000 vs 4,096 hidden | S | `cs` widths 1,024 / 1,152 |
| 108 | Likely cause of the poor O1 fusion is the shared penalty | C | Hypothesis; not tested (the paper says so in Limitations) |
| 109 | B1→B2: Llama NLL −0.017 (0.002–0.032), RMSE −0.009 (0.003–0.015); Qwen no detectable difference | S | `oc` C6 |

## J. Verifier audit

| # | Claim | Status | Evidence |
|---|---|---|---|
| 110 | Official check = all upper case and langdetect == English | S | `audit_ec_causes.csv` cause categories; `audit_ec_consistency.csv` 0 disagreements |
| 111 | Qwen 12,554 of 29,097 violations (43.1%); 6.1% of checks; Llama 1,281 (4.2%) | S | `audit_ec_causes.csv`, `tab_ec` |
| 112 | Detected as German: 92.5% (Qwen), 99.0% (Llama) | S | `audit_ec_langdetect_codes.csv` |
| 113 | None mainly non-Latin script | S | `audit_ec_script_by_status.csv` (mainly_cjk = mainly_other_script = 0 for the langdetect-only rows) |
| 114 | Lower-cased text returns English for 9,839 / 12,554 (Qwen) and 1,263 / 1,281 (Llama) | S | `audit_ec_langdetect_only_assessment.csv` sums 7,695+1,330+814 and 1,209+51+3 |
| 115 | 3,520 of the Qwen responses are repetition loops | S | Same file: 1,116 + 2,404 + 0 |
| 116 | Top ten tasks account for 92% of Qwen's cases | S | `audit_ec_concentration.csv` 0.9167 |
| 117 | 16.0% with letter-count rule vs 2.2% without (Qwen) | S | `audit_ec_cooccurring_types.csv` 0.1602 / 0.0222 |
| 118 | These responses follow the rule as a person would read it | C | Partly supported (63% "English" by the function-word heuristic, 78% English when lower-cased), but 30% have few English function words and 3,520 are loops. Not established for all 12,554 |
| 119 | Corrected Qwen rate 14.2% → 8.1% | S | `desc_english_capital.csv` |
| 120 | No label changed; plan includes a drop-this-type sensitivity analysis | S | analysis_plan §8(b); `sens_english_capital.csv` |

## K. Remaining work, limitations, conclusion, reproducibility

| # | Claim | Status | Evidence |
|---|---|---|---|
| 121 | Structure head picks smallest λ in all 4 cells; one step inwards 7.9–11.1% higher | S | `cv_l2_edges.csv` 7.88, 8.59, 9.19, 11.07 |
| 122 | Several O2 configurations sit at an edge | S | `cv_l2_counts.csv`, `tab_l2` (7/7 and 5/0 edge configurations) |
| 123 | Widened-grid CV rerun would take about three hours | C | No projection table; head-CV runtime in `runtime.csv` is 2.2–4.4 h per process at the current grid |
| 124 | Test application: 11,070 prompts and 46,170 rows per generator | S | analysis_plan §7 D-task counts |
| 125 | Metrics, calibration bins from OOF deciles, 10,000-replicate bootstrap, Holm per family | S | analysis_plan §3–4, §6 |
| 126 | Table 7: 37 contrasts in 7 families (6, 4, 2, 10, 4, 4, 7) | S | analysis_plan §5 |
| 127 | RQ expectations (A1 and H2 in O1, H1 in Qwen O2, not Llama O2) | S | Consistent with `oc` C1 rows (forecast, not a result) |
| 128 | Controlled-analysis definitions A–E and reliability criteria | S | analysis_plan §7 |
| 129 | Table 8 pair counts (22,950 / 2,700 / 25,380 / 46,170; 1,080 / 7,560 / 11,070) | S | analysis_plan §7 |
| 130 | Pre-specified sensitivity analyses (truncation, uppercase rule, loops) | S | analysis_plan §8 |
| 131 | Some contrasts compare identical feature sets; kept in Holm (conservative) | S | analysis_plan §9 |
| 132 | p̂ SE up to 0.11 with n=20 | S | sqrt(0.25/20) = 0.112 |
| 133 | Qwen 2.5% truncated, 2.1% loops; Llama 0.2% / 0.1% | S | `desc_truncation.csv` 0.0249 / 0.0018; `desc_loops.csv` 0.0205 / 0.0008 |
| 134 | Qwen title rule almost never violated; per-type heads fitted on very few violations | S | `desc_by_type.csv` 0.0004; `cv_sparse_types.csv` (1 violated response in a fold) |
| 135 | Test set shares rule types and many exact rules with training | C | Shared types: S (`tab_types`); "many exact rules" relies on the 50/90 figure (#35) |
| 136 | Conclusion: A1 improves on fine-tuned encoder for both generators | S | As #78 |
| 137 | Conclusion: the improvement concentrates on layout and length rules | C | Against B2 the gains span layout, length and punctuation types. Against structure the biggest gain is letter count, and A1 is *worse* than structure on sentence count (0.618 vs 0.550, Llama). Only partly backed |
| 138 | Conclusion: the improvement grows with the number of rules | S | As #97 |
| 139 | Conclusion: H1 no better than structured features; B2 wins where difficulty lies in wording; matches probes in Llama O2 | S | #71, #91, #10 |
| 140 | Every table is generated from saved files by a script; quoted numbers are checked automatically | S | `generated/*.tex` headers "generated by report/make_tables.py"; `report/check_numbers.py` ran with 466 numbers, 0 unmatched |
| 141 | Source tree hash differs from FROZEN.json only by one wording-only edit to the report generator made a minute after freeze | C | `FROZEN.json` records only the hash; no edit log in the sources |
| 142 | Head selection 4.4 h (O1) and 2.2 h (O2) per generator; B2 grid about 6 GPU-hours | S | `runtime.csv` |
| 143 | Training details: L-BFGS 500 iterations; Adam 1e-3, batch 1,024; LoRA α=32, AdamW wd 0.01, 5% warm-up | S | `heads.py` (LBFGS_MAX_ITER = 500), `b2.py` (lora_alpha = 2×rank, wd, warm-up), analysis_plan §2 |
| 144 | Appendix tables 10–13 (layer/pool, B2 grid, λ, full CV) values | S | `cv_layer_selection.csv`, `cv_pooling_selection.csv`, `cv_b2_grid.csv`, `cv_l2_*`, `cs` |
| 145 | Table types: training-group counts per type and violation rates with intervals | S | `desc_by_type.csv` (n_groups, est, lo, hi) |
| 146 | Table types: test-group counts per type and sampled parameter ranges | C | No CSV in the provided sources holds test-group counts or parameter ranges |

## Tally

- Supported: 115
- Unsupported: 0
- Can't tell: 31

No claim is contradicted outright by a saved table. The weakest wordings are #69 ("B2 does not help for Qwen", where the CV loss is in fact 1.5% lower for Objective 1), #137 (layout and length concentration) and #118 (verifier failures "follow the rule as a person would read it"). Most can't-tell items are generation-phase provenance (prompt reproduction, split search, decoding, backend cosine) whose evidence sits outside the files I was allowed to read.
