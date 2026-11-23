# ManyIFEval Research Project — Master Decision Tracker

**Tracker v1.16 | November 14, 2026 | Based on research plan v3.6 (v3.4 counts; Amendment A1 withdrawn by A2) + runtime Amendment R1 (Betty CUDA pilot)**  
**Legend:** Decided = current design choice; Provisional = working specification pending audit/pilot; Open = needs a specific decision; Optional = contingent extension.  
**Purpose:** Keep the main specification, counts, rationale, and unresolved choices in one versioned place. Append dated changes and update affected counts whenever a decision changes.

## 1. Project and dataset overview

| Decision | Current value | Status | Concise rationale |
|---|---|---|---|
| Primary question | Predict instruction-violation risk before generation; measure incremental value of the generating LLM's representations over prompt-only features | Decided | Test useful predictive signal, not causal access to new information |
| Source | ManyIFEval; 216 source task descriptions | Decided | Verifiable instruction-following tasks |
| Task coverage | **All 216 tasks** | Decided | Complete coverage of the ManyIFEval task inventory |
| Task families | Proposed six primary categories: explanation/informational; creative; persuasive; practical; analytical; transformation/other | Provisional | Preserve diverse task settings; actual bins need source-task audit |
| Task selection | None: every task is used. Tasks are formed into groups with the proposed family taxonomy (greedy max-diversity grouping, fixed seed) | Decided | No sampling step; family labels only shape grouping and split balance |
| Grouping | 3 tasks per freshly sampled six-rule pool, which must be satisfiable for each of them; preferably different families | Decided | Enables controlled task comparisons holding rules fixed |
| Groups | 72 total | Decided | 216 / 3 |
| Split | 171/45 tasks; equivalently 57/15 groups (train/test only) | Decided | Grouped holdout prevents task/rule-pool/ordering leakage; five-fold CV only inside train |
| Splitting and coverage | Freeze 57/15 group split before fitting; balance task families and 15 rule types where feasible; no dedicated validation set | Decided | Credible unseen-task evaluation |
| Generating LLMs | **2 candidates: Meta Llama 3.1 8B Instruct and Qwen3-8B in non-thinking mode** | Selected for pilot, full run contingent on preflight | Original ManyIFEval model family + newer similar-sized cross-family test |
| Generator contexts | **Llama: 128K native; Qwen3: 32,768 native (131,072 with optional YaRN)** | Verified model-card limits; usage must pass preflight | Native capacities comfortably exceed typical short-rule prompts; avoid unnecessary RoPE extension |
| Qwen3 decoding mode | `enable_thinking=False` for every sample | Selected | Exclude variable hidden reasoning generation from verification; comparable text-only responses |
| Generator pairing | Same prompt/task/rule/COA specifications and splits; independent completions and probe training per LLM | Selected | Test cross-model robustness without pooling incompatible internal representations |
| Generator context-length gate | Tokenize every prompt using EACH model's tokenizer/chat template; longest input + reserved output + margin must fit; never truncate | Decided requirement; test pending | Protect rule text and pre-generation features |
| Attention-access gate | Pilot `output_hidden_states` and rule-span `output_attentions` via supported explicit weights/eager backend | Decided requirement; test pending | Flash/fused attention may not expose A1's needed weights |
| BGE-M3 encoder length | Verify full prompts fit separate 8,192-token encoder limit; no silent truncation | Decided requirement | Both baselines require full-prompt features |
| Two targets | Per-rule violation probability; expected mean fraction of rules violated | Decided | Separate rule-level and prompt-level risk |

## 2. Instruction taxonomy and rule construction

| Decision | Current value | Status | Concise rationale |
|---|---|---|---|
| Rule taxonomy | 6 broad categories, 15 types (keywords 3; length/count 4; formatting 3; letter case 3; beginning/ending 1; punctuation 1) | Decided | Follow benchmark's verifiable taxonomy |
| Rule identity levels | Category, type, exact instantiated rule | Decided | Probe weights are shared BY TYPE; exact instance identifies a target and provides optional rule-text features |
| Global exact-rule inventory | **None**; draw exact parameters per group from IFEval's code-default parameter pool (each type's official `build_description()` called with no arguments, from the pinned ManyIFEval copy of the IFEval code) | **Decided** | Explicit, seed-reproducible generators from the released code; see *Rule-parameter distribution* |
| Candidate rule pool per group | **6 independently sampled exact rules per group; distinct types** | Decided | Broad parameter coverage and compatibility while retaining five-rule maximum |
| Rule count per prompt | k in {1,2,3,4,5}; maximum k=5 | Decided | Feasible multi-instruction complexity |
| Compatibility | Jointly compatible six-rule pool (D1, D3). Task level: **feasibility only**. The pool must not be impossible for any of the group's three tasks, confirmed at manifest build by running the official verifiers on the pool's witness with each task's text (`task_feasibility` in `manifest_summary.json`). **No relevance or appropriateness filter**: unusual but possible pairings (e.g. a tweet with ≥ 17 sentences) are kept | Decided | Every intended subset remains satisfiable. None of the 15 verifiers reads the task text, and the released ManyIFEval prompts contain the same kinds of pairings |
| Rule sampler | Sample 6 distinct types from ManyIFEval’s 15; instantiate each by calling the official instruction class's `build_description()` with no arguments (seeded per group); feasibility lookahead, resample provably infeasible draws (D3), reject/backtrack, verify | Decided; per-type D3 rejection rates in `data/manifest/pool_audit.json` | Prevent dead-end or contradictory combinations without a fixed global instance list |
| Rule-parameter distribution | **Released code's default generators only** (uniform integer ranges, 50/50 relations, 2 distinct `WORD_LIST` words, uniform lowercase letter), filtered only by D1/D3 checks with resampling. **Not** the empirical distribution of the published ManyIFEval parameter values (e.g. uniform between the observed minimum and maximum). Follows the released code's procedure; makes **no claim** to reproduce the original authors' dataset-generation distribution | **Decided** | (1) Reproducible: explicit distributions and ranges, exact with fixed seeds. (2) No invented distributions: the process behind the published out-of-range values is undocumented. (3) Goal is a new controlled benchmark on ManyIFEval's types and verifiers, not a replica of its empirical distribution. (4) Unsatisfiable or incompatible draws are rejected and resampled; every rejection and resampling decision is logged (`rule_pools.jsonl`) and summarized per type (`pool_audit.json`), since filtering shifts the final distribution |
| Exact rule stability | **Keep rule instances fixed only inside their three-task group**; independently sample again for the next group | Decided | Valid matched comparisons without requiring cross-group repetition |
| Per-rule labels | Verify every included rule in every response | Decided | Each prompt contributes k training records |
| Rule sampling coverage | Audit 15 types, six categories, parameter-variant diversity, per-type sample counts and incidental exact-instance recurrence | Decided audit | H1/H2/A1 have per-TYPE heads; instances need NOT recur across groups |

**Sampling scope:** For each group, propose six distinct ManyIFEval instruction types, draw their exact parameters from IFEval's code-default pool (official `build_description()` with no arguments), and reject/backtrack until the complete six-rule set is jointly satisfiable; then confirm it is satisfiable for each of the three tasks (feasibility only, no relevance filter). Parameters follow the released code's default generators, not the published dataset's empirical values (see *Rule-parameter distribution*). Keep those six exact instances fixed throughout that group; **do not require or engineer reuse in other groups**. Log the actual sampling/rejection distribution; filtering may change relative rule-type frequencies.

## 3. Chains, combinations, and position designs

| Decision | Current value | Status | Concise rationale |
|---|---|---|---|
| Chain design | 6 cyclic nested chains from one randomized 6-rule ordering, prefixes only through k=5 | Decided, revisit if composition diversity inadequate | Matched add-one-rule comparisons; balanced single-rule exposure |
| Chains per group | 6 | Decided | Each candidate begins one chain |
| Subsets at each k | 6 distinct sets for every k=1–5; 30 total across k | Decided | Controlled subset budget and k=5 composition variation |
| Max-k compositions | Six different 5-of-6 subsets | Decided | Swap companions even at k=5 |
| Per-rule subset frequency | Each candidate appears in k subsets at rule count k | Guaranteed by design | Balanced marginal inclusion by k |
| Order selection | Index-one, strength-two component orthogonal array (COA) per fixed subset | Decided | Pairwise-position balance without enumerating all k! orders |
| COA order counts for k=1–5 | 1, 2, 6, 12, 20 | Decided | Each rule occurs in every position k−1 times (k≥2) |
| COA implementation | k=4: 12 even permutations; k=5: 20 affine permutations modulo 5 | Decided implementation; verify in code | Reproducible exact balancing |
| Order design limitation | COA does NOT exhaust all permutations or isolate all higher-order order effects | Acknowledged | Avoid overstating experimental control |
| Reuse across tasks | Apply same 246 ordered configurations to all three tasks in a group | Decided | Clean task-change comparison |

### Counts per task

| k | Subsets | Orderings per subset | Prompts | Prompt–rule records | Target appearances per position per subset |
|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 1 | 6 | 6 | 1 |
| 2 | 6 | 2 | 12 | 24 | 1 |
| 3 | 6 | 6 | 36 | 108 | 2 |
| 4 | 6 | 12 | 72 | 288 | 3 |
| 5 | 6 | 20 | 120 | 600 | 4 |
| **Total** | **30 size-specific subsets** | — | **246** | **1,026** | — |

### Scale by split (per generating LLM)

| Split | Tasks | Groups | Unique prompts | Prompt–rule records | Sampled generations |
|---|---:|---:|---:|---:|---:|
| Train | 171 | 57 | 42,066 | 175,446 | 841,320 |
| Test | 45 | 15 | 11,070 | 46,170 | 221,400 |
| **All** | **216** | **72** | **53,136** | **221,616** | **1,062,720** |

Each group has 738 unique prompts (246 × 3 tasks) and 14,760 sampled responses (738 × 20). Across all groups there are 432 newly sampled candidate-rule slots; 342 are in training and 90 are in test. These slots are not a fixed set of 432 globally unique exact rules. Dataset size excludes supplementary controlled prompts. **Both generators together:** 53,136 unique prompt texts; **106,272 generator-prompt combinations**; **2,125,440 sampled completions**; **443,232 generator-specific prompt-rule label records**. Mandatory first 100-prompt pilot is 2,000 completions per LLM (**4,000 across both**). Optional 4-group extended pilot is 59,040 per LLM (**118,080 across both**), included in the full budget only if settings are unchanged.

## 3A. Training/test split and model-selection protocol

| Decision | Current value | Status | Concise rationale |
|---|---|---|---|
| Permanent partitions | **57 TRAIN groups (171 tasks) and 15 TEST groups (45 tasks)**; **NO permanent validation partition** | **Decided** | Approximately 80/20 at the correct independent design-block unit |
| Split proportions | **79.17% / 20.83% of groups and tasks** | Decided | Whole groups of three cannot be fractionally split |
| Train/test prompt counts | **42,066 / 11,070** prompt texts per LLM | Decided | 246 prompts per task |
| Train/test generation counts | **841,320 / 221,400** responses per LLM | Decided | 20 stochastic responses per prompt |
| Train/test prompt–rule records | **175,446 / 46,170** per LLM | Decided | 1,026 prompt–rule records per task |
| Train/test sampled candidate-rule slots | **342 / 90** | Decided | Six independently sampled exact instances per three-task group |
| Hyperparameter selection | **5-fold GroupKFold inside 57 TRAIN groups only**, 12/12/11/11/11 groups per fold | Decided | Optimize without test-set peeking and without third permanent split |
| Hyperparameter search | **Staged small grid**, specified in the table below | Recommended for pilot | Limit compute and maintain comparable tuning budgets |
| Training-fold fit groups | 45 or 46 groups depending on held-out fold | Decided | CV respects group-level dependence |
| Selection scores | Objective 1: grouped CV binomial NLL; Objective 2: grouped CV MSE | Decided | Objective-specific target-aligned model selection |
| Final fit | Refit each selected method on **all 57 train groups** | Decided | Use available training data after model selection |
| Early stopping | Choose epoch schedules within group CV (or prespecified fixed epochs), never using test labels | Decided | Avoid hidden test-set tuning |
| Model-fit counts | **46 final trained configurations/LLM; 92 final generator-specific fits** plus temporary CV fits | Decided | CV adds compute, not a new feature configuration or generations |
| Final evaluation | **Test 15 groups once**, for standard evaluation and prespecified matched controlled analyses | Decided | Honest held-out inference |
| Statistical caveat | 15 independent rule-set blocks; group-aware paired uncertainty | Decided | Thousands of prompts do not yield thousands of independent task pools |

### Compact hyperparameter grid (research-plan Section 9.1.1)

| Component | Recommended candidates | Why / control |
|---|---|---|
| All heads | Linear sigmoid versus one-hidden-layer MLP (128 units); L2 penalty {0.01, 0.1, 1}; initial MLP schedule 30 epochs | Control model complexity and overfitting without massive search |
| H1/H2 hidden layer | 50%, 75%, 100% depth; ONE shared selected layer for H1/H2 and combinations | Fair exact-rule conditioning ablation. **Selected (grouped CV, 2026-11-14):** llama Obj1 row 16 (50%); llama Obj2 row 16 (50%); qwen Obj1 row 18 (50%); qwen Obj2 row 27 (75%); 100% = row L+1 (pre-final-norm) |
| A1 attention pooling | Target rule-span SUM vs length-normalized MEAN | Rule-length sensitivity; reuse selection across combinations. **Selected (grouped CV, 2026-11-14):** llama Obj1 SUM; llama Obj2 SUM; qwen Obj1 SUM; qwen Obj2 SUM |
| B2 fine-tuning | Choose full or LoRA in pilot and hold fixed; full LR {1e-5,3e-5}, LoRA LR {1e-4,2e-4}; epochs {1,3}; weight decay 0.01 fixed; LoRA rank=16/dropout=0.05 fixed initially | Limited adaptation search; no new model family. **Decided 2026-11-14 (training-only): LoRA**, by the fixed rule (lower held-out NLL, Llama Objective 1, fold 0, 1 epoch, each method's lower LR); selected llama Obj1 LR 0.0001, 1 ep; llama Obj2 LR 0.0002, 3 ep; qwen Obj1 LR 0.0001, 1 ep; qwen Obj2 LR 0.0002, 1 ep |
| Fusion models | Same head grid; freeze selected feature extraction and B2 adaptation | Test incremental benefit without search-budget advantage |

**Fold hygiene:** For each of five folds, fit B2 encoders, preprocessing, and heads only on the 45–46 fitting groups; do not pretrain B2 on all 57 groups before CV. Select using mean held-out-fold binomial NLL for Objective 1 or MSE for Objective 2. Refit chosen settings on all 57 training groups, then use 15 test groups exactly once. Hyperparameter grids are recommended pilot starting values, subject to a training-only feasibility check. CV adds fitting compute but **no additional completions**.

**Methodological support:** [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) enforces non-overlapping groups across CV folds; [Cawley & Talbot (2010)](https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf) explain model-selection bias; [IFCC review (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10320011/) describes conventional ~75–80% training splits and grouping related observations. The exact 80/20 ratio is a reasonable design convention, **not** a statistically guaranteed optimum; preventing leakage and test-guided selection is more important.

## 4. Labels, feature extraction, and training

| Decision | Current value | Status | Concise rationale |
|---|---|---|---|
| Generation repeats | 20 stochastic completions per unique prompt | Decided initially; pilot-check noise | Obtain probability labels rather than a single binary outcome |
| Generation consistency | Fixed prompt/checkpoint/chat template/decoding policy; vary sampling randomness | Decided | Repeatable input conditions |
| Violation vector | k binary verifier results per completion (1=violated, 0=followed) | Decided | Explicit supervision for all included rules |
| Objective 1 label | Violations for a target rule / 20 | Decided | Empirical target-specific violation frequency |
| Objective 2 label | Sum of violations across all 20 completions and k rules / (20k) | Decided | Empirical expected fraction violated, not probability of any violation |
| Objective 1 training unit | Prompt–rule pair | Decided | One prediction for each included rule |
| Objective 2 training unit | Prompt | Decided | One prediction per prompt |
| Objective 1 loss | Aggregated binomial negative log-likelihood | Decided | Uses 20 underlying binary verifier results correctly |
| Objective 2 loss | Sigmoid-bounded mean squared error initially | Decided starting point | Regression on a fraction, allowing dependent rule outcomes |
| Input timing | Features only from prompt, before response generation | Decided | Prevent label and generation leakage |
| Hidden state | Generating LLM final prompt token at predetermined layer(s) | Decided family; layer choice open | Pre-generation contextual signal |
| Attention | Final prompt token attention to each exact rule span, across heads/layers | Decided family; aggregation details open | Rule-specific pre-generation signal |
| Source text encoder | BAAI/bge-m3 dense embeddings, 1,024 dimensions | Decided | Independent full-prompt and rule text representation |
| Prompt-only features | Full-prompt and target-rule embeddings, k, type, position, typed parameters, companion types, task family and length; exact IDs for audit only | Decided | Transfer to newly sampled exact instances without fixed-ID vocabulary |
| Objective 2 rule features | All rule vectors ordered into ≤5 slots, with padding and presence masks | Decided | Preserve order and membership for prompt-level prediction |
| Preprocessing | Fit feature encodings/scaling within each training CV fold, then refit on all training groups | Decided | Prevent leakage |
| Probe-only feature caching | Cache hidden states and rule-span attention once per prompt where possible | Decided approach | 46 configurations for a single generator share its 20-response labels; a second generator requires its own completions |

## 5. Five core model families

| Model | Objective 1 implementation | Objective 2 implementation | Rationale |
|---|---|---|---|
| **B1** | Frozen BGE-M3, full prompt + exact target rule + structure; learned head | Frozen BGE-M3, full prompt + all ordered exact rules + structure; learned overall head | Strong task-supervised frozen-text baseline |
| **B2** | Same raw inputs, fine-tune BGE-M3 and head | Same prompt-level raw inputs, fine-tune BGE-M3 on overall target | Challenge internal probes with task-adapted prompt encoder |
| **H1** | **One head per rule TYPE (15)**; generating LLM final-token hidden state; no explicit exact-rule vector | One shared overall head on hidden state + ordered rule-type information | Pool parameter variants under each type |
| **H2** | **One head per rule TYPE (15)**, using hidden state + frozen exact target-rule embedding; NO head per instance | One shared overall head on hidden state + embeddings of all exact rules | Test whether explicit rule-text conditioning helps within type |
| **A1** | Attention to target rule token span across heads/layers; **one head per rule TYPE (15)**, not per instance | Ordered, masked per-rule attention vectors across all rules; one shared overall head | Test attention signal while pooling independently sampled variants by type |

**IMPORTANT:** For Objective 1, “per-rule prediction” means one prediction TARGET per included rule; it does **not** mean one uniquely fitted probe per exact rule. H1, H2, A1, and configurations containing these probe signals use shared weights for each of the **15 instruction types**. Parameter variants within a type share the SAME type-specific probe weights, whether or not their exact text has occurred during training. The 23 per-rule configurations count feature combinations, not the number of trained per-type head tensors. B1/B2 baseline-only use their shared rule-conditioned prediction head. For Objective 2, there is one overall-fraction head per configuration, not 15 separate overall heads.

**Excluded:** A2 and B1+B2 combinations. H1+H2 combinations are retained but secondary due to redundant hidden-state sources.

## 6. Exactly 23 configurations for each objective and generating LLM

| Family | # | Configurations |
|---|---:|---|
| Probe-only | 7 | H1; H2; A1; H1+H2; H1+A1; H2+A1; H1+H2+A1 |
| B1 family | 8 | B1; B1+H1; B1+H2; B1+A1; B1+H1+H2; B1+H1+A1; B1+H2+A1; B1+H1+H2+A1 |
| B2 family | 8 | B2; B2+H1; B2+H2; B2+A1; B2+H1+H2; B2+H1+A1; B2+H2+A1; B2+H1+H2+A1 |
| **Total per objective** | **23** | Models combined by explicit feature fusion + objective-specific head |

| Output | Trained | Evaluated | Reason |
|---|---:|---:|---|
| Per-rule probabilities | 23 | 23 | Objective 1 |
| Direct overall fractions | 23 | 23 | Objective 2 |
| **Total per LLM** | **46** | **46** | Only two separately trained objectives; no derived averaged-prediction family |

**Across both generating LLMs:** 46 DISTINCT configuration definitions, evaluated/fitted separately on each model = **92 generator-specific model fits/evaluations**. There are no extra 23 averaged-prediction configurations.

## 7. Evaluations, metrics, and uncertainty

| Decision | Current value | Status | Rationale |
|---|---|---|---|
| Evaluation 1 | All 11,070 prompts from 45 unseen test tasks / fifteen groups | Decided | General predictive performance |
| Objective 1 metrics | Binomial NLL, Brier, calibration; AUROC and RiskTop/RiskBottom secondary | Decided | Accuracy, discrimination, calibrated risk |
| Objective 2 metrics | MAE, RMSE, proportion calibration; Pearson/Spearman secondary | Decided | Overall-fraction prediction accuracy |
| Main ablations | B1 vs B2; baseline vs probes; +H1/+H2/+A1; H1 vs H2; hidden+attention complementarity | Decided | Test incremental utility and architectural hypotheses |
| Evaluation 2 | Matched held-out prompts from the same 45 test tasks, with supplementary prompts only if necessary | Decided framework; matching algorithm open | Predict changes rather than levels of risk |
| Position (A) | Fix task, exact rule set, k, target; compare target positions using COA-balanced mean rates | Decided primary analysis | Test positional sensitivity; not a strict target-only move |
| Count (B) | Nested +1-rule prompt comparisons over k transitions 1→2→3→4→5 | Decided framework | Test response to added constraint; identity/count confounded in one pair |
| Composition (C) | Fix task, k, target and position; replace exactly one companion | Decided framework; strict matches audit | Test instruction interactions |
| Task (D) | Fix exact ordered rules; compare three different task texts within group | Decided | Test task-dependent risk changes |
| Difficulty (E) | Training-only k=1 failure rates summarized by **instruction type**; optional easy/medium/hard type buckets | Provisional | Test instances may be unseen; exact-instance training difficulty often unavailable |
| Change score | Absolute error between predicted and observed differences; sign accuracy only for reliable changes | Decided | Evaluate sensitivity to controlled differences |
| Uncertainty | Paired evaluation; bootstrap or similar confidence intervals clustered by the 15 held-out 3-task pools | Decided | Prompts within the 15 test groups are dependent |
| Supplementary panel | 30–50 matched pairs per factor as a pilot where strict pairs are absent | Optional, not power-justified | Improve strict isolation; adds generations if new prompts |
| Generalization claim | Unseen tasks with independently sampled ManyIFEval exact-rule pools; measure realized train/test rule-instance overlap | Decided scope | Do not claim all test instances are unseen without enforcing and verifying a disjoint split |

## 8. Generator model sources and protocol

| Model | Model card | Research relevance | Native context / use |
|---|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct | Llama 3.1 8B was evaluated in Harada et al., ManyIFEval (2025) | 128K native; gated-license access may be needed |
| `Qwen/Qwen3-8B` | https://huggingface.co/Qwen/Qwen3-8B | Contemporary similar-sized model; Qwen3 report gives strong IFEval instruction-following results | 32,768 native; YaRN extension optional; fix `enable_thinking=False` |

**Sources:** https://aclanthology.org/2025.findings-emnlp.896/ ; https://arxiv.org/abs/2505.09388 ; https://huggingface.co/docs/transformers/en/attention_interface . Qwen's IFEval scores should not be described as ManyIFEval scores.

**Preflight:** check actual context/tokenization and output reserve for all prompts using each model's template; run A1/H1/H2 extraction/memory pilot; fix decoding, tokenizer, checkpoint revision and response limits; avoid quantization unless audited for behavioral change. Cross-model robustness is of the probing PROCEDURE, not necessarily shared fitted weights.

## 9. Pilot and unresolved decisions

| Choice / next step | Current state | Why it matters |
|---|---|---|
| **Mandatory first pilot** | **100 TRAIN-only distinct prompts × 20 completions/model = 2,000/model, 4,000 total**; same prompt texts for both generators; 20 prompts at each k=1–5. **Status: pending on the frozen manifest (Mac `pilot`, Betty `pilot_betty_dgx-b200`); only outputs that match the manifest's hashes count.** | First measure actual M4 Max throughput, context fit, verifier quality and introspection feasibility |
| **CUDA re-run of the pilot (Amendment R1)** | **Status (2026-11-12): Betty pilot complete (Slurm job 10426227, all gates pass); bulk generation complete 2026-11-13 (Slurm 10510610, 24/24 tasks, dgx027 excluded; 1,062,720 responses per model) and verified (Slurm 10510611; 221,616 prompt-rule and 53,136 prompt labels per model); outputs copied to the Mac, all 272 files checksum-identical; bulk features complete on the Mac (all 53,136 prompts per model, 17.5 / 19.5 GB; every shard checked: prompt and rendered hashes match the manifest, no NaN or unwritten rows, rule masks match k). Only outputs that match the manifest's hashes count toward the totals.** Backend findings (`docs/pilot_report_betty.md`): tokens identical to the Mac; hidden-state cosine ≥ 0.998; attention-sum difference flagged (max 0.066). The official `english_capital` checker's langdetect step fails much all-caps English (Qwen: 48 of 84 violations come only from it); this is recorded for the methods team, and the checker is unchanged. **Bulk approved 2026-11-10 by the project owner: Betty, vLLM 0.25.0 BF16, vLLM sampler order (temperature first);** queued on 2026-11-11 alongside the Betty pilot (v1.15), which runs first as a pre-check. The main run generates all 53,136 prompts itself; pilot responses (Betty or Mac) are not imported into the totals. Bulk features: **Apple MPS on the Mac** for all 53,136 prompts (changed in v1.12 from "CUDA only, deferred"); never mixed with CUDA features. Seeding: one engine seed per chunk job. Same 100 TRAIN prompts and frozen token IDs, re-run on Penn PARCC Betty (NVIDIA B200, one GPU per model) under a new run id. Generation: vLLM 0.25.0, BF16. Features: HF transformers eager BF16 on CUDA. Mac pilot kept unchanged as run `pilot`. | Faster bulk generation; cross-backend checks quantify what changes |
| Pilot selection | Spread selected prompts across training groups, task families, rule types and prompt lengths; audit worst-case prompt lengths separately over all 53,136 | Representative speed and error benchmark without test leakage |
| Pilot completion gates | Verify full context and output reserve, sampling settings, no truncation, viable verifier/labels, final-token hidden states and rule-span attention, realistic memory and runtime | Decide if full generation is feasible before running ~1.42M completions |
| **Optional expanded pilot** | Four TRAIN groups × three tasks × 246 prompts × 20 = **59,040 responses/model, 118,080 across two** only if additional end-to-end checks are needed | Existing larger pilot is a contingency, not the mandatory first run |
| Pilot budget treatment | Count retained pilot generations within full 1,062,720/model, if checkpoint/precision/decoding/serialization identical; else rerun | Avoid double-counting and distribution shifts |
| Rule-instance sampling policy | **No frozen list.** Direct sampling of six compatible instances per group from the original ManyIFEval templates, with parameters from IFEval's code-default pool; log type-proposal distribution, seeds, and rejections | Reproducible sampling without restricting original benchmark variability |
| Per-type probe policy | **Decided: H1, H2, A1 per instruction TYPE (15)** for Objective 1; never per exact sampled rule. Objective 2 has a shared prompt-level head. | Pool examples across rule variants and generalize to new parameter instantiations |
| Task-family rubric and labels | **Provisional**; inspect 216 tasks and finalize before sampling | Ensures justified strata |
| Random seeds and selected IDs | **Open** | Reproducibility |
| Generating models and attention extraction | **Selected for pilot:** Llama 3.1 8B Instruct; Qwen3-8B non-thinking. Hidden layer(s), attention statistic and hardware capability open. | Two-model feasibility, honest cross-architecture comparison |
| Decoder settings | **Open** | Affects target probabilities |
| B2 adaptation strategy (full vs LoRA) | **Pilot choice pending**; select one and hold fixed, with limited LR/epoch grid in Section 3A | Fair feature comparisons and cost |
| Head/layer/attention hyperparameter grids | **Recommended and documented**; final values selected through training-only grouped CV | Keep model selection prespecified and feasible |
| Number and method of strict matched pairs for Evaluation 2 | **Open** | Avoid post-hoc comparison selection |
| Rule-difficulty bucket thresholds | **Open / optional** | Avoid arbitrary or label-leaking difficulty labels |
| Unseen-exact-rule-variant test | **Optional** | Different from existing unseen-task test |
| Parameter-pool conformance | **Audited at manifest build:** 432/432 exact rules inside IFEval's code-default pool; per-type value histograms, relation split and D3 rejections in `data/manifest/pool_audit.json` | Verify sampling uses exactly the code-default pool without a preset inventory size |
| Test precision | 15 independent groups in the primary test split | Interpret subgroup analyses cautiously |

## 10. Change log

| Date | Tracker version | Change |
|---|---|---|
| 2026-11-09 | 1.0 | Consolidated research plan v2.7 into a master decision tracker |
| 2026-11-09 | 1.1 | Updated plan v2.8: exactly 23 per-rule + 23 direct overall = 46 trained/evaluated configurations; removed 23 aggregate-prediction comparisons; required verified long-context generating model and separate encoder fit checks |
| 2026-11-09 | **1.2** | Selected **Llama 3.1 8B Instruct** (128K native, included model family in ManyIFEval) and **Qwen3-8B non-thinking** (32,768 native; YaRN extended only if essential); added tokenizer/attention preflight, two-model completion budget, and **46 designs × 2 generators = 92 fitted instances** |

| 2026-11-09 | **1.3** | Removed the global frozen exact-rule inventory; sample six compatible exact instances anew per three-task group from original ManyIFEval types and parameter options. Retained within-group exact-rule locking; updated type coverage, baseline input IDs, rule-difficulty definition and seen/unseen-variant claims. Counts unchanged. |
| 2026-11-09 | **1.4** | Clarified that Objective 1 H1/H2/A1 probes are trained **per instruction type**, never per exact rule instance; exact-rule text in H2 is conditioning data, not a dedicated head. Kept 23+23 configurations unchanged. |

| 2026-11-09 | **1.5** | Removed the dedicated validation partition. Finalized the grouped train/test split of three-task groups; run five-fold GroupKFold within training for all hyperparameter/layer/early-stopping decisions and refit on all training groups before untouched test evaluation. Updated all counts; generation totals and 46-per-LLM configurations unchanged. |

| 2026-11-09 | **1.6** | Updated plan v3.3 with a small, staged tuning specification: shared head complexity/L2 grid, common H1/H2 hidden layer, A1 pooling, fixed B2 strategy plus small LR × epoch search, consistent five-fold group-CV fold hygiene, and no new generations/configuration families. |
| 2026-11-09 | **1.7** | Updated plan v3.4: first run a 100-prompt training-only M4 Max pilot, 20 completions per prompt and generator (4,000 completions total); audit output throughput, context/truncation, verifier labels, memory, hidden/attention accessibility and runtime; optional 4-group expansion only if needed. Keep full-generation totals unchanged when compatible pilot outputs are reused. |
| 2026-11-09 | **1.8** | **Amendment A1 (plan v3.5): 10 completions per prompt instead of 20**, approved by the project owner after the pilot to halve runtime on one M4 Max. All response totals halved; labels use n = 10. Pilot stays as executed (20 per prompt); its sample indices 0–9 count toward the main dataset and 10–19 are excluded from main labels. Prompts, prompt–rule records, models, decoding, verifier, features and 46 configurations per generator are unchanged. |
| 2026-11-09 | **1.9** | **Amendment A2 (plan v3.6): Amendment A1 withdrawn; 20 completions per prompt restored**, by the project owner before any main-run generation. All counts return to the 20-per-prompt values; labels use n = 20. All 20 pilot samples per prompt may be retained within the main totals. |
| 2026-11-10 | **1.10** | **Amendment R1 (runtime/hardware): CUDA re-run of the 100-prompt TRAIN-only pilot on Penn PARCC Betty**, requested by the project owner. One GPU per model (dgx-b200 full B200 by default; MIG slices optional). Generation backend **vLLM 0.25.0** (torch 2.11.0+cu129; newest vLLM with a CUDA 12.x build for the cluster's CUDA 12.8 driver). Introspection **HF transformers 5.22.2, eager attention**. BF16 throughout, no quantization. Unchanged: prompts and frozen token IDs (so chat templates and Qwen3 `enable_thinking=False` are identical), checkpoints, decoding values, max_new_tokens 2,048, 20 samples per prompt, verifier, feature definitions. New run id (`pilot_betty_<partition>`); the Mac pilot is protected and kept. **Known differences, flagged:** (1) GPU kernels and RNG differ, so samples and representations are not identical. (2) **Sampler definition:** mlx_lm applies top-p/top-k to untempered probabilities and then samples with temperature, while vLLM/HF apply temperature first. Measured on Mac pilot responses, MLX's definition puts ≈ 5.7 (Llama) / 4.6 (Qwen) expected tokens per 512-token response outside vLLM's support (mean per-token TV ≈ 0.017–0.019). (3) Seeding: one engine seed per run (per-sample seeds optional; they slow vLLM's sampler). Cross-backend checks: token IDs/spans, hidden-state cosine and max-abs diff, rule-span attention, vLLM-vs-HF log-probs, verifier rates per k with prompt-bootstrap CIs. **Bulk generation not authorised;** which backend and sampler definition to use for the bulk run is open pending this pilot. |
| 2026-11-10 | **1.11** | **Bulk generation approved** by the project owner after the Betty pilot (`docs/pilot_report_betty.md`): Betty `dgx-b200`, vLLM 0.25.0 + torch 2.11.0+cu129, BF16, no quantization, **vLLM/HF sampler order** (temperature, then top-k, then top-p). Counts unchanged. The **Betty pilot** (run `pilot_betty_dgx-b200`, 2,000 per LLM) is imported into the main totals. The **Mac pilot is not** (different backend and sampler); it stays as the recorded feasibility run. Execution: contiguous manifest chunks per model, one B200 each, then CPU verification. Pre-generation features: computed on CUDA only (never mixed with MPS features), **deferred until project storage is available** (home quota 50 GB; the full feature set does not fit). Seeding: one engine seed per chunk job, derived from (generation seed, backend, model, run, chunk). Open: whether the flagged attention-sum differences need a fixed tolerance (methods team). |
| 2026-11-10 | **1.12** | **Feature backend changed: all bulk pre-generation features on Apple MPS (the Mac), not CUDA**, approved by the project owner. Reason: Betty home quota (50 GB, ≈ 2.8 GB free) cannot hold the full feature set, and project storage is not yet available. Same extractor (HF transformers 5.22.2, eager, BF16), same frozen token IDs, same layout (all-layer final-prompt-token hidden states + per-layer/head rule-span attention sum and mean). One backend for all manifest prompts × 2 models; the Betty pilot's CUDA features are kept as a diagnostic only and never mixed in. Expected cost from the Mac pilot: 0.19 / 0.17 s per prompt. Trade-off recorded in `docs/pilot_report_betty.md`: MPS vs CUDA hidden-state cosine ≥ 0.998, max rule-span attention-sum difference 0.09; generation used vLLM, whose numerics neither feature backend reproduces exactly. Extractor made resumable (per-shard index commits) for the multi-hour run. **FP32 reference check (100 pilot prompts, CPU):** Mac MPS and Betty CUDA BF16 features are equally far from FP32 (median relative error 1.13% vs 1.13% Llama, 1.48% vs 1.51% Qwen; Mac closer in 52% / 51% of prompt-layer cases), so the Mac choice carries no accuracy cost; details in `docs/pilot_report_betty.md`. No design, count, or label change. |
| 2026-11-11 | **1.13** | 2026-11-11: Rule parameters drawn from IFEval's code-default pool (official `build_description()` with no arguments, D2); manifest and tokenization built from it. |
| 2026-11-11 | **1.14** | 2026-11-11: Expanded the task set from 144 sampled tasks to all 216 ManyIFEval tasks (72 groups; 57 train / 15 test groups; 53,136 prompts; 1,062,720 responses per model); regenerated manifest, tokenization and pilot selection. |
| 2026-11-11 | **1.15** | 2026-11-11: Queued bulk generation on Betty and started the Mac bulk feature run before the Betty pilot report, approved by the project owner, because Slurm queue waits (≈ 1 day per submission) dominated the timeline. The Betty pilot is kept and runs first as a pre-check; the bulk array is cancelled if it fails. The main run generates all 53,136 prompts itself, so pilot responses are not imported. Bulk array time limit 40 min per chunk. No design, count, or label change. |
| 2026-11-14 | **1.16** | **Probing, evaluation and results phase authorised** by the project owner (`PROMPT_probing_and_results.md`); scope sections of `CLAUDE.md`, `AGENTS.md`, `README.md` updated; code in `src/mife/probing/` (`docs/probing_schema.md`). Frozen inputs: run `main` manifest, responses, verification and Apple MPS features; no pilot or CUDA features. All choices from 5-fold GroupKFold over the 57 TRAIN groups (fold sizes 12/12/11/11/11, seed 20261201). Selected: hidden row llama Obj1 row 16 (50%); llama Obj2 row 16 (50%); qwen Obj1 row 18 (50%); qwen Obj2 row 27 (75%); A1 pooling llama Obj1 SUM; llama Obj2 SUM; qwen Obj1 SUM; qwen Obj2 SUM; B2 LoRA (rank 16, dropout 0.05), llama Obj1 LR 0.0001, 1 ep; llama Obj2 LR 0.0002, 3 ep; qwen Obj1 LR 0.0001, 1 ep; qwen Obj2 LR 0.0002, 1 ep; head settings per configuration in `results/prespec/analysis_plan.md`. The L2 grid {0.01, 0.1, 1} is used as specified; edge selections are flagged in `results/tables/cv_l2_edges.csv`. B2 ran on Betty (dgx-b200) from a TRAIN-only bundle because the Mac projection exceeded 24 h (project owner's choice); the Betty generator checkpoints were deleted with approval to free quota. Pre-specification frozen 2026-11-14: `analysis_plan.md` SHA-256 `5070f05b7eb8ed1ec18528c5321053264e48909f4076ddc8279b381bcd265b7d`, source-tree SHA-256 `2cb1398ae4864633…` (no git repository). Fixed §7 and §9: the test split has **15** held-out groups (previously written as "ten"). TEST set not read; unblinding awaits the project owner's go. |
| 2026-11-23 | **1.17** | **Split amendment S3 proposed** by the project owner (`docs/amendment_split_s3_2026-11-23.md`): a ~70/30 group split (22 test groups, 16,236 test / 36,900 train prompts) with a new hard check that every compatible type pair co-occurs in at least 2 TEST pools (3 is infeasible: one pair occurs in only 2 of 72 pools). Proposed split in `results/split_s3/proposed_split.json` (`src/mife/split_v3.py`, `configs/split_s3.yaml`, seed 20261123). Status: **proposed only**; the manifest `split` column, the 2026-11-14 freeze, folds, caches and `configs/experiment.yaml` are unchanged until the owner adopts it. |

**Update protocol:** Whenever a decision changes, update its row, recompute affected counts, mark pending decisions as decided only after validation, and append a dated change-log entry. This file does not auto-update; replace it with the revised version when discussing changes in a future chat.
