# Pre-generation audit of the expanded held-out evaluation design

**Date:** 2026-11-22. **Status:** audit and planning only. No responses were generated, no verifier was run, and no TEST label, outcome, feature or prediction was read. The split, frozen data, configs and V1/V2 outputs are unchanged.

**What was read:** the frozen manifest metadata (`data/manifest/prompts.parquet`, whose SHA-256 matches `results/prespec/FROZEN.json`, plus `prompt_rules.parquet`, `rule_pools.jsonl` and `tasks.parquet`). That gives rule ids, types, texts, task keys, splits, chains and COA rows. Plausible effect sizes come from committed TRAIN-only tables (`results/tables/desc_by_position.csv`, `desc_by_k.csv`).

**Reproduce:** `PYTHONPATH=src python audits/evaluation_expansion_audit.py` (about 15 s). It writes every table in `results/audits/evaluation_expansion/` and asserts the expected counts. A second script, `audits/evaluation_expansion_train_variance.py`, estimates the two unknown variance parameters from TRAIN labels through the train-locked loaders. It has to run on the Mac and **has not been run yet** (see Recommendation 2).

Labels used below: **[V]** = verified from the manifest or code; **[A]** = depends on stated statistical assumptions; **[R]** = recommendation.

---

## Executive summary

**Verdict:** the proposed 456-configuration design is internally sound and the counts are correct. It is ready to generate for **aggregate** claims about all four factors, with one implementation detail to pin first. It is **not** adequate for rule-type-specific or type-pair-specific claims. The limit is the number of independent rule pools (15), not the number of responses per prompt.

1. **Counts verified [V].** 62 subsets and 456 ordered configurations per task. That gives 20,520 prompts, 410,400 generations per LLM and 820,800 in total. 246 of the 456 configurations per task already exist: the frozen test set of 11,070 prompts and 442,800 generations, generated on Betty with vLLM 0.25.0 and still sealed. So the expansion needs **378,000 new generations** (9,450 new prompts).
2. **Implementation trap [V].** The six existing subsets at each k must keep their *chain-rotation* base order when the COA template is applied. If every subset used the A–F chain-label order instead, 24 of the 246 existing configurations per group would not be reproduced (all at k = 4, where A4 is not closed under the 4-cycle that the chain rotation introduces). The audit's construction (existing subsets keep their chain base; new subsets use chain-label order) reproduces all 246 exactly.
3. **Factor coverage [V].** Position (A) is perfectly balanced at every k. Strict appends (B1) cover 100% of possible appends at 1→2 and 2→3, 50% at 3→4 (still balanced), and only 50% of k = 5 configurations at 4→5, where target-rule × position counts are mildly unbalanced (8–12, CV 0.16). Strict replacements (C) cover 100% at k = 2 and 3, 47% at k = 4 and 40% at k = 5, balanced by position. Task matching (D) is complete. The expansion is a large improvement over the current design: for example, the current design has **zero** strict replacement pairs at k = 4.
4. **Independent replication is the weak point [V].** Three rule types appear in only 3 of the 15 test groups. Of the 98 type pairs that are allowed to co-occur, 5 never co-occur in TEST and 28 co-occur in only one group. Five types have a single parameter setting across the whole test set.
5. **20 responses per prompt [A].** This is too few to read a *single* matched pair: the SE of a difference is about 10–13 pp, while the TRAIN effects are about 0.5–1.5 pp. It is enough for *aggregate* factor effects. Within a group, generation noise on an aggregate contrast is about 0.4–0.7 pp at n = 20. Once the effect varies across groups by 1 pp or more (τ ≥ 1 pp), going from 20 to 100 responses per prompt shrinks the across-group SE by 6–16% (depending on the contrast), at 4.5 times the cost.
6. **Best value for money [R].** First, spend nothing: run the TRAIN variance script (minutes) and adjust the analysis plan. Then generate the expansion as proposed. If rule-type-level claims matter, add **independent rule pools** on the existing TEST tasks. That changes the held-out protocol, so **it needs your explicit approval**. Do not add responses per prompt. Adding orderings (Option B) is optional and only helps the 4→5 strict append.

---

## 1. What exists and what would be new [V]

| k | Subsets | COA orderings per subset | Configurations per task | Already generated per task | New per task |
|---|---:|---:|---:|---:|---:|
| 1 | 6 | 1 | 6 | 6 | 0 |
| 2 | 15 | 2 | 30 | 12 | 18 |
| 3 | 20 | 6 | 120 | 36 | 84 |
| 4 | 15 | 12 | 180 | 72 | 108 |
| 5 | 6 | 20 | 120 | 120 | 0 |
| **Total** | **62** | | **456** | **246** | **210** |

- 45 test tasks × 456 = **20,520 prompts**. × 20 responses = 410,400 per LLM, **820,800** for both LLMs.
- Already generated and sealed: 11,070 prompts, **442,800** generations. New: 9,450 prompts, **378,000** generations. Your figures are correct.
- New prompts sit only at k = 2, 3 and 4. k = 1 and k = 5 are already complete, because six cyclic windows cover all C(6,1) and C(6,5) subsets.
- **Reuse condition [R].** The 442,800 existing generations can be pooled with new ones only if the new ones use the identical stack: Betty `dgx-b200`, vLLM 0.25.0 + torch 2.11.0+cu129, BF16, the same sampler order and decoding values. New configurations are concentrated in specific subsets, so any backend shift would be confounded with k, subset and companion identity. Before the test run, run a small TRAIN-only regeneration (for example the 100 pilot prompts) with the same stack and compare violation rates per k with the frozen TRAIN responses.
- Downstream steps the expansion also needs, each requiring its own authorisation: tokenisation and context checks for 9,450 new prompts; pre-generation features (MPS) and A1max for TEST prompts, which are currently sealed; verification; and an amendment to the 2026-11-14 pre-specification. Evaluation 2's pairing code (`controlled.py`) uses only COA row 0 for the count analysis and would need to enumerate all strict appends.

Tables: `design_counts_by_k.csv`, `design_validation_by_group.csv`, `expanded_manifest_configs.csv` (all 6,840 group-level configurations, each flagged as already generated or new).

---

## 2. Factor coverage scorecard (Audit 1) [V]

### Summary

| Factor | Exact comparison | Valid comparisons (15 groups, ×3 tasks) | Coverage | Main limitations | More orderings needed? |
|---|---|---|---|---|---|
| **A Position** | Same task, exact subset, k and target. Target at position j vs j′, each side averaged over its k−1 COA orderings | Task × subset × target cells: 1,350 (k=2), 2,700 (k=3), 2,700 (k=4), 1,350 (k=5); × C(k,2) position pairs | 100%; every target at every position k−1 times; companions perfectly balanced | Moving the target also moves it relative to the prompt end (about 77 characters per step) and changes which companions precede it (balanced on average, not per ordering) | No |
| **B1 Strict append** | X → X+r: X is an exact prefix of the larger prompt; retained targets keep position and wording | Target comparisons: 1,350 (1→2), 10,800 (2→3), 24,300 (3→4), 10,800 (4→5) | Of appends possible from present configurations: 100%, 100%, 50%, 17% (= 50% of k=5 configurations) | 4→5 is thinner and mildly unbalanced (target rule × position counts 8–12) | Optional (Option B-prefix fixes 4→5) |
| **B2 Marginal k** | Same target and position; average over every subset containing it, k vs k+1 | Cells: 270 (k=1) to 1,350 (k=5); 5, 20, 30, 20 configurations per task-target-position at k = 2–5 | 100% of subsets; companions exactly balanced | Mixes companion identities and orderings; not a single matched change | No |
| **C Strict replacement** | Same task, k, target, target position; one companion replaced in place; other companions fixed | Target comparisons: 5,400 (k=2), 48,600 (k=3), 45,360 (k=4), 21,600 (k=5) | Of potential: 100%, 100%, 47%, 40% | Each specific swap of companion types is replicated in a median of 2 groups (k ≤ 4) or 1 group (k = 5) | No (the gap is in groups, not orderings) |
| **C Marginal composition** | Same task, target, k, target position; mean with companion c present vs absent | Cells: 2,700 to 6,750, all valid | 100% | At k = 5 the "absent" side is a single subset (4 configurations) | No |
| **D Task** | Identical ordered configuration on two tasks of the same group | 20,520 prompt pairs; 78,570 target comparisons; 45 task pairs | 100% | Only 3 tasks per group and 45 tasks overall; task differences are not decomposable | No |

### A. Rule position

Example: in subset {A, B, C, D}, target A sits at position 1 in 3 of the 12 A4 orderings and at position 4 in another 3. The comparison is the mean over the first three against the mean over the second three, on the same task.

- **Coverage.** At every k ≥ 2, every target is at every position in exactly k−1 orderings (`A_position_coverage.csv`). Per position, that is 1,350 / 5,400 / 8,100 / 5,400 prompt-target observations at k = 2 / 3 / 4 / 5.
- **Companion balance.** At k ≥ 3, for every (subset, target, position), each companion occupies each other position exactly once (2,700 + 3,600 + 2,250 cells checked, 0 unbalanced; `A_companion_position_balance.csv`). This is what the sharply 2-transitive COA guarantees.
- **Comparisons.** Every position pair (j, j′) is available for every target. No transition is under-represented: adjacent and long-range moves have identical support (`A_position_comparisons.csv`).
- **Nuisance that remains.** The prompt's total length is unchanged, but the target's distance to the end of the prompt changes by about 77 characters per position (mean rule length). The distance also varies within a position, with an SD of 24–47 characters depending on which companions follow. The target's distance to the task text changes too, and so does the number of rules read before it. COA balances companion identity *on average* across the k−1 orderings, but not within any one ordering.

### B1. Strict append (primary)

Example: ABC → ABCD, where ABC is an exact prefix of ABCD and A, B and C keep their positions.

| Transition | Ordered pairs per group | Target comparisons | % of appends possible from present configs | % of full-permutation universe | k+1 configurations with their prefix present |
|---|---:|---:|---:|---:|---:|
| 1→2 | 30 | 1,350 | 100 | 100 | 100% |
| 2→3 | 120 | 10,800 | 100 | 100 | 100% |
| 3→4 | 180 | 24,300 | **50** | 50 | 100% |
| 4→5 | 60 | 10,800 | **17** | 8 | **50%** |
| *Current 246 design, for comparison* | 12 / 24 / 36 / 24 | | 40 / 50 / 33 / 17 | | |

- **Why 3→4 is 50%.** A4 keeps 12 of the 24 orderings of a 4-subset. For each 3-subset and appended rule, exactly 3 of the 6 prefix orderings get the append, namely the 3 cyclic rotations of one orientation. Those 3 put each retained target at each position once, so the 3→4 set is **exactly balanced** by target rule × position (CV 0) and by target × appended rule. The missing half costs precision, not balance.
- **Why 4→5 is weaker.** Only 60 of the 120 k = 5 orderings have their 4-prefix among the A4-based k = 4 orderings. The surviving pairs are balanced by target position (2,700 each) and by target × appended rule, but **not** by target rule × position: per group, counts range from 8 to 12 (CV 0.16; `B1_strict_append_balance.csv`). Gaps therefore fall on 4→5, not on 3→4.
- **Coverage by type, task and pool.** By type (`B1_strict_append_by_type.csv`), coverage is proportional to the number of groups holding that type: 3 groups give 810 target comparisons at 3→4, 9 groups give 2,430. Every task and group has the same counts (`B1_strict_append_by_group.csv`).
- **Note.** `controlled.py` currently builds count pairs from COA row 0 only (24 per task). The expanded design supports all the strict pairs above.

### B2. Marginal k (secondary)

- All C(6,k) subsets are present in every group. Per (task, target, fixed position), there are 1 / 5 / 20 / 30 / 20 configurations at k = 1–5, which is 20 / 100 / 400 / 600 / 400 responses (`B2_marginal_k_coverage.csv`).
- Companion identity is exactly balanced: at each (target, position, k), every other rule appears in the same share of configurations, (k−1)/5. The current design does not have this property (6 of 15 subsets at k = 2–4).
- **How it differs from B1.** B1 measures the change when a specific rule is appended to a specific ordered prompt, holding everything else fixed. B2 compares the average risk over *all* k-rule contexts with the average over *all* (k+1)-rule contexts. The two contexts share no single prompt, and the target's distance from the end differs by one rule on average. B2 is precise and balanced, but it is a difference of marginal means, not a matched change.
- More orderings would not improve B2's balance, which is already exact. They would only add precision, and the precision analysis below shows that is not the binding limit.

### C. Companion composition

Example: ABC → ADC. The target A stays at position 1, C stays at position 3, and B is replaced by D.

| k | Ordered pairs per group | Target comparisons | % of potential | Current design |
|---|---:|---:|---:|---:|
| 2 | 120 | 5,400 | 100 | 12 pairs per group |
| 3 | 540 | 48,600 | 100 | 36 |
| 4 | 336 | 45,360 | **47** | **0** |
| 5 | 120 | 21,600 | **40** | 120 |

- Every (target position, replaced position) combination has equal support at each k (`C_strict_replacement_by_position.csv`).
- **What is missing.** At k = 4 and 5, the COA subgroups do not contain both orderings of most replacement pairs. At k = 5 the only possible replacement is the single absent rule, and 40% of those exist. The missing comparisons are spread evenly across positions.
- **Weak support by type pair.** A specific swap such as "replace a bullet-list rule with a title rule" is replicated in a median of 2 groups at k = 2–4 and 1 group at k = 5. 28–40 such swap-type pairs per k occur in only one group (`C_strict_replacement_type_pairs.csv`). More orderings cannot fix this, because it is a group-count limit.
- **Marginal composition** (target with companion c present vs absent, holding task, target, k and position fixed) is valid in every cell. The present / absent sides have 1/4, 8/12, 18/12 and 16/4 configurations at k = 2–5 (`C_marginal_composition_coverage.csv`).
- **Trade-off.** Strict replacement isolates a single swap with everything else fixed, but each swap is specific to one pair of exact rules and has thin group replication. Marginal composition averages over all contexts, so it is more precise and can be summarised per companion type. It does mix the companion's effect with the subsets it happens to appear in, and at k = 5 its contrast leans on one subset.

### D. Task differences

- **Checks.** All 15 pools list the same three tasks as the manifest. Every task in a group has the identical 246 current configurations, and the expanded configurations are built per group, so they are identical across the three tasks by construction (`D_task_matching_by_group.csv`).
- **Counts.** 456 configurations × 3 task pairs × 15 groups = 20,520 prompt pairs and 78,570 target-level comparisons.
- **Replication.** There are only 3 tasks per group and 45 in total. Every group has three distinct (provisional) families. Task lengths range from 35 to 306 characters.
- **What can be concluded.** Whether violation risk for the identical rule configuration differs between tasks, and how much task-to-task variation there is relative to other factors. Within a group, a single task-pair difference is estimated very precisely (generation-noise SE about 0.4 pp).
- **What cannot be concluded.** *Why* tasks differ (semantics, length, genre, how well rules fit the task), because these vary together across only 45 tasks. Family labels are provisional and each family appears in only a few groups.

---

## 3. Independent-group coverage (Audit 2) [V]

**Unit of replication.** One test group is one six-rule pool on three tasks. 15 groups means 15 independent rule-pool environments. The 45 tasks are nested in those pools.

### Rule types (`rule_type_frequencies.csv`)

| Groups containing the type | Types |
|---|---|
| 3 | capital_word_frequency, english_lowercase |
| 4 | number_paragraphs (3 exact instances), number_sentences, number_words |
| 6 | english_capital, title, existence, frequency, quotation |
| 8 | number_placeholders, number_bullet_lists, no_comma |
| 9 | forbidden_words, letter_frequency |

- All 15 types appear, but support is uneven: 3 to 9 groups, against 13 to 32 in TRAIN.
- 90 exact rule instances in total. **Five types have exactly one parameterisation** across all test groups (english_capital, english_lowercase, title, no_comma, quotation). Their type-level estimates are estimates for that one instruction text.
- The spread follows the sampler. The three mutually exclusive families (case, length and keyword-count types) each contribute at most one type per pool.

### Type pairs (`rule_type_pair_frequencies.csv`)

| Test groups containing both | Number of type pairs (of 98 that can co-occur) |
|---|---:|
| 0 | 5 |
| 1 | 28 |
| 2 | 28 |
| 3 | 16 |
| 4–6 | 21 |

- Seven pairs can never co-occur, by the ManyIFEval exclusivity rules.
- Never observed together in TEST: english_lowercase + frequency, english_lowercase + number_words, number_placeholders + quotation, existence + number_sentences, number_paragraphs + quotation.
- **Prompts are not replications.** A pair held by one group still appears in 534 expanded prompts (21,360 generations), all from the same single pool and three tasks. Report support as groups first, then tasks, then prompts.

### Is 15 groups enough? [A]

- **Unit for intervals.** Resample groups, using the paired group bootstrap already in the plan. Tasks, prompts, targets and responses within a group are not independent.
- **Bootstrap with 15 groups** (`precision_group_bootstrap_coverage.csv`, 2,000 simulations each). The nominal 95% percentile bootstrap covers **92.2%** when group effects are normal, 89.8% with heavy tails and **83.4%** when they are skewed. It is also about 12% narrower than a t(14) interval. The t(14) interval covers 94.6% / 95.6% / 85.8%. At 30 groups, the bootstrap reaches 94% (normal). The plan already lists a leave-one-group-out jackknife / t(14) sensitivity check. **[R]** Report it next to the bootstrap for every Evaluation 2 headline.
- **Claims that hold up:** pooled, aggregate factor effects (all targets, per k or pooled over k) and their agreement with probe predictions, at the level of "on these 15 pools".
- **Weak claims:** anything per rule type (3–9 groups), per type pair (0–6 groups, mostly 1–2), or per specific companion swap. Per-type intervals at 3–4 groups are close to uninformative.
- **What three tasks per group buy.** They average out task idiosyncrasy within a pool and give the task factor (D). They do not make pools independent: all three tasks share the same six exact rules.

**Projection with added pools** (simulated with a type-level replica of `rule_sampler`, which matches the 72 frozen pools' type frequencies within sampling error; `pool_sampler_validation.csv`, `added_pool_coverage_projection.csv`):

| Total test pools | Every type in ≥ 5 groups (probability) | Feasible pairs never observed (mean) | Feasible pairs in ≥ 2 groups | in ≥ 3 groups |
|---:|---:|---:|---:|---:|
| 15 (current) | 0 | 5.0 | 66% | 38% |
| 20 | 0.04 | 2.5 | 81% | 57% |
| 25 | 0.57 | 1.4 | 89% | 71% |
| **30** | **0.92** | **0.7** | **94%** | **82%** |
| 45 | 1.00 | 0.1 | 99% | 95% |

**[R]** To support type-level claims, about **30 pools** (15 more) is where every type reliably reaches 5 groups and almost every pair reaches 2. Choose 15 groups with narrower claims if type-level results are descriptive only. Adding pools changes the held-out protocol and **needs your explicit approval**. All 216 ManyIFEval tasks are already used, so new pools would be new six-rule pools sampled with the same sampler on the existing 45 TEST tasks, regrouped into new task triples. That leaves the TRAIN/TEST task split and all V2 training untouched. Moving TRAIN groups to TEST is not recommended, because it would invalidate the V2 selections and refits.

---

## 4. Statistical precision (Audit 3)

### A single prompt [V, exact binomial] (`precision_single_prompt.csv`)

| True rate | SE at 20 responses | Typical observed range (central 95%) | P(0 violations observed) |
|---|---:|---|---:|
| 5% | 4.9 pp | 0–15% | 36% |
| 10% | 6.7 pp | 0–25% | 12% |
| 20% | 8.9 pp | 5–40% | 1% |
| 50% | 11.2 pp | 30–70% | 0% |

In plain terms, one prompt's observed rate is precise to about ±10–20 pp. Each response moves it by 5 pp.

### One matched pair [V, exact enumeration] (`precision_matched_pair.csv`)

At a 18% baseline and 20 responses per side, the SE of a difference is 12.5–13.8 pp. Using the project's Agresti–Caffo "reliable pair" rule (`controlled.py`):

| True difference | P(observed sign correct) | P(pair passes "reliable" with correct sign) | Same at 100 responses |
|---:|---:|---:|---:|
| 2.5 pp | 0.50 | 0.03 | 0.06 |
| 5 pp | 0.58 | 0.04 | 0.14 |
| 10 pp | 0.72 | 0.09 | 0.39 |
| 20 pp | 0.90 | 0.29 | 0.89 |

TRAIN effects are about 0.5–1.5 pp on average (`train_only_effect_anchors.csv`). At that size almost no single pair is "reliable", and the pairs that pass are selected for extreme noise. **[R]** Evaluation 2 should score predicted against observed changes on *aggregated* contrasts (cells or groups), not on single pairs filtered by reliability. That changes the analysis plan, not the data. It also needs an amendment, because the plan is frozen.

### Aggregate effects [A] (`precision_aggregate_scenarios.csv`)

**Model.** For each prompt and target, the observed rate equals the true rate plus binomial noise. Two rules in the same response have correlated violations (ρ). Each prompt-target has an idiosyncratic deviation (SD σᵤ) that matching cannot remove. The group-level effect varies across groups (SD τ). The weights for each contrast come from the actual expanded comparison sets, so prompt reuse across comparisons and multiple targets per response are counted exactly. The generation-noise part was checked by Monte Carlo (`precision_mc_check.csv`; analytic and simulated SDs agree within 2%). The rates p(1−p) use the TRAIN mean rate, which is conservative, because real rates cluster near zero.

At p = 0.18, ρ = 0.3, σᵤ = 5 pp, within one group and n = 20, generation noise contributes an SE of 0.44 pp (A, k=4), 0.46 (B1 3→4), 0.68 (B1 4→5), 0.40 (B2 4→5) and 0.55 (C marginal, k=4, one companion).

**95% CI half-width (t₍G−1₎) for an aggregate effect, B1 3→4** (other contrasts are similar):

| τ (between-group SD of the effect) | Groups | n = 20 | 40 | 60 | 100 |
|---|---:|---:|---:|---:|---:|
| 0 | 15 | 0.28 pp | 0.22 | 0.19 | 0.16 |
| 1 pp | 15 | 0.62 | 0.59 | 0.59 | 0.58 |
| 1 pp | 30 | 0.42 | 0.40 | 0.40 | 0.39 |
| 2 pp | 15 | 1.14 | 1.13 | 1.12 | 1.12 |
| 2 pp | 30 | 0.77 | 0.76 | 0.76 | 0.76 |

- If effects are homogeneous across pools (τ ≈ 0), more responses help, about √n. Even then, 20 responses already give ±0.3 pp.
- If τ ≥ 1 pp, the group term dominates. Going from 20 to 100 responses (4.5 times the cost) narrows this interval by 7%, and by at most 16% for any aggregate contrast at k ≥ 3. Doubling the number of groups narrows it by about 30%.
- **Rule-type-specific effects** have far fewer groups: a median of 6 and a minimum of 3. Example: the B1 3→4 effect for one type at τ = 1 pp has SE 0.59 pp with 6 groups and 0.83 pp with 3. Against TRAIN effects of about 1 pp, those intervals usually include zero.
- **τ is unknown.** TRAIN tables give mean effects but not their between-group spread. `audits/evaluation_expansion_train_variance.py` estimates τ and ρ from the 57 TRAIN groups (current design; train-locked loaders). It was checked on synthetic data with a known τ. Until it is run, read these rows as scenarios.

### Alternative budgets (Audit 3D)

| Responses per prompt | Total generations (both LLMs) | New generations | Aggregate precision gain vs 20 | Strengthens factor analyses? |
|---:|---:|---:|---|---|
| 20 | 820,800 | 378,000 | — | — |
| 40 | 1,641,600 | 1,198,800 | 19–25% narrower CI if τ = 0; ≤ 9% if τ ≥ 1 pp | Marginally |
| 60 | 2,462,400 | 2,019,600 | 26–35% if τ = 0; ≤ 13% if τ ≥ 1 pp | Marginally |
| 100 | 4,104,000 | 3,661,200 | 33–43% if τ = 0; ≤ 16% if τ ≥ 1 pp | Only if effects are homogeneous |

Additional pools are worth more than additional responses whenever τ ≳ 0.5 pp.

---

## 5. What are we actually measuring? (Audit 4)

| Factor | Held constant | Deliberately changed | Changes as a consequence | Estimand | Defensible conclusion | Too strong |
|---|---|---|---|---|---|---|
| **Position** | Task, exact rule texts, subset, k, prompt length | Target's slot in the rule list | Target's distance to prompt end (about 77 characters per slot) and to the task; which companions precede it (balanced only on average) | Mean violation of target at slot j′ minus slot j, averaged over the COA orderings | "Moving an instruction later in the list (with companions balanced) changes its violation risk by X" | "Serial position *per se* causes X, independent of distance from the end or of what precedes it" |
| **Append** | Task, ordered prefix (texts, order, positions) | One rule appended at the end | Prompt length, the identity and wording of the new companion, and the retained targets' distance to the end | Change in a retained target's violation rate when a specific rule is appended | "Appending an instruction changes existing rules' violation risk by X on average (B1); average risk differs by Y between k- and (k+1)-rule prompts (B2)" | "An abstract rule-count effect" independent of length, distance or the added rule's content |
| **Composition** | Task, k, target text and slot, the other companions and slots | One companion replaced in place | Companion wording, semantics, parameters, length (and so the target's offset if the swap precedes it) | Target's change when companion c is replaced by c′ (strict), or with c present vs absent (marginal) | "Replacing this companion instruction changes the target's risk by X" | "Companion *identity/type* causes X independent of its wording, length and parameters" |
| **Task** | Exact ordered configuration | Task text (one of the pool's 3) | Topic, genre, length, task–rule fit, expected response length | Difference in violation rate for the same configuration across tasks | "Violation risk for identical instructions varies across tasks by X (SD over 45 tasks)" | Attributing it to semantics, length or family |

**Suggested terms [R]:** "rule-position effect (COA-balanced)", "append effect" (strict) and "average k-step difference" (marginal), "companion-replacement effect" and "companion-presence effect", "task-context variation". Avoid "number-of-rules effect", "companion-type effect" and "task-semantics effect" unless they are qualified.

---

## 6. Recommended changes (Audit 5), ranked by value per cost

| Rank | Change | Weakness addressed | Added coverage | Added generations | Improves |
|---|---|---|---|---|---|
| 1 | **Pin the base-order convention** in the expansion spec (existing subsets keep their chain base) | Silent loss of 24 configurations per group (1,080 prompts, 43,200 already-generated responses) and an existing/new mismatch | Reuses all 442,800 existing generations | 0 | Correctness |
| 2 | **Run the TRAIN variance script** (τ, ρ) and decide on pools from τ | The precision scenarios hinge on unknown τ | — | 0 (minutes on the Mac) | Decision quality |
| 3 | **Analysis-plan amendment** (before unblinding): aggregate-level Evaluation 2 scoring instead of reliable single pairs; enumerate all strict appends; t(14)/jackknife alongside the bootstrap; claims limited to aggregate level, with per-type results labelled descriptive | Single-pair noise; bootstrap under-coverage at 15 groups | — | 0 | Validity |
| 4 | **Option A: generate the 456 design at 20 responses** | Thin B and C matching in the current design (for example zero k=4 replacements) | B1 target comparisons 11,880 → 47,250; C strict k=4 target comparisons 0 → 45,360 | **+378,000** | Matching coverage, balance |
| 5 | **Option C: about 15 new pools** on regrouped TEST tasks, same 456 design (*needs approval*) | Thin type and pair replication; bootstrap at G = 15 | Every type in ≥ 5 groups (p ≈ 0.92); 94% of pairs in ≥ 2 groups; across-group SE × 0.71 | +820,800 (or +442,800 with the 246 design, which loses B and C matching in new pools) | Statistical independence |
| 6 | Option B-prefix: add the 48 missing 4-prefixes of k=5 configurations per group | 4→5 strict append: half missing, mildly unbalanced | 4→5 pairs 60 → 120 per group, balance CV 0.16 → 0; k=4 replacements 336 → 560 | +86,400 | Matching for one transition. **Caveat:** augmented 4-subsets lose COA position balance (14–16 orderings), so position and marginal analyses must use the COA rows only |
| — | Option B-S4 (all 24 orderings at k = 4) | 3→4 and 4→5 | 3→4 360, 4→5 120 per group | +324,000 | Precision only; poor value |
| — | Option D (40–100 responses) | Single-prompt precision | — | +820,800 to +3,283,200 over A | Little for aggregate effects when τ ≥ 1 pp; not recommended |

---

## 7. Final generation budget

| Design | Unique test prompts | Total generations (2 LLMs) | Already generated | New |
|---|---:|---:|---:|---:|
| Current frozen design | 11,070 | 442,800 | 442,800 | 0 |
| **Proposed (Option A)** | **20,520** | **820,800** | 442,800 | **378,000** |
| A + B-prefix | 22,680 | 907,200 | 442,800 | 464,400 |
| A + 15 new pools (Option E core) | 41,040 | 1,641,600 | 442,800 | 1,198,800 |
| A + 15 pools + B-prefix in all 30 pools | 45,360 | 1,814,400 | 442,800 | 1,371,600 |

All scenarios are in `generation_budget_scenarios.csv`.

---

## 8. Final recommendation

**Proceed with the 820,800-generation design, not a larger number of responses per prompt.** Do three no-cost things first:

1. Pin the base-order convention.
2. Run the TRAIN variance script.
3. Draft the analysis-plan amendment.

The design is adequate as it stands for aggregate claims about position, appending, companion composition and task variation. If you want rule-type-specific or type-pair conclusions, or the TRAIN estimate shows τ ≳ 1 pp, the change worth paying for is **about 15 additional independent rule pools** on the existing TEST tasks (+820,800 generations). That changes the held-out protocol and needs your explicit approval. Adding orderings is optional (B-prefix, +86,400, only for the 4→5 strict append). More responses per prompt is the least cost-effective option.

Nothing has been generated or implemented, and none of this proceeds until you approve the audit.
