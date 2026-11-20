---
title: "Homework 11: Review Your Project Five Ways"
subtitle: "Project: Predicting Rule Violations Before Generation (ManyIFEval probing)"
geometry: margin=0.9in
fontsize: 10pt
---

**Model used.** Claude (Anthropic), Claude Code. Ways 1, 2, 3 (the three reviewers), 4 and 5 ran on Claude Fable 5.2 (`claude-fable-5-2`). The Way 3 area-chair meta-review ran on Claude Opus 5.5 (`claude-opus-5-5`) after a model switch mid-session; this is a deviation from "one chat assistant for all five ways" and is noted here. Each review ran in its own fresh agent session with no access to the other reviews (except the area chair, which was given the three reviewer reports as the assignment requires). Instead of uploading a PDF, each session read the paper's LaTeX source, its generated tables and its figures; the evidence check (Way 4) was also given the result CSVs and code.

**Paper reviewed.** `report/manyifeval_probing_report.pdf` (draft of 16 Nov 2026): proposal, methods, training-set results, figures and planned work. No test-set results exist yet.

# Part 3: Judge each way

| Way | What worked well? | What worked badly? | How could you make it work better? |
|---|---|---|---|
| **1. One prompt** (Rating 3/4 "reject", Conf. 4) | Surprisingly specific. It found the most important design flaw on its own: that the probe-vs-baseline comparison is confounded by head structure in C1, not only in C2/C3 as the paper says. It also proposed a concrete control: "If accuracy is largely retained, A1 is encoding the prompt, not the generating model's disposition." | Long and unprioritised; generic filler mixed in with real points, e.g. "Only two 8B models are studied, with English prompts, at most five rules, and default decoding that differs between the models." It also used a 1–4 rating together with a confidence of 4, mixing scales. | Ask for the top three weaknesses ranked by how much they would change the conclusion, and require each to cite a table or section. |
| **2. Rubric** (Pres. 3, Inf. 2, Sound. 2, Rating 2/4, Conf. 2) | The fixed fields forced a calibrated, comparable verdict, and the strengths/weaknesses split was clean. Its sharpest comment was quantitative: "The structure head selects the smallest penalty in all four cells and loses 7.9 to 11.1% one grid step inwards, while the probe margins over it are 8.4 to 14.2%." | The 1–3 scales are coarse, so scores barely discriminate; some text was boilerplate: "As submitted, the central claim is unverified." | Ask for a one-line justification per score, and for each weakness a "what result would fix this". |
| **3. Reviewer team** (Soundness 4/10, Novelty 4/10, Clarity 6/10; AC: reject, resubmission encouraged) | Specialisation produced the deepest critique. Soundness: "A per-type-head version of the structure head, B1 and B2 is an essential control and is currently listed only as an exploratory extension." Novelty proposed the decisive control: "predict Llama's violations from Qwen's hidden states and attention, and the reverse." The AC merged these into three clear fixes (run the test, make the comparison fair, show the signal is generator-specific). | Heavy overlap: all three reviewers and the AC repeat the same confound. The novelty reviewer suggested missing citations "from memory", which then had to be checked in Way 5. The clarity reviewer's best point ("A reader will take the abstract's comparison as features against features") is a soundness point in disguise. | Give each reviewer an explicit "do not discuss" list outside its focus, require web search for the novelty reviewer, and ask the AC to rank fixes by cost as well as importance. |
| **4. Evidence check** (146 claims: 115 supported, 0 unsupported, 31 can't tell) | The most useful way for honesty: it recomputed numbers from the result CSVs and found nothing contradicted ("No claim is contradicted outright by a saved table."). It also caught overstatements the other reviews missed: "B2 does not help for Qwen" when Objective 1 CV loss is 1.5% lower, and our conclusion that gains "concentrate on layout and length rules" while A1 is worse than the structure head on Llama sentence count (0.618 vs 0.550). | Most "can't tell" items are provenance it was not allowed to see: "Most can't-tell items are generation-phase provenance (prompt reproduction, split search, decoding, backend cosine) whose evidence sits outside the files I was allowed to read." A flat 146-row list is hard to act on, and it says nothing about whether the experiments are the right ones. | Give it the pilot report and manifest summary so provenance claims can be checked, and ask it to rank the can't-tell claims by importance and suggest a rewording for each overstatement. |
| **5. Related-work check** | Found three relevant papers we had not cited, including one that challenges our premise: Rocchetti & Ferrara (2026) report that constraint satisfaction looks like monitoring during generation rather than planning before it. It also caught an out-of-date bib entry (InFoBench is in Findings of ACL 2024, not just arXiv). Useful framing: "SAT Probe is the methodological ancestor of our best-performing component, and the draft could say so more prominently." | Two of its "five closest" were papers we already cite, so the genuinely new list is short. It could not resolve whether IFScale is a NeurIPS main-track or workshop paper. | Ask explicitly for papers *not* in the bibliography, and for papers that would contradict the result. |

**Citation check (Way 5).** All five listed papers exist; I confirmed the least familiar one ([Rocchetti & Ferrara, arXiv:2604.06015](https://arxiv.org/abs/2604.06015)) myself by web search, and the agent confirmed the others against arXiv records: Heo et al. ICLR 2025 ([2410.14516](https://arxiv.org/abs/2410.14516)), Heo et al. ICLR 2025 ([2410.14582](https://arxiv.org/abs/2410.14582)), Ashok & May NeurIPS 2025 ([2502.13329](https://arxiv.org/abs/2502.13329)), Yuksekgonul et al. ICLR 2024 ([2309.15098](https://arxiv.org/abs/2309.15098)). No invented citations.

# Part 4: Compare and plan

**1. How did the five reviews compare with the Stanford reviewer?**

**[TO FILL after the Stanford review is run: compare it with Ways 1–5. What it caught that ours missed, and what ours caught that it missed (e.g. our evidence check's overstatements and the uppercase-verifier issue).]**

**2. Three changes for the Draft Report (due 11/23).**

1. **Give the baselines the same per-type heads as the probes.** In Objective 1 the probes get 15 per-rule-type heads while B1, B2 and the structure head get one shared head, so every probe-vs-baseline contrast mixes features with head structure. We will refit the structure head, B1 and B2 with per-type heads (training data only) and report both versions. *Prompted by* the soundness reviewer: "A per-type-head version of the structure head, B1 and B2 is an essential control and is currently listed only as an exploratory extension."; the same point appears in Ways 1, 2 and the AC meta-review.
2. **Fix comparator tuning before unblinding: widen the penalty grid and give each feature block its own penalty.** The structure head sits at the edge of the grid, and B1/B2 lose to a subset of their own inputs, which suggests poor tuning rather than a real result. We will widen λ to {0.001, …, 10}, add block-wise penalties, rerun cross-validation on training groups and re-freeze the plan. *Prompted by* the rubric review: "The structure head selects the smallest penalty in all four cells and loses 7.9 to 11.1% one grid step inwards, while the probe margins over it are 8.4 to 14.2%."; and the soundness reviewer: "A superset of features doing worse is a tuning failure … not evidence that text adds nothing."
3. **Show the signal is the generator's own: a cross-model control, then report the test results.** We will predict Llama's violations from Qwen's features and the reverse (and an attention control built only from span length, position, k and rule type); if these do nearly as well, the internal-state claim fails. Then we will run the frozen test evaluation and put the held-out numbers in the draft, and discuss Rocchetti & Ferrara (2026) in related work. *Prompted by* the novelty reviewer: "The decisive control is cross-model: predict Llama's violations from Qwen's hidden states and attention, and the reverse."; and the AC's fix 3.

# Appendix

The prompts used (verbatim, after the instruction to read the paper) were:

- **Way 1:** "Review this paper as an ICLR reviewer."
- **Way 2:** "Review this paper by filling out the ICLR review form below. Fields: Paper Summary (written response); Presentation (score 1-3); Informativeness (score 1-3); Soundness (score 1-3); Critical Strengths (written); Critical Weaknesses (written); Questions For Authors (written); Flag For Ethics Review (Yes or No); Rating (1-4); Confidence (1-3)."
- **Way 3a:** "You are an ICLR reviewer whose focus is experiments and soundness. Review this paper: assess the experimental design, statistics, baselines, controls and whether the conclusions follow from the evidence. Give strengths, weaknesses, questions, and a score from 1 to 10."
- **Way 3b:** "You are an ICLR reviewer whose focus is novelty and related work. Review this paper: what is new relative to prior work, is the positioning accurate, what related work is missing. Give strengths, weaknesses, questions, and a score from 1 to 10."
- **Way 3c:** "You are an ICLR reviewer whose focus is clarity and presentation. Review this paper: writing, organisation, figures, tables, notation, and whether a reader can follow and reproduce it. Give strengths, weaknesses, questions, and a score from 1 to 10."
- **Way 3 AC:** "You are the area chair. Here is the paper and three reviews (experiments/soundness, novelty/related work, clarity/presentation). Write a meta-review with an accept or reject decision and the three most important fixes."
- **Way 4:** "Here is the paper plus our results (tables, logs, code). List every claim the paper makes and mark each one 'supported', 'unsupported' or 'can't tell', pointing to the number, table file or figure behind it."
- **Way 5:** "With web search on, find the five closest prior papers to this one, with links, and say what is new in ours. Check that every paper you list actually exists."
- **Stanford:** paper uploaded to paperreview.ai, target venue ICLR.

The full outputs follow.


\newpage

## Appendix — Way 1: One simple prompt

### ICLR-style review

**Paper:** Predicting Rule Violations Before Generation: Hidden-State and Attention Probes versus Prompt-Only Baselines on a Controlled Multi-Instruction Benchmark

#### Summary

The paper asks whether an LLM's violations of explicit prompt constraints can be predicted before generation, and whether the generator's internal state at the final prompt token predicts them better than predictors that only read the prompt. The authors build a balanced dataset from ManyIFEval (216 tasks, 72 groups of three tasks sharing six sampled rules, nested rule subsets for k = 1..5, orders from a strength-two component orthogonal array; 53,136 prompts), sample 20 responses per prompt from Llama 3.1 8B and Qwen3-8B, and label each (prompt, rule) with an empirical violation probability from the official verifiers. They compare 23 feature configurations per objective and generator: hidden-state probes (H1), hidden state plus a rule-text embedding (H2), attention from the final token to the rule span (A1), a frozen BGE-M3 baseline with structured features (B1), a LoRA-fine-tuned version (B2), and combinations, against rate-based references and a "structure head".

The analysis plan is frozen and the 15 test groups are unread, so all reported results are grouped 5-fold cross-validation on the 57 training groups. In those results A1 is the best single per-rule predictor for both generators (out-of-fold AUROC 0.85 / 0.84), internal-state probes beat the best prompt-only comparator by 8 to 14% in loss in three of four generator-objective cells, and B2 matches them for per-prompt prediction on Llama. The paper also reports that the official "all uppercase" verifier fails correct uppercase English through its language-detection step, which accounts for 43% of Qwen's violations of that rule.

#### Strengths

1. **The right question, posed incrementally.** Most probing papers compare against chance or a weak baseline. This one makes the comparator the point, and it reports that rule type and k alone remove 12 to 15% of the loss and a 75-feature structure head about 20%. That framing is the paper's main intellectual contribution.
2. **Careful experimental hygiene.** Grouped splits, probabilistic labels from repeated sampling, a hashed pre-specification, a guarded test set, a disclosed exception (token-limit share computed over all prompts), and a paired cluster bootstrap. The paper is unusually candid about what is descriptive and what is confirmatory.
3. **Honest reporting of unfavourable results.** H1 is no better than the structure head in Objective 1; H1+A1 is 8% worse than A1 alone; B2 wins on letter count; within-type AUROC for Qwen is 0.56 for A1 against 0.58 for B2; 8 of 23 configurations are duplicates. None of this is hidden.
4. **The verifier audit is concrete and useful.** The langdetect failure on uppercase text is quantified (12,554 of 29,097 Qwen violations), diagnosed (92.5% labelled German, most flip to English when lower-cased) and tied to a co-occurring rule. Anyone using IFEval-derived verifiers should know this.
5. **The design supports analyses that most datasets cannot.** Nested chains and balanced orders allow matched-pair tests of position, count, composition and task effects.

#### Weaknesses

##### W1. The paper has no held-out results, and says so

The status box, abstract and conclusion all state that the test groups are unread. Every number comes from the folds on which head type, penalty, epochs, layer and pooling were selected. The controlled analyses (A to E), which are the main justification for the elaborate design, are also not run. As submitted this is a registered-report stage-1 manuscript, which ICLR's main track does not have a category for. The contribution list reads "a pre-specified comparison" and "training-set evidence"; neither is a finding a reader can build on yet.

##### W2. Per-type heads are confounded with internal features in the headline comparison, not only in C2/C3

In Objective 1 every configuration containing H1, H2 or A1 uses 15 per-type heads, while B1, B2 and the structure head use one shared head. The paper flags this for the "baseline plus probe" contrasts (C2, C3), but the same confound applies to the C1 and reference contrasts that carry the abstract's claim (A1 against B2, A1 against the structure head). The control is listed as exploratory extension (iii). It should be a primary analysis: structure head, B1 and B2 with per-type heads, and ideally a per-type head on the rule embedding alone (ER) and on ER + S1. The H1 to H2 result makes this pressing: adding a frozen rule-text embedding to the hidden state accounts for a 0.037 NLL improvement on Llama, almost the whole gap between H1 and A1, and that embedding is prompt-only information.

##### W3. A1 may be a fingerprint of the rule text rather than a signal about the model's state

A1 is 1,024 to 1,152 attention values to the target rule's span, sum-pooled. Sum pooling scales with span length, and head-wise attention to a span depends on which tokens are in it and where it sits in the prompt. The feature vector can therefore encode rule identity, parameters and position, all of which are readable from the prompt. The letter-count result points this way: the paper says the difficulty of that rule depends on which letter is named, the structured features lack the letter, and A1 reaches within-type AUROC 0.92 on Llama and 0.88 on Qwen, close to B2's 0.93 and 0.90. A1 can only do that if it identifies the letter. The paper cites Hewitt and Liang's control-task caveat but runs no control. Three are cheap and decisive:
- **Cross-generator features.** Predict Qwen's labels from Llama's attention and the reverse. If accuracy is largely retained, A1 is encoding the prompt, not the generating model's disposition. This is exploratory item (iv) and deserves promotion.
- **A non-generating model's attention**, for instance a third LLM of similar size, or the same architecture at random initialisation.
- **Span-length and position controls**: add span length in tokens and absolute span position to the structure head, and report A1 with mean pooling restricted to rules of equal span length.

##### W4. The prompt-only baselines are weaker than "strong" suggests

- B1 and B2 both contain the structured features S1/S2, yet B1 is worse than the structure head in three of four cells and B2 is worse in both Qwen cells (Objective 2: 1.379 against 1.275). A model that is beaten by a strict subset of its own inputs is under-regularised or badly fused; the paper's own explanation for H1+A1 (one shared penalty lets a wide block swamp a narrow one) applies here too, with 2,048 embedding dimensions against 75 structured ones.
- The structure head selects the smallest penalty in all four cells, with loss 7.9 to 11.1% higher one step inwards. The authors raise this themselves. The grid should be widened before unblinding; reporting it as a limitation would leave the central comparator known to be under-tuned.
- The B2 fine-tuning grid has only two learning rates and two epoch counts, so every selection is at an edge and the grid cannot show an interior optimum. The selected setting also flips between cells (1 epoch at the lower rate for Objective 1, 3 epochs at the higher rate for Llama Objective 2). A wider grid is needed before B2 can be called a tuned baseline.
- BGE-M3 is a retrieval encoder. A more natural prompt-only comparator is a fine-tuned LM of comparable size reading the prompt, or a different LLM's hidden states (which also serves W3).
- The abstract quotes AUROC 0.85 / 0.84 against 0.79 / 0.77 for B2. The structure head has 0.809 / 0.817. The abstract's comparison is against the weaker of the two prompt-only numbers.

##### W5. Seed noise is the same size as several reported effects, and the paper has the data to show it

Configurations with identical inputs differ only by head seed. For Llama Objective 2, B1+H1, B1+H2 and B1+H1+H2 have identical inputs and CV losses of 1.642, 1.669 and 1.691, a 3% spread. For Qwen Objective 1, H2+A1 and H1+H2+A1 give 0.2386 and 0.2358, and the second is then named "best probe-only" over A1 at 0.2361. Effects of this size are interpreted elsewhere: the 1.4% gain of B2+probes on Qwen, the 1.7% probe advantage on Llama Objective 2, the 2.3 to 4.3% complementarity gain in Objective 2. The duplicates should be reported as an explicit noise floor, every MLP configuration should be run with several seeds, and "best of seven probe-only configurations" should not be selected and then evaluated on the same folds without a correction.

##### W6. The pooled metrics are dominated by k = 4 and 5 and by near-replicate prompts

Each task contributes 1,026 prompt-rule rows, of which 600 come from k = 5 and 288 from k = 4 (86% together), because the number of orders grows as 1, 2, 6, 12, 20. The paper's own descriptive result is that order barely matters (17.7% first against 18.9% last for Llama, flat for Qwen), so the 20 orders of a subset are close to repeated measurements of the same quantity. Headline NLL, AUROC and decile rates are therefore mostly statements about k = 4 and 5, and the statement that A1's advantage "grows with k" is partly a statement about where the rows are. Report k-balanced and subset-balanced versions (one row per task, subset and rule, averaging over orders) alongside the pooled ones.

##### W7. Hidden-state probes are explored too narrowly to support the negative claim

Only three depths are tried (50, 75, 100%), only the final token, and the shallowest option wins in three of four cells with the last layer worst everywhere. The optimum may well be earlier than 50%. With per-type heads on 4,096 dimensions and 13 to 32 training groups per type, each head sees a few dozen distinct rule instances; dimensionality reduction or a shared head with a type embedding would be fairer. The conclusion that "a hidden-state probe that is not told the exact rule does no better than structured prompt features" should be limited to this configuration.

##### W8. No noise ceiling

With n = 20, part of the binomial NLL and of the MSE against q-hat is irreducible sampling noise. Without an estimate of it (for instance from split-half labels, 10 against 10 responses), a reader cannot tell whether a 10% loss reduction closes a tenth or a half of the explainable gap. The same split-half computation gives the reliability of q-hat, which bounds the Spearman values of 0.50 to 0.55.

##### W9. The confirmatory plan has soft spots

- "Support requires C1 contrasts against B2 that favour the probe ... together with the corresponding structure-head contrasts" does not say whether all, any or a named contrast must pass. As written, the decision rule can be read after the fact.
- Holm is applied within family, generator and objective. RQ1 spans two generators and several probes; there is no stated rule for combining cells.
- Fifteen clusters is few for a percentile bootstrap, and several rule types appear in only 3 or 4 test groups. A power calculation from the out-of-fold effect sizes and between-group variance would say whether the planned test can detect the effects seen in training.
- 50 of 90 test rule instances also occur in training. The paper is clear that this is not a test of unseen rules, but it means a predictor that memorises exact-rule difficulty is rewarded on the test set. A split of test rows into seen and unseen exact rules should be pre-specified.

##### W10. Labels include a known artefact that a predictor can learn

The uppercase audit shows that 6.1% of Qwen's checks of that rule fail only on language detection, far more often when a letter-count rule co-occurs. The structure head sees rule co-occurrence, and attention features may too, so part of what is being predicted is langdetect behaviour. The planned sensitivity analysis drops the rule type; the corrected-label analysis is only "under consideration". It should be added, and the paper should check whether the letter-count verifier and the 2.5% truncated and 2.1% looping Qwen responses similarly reward predicting verifier or decoding failures rather than non-compliance.

##### W11. Motivation and practical value are asserted, not shown

All 15 rule types are checked by cheap deterministic programs. Generating and re-checking is therefore a strong and simple alternative to pre-generation prediction, and the paper does not compare against it or show a downstream use (routing, reordering, resampling) of the predicted risks. A top-decile violation rate of 71% is promising, but no decision-level result is given. Within-type AUROC averages 0.73 on Llama and 0.56 on Qwen, so for Qwen the predictor mostly tells the user which rule types are hard, which a lookup table does.

##### W12. Scope and smaller points

- Only two 8B models are studied, with English prompts, at most five rules, and default decoding that differs between the models.
- Internal features come from a different backend (Transformers on Apple silicon) from generation (vLLM on GPU). The 0.998 cosine check is for hidden states in a pilot; nothing is reported for attention, which is the winning feature.
- In Objective 2, "H1" includes the rule types and order (M2) and "A1" includes k, so "probe-only" configurations are not internal-only. The naming should say so.
- The task-family taxonomy is the authors' own and is an input feature; inter-annotator agreement or a sensitivity check without it would help.
- Novelty relative to Heo et al. (2025) and the SAT probe is mainly the multi-rule setting and the baseline comparison. That is a reasonable increment, but it depends entirely on the baselines and controls being convincing (W2 to W4).
- The manuscript is formatted as a project report with a status box and a "remaining work" section, not as a conference paper.

#### Questions for the authors

1. What are the Objective 1 losses for the structure head, B1 and B2 with per-type heads, and for a per-type head on the rule embedding alone?
2. How well does Llama's attention predict Qwen's labels, and the reverse?
3. What is the split-half noise floor for NLL and RMSE in each cell?
4. With a widened penalty grid and block-specific penalties, do B1 and B2 at least match the structure head that they contain?
5. What are the seed-to-seed standard deviations of the CV loss for A1, H2, B2 and the structure head?
6. Do the conclusions hold on k-balanced metrics, and on rows restricted to rule instances that do not appear in the fitting folds?
7. Which exact contrasts must pass for RQ1 to count as supported, and what is the power on 15 groups?
8. Do layers below 50% depth do better for H1?
9. Does A1 survive adding span length and span position to the structure head?
10. How do backend differences affect the stored attention values?

#### Assessment

- **Soundness:** 2 (fair). Careful protocol, but the headline comparison is confounded by head structure, the baselines are under-tuned by the paper's own evidence, and there is no control showing A1 reflects the generator rather than the prompt.
- **Presentation:** 3 (good). Clear, precise and candid; not in conference form.
- **Contribution:** 2 (fair). A well-designed dataset and a useful verifier finding; the central claim is not yet tested.

**Rating: 3 (reject, not good enough).** The work is not finished: there are no held-out results and none of the controlled analyses. The preliminary evidence is also less clean than the abstract suggests, for the reasons in W2 to W6. I would expect a version with the confirmatory evaluation, per-type-head baselines, a cross-model attention control, a seed noise floor and a wider penalty grid to be a solid accept-range paper whichever way the result falls, because the question is well posed and the design is strong.

**Confidence: 4.** I read the main text, all generated tables and the figures closely, and checked the quoted numbers against the tables. I did not have access to code, the pre-specification file or the bibliography.


\newpage

## Appendix — Way 2: Rubric

### ICLR review form: "Predicting Rule Violations Before Generation"

Reviewed from `paper.tex`, the 18 generated tables and the 6 figures only. The bibliography and code were not inspected.

#### Paper Summary

The paper asks whether a model's violations of prompt instructions can be predicted before generation, and whether the generator's own internals (final-prompt-token hidden state, attention from that token to each rule's span) predict them better than the prompt text does. It builds a balanced dataset from ManyIFEval: 216 tasks in 72 groups of three, a pool of six sampled rules per group, nested rule subsets for k = 1 to 5 and strength-two orderings, giving 53,136 prompts. Llama 3.1 8B and Qwen3-8B each produce 20 responses per prompt, checked with the official verifiers to give per-rule (Objective 1) and per-prompt (Objective 2) violation probabilities. Twenty-three combinations of three probe components (H1, H2, A1) and two prompt-only baselines (frozen and LoRA-tuned BGE-M3 plus structured features) are compared with rate references and a "structure head".

The analysis plan is frozen and the 15 test groups are unread, so every result is grouped 5-fold cross-validation on the 57 training groups. There, the attention probe is the best single per-rule predictor (out-of-fold AUROC 0.852 and 0.844; NLL 0.039 and 0.024 below the fine-tuned baseline). Probe-only configurations beat the strongest prompt-only comparator by 8 to 14% in three of four cells, and the fine-tuned baseline ties them on Llama's per-prompt objective. The paper also reports that the official "all uppercase" verifier fails uppercase English text at its language-detection step (43.1% of Qwen's violations of that rule).

#### Presentation: 3

#### Informativeness: 2

#### Soundness: 2

#### Critical Strengths

1. **The right question, posed against the right kind of baseline.** The paper states that a hidden-state probe may succeed only because the prompt is encoded in it, and it reports every result against type-rate references and a structure head. The finding that type-by-k rates and structured features already remove 12 to 21% of the constant-rate loss is useful on its own.

2. **Careful design and leakage control.** Splitting by group, the nested chains, the strength-two orderings, the witness-response feasibility check and the byte-for-byte reproduction of the 2,160 released prompts are all sound. I recomputed the counts in the dataset table and the uppercase audit (for example 12,554 / 205,200 = 6.1%, and 171 rows x 3 tasks x 20 responses x 20 groups = 205,200) and they agree.

3. **Unusual candour.** The draft discloses the one look at test-inclusive data, the edge-of-grid selections, the 15 distinct feature sets among 23 names, the shared-versus-per-type head confound and the post-freeze edit. The text matches the tables in every place I checked.

4. **Negative and mixed results are reported.** H1 is indistinguishable from the structure head in Objective 1, H1+A1 is 8% worse than A1 alone, the fine-tuned encoder wins on letter count, and within-type AUROC for Qwen is 0.56 against 0.54.

5. **The verifier audit is concrete and reusable.** It is supported by counts, the detected language, the lower-casing check and the interaction with the letter-count rule, and it matters to anyone using IFEval-derived labels.

#### Critical Weaknesses

1. **There is no held-out result, so none of the four research questions is answered.** The paper is a pre-registration plus training-set preview. The out-of-fold intervals are descriptive, the "best probe-only" row of the best-versus-comparator table is chosen after seeing the same folds, and no power analysis is given for a confirmatory test on only 15 groups (3 test groups for two rule types, 4 for three more). As submitted, the central claim is unverified.

2. **The key comparator is demonstrably under-tuned, at the scale of the headline effect.** The structure head selects the smallest penalty in all four cells and loses 7.9 to 11.1% one grid step inwards, while the probe margins over it are 8.4 to 14.2%. A one-hidden-layer MLP on 75 tabular columns is also a weak learner for this input; gradient-boosted trees are the standard comparator. Until this is fixed, the 8 to 14% figure in the abstract is an upper bound on the effect.

3. **The per-type-head confound affects the headline C1 comparison, not only C2 and C3.** In Objective 1 the structure head, B1 and B2 each use one shared head, whereas H1, H2 and A1 use 15 per-type heads. The paper admits this for "baseline plus probe" but A1 versus B2 has the same problem. The missing control, per-type heads on the rule embedding and structured features with no internal features, is listed only as exploratory extension (iii). It decides the interpretation: H1 with per-type heads scores 0.3761, the structure head 0.3779, and adding the frozen rule embedding (H2) gives 0.3390, so most of H2's gain may come from rule text routed through per-type heads.

4. **Nothing shows that the attention signal is internal to the generator.** A 1,024-dimensional vector of attention to a span, summed over tokens, encodes span length, distance from the final token, the number of competing rules and probably the rule's wording. A1's within-type AUROC on letter count is 0.92 against 0.82 for the structure head and 0.93 for B2, which reads the letter from the text; this suggests A1 is partly reading rule identity. The paper cites control tasks (Hewitt and Liang) but runs none. The cheapest decisive control is a swap: predict Llama's labels from Qwen's H and A features and the reverse. Another model's activations are a prompt-only feature set, so if they do nearly as well the claim of generator-specific information fails.

5. **Seed noise is as large as several reported differences.** Configurations with identical inputs differ only by head seed, and they disagree by up to 3%: for Llama Objective 2, B1+H1, B1+H2 and B1+H1+H2 score 1.642, 1.669 and 1.691. The paper nonetheless interprets a 1.4% gap (Qwen Objective 1, 0.2324 against 0.2358), a 1.7% gap (Llama Objective 2, probe against B2) and names H1+H2+A1 as Qwen's best probe-only configuration although it has the same inputs as H2+A1 (0.2386). Each distinct feature set needs several seeds, and the 23 names should be reduced to 15.

6. **B2 is not a strong baseline for Qwen and may be handicapped by its construction.** For Qwen, B2 contains the structured features and still does worse than the structure head in both objectives (0.2606 against 0.2573; 1.379 against 1.275), and its AUROC is below the structure head for both generators (0.793 against 0.809; 0.774 against 0.817). The abstract contrasts A1's AUROC with B2 only, which is the weaker comparator on that metric. Adding H1 to B2 gives 1.375 where H1 alone gives 1.130, which is a symptom of a fitting problem. A likely cause is that the encoder is fine-tuned on the same rows whose embeddings then train the head, so in-fold embeddings carry memorised label information that held-out embeddings lack. The fine-tuning grid is also at its edge in Objective 1 (lowest learning rate, one epoch).

7. **Other grids are at an edge as well, on the probe side.** Only 50, 75 and 100% depth are tried, 50% wins in three of four cells and the loss rises steeply after it (0.3761 to 0.3973 for Llama), so earlier layers may be better. That H1+A1 is 8% worse than A1 alone with the penalty tuned says the heads, not the features, limit several comparisons.

8. **No noise ceiling.** With 20 samples and 53 to 75% of rows at zero violations, it is unclear how much of the gap between 0.378 and 0.331 is closable. A split-half estimate (predict 10 samples from the other 10) and the exact-rule-instance rate as an oracle would place the gains on an interpretable scale.

9. **The pooled metrics are dominated by k = 4 and 5.** The design gives 6, 24, 108, 288 and 600 rows per task for k = 1 to 5, so k = 5 is 58% of Objective 1 rows and 49% of prompts, and most of these are reorderings of the same rule set with nearly identical labels (the position effect is about one point). Since A1's advantage "grows with k", the pooled headline is partly a product of this weighting. The effective sample is about 340 training rule instances, 13 to 32 per type.

10. **The uppercase correction is overstated.** The text says the affected responses "follow the rule as a person would read it", but 3,520 of the 12,554 are repetition loops and 2,715 are not detected as English even after lower-casing. Counting only the 9,839 that pass after lower-casing gives a corrected Qwen rate of about 9.4%, not 8.1%.

11. **The practical motivation is untested.** The introduction motivates reordering, routing and resampling, but no intervention is evaluated, and Objective 2 predicts the expected fraction of rules violated instead of the probability that all rules are followed. Only two 8B open models are studied, so generality is limited.

#### Questions For Authors

1. Will you widen the penalty grid before unblinding? I recommend it strongly: the test set is unread, so it costs nothing in validity, and the current grid biases the main comparison in the probes' favour. Please add a tree-based structure comparator at the same time.
2. What does a per-type-head model on the rule embedding plus structured features, with no internal features, score in Objective 1? Can this be promoted from exploratory to pre-specified?
3. How well do Qwen's hidden states and attention predict Llama's violations, and the reverse?
4. How well does A1 do after regressing out span length, span position and k, or when restricted to a small set of layers and heads?
5. What is the spread over five or more seeds for each of the 15 distinct feature sets, and which differences in the cross-validation table survive it?
6. Are B2 embeddings for head-training rows produced by an encoder that was trained on those rows? If so, what happens with cross-fitted embeddings, or when the jointly trained head is used directly?
7. What is the split-half noise floor of the binomial NLL and of the Objective 2 RMSE for each generator?
8. Drawing 15 of the 57 training groups repeatedly, how often do the out-of-fold C1 contrasts reach Holm-adjusted p < 0.05? Percentile intervals from 15 clusters tend to under-cover; have you considered BCa or a group-level permutation test?
9. Do the results change with a k-balanced metric, or with one row per (task, rule set, target rule) averaged over orders?
10. How does the mean of the Objective 1 per-rule predictions perform as an Objective 2 predictor?
11. Does the "all lowercase" rule, which uses the same language check, show the same artefact? How many uppercase labels flip under a different language-detector seed?
12. How close are attention weights, not only hidden states, between the generation and feature-extraction backends?
13. For Llama's title rule, A1 reaches a within-type AUROC of 0.97 against 0.67 for the structure head. What distinguishes the 3% of violating cases?

#### Flag For Ethics Review

No

#### Rating: 2

Below the acceptance threshold in its current form. The design and reporting are of high quality, but the confirmatory evaluation has not been run and the training-set evidence depends on an under-tuned comparator and uncontrolled confounds (head structure, seed noise, prompt information carried by attention). Fixing weaknesses 2 to 5 before unblinding and then reporting the test results would make this a strong submission.

#### Confidence: 2

I read the paper, all tables and all figures closely and recomputed several numbers. I did not see the code, the frozen plan or the bibliography, and I cannot verify the related-work claims.


\newpage

## Appendix — Way 3a: Reviewer — experiments and soundness

### Review (experiments and soundness): "Predicting Rule Violations Before Generation"

**Score: 4 / 10** (reject in current form; the design is strong enough that a completed version with the controls below could be a clear accept)

#### Summary

The paper asks whether a generator's pre-generation internals (final-prompt-token hidden state, attention to each rule's span) predict rule violations better than prompt-only predictors. It builds a balanced 53,136-prompt design from ManyIFEval, labels it with 20 samples per prompt from two 8B models, and compares 23 feature configurations per objective against structure-only references. The held-out test groups are deliberately unread, so every result is grouped cross-validation on the 57 training groups. The headline: the attention probe A1 is the best single per-rule predictor (out-of-fold AUROC 0.85/0.84), and internal-state probes beat the strongest prompt-only comparator by 8 to 14% in three of four cells.

#### Strengths

1. **Unusually careful design.** Soft labels from repeated sampling, a strength-two order design, nested chains, group-level splitting, and a group-level paired bootstrap are all the right choices. Groups, not rows, are treated as the unit of independence throughout.
2. **The right question.** The paper compares against prompt-only predictors and rate references rather than chance, and shows that type-by-k rates and a structure head already remove 12 to 21% of the constant-rate loss.
3. **Pre-specification and disclosure.** A hashed plan, a loader that refuses test rows, and candid disclosure of the one test-touching statistic, the narrow penalty grid, the duplicate configurations and the head-structure confound.
4. **Honest negative and qualifying results.** H1 does not beat the structure head in Objective 1; B2 matches the probes for Llama Objective 2; within-type AUROC for Qwen is 0.56 (A1) against 0.58 (B2); H1+A1 is worse than A1.
5. **The verifier audit** (43.1% of Qwen's all-uppercase violations fail only the language-detection step) is concrete, checkable and useful to anyone using IFEval-family verifiers.

#### Weaknesses

**W1. There is no held-out evidence.** The paper's central claims rest on cross-validation losses whose head type, penalty, epoch count, hidden-state layer, attention pooling and "best probe-only" configuration were all selected on the same five folds that are then used to report effect sizes. The authors say this, but it means the submission is a registered-report stage 1, not a results paper. Nested cross-validation on the training groups would have given unbiased training-side estimates at no cost to the test set.

**W2. The head-structure confound contaminates the headline contrasts, not only C2 and C3.** Every configuration containing H1, H2 or A1 uses 15 per-type heads, while the structure head, B1 and B2 use one shared head, so the headline comparisons (A1 against B2 and against the structure head) change the model class and the features at the same time. The limitation section acknowledges this only for the "add a probe to a baseline" contrasts. A per-type-head version of the structure head, B1 and B2 is an essential control and is currently listed only as an exploratory extension.

**W3. The prompt-only baselines are not as strong as claimed.** Three pieces of the paper's own evidence:
- B1 contains the structured features as a strict subset of its inputs, yet has a higher loss than the structure head in three of four cells (0.3868 vs 0.3779; 0.2646 vs 0.2573; 1.360 vs 1.275). A superset of features doing worse is a tuning failure (one shared penalty over 2,048 standardised embedding columns and 75 structured ones), not evidence that text adds nothing.
- The structure head selects the smallest penalty in all four cells, with loss 7.9 to 11.1% higher one step inward. The possible gain from a wider grid is of the same order as the reported probe margins (8.4 to 13.5% over the structure head).
- For Qwen Objective 2, adding B2 embeddings to H1 raises the loss from 1.130 to 1.375, and no B2-family configuration beats the structure head. This is the signature of stacking on in-sample embeddings: within a fold, the head is trained on embeddings of rows the encoder was fine-tuned on and evaluated on embeddings of rows it never saw. Heads on B2 features should be trained on cross-fitted (inner out-of-fold) embeddings. The B2 grid (two learning rates, one or three epochs) also selects its edge in three of four cells.

The abstract compares A1's AUROC with B2 (0.79, 0.77), although the structure head is the stronger comparator on that metric (0.809, 0.817). There is also no tree-based tabular baseline on the structured features, and the structure head is denied the letter and keywords, which is exactly where it loses most (letter count: 0.491 vs 0.281 for B2).

**W4. Nothing tests that the signal is the model's *own* state.** A1 sums attention over a span, so it mechanically carries span length, position, k and the number of competing rules, and plausibly rule identity. The paper cites control tasks (Hewitt and Liang) but runs none. The decisive control is cross-model: predict Llama's labels from Qwen's features and vice versa (same dimensionality, same pipeline, same heads). The comparison is otherwise confounded with encoder capacity (a 0.6B text encoder against an 8B model's representations). This is listed as exploratory; it should be primary.

**W5. The planned confirmatory test looks underpowered for its own success criterion.** Training-side interval half-widths with 57 groups are about 0.015 NLL for effects of 0.024 (Qwen) and 0.039 (Llama). Scaling by sqrt(57/15) gives about 0.03 on 15 test groups, before the Holm adjustment and before the conjunction with the structure-head contrasts. The Objective 2 effects (about 0.011 RMSE, half-width 0.007) fare similarly. No power analysis is reported, though the out-of-fold predictions allow one by subsampling 15 groups. Percentile bootstraps with 15 clusters also tend to undercover, and several rule types appear in only three or four test groups. The criterion itself ("C1 contrasts against B2 that favour the probe") does not say which or how many contrasts must pass.

**W6. Seed noise is unquantified and is as large as some reported differences.** Configurations with identical inputs differ by up to 0.0036 NLL (Llama B1+H1 0.3429 vs B1+H1+H2 0.3393; Qwen H2+A1 0.2386 vs H1+H2+A1 0.2358). The "best probe-only" choice for Qwen Objective 1 (H1+H2+A1 over A1 by 0.0003) and the bolded lowest loss (B2+H1+H2+A1, 0.2324, vs its twin at 0.2348) are therefore within seed noise. Everything is a single seed and a single fold partition.

**W7. Effective sample size and pooling.** Each rule type has one instance per group, so a per-type head sees only 13 to 32 distinct rule instances (fewer inside a fold), with 1,000 to 8,000 inputs. Rows at k=4 and k=5 make up 86% of Objective 1 rows (888 of 1,026 per task) and are order permutations of a handful of subsets, so pooled metrics are essentially a k>=4 result. Fold-to-fold SD (0.03 to 0.04) is the size of the effects; the Qwen Objective 1 best probe beats B2 in only three of five folds.

**W8. No noise ceiling, and label artefacts may carry signal.** With binomial labels from n=20, the irreducible NLL and RMSE can be estimated (for example by split-half), which would show how much explainable loss remains. For Qwen, 2.5% truncated and 2.1% looping responses fail several rules at once, and the uppercase artefact is concentrated in ten tasks and co-occurs with letter-count rules. The Qwen Objective 2 advantage of H1 may partly be prediction of degenerate outputs rather than of rule-specific non-compliance. The sensitivity analyses are planned only for the test set; they can be run out-of-fold now.

**W9. Smaller points.** Feature extraction used a different backend from generation and only hidden-state agreement is reported, not attention agreement. Several pre-specified contrasts compare identical inputs (Qwen Objective 2, B1+H1 vs B1+H2: difference 0.0000 with interval [0, 0]) and should be removed from the Holm families rather than kept. 50 of 90 test rule instances duplicate training rules.

#### Do the conclusions follow?

Mostly yes as worded, because the wording is hedged ("so far", "qualified yes"). The claim that survives scrutiny is narrow: for Llama, per-rule, A1 and H2 carry within-type information beyond structure (within-type AUROC 0.73 vs 0.63). For Qwen the within-type gain is 0.02 and smaller than B2's, so "for both generators" overstates the per-rule result; much of the pooled AUROC is knowing which rule is asked about. The statement that probes "improve on a text encoder fine-tuned for the same task" does not yet follow, given W2 and W3.

#### Questions

1. What are the losses of the structure head, B1 and B2 with 15 per-type heads?
2. What happens to all margins with the widened penalty grid, block-specific penalties, and cross-fitted B2 embeddings?
3. How well do Qwen's features predict Llama's labels, and the reverse?
4. What is the split-half noise ceiling for each objective and generator?
5. What is the simulated power of the 15-group test for each C1 contrast, using subsamples of the out-of-fold predictions?
6. What is the spread over head seeds and over repeated fold partitions?
7. Does A1 still beat a head given only span length, span position, k and rule type (a no-model attention control)?
8. Do the Qwen Objective 2 results survive, out of fold, when truncated and looping responses and the uppercase artefact are excluded?
9. Was any verifier other than "all uppercase" audited by cause of failure?

#### Recommendation

Reject in the present form: the experiments are well designed but unfinished, and the two comparisons that carry the claim (probe vs fine-tuned text baseline, probe vs structure head) are confounded by head structure and by under-tuned comparators. Running the controls in questions 1 to 3 on training data, freezing again, and then reporting the test evaluation would make this a much stronger paper.


\newpage

## Appendix — Way 3b: Reviewer — novelty and related work

### Review (novelty and related work focus)

**Paper:** Predicting Rule Violations Before Generation: Hidden-State and Attention Probes versus Prompt-Only Baselines on a Controlled Multi-Instruction Benchmark

**Score: 4 / 10** (below the acceptance threshold in its current form; the main reason is that it has no held-out results, the second is that the central novelty claim is not yet isolated)

**Confidence:** 4 / 5. Citations I suggest below are from memory; the authors should verify titles and venues before citing.

#### Summary

The paper asks whether an LLM's violation of each rule in a multi-rule prompt can be predicted before generation, and whether the generator's own internal state (final-prompt-token hidden state, attention from that token to each rule span) predicts it better than predictors that see only the prompt. It builds a balanced design on ManyIFEval (53,136 prompts, 2 generators, 20 samples per prompt, soft labels), compares 23 feature configurations per objective against structure-only references and a frozen and fine-tuned BGE-M3 baseline, and freezes an analysis plan. Only grouped cross-validation results on training groups are reported; the test set is unread. On those, the attention probe is the best single per-rule predictor (OOF AUROC 0.85/0.84 vs 0.79/0.77 for the fine-tuned text baseline).

#### What is new relative to prior work

1. **Per-rule, multi-rule, pre-generation prediction.** Heo et al. (2025) probe single-instruction IFEval prompts with a binary outcome. Here each of up to five co-occurring rules is a separate target. This is a real extension, and the per-type result (Figure 5) and the growth of the gain with k (0.332 vs 0.341 at k=1, 0.338 vs 0.387 at k=5 for Llama) are the most interesting new observations.
2. **Probability labels from 20 samples** instead of one greedy outcome. Useful, though not new in kind (semantic-entropy probes and P(IK) already train on sampled-outcome targets).
3. **The incremental question with a supervised prompt-only comparator and a structure-only reference.** This is the paper's main conceptual contribution, and the finding that "known structure already explains a lot" (structure head removes about 20% of the constant-rate loss; plain H1 does not beat it) is a valuable negative result for the probing literature.
4. **Design-of-experiments construction** (nested chains, strength-two component orthogonal arrays for order). I am not aware of this being used in instruction-following evaluation; it is new as tooling, but its payoff (controlled analyses A to E) is not yet delivered.
5. **The langdetect artefact** in the "all uppercase" verifier (43% of Qwen's violations of that rule). A concrete, reusable finding for anyone using IFEval or ManyIFEval.

Items 1 and 3 are the novelty that would carry an ICLR paper. Methodologically the probes themselves are not new: A1 is SAT Probe applied to instruction spans (the paper says so), H1 is a standard last-token probe.

#### Is the positioning accurate?

Mostly honest, and unusually careful about what is not claimed. Four places where it overstates or under-specifies:

- **"Strong" prompt-only baselines.** The fine-tuned baseline B2 has lower OOF AUROC than the structure head on both generators (0.793 vs 0.809; 0.774 vs 0.817), and B1 is worse than the structure head in three of four cells. A baseline that loses to 75 hand-built columns is not a strong text baseline; it suggests an embedding-plus-shared-penalty fusion problem. The headline "8 to 14% over the strongest comparator without internal access" is therefore a claim against a comparator that is probably under-tuned (the paper itself notes the structure head sits at the edge of the λ grid).
- **"The model's own internal state."** No experiment separates "the generator's state" from "any 8B LLM's encoding of the prompt". The decisive control is cross-model: predict Llama's violations from Qwen's hidden states and attention, and the reverse. This is listed only as exploratory item (iv). Without it, A1 beating BGE-M3 may reflect a 16x larger reader, not self-knowledge. The moderate cross-generator agreement (Spearman 0.43) makes this control informative.
- **Head-structure confound.** Probe configurations get 15 per-type heads; B1, B2 and the structure head get one shared head. The paper discloses this for C2/C3 but the same confound applies to the headline C1 comparison (A1 vs B2), which is what the abstract reports.
- **What attention-to-span encodes.** Summed attention over a span carries span length and position, which are prompt-derivable. Within-type AUROC is 0.56 for A1 on Qwen versus 0.58 for B2, so for Qwen the "internal" advantage is almost entirely between-type or calibration. The abstract's symmetric phrasing ("for both generators") hides this.

The description of Heo et al. and of Stolfo et al. is accurate as far as I can tell. The statement that prior multi-instruction work "measures or estimates aggregate compliance" is fair for ManyIFEval and IFScale.

#### Missing related work

- **Prompt-only predictors of model success (the baseline's own literature).** Assessors: Zhou et al., "Reject Before You Run: Small Assessors Anticipate Big Language Models" (2022); Schellaert et al. on predictability of LLM performance. Routing: FrugalGPT (Chen et al., 2023), Hybrid LLM (Ding et al., 2024), RouteLLM (Ong et al., 2024). The introduction motivates routing but cites none of it, and these are exactly B1/B2-style predictors.
- **Conditional / baselined probing.** Hewitt et al., "Conditional Probing: Measuring Usable Information Beyond a Baseline" (EMNLP 2021) formalises RQ3; also Voita and Titov (MDL probing) and Pimentel et al. (information-theoretic probing). RQ3 should be framed against these.
- **Pre-generation probes of own behaviour.** Ji et al., "LLM Internal States Reveal Hallucination Risk Faced With a Query" (2024); Slobodkin et al. on (un)answerability (2023); Snyder et al. on early hallucination detection; and, closest, Ashok and May, "Language Models Can Predict Their Own Behavior" (2025), which to my recollection includes format-following failures predicted from input-token states.
- **Uncertainty for instruction following.** Heo et al., "Do LLMs Estimate Uncertainty Well in Instruction-Following?" (ICLR 2025), by the same group as the closest cited work, evaluates verbalised, logit and probe-based uncertainty for this task. No verbalised self-prediction or P(IK)-style baseline is included here.
- **Attention features as detectors.** Lookback Lens (Chuang et al., 2024) trains linear classifiers on per-head attention ratios; Attention Tracker (Hung et al., 2024) uses attention to the instruction for prompt-injection detection; instruction-attention steering (e.g. Spotlight, Venkateswaran and Contractor, 2025) alongside PASTA.
- **Multi-constraint benchmarks and order effects.** ComplexBench (Wen et al., NeurIPS 2024), COLLIE (Yao et al., 2024), Sun et al. (2023) on controlled generation, and work on constraint-order effects (I recall "Order Matters", Zeng et al., 2025). The paper's position result (17.7% first vs 18.9% last) should be compared with these.

#### Strengths

- Clear, well-motivated question; the incremental framing is the right one and rarely done properly.
- Structure-only references and the candid negative results (H1 equals structure head; Llama Objective 2 tie) raise the bar for this literature.
- Group-level splits, pre-specification, hashed plan, disclosed deviations.
- Scale and design quality of the dataset; the verifier audit.
- Limitations section is accurate and complete.

#### Weaknesses

- No test results. All evidence is cross-validation on data used for selection; the paper says so, but a venue submission cannot be judged on a plan.
- The novelty-bearing claim (internal beats prompt-only) is confounded by head structure, baseline strength, λ-grid edges and the absence of a cross-model control.
- 15 of 23 configurations have distinct inputs; the "23 predictors" framing inflates the comparison.
- Related work omits the assessor/routing and conditional-probing literatures that define the baseline and the question.
- Two 8B models, 15 templated rule types, 50 of 90 test rule instances seen in training: generality of the attention finding is untested.

#### Questions

1. What does A1 score against B2 and the structure head when all three use per-type heads?
2. Cross-model control: how well do Qwen's H and A features predict Llama's labels?
3. Does a structure head given span length and token position of the target rule close the gap to A1?
4. Why does fine-tuned B2 rank worse (AUROC) than the structure head? Was a stronger text predictor (larger encoder, or a LoRA-tuned small LLM) tried?
5. Is the langdetect failure on uppercase text already reported for IFEval, and does it affect the released ManyIFEval numbers?
6. How does the per-rule result compare with Heo et al.'s direction on the k=1 subset, where the settings coincide?

#### Recommendation

A careful and honest study with a worthwhile question, but as submitted it is a registered report without the result. With test-set numbers, the per-type-head and cross-model controls, and the missing literature, this could be a clear accept.


\newpage

## Appendix — Way 3c: Reviewer — clarity and presentation

### Review (clarity and presentation focus)

**Paper:** Predicting Rule Violations Before Generation: Hidden-State and Attention Probes versus Prompt-Only Baselines on a Controlled Multi-Instruction Benchmark

**Basis of this review:** `paper.tex`, the 18 generated tables it inputs, and the six figure PNGs. I did not see the bibliography or a compiled PDF, so I cannot comment on page layout, float placement or reference formatting. The source includes the figures as `.pdf`; I reviewed the `.png` versions.

**Score: 6 / 10** (marginally above the acceptance threshold on clarity alone; see "Reason for the score")

---

#### Summary of the paper

The paper asks whether a language model's internal state at the end of a prompt predicts which of several instructions it will violate, better than a strong predictor that sees only the prompt. It builds 53,136 prompts from ManyIFEval in a balanced design (72 groups, k = 1 to 5 rules, orthogonal-array orderings), samples 20 responses per prompt from two 8B models, and compares 23 feature configurations (hidden state H1, hidden state plus rule embedding H2, attention to the rule span A1, frozen text baseline B1, fine-tuned text baseline B2, and combinations) against structure-only references. The manuscript is explicitly a pre-unblinding draft: all results are grouped cross-validation or out-of-fold on the 57 training groups, and the 15 test groups are unread.

---

#### Strengths

1. **The status of every claim is stated, repeatedly and in the right places.** The boxed note before the abstract, the section title "Results so far (training groups only)", the phrase "Descriptive, not confirmatory" in captions, and the paragraph on why out-of-fold intervals are not confirmatory leave no room to misread training-set numbers as test results. This is rarer than it should be.

2. **The question is sharpened before any method appears.** The introduction's third paragraph (a probe may do well "simply because the prompt is encoded" in the hidden state) sets up the incremental question, and the four numbered design choices follow from it. A reader knows why each baseline exists before meeting it.

3. **Problem setup is compact and unambiguous.** Section 3 defines p_r(x), q(x), their estimators and the two objectives in half a page, and heads off the obvious misreading of q ("expected fraction of rules violated, not the probability that at least one is").

4. **The dataset construction can be followed and audited by counting.** Table 1 gives per-task, per-group, train, test and total counts for every unit, and they multiply out (246 x 3 x 72 = 53,136; 6 x (1+2+6+12+20) = 246). The overview figure maps cleanly onto Sections 4 to 6.

5. **Prose style.** Short declarative sentences, one idea per paragraph, bold run-in heads that state the finding ("Known structure already explains a lot"). Numbers in the text match the tables I checked (Table 2, Table 3, the contrast tables, the per-k table, the uppercase audit).

6. **Unflattering facts are surfaced, not buried.** Only 15 of 23 configurations have distinct inputs; the regularisation grid is edge-selected for the key comparator; head structure is confounded with probe features; one test-set statistic was computed before the firewall. Each is disclosed in the main text and again in Limitations.

7. **Reproducibility detail is above average.** Decoding settings, token limits, optimiser settings, LoRA hyperparameters, bootstrap replicate counts, fold sizes, feature-block widths and compute cost are all given. The appendix reports the selected head for every configuration.

8. **Figures are clean.** Consistent colour per generator, direct labelling, intervals drawn, and captions that warn when axes differ between panels.

---

#### Weaknesses

##### A. Things that affect whether the main claim can be read correctly

1. **The per-type-head confound is disclosed for the wrong (or too few) contrasts.** Section 5.2 says any Objective 1 configuration containing H1, H2 or A1 uses 15 per-type heads, while B1, B2 and the structure head use one shared head. The paper flags this for C2 and C3 (adding a probe to a baseline). But the headline comparisons (A1 vs B2, A1 vs structure head, Table 3, Figure 4, the abstract's AUROC 0.85 vs 0.79) are also per-type heads against a shared head. The text never says so at the point where those numbers are presented, and the Limitations entry names only C2 and C3. A reader will take the abstract's comparison as features against features. Either state the confound next to Table 3 and in the abstract, or explain why it does not apply to C1 and the reference contrasts.

2. **Seed noise between identical-input configurations is as large as several reported effects, and the paper does not give the reader a noise floor.** Examples from the appendix tables: for Llama Objective 2, B1+H1, B1+H2 and B1+H1+H2 are stated to have identical inputs but score 1.642, 1.669 and 1.691 (a 3% spread). For Qwen Objective 1, H2+A1 and H1+H2+A1 are identical inputs and score 0.2386 and 0.2358; the latter is then named "best probe-only" and the text reports the B2-plus-probe optimum as "1.4% below the best probe-only configuration", a margin of the same size. Meanwhile some identical pairs agree exactly (the Qwen Objective 2 contrast B1+H1 vs B1+H2 is +0.0000 [+0.0000, +0.0000]). The reader cannot tell which duplicates are deterministic (linear heads?) and which are not, or how much any single CV number should be trusted to the third digit. Please (i) state the seed-to-seed spread explicitly, (ii) collapse the 23 names to the 15 distinct feature sets in Figure 3 and the appendix tables, or mark duplicates visually, and (iii) avoid naming a "best" among duplicates.

3. **Three different loss scales are used for Objective 2 without a bridge.** Table 2 and the appendix report MSE in units of 10^-2 (1.613); Figure 3 plots the same quantity as raw MSE (0.016); Tables 3, 5-appendix and Figure 4 use RMSE (0.127). Table 3 puts a relative change in CV MSE next to an absolute change in OOF RMSE in adjacent columns. A 14% MSE reduction is roughly a 7% RMSE reduction, so the two columns look inconsistent to a reader who does not stop to convert. Pick one scale for Objective 2, or add a sentence and a consistent axis.

4. **"Best probe-only configuration" and "A1" are used interchangeably in places where they are not the same.** Under the bold head "attention is the best single signal", the sentence "The best probe-only configuration lowers the loss of the structure head by 12.3% (Llama) and 8.4% (Qwen)" refers to A1 for Llama but H1+H2+A1 for Qwen. Table 3 names the configuration, but the prose and the abstract ("Probes built on internal states lower the loss of the strongest comparator ... by 8 to 14%") do not, and "strongest comparator" changes identity across cells (B2, structure head, structure head) without being named.

##### B. Notation and terminology

5. **Too many names for units of analysis.** "prompt-rule row", "(prompt, rule) pair", "(response, rule) check", "rule instance", "rule-level pair", "cell" (used in the abstract before it is defined as a generator-objective combination). A one-line glossary in Section 3, and using "cell" only after defining it, would help.

6. **Rule-type names are inconsistent.** Section 4.1 uses "keyword existence, keyword frequency"; Section 7.4 uses "keyword inclusion, keyword count"; the tables and figures use "Include keywords", "Keyword count". The same for "letter-case types" versus the three named types. Use the table names everywhere.

7. **Component versus block versus configuration.** The H1/H2/A1/B1/B2 labels are opaque on first contact, and Table 2 introduces ten more block codes (H, A, A5, EP, ER, ER5, S1, S2, M2, KM) in a caption. A reader has to hold fifteen abbreviations to parse Table 2. Dimensions are given for some blocks and not others (M2 and KM have none; the Objective 2 widths in the appendix, e.g. 4,197 for H1, can only be reconciled by guessing M2 = 101). In Table 6, family C4 uses a bare "H" that is never defined. Consider descriptive names (Hidden, Hidden+Rule, Attention, Text-frozen, Text-tuned) at least in figures and the abstract.

8. **Sign conventions for differences are described three ways.** Table 3: "probe minus comparator". Figure 4: "second configuration minus first" with arrows "Structure -> H1". Appendix: "b minus a". They agree, but a single convention stated once would save the reader the check.

9. **Metrics on soft labels are not defined.** AUROC, Brier score, ECE and "top/bottom 10%" are reported for a target that is a count out of 20. The text implies AUROC is computed over individual (response, rule) checks with the row's prediction replicated, but does not say so; likewise for ECE in "decile bins". The binomial NLL "reported per check" is never written as a formula. These need one equation each.

##### C. Reproducibility gaps

10. **Rule-span definition.** "The exact token span of each rule" is not specified: does it include the bullet marker and newline, how are sub-word boundaries at span edges handled, and are spans identical in definition for both tokenisers? Since A1 is the headline feature this matters.

11. **B2 procedure is hard to reconstruct.** It is unclear (i) whether the encoder hyperparameters were selected on the same held-out fold whose loss is then reported for B2 heads, (ii) which encoder produces B2 embeddings for the final all-train refit, (iii) what "a rule fixed in advance on one fold" for LoRA versus full fine-tuning was, and (iv) whether Objective 1 and Objective 2 use separately tuned encoders (Table "B2 grid" suggests yes; the text does not say).

12. **Unspecified procedures.** The written rubric for task families; how the eight near-duplicate clusters were identified; how the five folds were assigned (random, stratified?); what "provably unsatisfiable" means operationally; how repetition loops are detected; how A5 and ER5 pad prompts with fewer than five rules; what the "structure head" input is for Objective 2 beyond "S2". None is long, and an appendix paragraph each would do.

13. **Per-type interval counts cannot be verified.** Section 7.4 reports "13 of the 15 types (with an interval excluding zero on 8)", but neither Figure 5 nor the per-type appendix tables show intervals. Add them to the tables or the figure.

14. **Hardware statement.** Responses come from vLLM on NVIDIA GPUs and features from Transformers on Apple silicon. The 0.998 cosine check is described as coming from "a pilot" with no sample size, layer, or statement of whether it was measured at the selected layers. Attention agreement between backends is not reported at all, though attention is the main result.

##### D. Organisation

15. **The paper is long and its centre of gravity is off for a results paper.** Section 8 (remaining work) is about as long as Section 7.3 to 7.5 combined, includes two tables, and discusses decisions not yet taken. As a registered-report style draft this is defensible, but for a conference submission the reader finishes without a confirmatory result and with two open protocol decisions. The verifier audit (Section 7.6) is interesting but sits oddly as the last results subsection; it reads as a data-quality finding and would fit better in Section 4 or its own short section.

16. **Key tables are in the appendix while the main text cites their numbers.** Per-k results (Section 7.4), the Objective 2 OOF table, layer selection and the lambda-edge table all drive main-text claims but live in appendices. Conversely Table 4 (complementarity) could move out.

17. **Figure-specific issues.**
    - Figure 3: 23 rows per panel with duplicates plotted as if distinct; x-axis "CV MSE" in raw units while the tables use 10^-2; the "lowest loss" ring in Qwen Objective 1 sits on a point visually indistinguishable from five neighbours.
    - Figure 4: filled versus open markers is a significance-style encoding in an analysis the text says has no p-values; say plainly that fill is only a reading aid. The row labels "Structure -> H1" need the caption to decode.
    - Figure 5: structure-head markers are hidden where points overlap (several Qwen rows); no uncertainty shown. The caption says types are ordered by Llama's violation rate, which makes the Qwen panel look unordered; say so.
    - Figure 2a: only the endpoints are labelled; the "dip at k = 2" discussed in the text is within the band and probably should not be mentioned as a feature.
    - Reliability figure: decile bins crowd near zero for Qwen; a log or inset axis would show the region where most mass lies.

18. **Small inconsistencies in the prose.** The largest Llama gains are listed as "letter count, paragraph count, bullet points, title and quotation marks, which are mostly rules about the layout of the response"; letter count is not layout, and the next sentence says B2 wins on letter count. The conclusion then says the improvement "concentrates on layout and length rules", but sentence count (a length rule) is one where the structure head beats A1 for Llama (0.550 vs 0.618).

---

#### Questions for the authors

1. Do the A1-versus-B2 and A1-versus-structure-head comparisons in Objective 1 compare per-type heads against a single shared head? If so, what is B2's loss with per-type heads?
2. What is the seed-to-seed standard deviation of the CV loss for a fixed configuration and head setting? Why do some identical-input pairs match to four decimals and others differ by 3%?
3. How exactly is a rule's token span delimited, and how well do attention features agree between the generation backend and the extraction backend?
4. For AUROC, ECE and the decile rates, what is the unit (prompt-rule row with a soft label, or expanded binary checks)? Please give the formulas.
5. For B2, is the reported fold loss on the same fold used to pick the learning rate and epoch count? Which encoder is used in the final refit?
6. Which contrasts among the 37 compare identical feature sets, and will they be removed or kept in the Holm families in the final version?
7. If the lambda grid is widened, will all main-text tables be regenerated, and will the paper show both versions?
8. Why is Objective 2 trained with squared error and reported as MSE in selection but RMSE in evaluation? Could one be used throughout?
9. Is the 0.43 cross-generator rank correlation of q-hat computed on the same prompts with the same label noise correction, or raw?

---

#### Suggestions (concrete, in priority order)

1. State the head-structure asymmetry beside the headline numbers, or add the per-type-head baseline to the main table.
2. Report results on the 15 distinct feature sets and give a seed-noise floor.
3. Use one Objective 2 scale throughout.
4. Add a notation table (units of analysis, blocks with dimensions, components, configurations) and unify rule-type names.
5. Add equations for the binomial NLL and for each secondary metric.
6. Add an appendix paragraph for each unspecified procedure in item 12, and the span definition in the main text.
7. Shorten Section 8 to a half-page plan plus Table 6; move the pair-count table to the appendix.

---

#### Reason for the score

Judged on writing alone, this is a well above average manuscript: clear question, clean setup, honest framing, verifiable counts, and text that matches its tables. What holds the score down is (i) the headline comparison can be misread because the head-structure asymmetry is flagged only for a subset of contrasts, (ii) the 23-configuration presentation carries eight duplicates whose seed noise is comparable to effects the text interprets, (iii) mixed MSE/RMSE scales and a heavy abbreviation load, (iv) several procedure gaps around the main feature (span definition, B2 selection, soft-label metrics), and (v) the manuscript is by its own statement incomplete, with no held-out result and two protocol decisions open. Items (i) to (iv) are fixable in a revision without new experiments beyond one control.

**Score: 6 / 10. Confidence: 4 / 5** (I read the full source, all generated tables and all figures; I did not see the bibliography or a compiled PDF).


\newpage

## Appendix — Way 3: Area-chair meta-review

### Meta-review: "Predicting Rule Violations Before Generation: Hidden-State and Attention Probes versus Prompt-Only Baselines on a Controlled Multi-Instruction Benchmark"

**Decision: Reject** (encourage resubmission once the confirmatory evaluation and the controls below are done)

Reviewer scores: 4 (experiments/soundness), 4 (novelty/related work), 6 (clarity/presentation).

#### Summary of the submission

The paper asks whether a generator's internal state at the last prompt token predicts which rules in a multi-rule prompt it will break, and whether it does so better than predictors that see only the prompt. It builds 53,136 prompts from all 216 ManyIFEval tasks in a balanced design (72 groups, nested rule subsets for k = 1 to 5, orders from a strength-two orthogonal array). It labels them with 20 sampled responses per prompt from Llama 3.1 8B and Qwen3-8B, checked by the official verifiers. It then compares 23 feature configurations per objective (hidden state H1, hidden state plus rule embedding H2, attention to the rule span A1, frozen and fine-tuned BGE-M3 baselines B1/B2, and their combinations) against rate references and a structure-only head. The 15 test groups are deliberately unread. Every result comes from grouped cross-validation or out-of-fold predictions on the 57 training groups. The headline is that A1 is the best single per-rule predictor (OOF AUROC 0.852 / 0.844), and that internal-state probes beat the strongest comparator without internal access by 8 to 14% in three of four generator-objective cells. The paper also reports a verifier artefact: 43.1% of Qwen's "all uppercase" violations fail only `langdetect`'s English check.

#### Consensus strengths

All three reviewers agree on these, and I do too.

- **The right question.** The paper asks whether internal state adds anything beyond a strong prompt-only predictor, not whether it beats chance. The structure-only reference already removes about 20% of the constant-rate loss, and showing this raises the bar for the probing literature (R1, R2).
- **Careful design and discipline.** Soft labels from repeated sampling, an orthogonal-array order design, group-level splits, a group-level paired bootstrap, a hashed pre-specified plan and a loader that refuses test rows. This is better than most work in this area.
- **Candour.** The paper reports its negative and qualifying results openly: H1 does not beat the structure head in Objective 1, B2 ties the probes for Llama Objective 2, and H1+A1 is worse than A1. It also discloses the narrow λ grid, the duplicate configurations and the one statistic that touched the test set. R3 notes that the status of every claim is stated wherever it matters.
- **A useful by-product.** The verifier audit is concrete, checkable and relevant to everyone who uses IFEval-family verifiers.

#### Why the decision is reject

The reviewers converge, independently, on one assessment. As submitted, this is a stage-1 registered report: a sound design and a frozen plan, but no confirmatory result. The training-side evidence that stands in for a result is also confounded in ways that bear directly on the headline claim. In my judgement no single issue is fatal. Together they mean the paper's central claim, that the generator's internals predict violations better than a prompt-only predictor, has not yet been shown.

1. **No held-out evidence (R1 W1, R2, R3 D15).** Head type, penalty, epochs, layer, attention pooling and the "best probe-only" label were all chosen on the same five folds that then report effect sizes. The authors acknowledge the optimism. Two protocol decisions (widening the λ grid, a verifier-corrected sensitivity analysis) are still open. R1 also estimates that the planned 15-group test may be underpowered for its own success criterion: interval half-widths of about 0.03 NLL against effects of 0.024 to 0.039, before Holm adjustment. In addition, "C1 contrasts against B2 that favour the probe" does not say which or how many contrasts must pass.

2. **The headline comparison mixes model class with features (R1 W2, R2, R3 A1).** In Objective 1, every configuration containing H1, H2 or A1 uses 15 per-type heads, while the structure head, B1 and B2 use one shared head. The paper discloses this confound only for the C2/C3 "add a probe" contrasts. It applies equally to the comparisons in the abstract and in Table 3 (A1 vs B2, A1 vs structure head).

3. **The comparators look under-tuned (R1 W3, R2).** I checked the reviewers' figures against the generated tables, and they hold.
   - B1, whose inputs are a strict superset of the structure head's, has a higher loss than the structure head in three of four cells (for example 0.3868 vs 0.3779 for Llama Objective 1).
   - The structure head selects the smallest λ in all four cells, and its loss one step inwards is 7.9 to 11.1% higher. That is the same size as the reported probe margins.
   - The "strong" fine-tuned baseline B2 has a lower AUROC than the structure head for both generators (0.793 vs 0.809; 0.774 vs 0.817). The abstract nonetheless contrasts A1 with B2's 0.79/0.77.
   - For Qwen Objective 2, adding B2 embeddings to H1 raises the loss from 1.130 to 1.375. That pattern is consistent with heads being fitted on in-sample fine-tuned embeddings.

4. **Nothing yet shows that the signal is the generator's *own* state (R1 W4, R2).** Summed attention over a span mechanically encodes span length, position, k and plausibly rule identity. The comparison also pits a 0.6B text encoder against an 8B model's representations. The cross-model control (predicting Llama's labels from Qwen's features and the reverse) and a "no-model attention" control are both listed only as exploratory. For Qwen, within-type AUROC is 0.56 for A1 against 0.58 for B2, so the "for both generators" phrasing overstates the per-rule result.

5. **Noise floor and presentation (R1 W6, R3 A2 to A4, R2).** Only 15 of the 23 named configurations have distinct inputs. Identical-input twins differ by up to 0.0036 NLL in Objective 1 (Qwen H2+A1 0.2386 vs H1+H2+A1 0.2358) and by up to 3% in Objective 2 (Llama B1+H1 1.642 vs B1+H1+H2 1.691). Yet the paper names a "best probe-only" and a bolded optimum among such twins. Objective 2 switches between three loss scales (MSE ×10-², raw MSE, RMSE). The related-work section omits the assessor/routing literature (FrugalGPT, RouteLLM, small assessors) that defines the prompt-only baseline, and the conditional-probing literature (Hewitt et al. 2021) that formalises RQ3. These are fixable without new experiments, but they currently make the main claim easy to misread.

R3's score of 6 is explicitly "on clarity alone" and agrees that the manuscript is incomplete. I therefore read the three reviews as consistent with rejection rather than split.

#### The three most important fixes

1. **Run and report the confirmatory test evaluation, after a power check.** Settle the λ-grid decision on training data and re-freeze. Run a simulated power analysis by subsampling 15 of the out-of-fold groups. State exactly which C1 and structure-head contrasts must pass for RQ1/RQ2 to count as supported. Then unblind once and report the Holm-adjusted test results alongside the controlled analyses (A to E).

2. **Make the probe-vs-prompt-only comparison fair before unblinding.** Give the structure head, B1 and B2 the same 15 per-type heads that the probes use, so features are compared with features. Widen the penalty grid for every configuration. Use block-specific penalties, so that 2,048 embedding columns cannot swamp 75 structured ones. Train heads on B2 embeddings that were cross-fitted within each fold. Add one non-linear tabular baseline on the structured features. Restate the abstract's AUROC contrast against the strongest comparator, which is the structure head, not B2.

3. **Show that the signal belongs to the generator, not to any large reader of the prompt.** Promote two controls from "exploratory" to primary: the cross-model control (Qwen's H and A features predicting Llama's labels and the reverse) and a no-model attention control (span length, span position, k and rule type). Report the seed-to-seed spread, collapse the 23 names to the 15 distinct feature sets, and only interpret differences above that noise floor. Qualify the per-rule claim for Qwen, where the within-type gain is nil.

#### Further requested changes (lower priority)

- Run the pre-specified sensitivity analyses now, out of fold: exclude truncated and looping responses, and drop or correct the uppercase artefact. This tests whether Qwen's Objective 2 advantage partly reflects predicting degenerate outputs.
- Report a split-half noise ceiling for each objective and generator.
- Report attention agreement between the generation and extraction backends, not only hidden-state cosine. Define the rule token span precisely.
- Use one Objective 2 scale throughout. Write out the binomial NLL and the soft-label AUROC/ECE as formulas. Unify rule-type names and add a notation table.
- Add the missing related work: assessors and routing, conditional probing, pre-generation self-prediction (Ji et al. 2024; Ashok and May 2025), Heo et al. on uncertainty in instruction following, and attention-feature detectors (Lookback Lens, Attention Tracker).
- Shorten the remaining-work section. Move the verifier audit to the dataset section or a section of its own.

#### Recommendation to the authors

The design and the incremental framing are the strengths of this paper, and the reviewers agree that a completed version could be a clear accept. The resubmission should lead with the test-set result, obtained after the per-type-head, tuning and cross-model controls have been run on training data and the plan re-frozen.


\newpage

## Appendix — Way 4: Evidence check

### Evidence check of `report/paper.tex` (draft of 16 Nov 2026)

How I checked: I read the paper, every `\input` table in `report/generated/`, `results/tables/*.csv`, `report/tables/*.csv`, `results/RESULTS.md`, `results/prespec/analysis_plan.md`, `results/prespec/FROZEN.json` and the code in `report/*.py` and `src/mife/probing/`. I recomputed the quantitative claims from the CSVs with pandas and also ran `report/check_numbers.py`, which reads only the CSVs and the paper. It reported "prose numbers checked: 466; unmatched: []" and all its assertions passed. Nothing under `data/` was read and no test-split data was loaded. Figures were not inspected pixel by pixel. Each figure claim is checked against the CSV its caption names.

Status labels: **S** = supported, **U** = unsupported, **C** = can't tell. When a claim appears more than once (abstract, introduction, body, conclusion), it is listed once at its main location and the repeats are noted.

Abbreviations: `cs` = `results/tables/cv_selected.csv`, `oc` = `report/tables/oof_contrasts.csv`, `om` = `report/tables/oof_metrics.csv`, `opt` = `report/tables/oof_per_type.csv`, `optd` = `report/tables/oof_per_type_diff.csv`.

#### A. Status, abstract and introduction

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

#### B. Related work

| # | Claim | Status | Evidence |
|---|---|---|---|
| 17 | Characterisations of IFEval, ManyIFEval, FollowBench, InFoBench, IFScale | C | External literature |
| 18 | Characterisations of probing work (control tasks, truthfulness, KEEN, etc.) | C | External literature |
| 19 | Characterisations of Heo et al. and Stolfo et al.; how this study differs | C | External literature (the "differs" half matches the design) |
| 20 | Characterisations of SAT Probe, lost-in-the-middle, PASTA, attention-as-explanation debate | C | External literature |

#### C. Setup and dataset

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

#### D. Predictors and protocol

| # | Claim | Status | Evidence |
|---|---|---|---|
| 42 | Component and block definitions (H1, H2, A1, B1, B2; Table 2) | S | `cv_identical_features.csv`; widths in `cs`; `src/mife/probing/features.py` |
| 43 | Structured features: 75 columns (O1) and 175 (O2); no exact rule, group or task IDs | S | `cs` n_features 75 / 175; analysis_plan §2 "Exact rule IDs are never model inputs" |
| 44 | 23 = 7 + 8 + 8; 15 distinct feature sets; duplicates listed in the appendix | S | `cv_identical_features.csv` (the 12 rows collapse 23 to 15 per objective) |
| 45 | O1 per-type heads for configurations with probes; shared head for B1, B2 and structure; O2 always shared | S | `cs` per_type_heads column; analysis_plan §2 |
| 46 | Four references (constant, type, type×k, structure head) | S | `cs` |
| 47 | Linear and MLP(128) heads × λ in {0.01, 0.1, 1}; sigmoid; binomial NLL / squared error; standardisation on fitting rows | S | analysis_plan §2; `src/mife/probing/heads.py` |
| 48 | Folds of 12, 12, 11, 11, 11 groups | S | `cv_folds.csv` |
| 49 | MLP epochs up to 30, chosen in CV; layer chosen on H1, pooling chosen on A1 | S | `cv_boundary_hits.csv` (epoch 30 edge); `cv_layer_selection.csv`; `cv_pooling_selection.csv` |
| 50 | B2: LoRA rank 16 with a linear head, grid of 2 LRs × {1, 3} epochs, inside each fold, 80 runs | S | `cv_b2_grid.csv`; `runtime.csv` "80 LoRA … runs"; `b2.py` |
| 51 | LoRA over full fine-tuning chosen by a rule fixed in advance on one fold | S | `b2_betty_check.csv` (fold-0 NLL 0.4141 vs 0.4173, LoRA chosen); analysis_plan §1 |
| 52 | Final models refitted on all 57 groups | S | `runtime.csv` refit_* rows |
| 53 | Loaders refuse test rows until unblinding; an automated test guards this | S | `src/mife/probing/data.py` (`test_unblinded`, guard); `tests/test_no_test_leakage.py` exists (not run) |
| 54 | B2 fine-tuning jobs received a train-only bundle | C | Stated in RESULTS.md §10, but the bundle itself is not checkable from sources |
| 55 | Disclosed exception: truncation share computed over all prompts before probing; unused | S | `open_issues.csv` "Test-set statistic seen before this phase" (disclosure; non-use cannot be proven) |
| 56 | OOF analysis: paired group bootstrap, 2,000 replicates; not part of the frozen plan | C | Intervals exist in `om`/`oc`; replicate count only in `report/analysis_oof.py` (not checked line by line). Absence from the plan is confirmed |

#### E. Results: violation rates

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

#### F. Results: cross-validation

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

#### G. Results: out-of-fold effect sizes

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

#### H. Results: where the signal helps

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

#### I. Design choices and contrasts

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

#### J. Verifier audit

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

#### K. Remaining work, limitations, conclusion, reproducibility

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

#### Tally

- Supported: 115
- Unsupported: 0
- Can't tell: 31

No claim is contradicted outright by a saved table. The weakest wordings are #69 ("B2 does not help for Qwen", where the CV loss is in fact 1.5% lower for Objective 1), #137 (layout and length concentration) and #118 (verifier failures "follow the rule as a person would read it"). Most can't-tell items are generation-phase provenance (prompt reproduction, split search, decoding, backend cosine) whose evidence sits outside the files I was allowed to read.


\newpage

## Appendix — Way 5: Related-work check

### Related-work check: five closest prior papers

Prompt: "With web search on, find the five closest prior papers to this one, with links, and say what is new in ours. Check that every paper you list actually exists."

Scope: I read the abstract, introduction, related work and contributions of `report/paper.tex`, and `report/refs.bib`. Searches used topic keywords and paper titles only. I confirmed that each listed paper exists by its arXiv API record, which gives the title, the full author list and the posting date. Venues are confirmed by the ACL Anthology, the ICLR or NeurIPS virtual-site pages, or the arXiv comment field, as noted.

#### The five closest papers

##### 1. Do LLMs "know" internally when they follow instructions?
- **Authors:** Juyeon Heo, Christina Heinze-Deml, Oussama Elachqar, Kwan Ho Ryan Chan, Shirley Ren, Udhay Nallasamy, Andy Miller, Jaya Narain
- **Venue/year:** ICLR 2025 (poster: https://iclr.cc/virtual/2025/poster/28257)
- **Link:** https://arxiv.org/abs/2410.14516
- **Already cited:** yes (`heo2025know`)
- **What it does:** It trains linear probes on input-side representations to find an "instruction-following dimension" that separates IFEval-style successes from failures, with AUROC around 0.7 to 0.8. The probe generalises to unseen tasks but not to unseen instruction types. Steering along the dimension raises compliance. A sensitivity analysis suggests the signal reflects how the prompt is phrased more than how hard the task is.
- **How ours differs:** Heo et al. study one instruction per prompt with a binary success label and no strong prompt-only competitor. Our prompts carry one to five rules, and each rule is its own target. Our labels are 20-sample violation probabilities. Our probes are compared against text encoders, one of them fine-tuned, that see the same prompt, the same rule text and the same supervision. Heo et al.'s own finding that the signal tracks phrasing is exactly why the incremental comparison (RQ3) matters.

##### 2. Do LLMs estimate uncertainty well in instruction-following?
- **Authors:** Juyeon Heo, Miao Xiong, Christina Heinze-Deml, Jaya Narain
- **Venue/year:** ICLR 2025 (poster: https://iclr.cc/virtual/2025/poster/30179)
- **Link:** https://arxiv.org/abs/2410.14582
- **Already cited:** **no**
- **What it does:** This is the first systematic evaluation of uncertainty estimation for instruction following. It builds a controlled benchmark that separates task quality from instruction compliance. It compares verbalised, logit-based and internal-state probe estimators. Internal-state probes beat the other estimators, but they fall to near-chance AUROC in the harder controlled setting.
- **How ours differs:** Their estimators mostly score a response after it has been generated. We predict before generation. We use multi-rule prompts with probabilistic labels, and our comparators are trained prompt-only predictors rather than the model's own confidence. Their near-chance result in the hard setting is a useful counterpoint to our training-set AUROCs of about 0.85.

##### 3. How LLMs Follow Instructions: Skillful Coordination, Not a Universal Mechanism
- **Authors:** Elisabetta Rocchetti, Alfio Ferrara
- **Venue/year:** arXiv preprint, April 2026 (2604.06015; I found no peer-reviewed venue)
- **Link:** https://arxiv.org/abs/2604.06015
- **Already cited:** **no**
- **What it does:** It probes attention and MLP streams across nine instruction-following tasks. Task-specific probes beat a general probe. Over the course of generation, the signal for constraint satisfaction is largely absent at pre-generation positions and appears only once generation starts, which the authors read as monitoring rather than planning. Causal ablations show sparse, asymmetric dependencies between tasks.
- **How ours differs:** It is the most direct challenge to our premise, and the draft should address it. Our training-set results (attention-probe AUROC of about 0.85, internal probes beating prompt-only comparators in three of four cells) show a usable pre-generation signal on ManyIFEval rules. Our design (one shared type-specific head per rule type, and contrasts against a fine-tuned text encoder) also speaks to their specialist-versus-general finding. The two accounts can be reconciled if much of the pre-generation signal is prompt-readable rule difficulty and not planning. Our RQ3 contrast tests exactly that.

##### 4. Language Models Can Predict Their Own Behavior
- **Authors:** Dhananjay Ashok, Jonathan May
- **Venue/year:** NeurIPS 2025 (per the arXiv comment on v2)
- **Link:** https://arxiv.org/abs/2502.13329
- **Already cited:** **no**
- **What it does:** It trains linear probes on the internal states of input tokens to forecast behaviour over the whole output, such as abstaining or complying with a request and alignment-relevant behaviours. It calibrates the probes with conformal prediction and uses confident predictions to stop computation early.
- **How ours differs:** The framing is the same (predict before generating, for routing or early warning), but the targets are different. Ashok and May predict coarse behaviours of the whole output. We predict compliance with each of several explicit, verifier-checked formatting and content rules. We also benchmark against prompt-only models that receive the same supervision, a comparison their setup does not centre on.

##### 5. Attention Satisfies: A Constraint-Satisfaction Lens on Factual Errors of Language Models
- **Authors:** Mert Yuksekgonul, Varun Chandrasekaran, Erik Jones, Suriya Gunasekar, Ranjita Naik, Hamid Palangi, Ece Kamar, Besmira Nushi
- **Venue/year:** ICLR 2024
- **Link:** https://arxiv.org/abs/2309.15098
- **Already cited:** yes (`yuksekgonul2024attention`)
- **What it does:** It treats factual queries as sets of constraints and shows that attention to constraint tokens predicts whether the model satisfies them. SAT Probe, a probe on that attention, predicts factual errors about as well as the model's own confidence, on single- and multi-constraint queries.
- **How ours differs:** Our attention probe carries the SAT Probe idea from entity constraints in factual queries to instruction-rule spans in generation prompts. We summarise attention per layer and head as both a sum and a token-normalised mean over each exact rule span. We also find that it is the best single per-rule predictor. SAT Probe is the methodological ancestor of our best-performing component, and the draft could say so more prominently.

#### Also close (not in the top five)
- Harada et al., *When Instructions Multiply* (Findings of EMNLP 2025; https://aclanthology.org/2025.findings-emnlp.896/), already cited. It is our data source. Its logistic regression on instruction count is the aggregate, structure-only analogue of our structure-only reference.
- Stolfo et al., *Improving Instruction-Following in Language Models through Activation Steering* (ICLR 2025; https://arxiv.org/abs/2410.12877), already cited. It intervenes on activations rather than predicting from them.
- Gottesman and Geva, *Estimating Knowledge in Large Language Models Without Generating a Single Token* (EMNLP 2024; https://aclanthology.org/2024.emnlp-main.232), already cited. It is pre-generation probing, but for factual knowledge.
- Jaroslawicz et al., *How Many Instructions Can LLMs Follow at Once?* (https://arxiv.org/abs/2507.11538), already cited. It is a benchmark only.

#### What is new in this paper
1. It predicts the violation probability of each rule and the expected fraction of rules violated, before generation, in prompts with up to five simultaneous verifiable rules. Labels are probabilities from 20 sampled responses, not single binary outcomes.
2. It tests whether the signal is incremental. Hidden-state and rule-span-attention probes are compared against strong prompt-only predictors that receive the same text and supervision, one of them fine-tuned. The design balances rule subsets, rule orders and tasks, and the evaluation plan is pre-registered with test groups held out by group.
3. It is the first evidence on ManyIFEval-style multi-rule prompts that pre-generation attention to rule spans is the strongest single per-rule predictor. This bears on a live disagreement (Rocchetti and Ferrara, 2026) about whether compliance is decided before generation at all.

#### Recommended additions to Related work
Cite and discuss Heo, Xiong, Heinze-Deml and Narain (ICLR 2025), Rocchetti and Ferrara (2026), and Ashok and May (NeurIPS 2025). Rocchetti and Ferrara matters most because its "monitoring, not planning" conclusion directly contests the paper's premise.

#### Bibliography check (`refs.bib`)
No entry has a wrong title, a wrong year or a wrong page range among those I checked. The following entries are outdated or incomplete:
- `qin2024infobench`: cited as an arXiv preprint, but it is published in Findings of ACL 2024, pp. 13025–13048 (https://aclanthology.org/2024.findings-acl.772).
- `jaroslawicz2025many`: cited as arXiv only. The paper is listed on the NeurIPS 2025 virtual site (https://neurips.cc/virtual/2025/122417), but I could not confirm whether it is a main-track or workshop paper. Check before updating.
- `harada2025instructions`: correct (Findings of EMNLP 2025, pp. 16506–16526, confirmed in the ACL Anthology bib). The full author list is available if "and others" is not wanted: Harada, Yamazaki, Taniguchi, Marrese-Taylor, Kojima, Iwasawa, Matsuo.
- `kossen2024sep`: the arXiv record lists no journal reference. I did not find a confirmed peer-reviewed venue, so the arXiv citation is acceptable as it stands.
- Entries I confirmed correct: `zhou2023ifeval` (arXiv 2311.07911), `jiang2024followbench` (ACL 2024 main), `heo2025know` (ICLR 2025), `stolfo2025steering` (ICLR 2025, authors match), `yuksekgonul2024attention` (ICLR 2024), `gottesman2024estimating` (EMNLP 2024, pp. 3994–4019), `marks2024geometry` (COLM 2024), `yang2021coa` (Technometrics 63(2):212–224).
- I did not check the remaining classic entries (Alain and Bengio, Belinkov, Hewitt and Liang, Jain and Wallace, Holm, Efron, and so on) online. They look standard.


\newpage

## Appendix — Stanford Agentic Reviewer (paperreview.ai, ICLR)

**[TO FILL: paste the Stanford review and its score here.]**
