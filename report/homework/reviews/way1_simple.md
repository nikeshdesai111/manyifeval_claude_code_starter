# ICLR-style review

**Paper:** Predicting Rule Violations Before Generation: Hidden-State and Attention Probes versus Prompt-Only Baselines on a Controlled Multi-Instruction Benchmark

## Summary

The paper asks whether an LLM's violations of explicit prompt constraints can be predicted before generation, and whether the generator's internal state at the final prompt token predicts them better than predictors that only read the prompt. The authors build a balanced dataset from ManyIFEval (216 tasks, 72 groups of three tasks sharing six sampled rules, nested rule subsets for k = 1..5, orders from a strength-two component orthogonal array; 53,136 prompts), sample 20 responses per prompt from Llama 3.1 8B and Qwen3-8B, and label each (prompt, rule) with an empirical violation probability from the official verifiers. They compare 23 feature configurations per objective and generator: hidden-state probes (H1), hidden state plus a rule-text embedding (H2), attention from the final token to the rule span (A1), a frozen BGE-M3 baseline with structured features (B1), a LoRA-fine-tuned version (B2), and combinations, against rate-based references and a "structure head".

The analysis plan is frozen and the 15 test groups are unread, so all reported results are grouped 5-fold cross-validation on the 57 training groups. In those results A1 is the best single per-rule predictor for both generators (out-of-fold AUROC 0.85 / 0.84), internal-state probes beat the best prompt-only comparator by 8 to 14% in loss in three of four generator-objective cells, and B2 matches them for per-prompt prediction on Llama. The paper also reports that the official "all uppercase" verifier fails correct uppercase English through its language-detection step, which accounts for 43% of Qwen's violations of that rule.

## Strengths

1. **The right question, posed incrementally.** Most probing papers compare against chance or a weak baseline. This one makes the comparator the point, and it reports that rule type and k alone remove 12 to 15% of the loss and a 75-feature structure head about 20%. That framing is the paper's main intellectual contribution.
2. **Careful experimental hygiene.** Grouped splits, probabilistic labels from repeated sampling, a hashed pre-specification, a guarded test set, a disclosed exception (token-limit share computed over all prompts), and a paired cluster bootstrap. The paper is unusually candid about what is descriptive and what is confirmatory.
3. **Honest reporting of unfavourable results.** H1 is no better than the structure head in Objective 1; H1+A1 is 8% worse than A1 alone; B2 wins on letter count; within-type AUROC for Qwen is 0.56 for A1 against 0.58 for B2; 8 of 23 configurations are duplicates. None of this is hidden.
4. **The verifier audit is concrete and useful.** The langdetect failure on uppercase text is quantified (12,554 of 29,097 Qwen violations), diagnosed (92.5% labelled German, most flip to English when lower-cased) and tied to a co-occurring rule. Anyone using IFEval-derived verifiers should know this.
5. **The design supports analyses that most datasets cannot.** Nested chains and balanced orders allow matched-pair tests of position, count, composition and task effects.

## Weaknesses

### W1. The paper has no held-out results, and says so

The status box, abstract and conclusion all state that the test groups are unread. Every number comes from the folds on which head type, penalty, epochs, layer and pooling were selected. The controlled analyses (A to E), which are the main justification for the elaborate design, are also not run. As submitted this is a registered-report stage-1 manuscript, which ICLR's main track does not have a category for. The contribution list reads "a pre-specified comparison" and "training-set evidence"; neither is a finding a reader can build on yet.

### W2. Per-type heads are confounded with internal features in the headline comparison, not only in C2/C3

In Objective 1 every configuration containing H1, H2 or A1 uses 15 per-type heads, while B1, B2 and the structure head use one shared head. The paper flags this for the "baseline plus probe" contrasts (C2, C3), but the same confound applies to the C1 and reference contrasts that carry the abstract's claim (A1 against B2, A1 against the structure head). The control is listed as exploratory extension (iii). It should be a primary analysis: structure head, B1 and B2 with per-type heads, and ideally a per-type head on the rule embedding alone (ER) and on ER + S1. The H1 to H2 result makes this pressing: adding a frozen rule-text embedding to the hidden state accounts for a 0.037 NLL improvement on Llama, almost the whole gap between H1 and A1, and that embedding is prompt-only information.

### W3. A1 may be a fingerprint of the rule text rather than a signal about the model's state

A1 is 1,024 to 1,152 attention values to the target rule's span, sum-pooled. Sum pooling scales with span length, and head-wise attention to a span depends on which tokens are in it and where it sits in the prompt. The feature vector can therefore encode rule identity, parameters and position, all of which are readable from the prompt. The letter-count result points this way: the paper says the difficulty of that rule depends on which letter is named, the structured features lack the letter, and A1 reaches within-type AUROC 0.92 on Llama and 0.88 on Qwen, close to B2's 0.93 and 0.90. A1 can only do that if it identifies the letter. The paper cites Hewitt and Liang's control-task caveat but runs no control. Three are cheap and decisive:
- **Cross-generator features.** Predict Qwen's labels from Llama's attention and the reverse. If accuracy is largely retained, A1 is encoding the prompt, not the generating model's disposition. This is exploratory item (iv) and deserves promotion.
- **A non-generating model's attention**, for instance a third LLM of similar size, or the same architecture at random initialisation.
- **Span-length and position controls**: add span length in tokens and absolute span position to the structure head, and report A1 with mean pooling restricted to rules of equal span length.

### W4. The prompt-only baselines are weaker than "strong" suggests

- B1 and B2 both contain the structured features S1/S2, yet B1 is worse than the structure head in three of four cells and B2 is worse in both Qwen cells (Objective 2: 1.379 against 1.275). A model that is beaten by a strict subset of its own inputs is under-regularised or badly fused; the paper's own explanation for H1+A1 (one shared penalty lets a wide block swamp a narrow one) applies here too, with 2,048 embedding dimensions against 75 structured ones.
- The structure head selects the smallest penalty in all four cells, with loss 7.9 to 11.1% higher one step inwards. The authors raise this themselves. The grid should be widened before unblinding; reporting it as a limitation would leave the central comparator known to be under-tuned.
- The B2 fine-tuning grid has only two learning rates and two epoch counts, so every selection is at an edge and the grid cannot show an interior optimum. The selected setting also flips between cells (1 epoch at the lower rate for Objective 1, 3 epochs at the higher rate for Llama Objective 2). A wider grid is needed before B2 can be called a tuned baseline.
- BGE-M3 is a retrieval encoder. A more natural prompt-only comparator is a fine-tuned LM of comparable size reading the prompt, or a different LLM's hidden states (which also serves W3).
- The abstract quotes AUROC 0.85 / 0.84 against 0.79 / 0.77 for B2. The structure head has 0.809 / 0.817. The abstract's comparison is against the weaker of the two prompt-only numbers.

### W5. Seed noise is the same size as several reported effects, and the paper has the data to show it

Configurations with identical inputs differ only by head seed. For Llama Objective 2, B1+H1, B1+H2 and B1+H1+H2 have identical inputs and CV losses of 1.642, 1.669 and 1.691, a 3% spread. For Qwen Objective 1, H2+A1 and H1+H2+A1 give 0.2386 and 0.2358, and the second is then named "best probe-only" over A1 at 0.2361. Effects of this size are interpreted elsewhere: the 1.4% gain of B2+probes on Qwen, the 1.7% probe advantage on Llama Objective 2, the 2.3 to 4.3% complementarity gain in Objective 2. The duplicates should be reported as an explicit noise floor, every MLP configuration should be run with several seeds, and "best of seven probe-only configurations" should not be selected and then evaluated on the same folds without a correction.

### W6. The pooled metrics are dominated by k = 4 and 5 and by near-replicate prompts

Each task contributes 1,026 prompt-rule rows, of which 600 come from k = 5 and 288 from k = 4 (86% together), because the number of orders grows as 1, 2, 6, 12, 20. The paper's own descriptive result is that order barely matters (17.7% first against 18.9% last for Llama, flat for Qwen), so the 20 orders of a subset are close to repeated measurements of the same quantity. Headline NLL, AUROC and decile rates are therefore mostly statements about k = 4 and 5, and the statement that A1's advantage "grows with k" is partly a statement about where the rows are. Report k-balanced and subset-balanced versions (one row per task, subset and rule, averaging over orders) alongside the pooled ones.

### W7. Hidden-state probes are explored too narrowly to support the negative claim

Only three depths are tried (50, 75, 100%), only the final token, and the shallowest option wins in three of four cells with the last layer worst everywhere. The optimum may well be earlier than 50%. With per-type heads on 4,096 dimensions and 13 to 32 training groups per type, each head sees a few dozen distinct rule instances; dimensionality reduction or a shared head with a type embedding would be fairer. The conclusion that "a hidden-state probe that is not told the exact rule does no better than structured prompt features" should be limited to this configuration.

### W8. No noise ceiling

With n = 20, part of the binomial NLL and of the MSE against q-hat is irreducible sampling noise. Without an estimate of it (for instance from split-half labels, 10 against 10 responses), a reader cannot tell whether a 10% loss reduction closes a tenth or a half of the explainable gap. The same split-half computation gives the reliability of q-hat, which bounds the Spearman values of 0.50 to 0.55.

### W9. The confirmatory plan has soft spots

- "Support requires C1 contrasts against B2 that favour the probe ... together with the corresponding structure-head contrasts" does not say whether all, any or a named contrast must pass. As written, the decision rule can be read after the fact.
- Holm is applied within family, generator and objective. RQ1 spans two generators and several probes; there is no stated rule for combining cells.
- Fifteen clusters is few for a percentile bootstrap, and several rule types appear in only 3 or 4 test groups. A power calculation from the out-of-fold effect sizes and between-group variance would say whether the planned test can detect the effects seen in training.
- 50 of 90 test rule instances also occur in training. The paper is clear that this is not a test of unseen rules, but it means a predictor that memorises exact-rule difficulty is rewarded on the test set. A split of test rows into seen and unseen exact rules should be pre-specified.

### W10. Labels include a known artefact that a predictor can learn

The uppercase audit shows that 6.1% of Qwen's checks of that rule fail only on language detection, far more often when a letter-count rule co-occurs. The structure head sees rule co-occurrence, and attention features may too, so part of what is being predicted is langdetect behaviour. The planned sensitivity analysis drops the rule type; the corrected-label analysis is only "under consideration". It should be added, and the paper should check whether the letter-count verifier and the 2.5% truncated and 2.1% looping Qwen responses similarly reward predicting verifier or decoding failures rather than non-compliance.

### W11. Motivation and practical value are asserted, not shown

All 15 rule types are checked by cheap deterministic programs. Generating and re-checking is therefore a strong and simple alternative to pre-generation prediction, and the paper does not compare against it or show a downstream use (routing, reordering, resampling) of the predicted risks. A top-decile violation rate of 71% is promising, but no decision-level result is given. Within-type AUROC averages 0.73 on Llama and 0.56 on Qwen, so for Qwen the predictor mostly tells the user which rule types are hard, which a lookup table does.

### W12. Scope and smaller points

- Only two 8B models are studied, with English prompts, at most five rules, and default decoding that differs between the models.
- Internal features come from a different backend (Transformers on Apple silicon) from generation (vLLM on GPU). The 0.998 cosine check is for hidden states in a pilot; nothing is reported for attention, which is the winning feature.
- In Objective 2, "H1" includes the rule types and order (M2) and "A1" includes k, so "probe-only" configurations are not internal-only. The naming should say so.
- The task-family taxonomy is the authors' own and is an input feature; inter-annotator agreement or a sensitivity check without it would help.
- Novelty relative to Heo et al. (2025) and the SAT probe is mainly the multi-rule setting and the baseline comparison. That is a reasonable increment, but it depends entirely on the baselines and controls being convincing (W2 to W4).
- The manuscript is formatted as a project report with a status box and a "remaining work" section, not as a conference paper.

## Questions for the authors

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

## Assessment

- **Soundness:** 2 (fair). Careful protocol, but the headline comparison is confounded by head structure, the baselines are under-tuned by the paper's own evidence, and there is no control showing A1 reflects the generator rather than the prompt.
- **Presentation:** 3 (good). Clear, precise and candid; not in conference form.
- **Contribution:** 2 (fair). A well-designed dataset and a useful verifier finding; the central claim is not yet tested.

**Rating: 3 (reject, not good enough).** The work is not finished: there are no held-out results and none of the controlled analyses. The preliminary evidence is also less clean than the abstract suggests, for the reasons in W2 to W6. I would expect a version with the confirmatory evaluation, per-type-head baselines, a cross-model attention control, a seed noise floor and a wider penalty grid to be a solid accept-range paper whichever way the result falls, because the question is well posed and the design is strong.

**Confidence: 4.** I read the main text, all generated tables and the figures closely, and checked the quoted numbers against the tables. I did not have access to code, the pre-specification file or the bibliography.
