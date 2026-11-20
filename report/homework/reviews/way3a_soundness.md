# Review (experiments and soundness): "Predicting Rule Violations Before Generation"

**Score: 4 / 10** (reject in current form; the design is strong enough that a completed version with the controls below could be a clear accept)

## Summary

The paper asks whether a generator's pre-generation internals (final-prompt-token hidden state, attention to each rule's span) predict rule violations better than prompt-only predictors. It builds a balanced 53,136-prompt design from ManyIFEval, labels it with 20 samples per prompt from two 8B models, and compares 23 feature configurations per objective against structure-only references. The held-out test groups are deliberately unread, so every result is grouped cross-validation on the 57 training groups. The headline: the attention probe A1 is the best single per-rule predictor (out-of-fold AUROC 0.85/0.84), and internal-state probes beat the strongest prompt-only comparator by 8 to 14% in three of four cells.

## Strengths

1. **Unusually careful design.** Soft labels from repeated sampling, a strength-two order design, nested chains, group-level splitting, and a group-level paired bootstrap are all the right choices. Groups, not rows, are treated as the unit of independence throughout.
2. **The right question.** The paper compares against prompt-only predictors and rate references rather than chance, and shows that type-by-k rates and a structure head already remove 12 to 21% of the constant-rate loss.
3. **Pre-specification and disclosure.** A hashed plan, a loader that refuses test rows, and candid disclosure of the one test-touching statistic, the narrow penalty grid, the duplicate configurations and the head-structure confound.
4. **Honest negative and qualifying results.** H1 does not beat the structure head in Objective 1; B2 matches the probes for Llama Objective 2; within-type AUROC for Qwen is 0.56 (A1) against 0.58 (B2); H1+A1 is worse than A1.
5. **The verifier audit** (43.1% of Qwen's all-uppercase violations fail only the language-detection step) is concrete, checkable and useful to anyone using IFEval-family verifiers.

## Weaknesses

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

## Do the conclusions follow?

Mostly yes as worded, because the wording is hedged ("so far", "qualified yes"). The claim that survives scrutiny is narrow: for Llama, per-rule, A1 and H2 carry within-type information beyond structure (within-type AUROC 0.73 vs 0.63). For Qwen the within-type gain is 0.02 and smaller than B2's, so "for both generators" overstates the per-rule result; much of the pooled AUROC is knowing which rule is asked about. The statement that probes "improve on a text encoder fine-tuned for the same task" does not yet follow, given W2 and W3.

## Questions

1. What are the losses of the structure head, B1 and B2 with 15 per-type heads?
2. What happens to all margins with the widened penalty grid, block-specific penalties, and cross-fitted B2 embeddings?
3. How well do Qwen's features predict Llama's labels, and the reverse?
4. What is the split-half noise ceiling for each objective and generator?
5. What is the simulated power of the 15-group test for each C1 contrast, using subsamples of the out-of-fold predictions?
6. What is the spread over head seeds and over repeated fold partitions?
7. Does A1 still beat a head given only span length, span position, k and rule type (a no-model attention control)?
8. Do the Qwen Objective 2 results survive, out of fold, when truncated and looping responses and the uppercase artefact are excluded?
9. Was any verifier other than "all uppercase" audited by cause of failure?

## Recommendation

Reject in the present form: the experiments are well designed but unfinished, and the two comparisons that carry the claim (probe vs fine-tuned text baseline, probe vs structure head) are confounded by head structure and by under-tuned comparators. Running the controls in questions 1 to 3 on training data, freezing again, and then reporting the test evaluation would make this a much stronger paper.
