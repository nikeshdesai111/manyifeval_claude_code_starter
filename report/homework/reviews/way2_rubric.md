# ICLR review form: "Predicting Rule Violations Before Generation"

Reviewed from `paper.tex`, the 18 generated tables and the 6 figures only. The bibliography and code were not inspected.

## Paper Summary

The paper asks whether a model's violations of prompt instructions can be predicted before generation, and whether the generator's own internals (final-prompt-token hidden state, attention from that token to each rule's span) predict them better than the prompt text does. It builds a balanced dataset from ManyIFEval: 216 tasks in 72 groups of three, a pool of six sampled rules per group, nested rule subsets for k = 1 to 5 and strength-two orderings, giving 53,136 prompts. Llama 3.1 8B and Qwen3-8B each produce 20 responses per prompt, checked with the official verifiers to give per-rule (Objective 1) and per-prompt (Objective 2) violation probabilities. Twenty-three combinations of three probe components (H1, H2, A1) and two prompt-only baselines (frozen and LoRA-tuned BGE-M3 plus structured features) are compared with rate references and a "structure head".

The analysis plan is frozen and the 15 test groups are unread, so every result is grouped 5-fold cross-validation on the 57 training groups. There, the attention probe is the best single per-rule predictor (out-of-fold AUROC 0.852 and 0.844; NLL 0.039 and 0.024 below the fine-tuned baseline). Probe-only configurations beat the strongest prompt-only comparator by 8 to 14% in three of four cells, and the fine-tuned baseline ties them on Llama's per-prompt objective. The paper also reports that the official "all uppercase" verifier fails uppercase English text at its language-detection step (43.1% of Qwen's violations of that rule).

## Presentation: 3

## Informativeness: 2

## Soundness: 2

## Critical Strengths

1. **The right question, posed against the right kind of baseline.** The paper states that a hidden-state probe may succeed only because the prompt is encoded in it, and it reports every result against type-rate references and a structure head. The finding that type-by-k rates and structured features already remove 12 to 21% of the constant-rate loss is useful on its own.

2. **Careful design and leakage control.** Splitting by group, the nested chains, the strength-two orderings, the witness-response feasibility check and the byte-for-byte reproduction of the 2,160 released prompts are all sound. I recomputed the counts in the dataset table and the uppercase audit (for example 12,554 / 205,200 = 6.1%, and 171 rows x 3 tasks x 20 responses x 20 groups = 205,200) and they agree.

3. **Unusual candour.** The draft discloses the one look at test-inclusive data, the edge-of-grid selections, the 15 distinct feature sets among 23 names, the shared-versus-per-type head confound and the post-freeze edit. The text matches the tables in every place I checked.

4. **Negative and mixed results are reported.** H1 is indistinguishable from the structure head in Objective 1, H1+A1 is 8% worse than A1 alone, the fine-tuned encoder wins on letter count, and within-type AUROC for Qwen is 0.56 against 0.54.

5. **The verifier audit is concrete and reusable.** It is supported by counts, the detected language, the lower-casing check and the interaction with the letter-count rule, and it matters to anyone using IFEval-derived labels.

## Critical Weaknesses

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

## Questions For Authors

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

## Flag For Ethics Review

No

## Rating: 2

Below the acceptance threshold in its current form. The design and reporting are of high quality, but the confirmatory evaluation has not been run and the training-set evidence depends on an under-tuned comparator and uncontrolled confounds (head structure, seed noise, prompt information carried by attention). Fixing weaknesses 2 to 5 before unblinding and then reporting the test results would make this a strong submission.

## Confidence: 2

I read the paper, all tables and all figures closely and recomputed several numbers. I did not see the code, the frozen plan or the bibliography, and I cannot verify the related-work claims.
