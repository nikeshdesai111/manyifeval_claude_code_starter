# Meta-review: "Predicting Rule Violations Before Generation: Hidden-State and Attention Probes versus Prompt-Only Baselines on a Controlled Multi-Instruction Benchmark"

**Decision: Reject** (encourage resubmission once the confirmatory evaluation and the controls below are done)

Reviewer scores: 4 (experiments/soundness), 4 (novelty/related work), 6 (clarity/presentation).

## Summary of the submission

The paper asks whether a generator's internal state at the last prompt token predicts which rules in a multi-rule prompt it will break, and whether it does so better than predictors that see only the prompt. It builds 53,136 prompts from all 216 ManyIFEval tasks in a balanced design (72 groups, nested rule subsets for k = 1 to 5, orders from a strength-two orthogonal array). It labels them with 20 sampled responses per prompt from Llama 3.1 8B and Qwen3-8B, checked by the official verifiers. It then compares 23 feature configurations per objective (hidden state H1, hidden state plus rule embedding H2, attention to the rule span A1, frozen and fine-tuned BGE-M3 baselines B1/B2, and their combinations) against rate references and a structure-only head. The 15 test groups are deliberately unread. Every result comes from grouped cross-validation or out-of-fold predictions on the 57 training groups. The headline is that A1 is the best single per-rule predictor (OOF AUROC 0.852 / 0.844), and that internal-state probes beat the strongest comparator without internal access by 8 to 14% in three of four generator-objective cells. The paper also reports a verifier artefact: 43.1% of Qwen's "all uppercase" violations fail only `langdetect`'s English check.

## Consensus strengths

All three reviewers agree on these, and I do too.

- **The right question.** The paper asks whether internal state adds anything beyond a strong prompt-only predictor, not whether it beats chance. The structure-only reference already removes about 20% of the constant-rate loss, and showing this raises the bar for the probing literature (R1, R2).
- **Careful design and discipline.** Soft labels from repeated sampling, an orthogonal-array order design, group-level splits, a group-level paired bootstrap, a hashed pre-specified plan and a loader that refuses test rows. This is better than most work in this area.
- **Candour.** The paper reports its negative and qualifying results openly: H1 does not beat the structure head in Objective 1, B2 ties the probes for Llama Objective 2, and H1+A1 is worse than A1. It also discloses the narrow λ grid, the duplicate configurations and the one statistic that touched the test set. R3 notes that the status of every claim is stated wherever it matters.
- **A useful by-product.** The verifier audit is concrete, checkable and relevant to everyone who uses IFEval-family verifiers.

## Why the decision is reject

The reviewers converge, independently, on one assessment. As submitted, this is a stage-1 registered report: a sound design and a frozen plan, but no confirmatory result. The training-side evidence that stands in for a result is also confounded in ways that bear directly on the headline claim. In my judgement no single issue is fatal. Together they mean the paper's central claim, that the generator's internals predict violations better than a prompt-only predictor, has not yet been shown.

1. **No held-out evidence (R1 W1, R2, R3 D15).** Head type, penalty, epochs, layer, attention pooling and the "best probe-only" label were all chosen on the same five folds that then report effect sizes. The authors acknowledge the optimism. Two protocol decisions (widening the λ grid, a verifier-corrected sensitivity analysis) are still open. R1 also estimates that the planned 15-group test may be underpowered for its own success criterion: interval half-widths of about 0.03 NLL against effects of 0.024 to 0.039, before Holm adjustment. In addition, "C1 contrasts against B2 that favour the probe" does not say which or how many contrasts must pass.

2. **The headline comparison mixes model class with features (R1 W2, R2, R3 A1).** In Objective 1, every configuration containing H1, H2 or A1 uses 15 per-type heads, while the structure head, B1 and B2 use one shared head. The paper discloses this confound only for the C2/C3 "add a probe" contrasts. It applies equally to the comparisons in the abstract and in Table 3 (A1 vs B2, A1 vs structure head).

3. **The comparators look under-tuned (R1 W3, R2).** I checked the reviewers' figures against the generated tables, and they hold.
   - B1, whose inputs are a strict superset of the structure head's, has a higher loss than the structure head in three of four cells (for example 0.3868 vs 0.3779 for Llama Objective 1).
   - The structure head selects the smallest λ in all four cells, and its loss one step inwards is 7.9 to 11.1% higher. That is the same size as the reported probe margins.
   - The "strong" fine-tuned baseline B2 has a lower AUROC than the structure head for both generators (0.793 vs 0.809; 0.774 vs 0.817). The abstract nonetheless contrasts A1 with B2's 0.79/0.77.
   - For Qwen Objective 2, adding B2 embeddings to H1 raises the loss from 1.130 to 1.375. That pattern is consistent with heads being fitted on in-sample fine-tuned embeddings.

4. **Nothing yet shows that the signal is the generator's *own* state (R1 W4, R2).** Summed attention over a span mechanically encodes span length, position, k and plausibly rule identity. The comparison also pits a 0.6B text encoder against an 8B model's representations. The cross-model control (predicting Llama's labels from Qwen's features and the reverse) and a "no-model attention" control are both listed only as exploratory. For Qwen, within-type AUROC is 0.56 for A1 against 0.58 for B2, so the "for both generators" phrasing overstates the per-rule result.

5. **Noise floor and presentation (R1 W6, R3 A2 to A4, R2).** Only 15 of the 23 named configurations have distinct inputs. Identical-input twins differ by up to 0.0036 NLL in Objective 1 (Qwen H2+A1 0.2386 vs H1+H2+A1 0.2358) and by up to 3% in Objective 2 (Llama B1+H1 1.642 vs B1+H1+H2 1.691). Yet the paper names a "best probe-only" and a bolded optimum among such twins. Objective 2 switches between three loss scales (MSE ×10⁻², raw MSE, RMSE). The related-work section omits the assessor/routing literature (FrugalGPT, RouteLLM, small assessors) that defines the prompt-only baseline, and the conditional-probing literature (Hewitt et al. 2021) that formalises RQ3. These are fixable without new experiments, but they currently make the main claim easy to misread.

R3's score of 6 is explicitly "on clarity alone" and agrees that the manuscript is incomplete. I therefore read the three reviews as consistent with rejection rather than split.

## The three most important fixes

1. **Run and report the confirmatory test evaluation, after a power check.** Settle the λ-grid decision on training data and re-freeze. Run a simulated power analysis by subsampling 15 of the out-of-fold groups. State exactly which C1 and structure-head contrasts must pass for RQ1/RQ2 to count as supported. Then unblind once and report the Holm-adjusted test results alongside the controlled analyses (A to E).

2. **Make the probe-vs-prompt-only comparison fair before unblinding.** Give the structure head, B1 and B2 the same 15 per-type heads that the probes use, so features are compared with features. Widen the penalty grid for every configuration. Use block-specific penalties, so that 2,048 embedding columns cannot swamp 75 structured ones. Train heads on B2 embeddings that were cross-fitted within each fold. Add one non-linear tabular baseline on the structured features. Restate the abstract's AUROC contrast against the strongest comparator, which is the structure head, not B2.

3. **Show that the signal belongs to the generator, not to any large reader of the prompt.** Promote two controls from "exploratory" to primary: the cross-model control (Qwen's H and A features predicting Llama's labels and the reverse) and a no-model attention control (span length, span position, k and rule type). Report the seed-to-seed spread, collapse the 23 names to the 15 distinct feature sets, and only interpret differences above that noise floor. Qualify the per-rule claim for Qwen, where the within-type gain is nil.

## Further requested changes (lower priority)

- Run the pre-specified sensitivity analyses now, out of fold: exclude truncated and looping responses, and drop or correct the uppercase artefact. This tests whether Qwen's Objective 2 advantage partly reflects predicting degenerate outputs.
- Report a split-half noise ceiling for each objective and generator.
- Report attention agreement between the generation and extraction backends, not only hidden-state cosine. Define the rule token span precisely.
- Use one Objective 2 scale throughout. Write out the binomial NLL and the soft-label AUROC/ECE as formulas. Unify rule-type names and add a notation table.
- Add the missing related work: assessors and routing, conditional probing, pre-generation self-prediction (Ji et al. 2024; Ashok and May 2025), Heo et al. on uncertainty in instruction following, and attention-feature detectors (Lookback Lens, Attention Tracker).
- Shorten the remaining-work section. Move the verifier audit to the dataset section or a section of its own.

## Recommendation to the authors

The design and the incremental framing are the strengths of this paper, and the reviewers agree that a completed version could be a clear accept. The resubmission should lead with the test-set result, obtained after the per-type-head, tuning and cross-model controls have been run on training data and the plan re-frozen.
