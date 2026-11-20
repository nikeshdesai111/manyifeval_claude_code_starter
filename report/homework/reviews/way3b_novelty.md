# Review (novelty and related work focus)

**Paper:** Predicting Rule Violations Before Generation: Hidden-State and Attention Probes versus Prompt-Only Baselines on a Controlled Multi-Instruction Benchmark

**Score: 4 / 10** (below the acceptance threshold in its current form; the main reason is that it has no held-out results, the second is that the central novelty claim is not yet isolated)

**Confidence:** 4 / 5. Citations I suggest below are from memory; the authors should verify titles and venues before citing.

## Summary

The paper asks whether an LLM's violation of each rule in a multi-rule prompt can be predicted before generation, and whether the generator's own internal state (final-prompt-token hidden state, attention from that token to each rule span) predicts it better than predictors that see only the prompt. It builds a balanced design on ManyIFEval (53,136 prompts, 2 generators, 20 samples per prompt, soft labels), compares 23 feature configurations per objective against structure-only references and a frozen and fine-tuned BGE-M3 baseline, and freezes an analysis plan. Only grouped cross-validation results on training groups are reported; the test set is unread. On those, the attention probe is the best single per-rule predictor (OOF AUROC 0.85/0.84 vs 0.79/0.77 for the fine-tuned text baseline).

## What is new relative to prior work

1. **Per-rule, multi-rule, pre-generation prediction.** Heo et al. (2025) probe single-instruction IFEval prompts with a binary outcome. Here each of up to five co-occurring rules is a separate target. This is a real extension, and the per-type result (Figure 5) and the growth of the gain with k (0.332 vs 0.341 at k=1, 0.338 vs 0.387 at k=5 for Llama) are the most interesting new observations.
2. **Probability labels from 20 samples** instead of one greedy outcome. Useful, though not new in kind (semantic-entropy probes and P(IK) already train on sampled-outcome targets).
3. **The incremental question with a supervised prompt-only comparator and a structure-only reference.** This is the paper's main conceptual contribution, and the finding that "known structure already explains a lot" (structure head removes about 20% of the constant-rate loss; plain H1 does not beat it) is a valuable negative result for the probing literature.
4. **Design-of-experiments construction** (nested chains, strength-two component orthogonal arrays for order). I am not aware of this being used in instruction-following evaluation; it is new as tooling, but its payoff (controlled analyses A to E) is not yet delivered.
5. **The langdetect artefact** in the "all uppercase" verifier (43% of Qwen's violations of that rule). A concrete, reusable finding for anyone using IFEval or ManyIFEval.

Items 1 and 3 are the novelty that would carry an ICLR paper. Methodologically the probes themselves are not new: A1 is SAT Probe applied to instruction spans (the paper says so), H1 is a standard last-token probe.

## Is the positioning accurate?

Mostly honest, and unusually careful about what is not claimed. Four places where it overstates or under-specifies:

- **"Strong" prompt-only baselines.** The fine-tuned baseline B2 has lower OOF AUROC than the structure head on both generators (0.793 vs 0.809; 0.774 vs 0.817), and B1 is worse than the structure head in three of four cells. A baseline that loses to 75 hand-built columns is not a strong text baseline; it suggests an embedding-plus-shared-penalty fusion problem. The headline "8 to 14% over the strongest comparator without internal access" is therefore a claim against a comparator that is probably under-tuned (the paper itself notes the structure head sits at the edge of the λ grid).
- **"The model's own internal state."** No experiment separates "the generator's state" from "any 8B LLM's encoding of the prompt". The decisive control is cross-model: predict Llama's violations from Qwen's hidden states and attention, and the reverse. This is listed only as exploratory item (iv). Without it, A1 beating BGE-M3 may reflect a 16x larger reader, not self-knowledge. The moderate cross-generator agreement (Spearman 0.43) makes this control informative.
- **Head-structure confound.** Probe configurations get 15 per-type heads; B1, B2 and the structure head get one shared head. The paper discloses this for C2/C3 but the same confound applies to the headline C1 comparison (A1 vs B2), which is what the abstract reports.
- **What attention-to-span encodes.** Summed attention over a span carries span length and position, which are prompt-derivable. Within-type AUROC is 0.56 for A1 on Qwen versus 0.58 for B2, so for Qwen the "internal" advantage is almost entirely between-type or calibration. The abstract's symmetric phrasing ("for both generators") hides this.

The description of Heo et al. and of Stolfo et al. is accurate as far as I can tell. The statement that prior multi-instruction work "measures or estimates aggregate compliance" is fair for ManyIFEval and IFScale.

## Missing related work

- **Prompt-only predictors of model success (the baseline's own literature).** Assessors: Zhou et al., "Reject Before You Run: Small Assessors Anticipate Big Language Models" (2022); Schellaert et al. on predictability of LLM performance. Routing: FrugalGPT (Chen et al., 2023), Hybrid LLM (Ding et al., 2024), RouteLLM (Ong et al., 2024). The introduction motivates routing but cites none of it, and these are exactly B1/B2-style predictors.
- **Conditional / baselined probing.** Hewitt et al., "Conditional Probing: Measuring Usable Information Beyond a Baseline" (EMNLP 2021) formalises RQ3; also Voita and Titov (MDL probing) and Pimentel et al. (information-theoretic probing). RQ3 should be framed against these.
- **Pre-generation probes of own behaviour.** Ji et al., "LLM Internal States Reveal Hallucination Risk Faced With a Query" (2024); Slobodkin et al. on (un)answerability (2023); Snyder et al. on early hallucination detection; and, closest, Ashok and May, "Language Models Can Predict Their Own Behavior" (2025), which to my recollection includes format-following failures predicted from input-token states.
- **Uncertainty for instruction following.** Heo et al., "Do LLMs Estimate Uncertainty Well in Instruction-Following?" (ICLR 2025), by the same group as the closest cited work, evaluates verbalised, logit and probe-based uncertainty for this task. No verbalised self-prediction or P(IK)-style baseline is included here.
- **Attention features as detectors.** Lookback Lens (Chuang et al., 2024) trains linear classifiers on per-head attention ratios; Attention Tracker (Hung et al., 2024) uses attention to the instruction for prompt-injection detection; instruction-attention steering (e.g. Spotlight, Venkateswaran and Contractor, 2025) alongside PASTA.
- **Multi-constraint benchmarks and order effects.** ComplexBench (Wen et al., NeurIPS 2024), COLLIE (Yao et al., 2024), Sun et al. (2023) on controlled generation, and work on constraint-order effects (I recall "Order Matters", Zeng et al., 2025). The paper's position result (17.7% first vs 18.9% last) should be compared with these.

## Strengths

- Clear, well-motivated question; the incremental framing is the right one and rarely done properly.
- Structure-only references and the candid negative results (H1 equals structure head; Llama Objective 2 tie) raise the bar for this literature.
- Group-level splits, pre-specification, hashed plan, disclosed deviations.
- Scale and design quality of the dataset; the verifier audit.
- Limitations section is accurate and complete.

## Weaknesses

- No test results. All evidence is cross-validation on data used for selection; the paper says so, but a venue submission cannot be judged on a plan.
- The novelty-bearing claim (internal beats prompt-only) is confounded by head structure, baseline strength, λ-grid edges and the absence of a cross-model control.
- 15 of 23 configurations have distinct inputs; the "23 predictors" framing inflates the comparison.
- Related work omits the assessor/routing and conditional-probing literatures that define the baseline and the question.
- Two 8B models, 15 templated rule types, 50 of 90 test rule instances seen in training: generality of the attention finding is untested.

## Questions

1. What does A1 score against B2 and the structure head when all three use per-type heads?
2. Cross-model control: how well do Qwen's H and A features predict Llama's labels?
3. Does a structure head given span length and token position of the target rule close the gap to A1?
4. Why does fine-tuned B2 rank worse (AUROC) than the structure head? Was a stronger text predictor (larger encoder, or a LoRA-tuned small LLM) tried?
5. Is the langdetect failure on uppercase text already reported for IFEval, and does it affect the released ManyIFEval numbers?
6. How does the per-rule result compare with Heo et al.'s direction on the k=1 subset, where the settings coincide?

## Recommendation

A careful and honest study with a worthwhile question, but as submitted it is a registered report without the result. With test-set numbers, the per-type-head and cross-model controls, and the missing literature, this could be a clear accept.
