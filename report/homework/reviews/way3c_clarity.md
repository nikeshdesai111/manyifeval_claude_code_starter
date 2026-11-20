# Review (clarity and presentation focus)

**Paper:** Predicting Rule Violations Before Generation: Hidden-State and Attention Probes versus Prompt-Only Baselines on a Controlled Multi-Instruction Benchmark

**Basis of this review:** `paper.tex`, the 18 generated tables it inputs, and the six figure PNGs. I did not see the bibliography or a compiled PDF, so I cannot comment on page layout, float placement or reference formatting. The source includes the figures as `.pdf`; I reviewed the `.png` versions.

**Score: 6 / 10** (marginally above the acceptance threshold on clarity alone; see "Reason for the score")

---

## Summary of the paper

The paper asks whether a language model's internal state at the end of a prompt predicts which of several instructions it will violate, better than a strong predictor that sees only the prompt. It builds 53,136 prompts from ManyIFEval in a balanced design (72 groups, k = 1 to 5 rules, orthogonal-array orderings), samples 20 responses per prompt from two 8B models, and compares 23 feature configurations (hidden state H1, hidden state plus rule embedding H2, attention to the rule span A1, frozen text baseline B1, fine-tuned text baseline B2, and combinations) against structure-only references. The manuscript is explicitly a pre-unblinding draft: all results are grouped cross-validation or out-of-fold on the 57 training groups, and the 15 test groups are unread.

---

## Strengths

1. **The status of every claim is stated, repeatedly and in the right places.** The boxed note before the abstract, the section title "Results so far (training groups only)", the phrase "Descriptive, not confirmatory" in captions, and the paragraph on why out-of-fold intervals are not confirmatory leave no room to misread training-set numbers as test results. This is rarer than it should be.

2. **The question is sharpened before any method appears.** The introduction's third paragraph (a probe may do well "simply because the prompt is encoded" in the hidden state) sets up the incremental question, and the four numbered design choices follow from it. A reader knows why each baseline exists before meeting it.

3. **Problem setup is compact and unambiguous.** Section 3 defines p_r(x), q(x), their estimators and the two objectives in half a page, and heads off the obvious misreading of q ("expected fraction of rules violated, not the probability that at least one is").

4. **The dataset construction can be followed and audited by counting.** Table 1 gives per-task, per-group, train, test and total counts for every unit, and they multiply out (246 x 3 x 72 = 53,136; 6 x (1+2+6+12+20) = 246). The overview figure maps cleanly onto Sections 4 to 6.

5. **Prose style.** Short declarative sentences, one idea per paragraph, bold run-in heads that state the finding ("Known structure already explains a lot"). Numbers in the text match the tables I checked (Table 2, Table 3, the contrast tables, the per-k table, the uppercase audit).

6. **Unflattering facts are surfaced, not buried.** Only 15 of 23 configurations have distinct inputs; the regularisation grid is edge-selected for the key comparator; head structure is confounded with probe features; one test-set statistic was computed before the firewall. Each is disclosed in the main text and again in Limitations.

7. **Reproducibility detail is above average.** Decoding settings, token limits, optimiser settings, LoRA hyperparameters, bootstrap replicate counts, fold sizes, feature-block widths and compute cost are all given. The appendix reports the selected head for every configuration.

8. **Figures are clean.** Consistent colour per generator, direct labelling, intervals drawn, and captions that warn when axes differ between panels.

---

## Weaknesses

### A. Things that affect whether the main claim can be read correctly

1. **The per-type-head confound is disclosed for the wrong (or too few) contrasts.** Section 5.2 says any Objective 1 configuration containing H1, H2 or A1 uses 15 per-type heads, while B1, B2 and the structure head use one shared head. The paper flags this for C2 and C3 (adding a probe to a baseline). But the headline comparisons (A1 vs B2, A1 vs structure head, Table 3, Figure 4, the abstract's AUROC 0.85 vs 0.79) are also per-type heads against a shared head. The text never says so at the point where those numbers are presented, and the Limitations entry names only C2 and C3. A reader will take the abstract's comparison as features against features. Either state the confound next to Table 3 and in the abstract, or explain why it does not apply to C1 and the reference contrasts.

2. **Seed noise between identical-input configurations is as large as several reported effects, and the paper does not give the reader a noise floor.** Examples from the appendix tables: for Llama Objective 2, B1+H1, B1+H2 and B1+H1+H2 are stated to have identical inputs but score 1.642, 1.669 and 1.691 (a 3% spread). For Qwen Objective 1, H2+A1 and H1+H2+A1 are identical inputs and score 0.2386 and 0.2358; the latter is then named "best probe-only" and the text reports the B2-plus-probe optimum as "1.4% below the best probe-only configuration", a margin of the same size. Meanwhile some identical pairs agree exactly (the Qwen Objective 2 contrast B1+H1 vs B1+H2 is +0.0000 [+0.0000, +0.0000]). The reader cannot tell which duplicates are deterministic (linear heads?) and which are not, or how much any single CV number should be trusted to the third digit. Please (i) state the seed-to-seed spread explicitly, (ii) collapse the 23 names to the 15 distinct feature sets in Figure 3 and the appendix tables, or mark duplicates visually, and (iii) avoid naming a "best" among duplicates.

3. **Three different loss scales are used for Objective 2 without a bridge.** Table 2 and the appendix report MSE in units of 10^-2 (1.613); Figure 3 plots the same quantity as raw MSE (0.016); Tables 3, 5-appendix and Figure 4 use RMSE (0.127). Table 3 puts a relative change in CV MSE next to an absolute change in OOF RMSE in adjacent columns. A 14% MSE reduction is roughly a 7% RMSE reduction, so the two columns look inconsistent to a reader who does not stop to convert. Pick one scale for Objective 2, or add a sentence and a consistent axis.

4. **"Best probe-only configuration" and "A1" are used interchangeably in places where they are not the same.** Under the bold head "attention is the best single signal", the sentence "The best probe-only configuration lowers the loss of the structure head by 12.3% (Llama) and 8.4% (Qwen)" refers to A1 for Llama but H1+H2+A1 for Qwen. Table 3 names the configuration, but the prose and the abstract ("Probes built on internal states lower the loss of the strongest comparator ... by 8 to 14%") do not, and "strongest comparator" changes identity across cells (B2, structure head, structure head) without being named.

### B. Notation and terminology

5. **Too many names for units of analysis.** "prompt-rule row", "(prompt, rule) pair", "(response, rule) check", "rule instance", "rule-level pair", "cell" (used in the abstract before it is defined as a generator-objective combination). A one-line glossary in Section 3, and using "cell" only after defining it, would help.

6. **Rule-type names are inconsistent.** Section 4.1 uses "keyword existence, keyword frequency"; Section 7.4 uses "keyword inclusion, keyword count"; the tables and figures use "Include keywords", "Keyword count". The same for "letter-case types" versus the three named types. Use the table names everywhere.

7. **Component versus block versus configuration.** The H1/H2/A1/B1/B2 labels are opaque on first contact, and Table 2 introduces ten more block codes (H, A, A5, EP, ER, ER5, S1, S2, M2, KM) in a caption. A reader has to hold fifteen abbreviations to parse Table 2. Dimensions are given for some blocks and not others (M2 and KM have none; the Objective 2 widths in the appendix, e.g. 4,197 for H1, can only be reconciled by guessing M2 = 101). In Table 6, family C4 uses a bare "H" that is never defined. Consider descriptive names (Hidden, Hidden+Rule, Attention, Text-frozen, Text-tuned) at least in figures and the abstract.

8. **Sign conventions for differences are described three ways.** Table 3: "probe minus comparator". Figure 4: "second configuration minus first" with arrows "Structure -> H1". Appendix: "b minus a". They agree, but a single convention stated once would save the reader the check.

9. **Metrics on soft labels are not defined.** AUROC, Brier score, ECE and "top/bottom 10%" are reported for a target that is a count out of 20. The text implies AUROC is computed over individual (response, rule) checks with the row's prediction replicated, but does not say so; likewise for ECE in "decile bins". The binomial NLL "reported per check" is never written as a formula. These need one equation each.

### C. Reproducibility gaps

10. **Rule-span definition.** "The exact token span of each rule" is not specified: does it include the bullet marker and newline, how are sub-word boundaries at span edges handled, and are spans identical in definition for both tokenisers? Since A1 is the headline feature this matters.

11. **B2 procedure is hard to reconstruct.** It is unclear (i) whether the encoder hyperparameters were selected on the same held-out fold whose loss is then reported for B2 heads, (ii) which encoder produces B2 embeddings for the final all-train refit, (iii) what "a rule fixed in advance on one fold" for LoRA versus full fine-tuning was, and (iv) whether Objective 1 and Objective 2 use separately tuned encoders (Table "B2 grid" suggests yes; the text does not say).

12. **Unspecified procedures.** The written rubric for task families; how the eight near-duplicate clusters were identified; how the five folds were assigned (random, stratified?); what "provably unsatisfiable" means operationally; how repetition loops are detected; how A5 and ER5 pad prompts with fewer than five rules; what the "structure head" input is for Objective 2 beyond "S2". None is long, and an appendix paragraph each would do.

13. **Per-type interval counts cannot be verified.** Section 7.4 reports "13 of the 15 types (with an interval excluding zero on 8)", but neither Figure 5 nor the per-type appendix tables show intervals. Add them to the tables or the figure.

14. **Hardware statement.** Responses come from vLLM on NVIDIA GPUs and features from Transformers on Apple silicon. The 0.998 cosine check is described as coming from "a pilot" with no sample size, layer, or statement of whether it was measured at the selected layers. Attention agreement between backends is not reported at all, though attention is the main result.

### D. Organisation

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

## Questions for the authors

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

## Suggestions (concrete, in priority order)

1. State the head-structure asymmetry beside the headline numbers, or add the per-type-head baseline to the main table.
2. Report results on the 15 distinct feature sets and give a seed-noise floor.
3. Use one Objective 2 scale throughout.
4. Add a notation table (units of analysis, blocks with dimensions, components, configurations) and unify rule-type names.
5. Add equations for the binomial NLL and for each secondary metric.
6. Add an appendix paragraph for each unspecified procedure in item 12, and the span definition in the main text.
7. Shorten Section 8 to a half-page plan plus Table 6; move the pair-count table to the appendix.

---

## Reason for the score

Judged on writing alone, this is a well above average manuscript: clear question, clean setup, honest framing, verifiable counts, and text that matches its tables. What holds the score down is (i) the headline comparison can be misread because the head-structure asymmetry is flagged only for a subset of contrasts, (ii) the 23-configuration presentation carries eight duplicates whose seed noise is comparable to effects the text interprets, (iii) mixed MSE/RMSE scales and a heavy abbreviation load, (iv) several procedure gaps around the main feature (span definition, B2 selection, soft-label metrics), and (v) the manuscript is by its own statement incomplete, with no held-out result and two protocol decisions open. Items (i) to (iv) are fixable in a revision without new experiments beyond one control.

**Score: 6 / 10. Confidence: 4 / 5** (I read the full source, all generated tables and all figures; I did not see the bibliography or a compiled PDF).
