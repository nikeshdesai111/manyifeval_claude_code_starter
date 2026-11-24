# Preliminary Results (S3 TRAIN, grouped-CV out-of-fold) — paper-ready text

All numbers come from `results/preliminary_train_report/tables/`. Intervals are paired 95% 50-group (rule-pool)
bootstrap intervals, pointwise and unadjusted. These are **not** held-out TEST results; see `README.md` for the
disclosure and provenance.

## Evaluation 1, Objective 1 (per-rule violation probability; 153,900 prompt–rule rows, 3,078,000 outcomes per generator)

Replacing the frozen prompt embedding with the generator's hidden state (comparison 1, B1→H1) reduces NLL by
68.3 [37.9, 102.7] ×10⁻³ nats per outcome for Llama and 58.3 [26.7, 94.4] for Qwen, raises AUROC by 5.9 [3.8, 7.9]
and 7.7 [4.2, 11.6] points, and lowers NLL on 10 and 11 of 15 rule types. The gain is concentrated in the ten
parameterised types (Llama 100 vs 7 ×10⁻³ for the fixed-wording types; Qwen 85 vs 8). Against B2, which also embeds the
exact rule, the hidden state (comparison 2) gives 7.9 [−2.2, 18.2] for Llama and 18.6 [3.8, 34.6] for Qwen, with AUROC
gains of 1.3 [0.5, 2.1] and 4.0 [2.1, 6.0]; rule attention (comparison 3) gives 11.6 [−3.9, 27.5] and
18.6 [1.2, 38.4]. Hidden state and attention are indistinguishable (comparison 4: −3.7 [−19.9, 10.4] and
0.0 [−13.0, 12.9]). Added to B2, the hidden state (comparison 6: 16.3 [7.8, 25.4]; 18.6 [3.9, 34.4]) and attention
(comparison 7: 19.2 [8.3, 31.4]; 19.2 [1.8, 39.2]) help both generators.

## Evaluation 1, Objective 2 (expected fraction of rules violated; 36,900 prompts)

Primary candidates of comparisons 1–3 raise Spearman correlation by 28.6/8.5/8.3 points (Llama) and 12.1/11.3/10.5
(Qwen), all intervals above zero. RMSE reductions are 1.94 [0.84, 3.06], 0.60 [0.04, 1.18] and 0.76 [0.06, 1.45] points
for Llama and 0.76 [−0.11, 1.61], 0.49 [−0.45, 1.35] and 0.50 [−0.04, 1.05] for Qwen. Comparison 4 is null for both.
The mean of the Objective 1 H2 predictions over a prompt's rules is as accurate as the dedicated prompt-level H2 head
(RMSE 12.22 vs 12.67 points for Llama; 11.02 vs 11.01 for Qwen), and every learned predictor beats the training-fold mean.

## Evaluation 2, Objective 1 (matched changes)

*Observed (descriptive, exploratory).* With everything else fixed, moving a rule changes its violation probability by a
median-over-types noise-adjusted RMS of 7.5 pp (Llama) and 6.5 pp (Qwen); most for paragraph count (Llama 24.4) and
bullet points (Qwen 20.9). Replacing one companion: 12.9 and 12.4 pp. Changing the task with the rule configuration
fixed: 21.1 and 12.3 pp, up to ~50 pp (bullet points for Llama, uppercase-word count for Qwen). Appending a rule changes
retained-rule risk by at most 1.4 pp: Qwen 0.4 [−0.3, 1.1] (1→2), 0.8 [0.2, 1.5], 1.2 [0.7, 1.8], 1.4 [0.9, 2.0] (4→5);
Llama −0.5 [−1.7, 0.6], 0.0 [−0.7, 0.6], 0.4 [−0.2, 1.1], 0.7 [0.2, 1.3]. Task-family differences are associations
(e.g. Analytical tasks −5.2 [−9.6, −0.9] pp for Llama and −3.5 [−6.1, −1.5] for Qwen relative to the matched other tasks).

*Predictive (Δ-MSE ×10⁴, positive favours candidate).* Generator- and factor-specific. Llama: hidden-state probes
forecast task-induced changes better than both text baselines (comparison 1: 161.8 [81.0, 247.8]; comparison 2:
127.1 [48.7, 206.7]); H2 beats the attention probe on all four factors (comparison 4: 43.4, 25.1, 27.6, 69.8, all
intervals above zero). Qwen: H2 is worse than B2 on task (−30.6 [−53.6, −8.1]) and position (−5.3 [−10.7, −1.8]).
Both generators: the attention probe is worse than B2 on position and append changes, and H1 is worse than B1 on append
and companion changes.

## Evaluation 2, Objective 2 (whole-prompt fraction; preliminary design, not approved in the plan)

Reordering the same rules changes the violated fraction by a noise-adjusted RMS of 6.9 [6.0, 7.7] pp (Llama) and
6.1 [5.4, 6.8] (Qwen); one-companion replacement 16.7 and 13.4 pp; task change 18.8 and 12.6 pp. Appending a rule (new
rule included in the outcome) raises the fraction by 0.5 [−0.0, 1.0] pp (Llama) and 1.1 [0.7, 1.5] (Qwen). Full
Δ-MSE table: `obj2_eval2.csv` and paper Appendix H.

## Limitations

Grouped-CV estimates on groups whose folds also selected settings (optimistic); 50 TRAIN pools, original 246-configuration
design (no strict companion replacements at k = 4); the Objective 2 matched-change analysis is a preliminary design;
intervals unadjusted. Final evaluation on the 22 S3 TEST groups is pending and will replace these numbers.
