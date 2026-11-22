# Baseline 1 (B1) in the ManyIFEval probing study: what it was in V1, what changed in V2, and how it runs

*Written 2026-11-21. Self-contained: no access to the code repository is needed. Every number comes from the project's saved training-only cross-validation outputs. The test set has not been touched.*

---

## 1. Background: the study in one page

**What is being predicted.** A large language model (LLM) receives a prompt: a short task ("Write a song about being excited to go on vacation.") followed by a list of 1 to 5 formatting rules ("Answer with at least 425 words.", "Wrap your entire response with double quotation marks.", and so on). We want to predict, **before the model writes anything**, how likely it is to break each rule.

**Data.**
- Rules come from ManyIFEval, an instruction-following benchmark. It has 15 *rule types*, for example `length_constraints:number_words`, `punctuation:no_comma` and `startend:quotation`. Each type has parameters, such as the word count and whether it is "at least" or "less than".
- The 216 tasks are arranged in 72 *groups* of 3 tasks. Each group has its own pool of 6 exact rules, with 6 different types, checked to be jointly satisfiable. From that pool, 246 ordered rule lists (k = 1 to 5 rules) are formed, and each is combined with each of the group's 3 tasks.
- Two generators answer every prompt 20 times: Llama-3.1-8B-Instruct and Qwen3-8B. An official verifier marks each of the 20 answers as following or violating each rule.
- The split is by group. 57 *training* groups give 42,066 prompts and 175,446 prompt–rule pairs. 15 *test* groups are sealed and have not been used.

**Objective 1 (the focus of this document).** The unit is a *prompt–rule row*: one prompt plus one of its rules, the *target rule*. The label is the number of the 20 completions that violated the target rule (v out of n = 20). The prediction is a probability p that a completion violates it.

**Metric.** Mean log loss per completion (binomial negative log likelihood), where lower is better. A row with v = 8 violations contributes 8 × (−log p) + 12 × (−log(1 − p)), and the total is divided by the number of completions.

**The question the study asks.** Does the generator's own internal state before it starts writing predict rule violations better than an outside reading of the prompt text? The internal state is either its final-token *hidden state* or its *attention* to the rule. **Baseline 1 is that "outside reading of the prompt text."**

**Cross-validation (CV).** All model choices are made with 5-fold *grouped* CV over the 57 training groups: whole groups are held out together (12/12/11/11/11 groups per fold). Every training row receives an *out-of-fold* (OOF) prediction from a model that never saw its group.

---

## 2. What B1 was in V1 (the 2026-11-14 plan)

### 2.1 Inputs (Objective 1)

V1's B1 concatenated three blocks, 2,123 numbers per row:

| Block | Size | What it is |
|---|---|---|
| EP: prompt embedding | 1,024 | Frozen BGE-M3 sentence-embedding model (a general-purpose text encoder, not fine-tuned) applied to the **full prompt text**. The same vector for every rule of that prompt. |
| ER: rule embedding | 1,024 | The same frozen BGE-M3 encoder applied to **the target rule's text only**, e.g. "Answer with at least 425 words." |
| S1: structure | 75 | Hand-built features: number of rules k (as a number and one-hot); **target rule type one-hot** (15); target position, position/k, is-first, is-last; **parsed target parameters** (log(1 + number) and a ±1 "less than / at least" flag in the type's own column, 13 columns); which other types are present (15); the companions' parsed parameters (13); **task family** one-hot (6, a proposed taxonomy); prompt length in **BGE-M3 tokens**, prompt characters, task characters. |

### 2.2 The prediction head

- **One shared head for all 15 rule types.** Every prompt–rule row, whatever its type, went into the same model. The type was only an input feature (the one-hot in S1), so the head had to learn how a rule's type changes its violation rate through a single set of weights.
- **Head family:** chosen by CV from 6 settings: {linear logistic, 1-hidden-layer MLP with 128 ReLU units} × L2 penalty {0.01, 0.1, 1}. The penalty was applied on the mean-loss scale to all weights at once (prompt, rule and structure alike). For the MLP, the number of training epochs (1–30) was also chosen by CV.
- **Selected in V1:** MLP with L2 = 0.1, stopped at epoch 4 (Llama) or epoch 2 (Qwen). Training-CV loss: **0.387 (Llama), 0.265 (Qwen).**

### 2.3 Why that was a problem

The V1 B1 was a reasonable predictor on its own. It was a poor **comparator** for the question the study asks, for these reasons, all visible in V1's training-CV tables:

1. **Different architecture from the thing it was compared with.** The probes it was compared against (H1 hidden state, H2, A1 attention) used **15 per-type heads**: one separate model per rule type. B1 and B2 used **one shared head**. A difference between "H1 vs B1" therefore mixed two things: a better input representation, and a different model structure (separate intercepts and slopes per type versus a single shared model). The effect of the representation could not be isolated.
2. **Different information on the two sides.** B1 received the target rule's embedding, its parsed parameters and the task family. H1 + structure received none of these. So "H1 vs B1" was not like-for-like on information either.
3. **Unstable and uneven tuning.**
   - Many selected penalties sat on the edge of the grid.
   - Several MLP heads chose epoch 1 or 2, also an edge.
   - Different predictors had different search spaces.
   - A single L2 value penalised a 75-column structure block and a 1,024-column embedding block equally.
4. **The structure block used the encoder's tokenizer, not the generator's.** The prompt length was measured in BGE-M3 tokens. Nothing measured the target span's length, or its distance from the end of the prompt, in the tokens the generator actually reads.
5. **Many of the 23 V1 configurations were literally identical** once duplicate blocks were removed (e.g. B1 + H1 = B1 + H2), which inflated the comparison list.
6. **Fine-tuned B2 was expensive.** It needed more than 24 hours on the Mac and ran on a GPU cluster.

---

## 3. What V2 changed, and why

V2 was adopted on 2026-11-20 after an independent audit. Its goal for the baselines is that **each comparison changes exactly one thing.**

### 3.1 New definitions

| | V2 definition |
|---|---|
| **Structure (S)** | 20 columns, identical in every Objective 1 predictor: k; target position; 15 companion-type indicators; rendered prompt length; target-span length; distance (in tokens) from the end of the target span to the final prompt token. All token counts use **the target generator's own tokenizer**. |
| **B1** | **EP (1,024) + S (20)**: the frozen full-prompt embedding and structure. |
| **B2** | B1 + **ER**: the frozen exact-rule embedding, the same pinned encoder, no fine-tuning. |
| **H1 + structure** | Generator's final-token hidden state (4,096) + S |
| **H2 + structure** | Hidden state + ER + S |

**Deliberately excluded from every predictor:** parsed rule parameters (the number 425 and the "at least" flag), task and group identifiers, and task family.

### 3.2 Same architecture for every predictor

Every Objective 1 predictor, B1 included, now has **15 ridge-logistic heads per generator, one per rule type.** The target rule's type selects the head. Each head:
- is fitted only on rows of its own type;
- has its own intercept (its own base violation rate) and its own coefficients;
- standardises its features (mean 0, sd 1) using only its own fitting rows.

**Rationale.** Rule types differ enormously. In the fitting rows of one Llama fold, 7.0% of completions violate a word-count rule but 29.2% violate an "exactly N bullet points" rule. The prompt cues that matter also differ by type. Per-type heads let each type have its own base rate and its own cues. More importantly, H1, H2, the attention probe, B1 and B2 now all use this same structure, so a comparison such as **"H1 + structure vs B1"** differs only in the prompt representation: the generator's hidden state versus an independent text encoder's embedding.

### 3.3 Same fitting recipe for every predictor

- **Loss.** The loss is summed binomial negative log likelihood over the head's fitting rows, with a ridge penalty:
  `J(w, b) = Σᵢ [ nᵢ·softplus(zᵢ) − vᵢ·zᵢ ] + ½ Σⱼ λⱼ wⱼ²`, where `zᵢ = b + Σⱼ wⱼ xᵢⱼ` and `p = sigmoid(z)`.
  It is implemented, equivalently, as the *mean* loss per completion plus a penalty of λ / N, where N is that head's number of fitting completions. This gives every head **the same prior strength regardless of its size**. A rare type with 6,000 rows and a common type with 11,000 rows are regularised consistently.
- **Three penalty "slots," identical across methods:**
  - *structure*: fixed weak penalty λ = 10, never tuned;
  - *prompt representation*: EP for B1/B2, hidden state for H1/H2, attention for the attention probe; tuned over the same 6 values {10², 10³, 10⁴, 10⁵, 10⁶, 10⁷};
  - *rule embedding*: B2/H2/attention probe only; tuned over the same 6 values.

  B1 and H1 + structure therefore search exactly the same 6-value grid.
- **Linear heads only, no MLPs and no epoch tuning.** The inputs are high-dimensional (1,024–6,200 columns) with roughly 5,000–9,500 fitting rows per head. A well-regularised linear model is the standard, stable choice there. It also means no predictor gets extra capacity or extra tuning knobs.
- **Selection.** One penalty setting per generator and predictor, shared by all 15 heads. It is the setting with the lowest pooled OOF log loss (pooled over all rows of all types and folds). Ties are broken deterministically (rounded to 1e-6; prefer the stronger penalty).
- **Boundary rule.** If the winner sits on the edge of the grid, extend that slot once by a factor of 10 and reselect. For B1 this never triggered, because the winner, 10⁶, is inside the grid.
- **Sparse-type fallback, identical for all methods.** If a type's fitting rows have fewer than 50 violated or 50 compliant completions, or fewer than 5 groups with any violation, that head becomes intercept-only and predicts (v + 0.5) / (n + 1). For Llama this never fired. For Qwen it fires for the title rule in every fold, and then applies to B1 and every other method alike.
- **Final model.** The chosen setting is refitted on all 57 training groups, one head per type.

---

## 4. How B1 runs, step by step

```
For each generator (Llama, Qwen) separately:
  Rows: the 175,446 training prompt–rule pairs, each with label (v violations out of n = 20)
  Features per row: x = [ EP(prompt) (1,024 numbers) , S(prompt, target) (20 numbers) ]

  For each candidate prompt penalty P in {1e2, 1e3, 1e4, 1e5, 1e6, 1e7}:
    For each CV fold f = 0..4:                      # held-out = the groups of fold f
      For each rule type t (15 types):
        fit rows  = rows of type t whose group is NOT in fold f
        held rows = rows of type t whose group IS in fold f
        if fallback(fit rows): predict (v+0.5)/(n+1) for held rows; continue
        standardise the 1,044 columns using fit rows' mean and sd
        minimise  mean NLL per completion + (1/2N)[ P·||w_EP||² + 10·||w_S||² ]
                  (damped Newton's method, exact Hessian)
        predict p = sigmoid(b + w·x) for held rows    -> out-of-fold predictions
    pooled loss(P) = Σ NLL over all held rows / Σ completions

  choose P* with the lowest pooled loss (tie -> larger P)
  refit 15 heads with P* on all 57 groups -> final B1 model
```

**Training-CV results (pooled mean log loss per completion; lower is better):**

| Prompt penalty P | 1e2 | 1e3 | 1e4 | 1e5 | **1e6** | 1e7 |
|---|---|---|---|---|---|---|
| Llama B1 | 1.009 | 0.644 | 0.516 | 0.456 | **0.440** | 0.446 |
| Qwen B1 | 0.869 | 0.498 | 0.383 | 0.332 | **0.318** | 0.322 |

Weak penalties overfit badly: 1,024 embedding dimensions against a few thousand groups' worth of rows. Very strong penalties shrink the embedding to almost nothing. The optimum is 10⁶ for both generators.

**For context (same folds, same metric):**

| Predictor | Llama | Qwen |
|---|---|---|
| Structure only (S, same 15 heads) | 0.448 | 0.323 |
| **B1 (EP + S)** | **0.440** | **0.318** |
| B2 (B1 + exact-rule embedding) | 0.382 | 0.283 |
| H1 + structure (generator hidden state + S) | 0.393 | 0.289 |
| V1 B1 (shared MLP head, EP + ER + parameters + family + S1) | 0.387 | 0.265 |

The full-prompt embedding adds only a little over structure alone (0.448 → 0.440 for Llama). Most of the extra information comes from the exact rule's wording (B2) or from the generator's own state (H1). V2's B1 is weaker than V1's B1 **by design**: it no longer sees the rule's own embedding, its parsed parameters or the task family. These are training-CV numbers that were used for selection, so they describe the fitting, not final test performance.

---

## 5. Worked example (Llama, a real training row)

**Prompt** `g20-c4-k3-o03-t1883` (group 20, so CV fold 3 holds it out). It has 3 rules:

```
Write a song about being excited to go on vacation.

Your response should follow the instructions below:
- Wrap your entire response with double quotation marks.
- Answer with at least 425 words.
- Your answer must contain exactly 2 bullet points. Use the markdown bullet points such as: * This is point 1.  * This is point 2
```

This one prompt produces **three Objective 1 rows**, one per rule. Each is routed to a different head:

| Target rule (position) | Head used | Observed violations (of 20) | B1 OOF prediction |
|---|---|---|---|
| quotation marks (1) | `startend:quotation` | 1 | 0.090 |
| at least 425 words (2) | `length_constraints:number_words` | 8 | 0.003 |
| exactly 2 bullets (3) | `detectable_format:number_bullet_lists` | 4 | 0.421 |

Follow row 2, "Answer with at least 425 words.":

1. **Routing.** The target type is `number_words`, so the word-count head is used. In fold 3, that head was fitted on **6,156 rows from 12 other groups** (123,120 completions, of which 8,593, or 7.0%, violated). Its fallback does not fire: there are well over 50 violations and the rule is violated in more than 5 groups.
2. **Features.**
   - *EP*: the 1,024-number BGE-M3 embedding of the whole prompt text above. The rows for rules 1 and 3 use the same vector.
   - *S*, raw values: k = 3; position = 2; companions = {quotation, bullet lists} (two 1s, the other 13 indicators 0); prompt length = 106 Llama tokens; target span = 8 tokens; 38 tokens from the end of the span to the final prompt token.
3. **Standardise** each column with the head's fitting-row mean and sd. For example, k = 3 becomes −1.74, because word-count rows in this fold mostly have more rules. The quotation companion becomes +2.93, because it is rare among word-count rows.
4. **Linear score** with the fold-3 word-count head (P = 10⁶, i.e. λ/N ≈ 8 on the mean scale):
   - intercept b = −3.67, the type's base rate on the logit scale;
   - prompt-embedding contribution w_EP · x_EP = +0.11;
   - structure contribution w_S · x_S = −2.35. The largest terms are the quotation companion (+2.93 × −0.64) and k (−1.74 × +0.25).
   - total z = −5.91, so **p = sigmoid(−5.91) = 0.0027**.
5. **Scoring.** The row's log-loss contribution uses v = 8 and n = 20. It is large here, because the model predicted 0.3% and the observed rate was 40%.

**Why this row is badly missed, and what it teaches.** In the training data, Llama's word-count violations depend strongly on the number requested:

| Word-count rule | Violation rate (Llama, training) |
|---|---|
| at least 200–300 words | 5.4% |
| at least 300–400 words | 9.2% |
| **at least 400–500 words** | **28.3%** |
| less than 300–500 words | about 0.5% |

V2 B1 is not given "425" or "at least" as features. That was a deliberate decision: parsed parameters stay outside every predictor. B1 can only pick them up indirectly through the full-prompt embedding. Here the embedding's whole contribution is +0.11 on the logit scale, so it effectively cannot read the number. V1's B1 did have the parsed number and relation as features, and predicted 0.089 for this row, closer to the observed 0.40. B2's rule-text embedding also misses (0.007). H1, the generator's hidden state, gives 0.027. This is a single example, not evidence about averages. It does show concretely what B1 represents in V2: what a generic text encoder's view of the whole prompt, plus simple structure, can predict for each rule type. It deliberately excludes any hand-parsed knowledge of the rule's parameters.

---

## 6. Summary

| | V1 B1 | V2 B1 |
|---|---|---|
| Inputs | Prompt embedding + **rule embedding** + 75 structure columns (type one-hot, **parsed parameters**, **task family**, BGE-token lengths) | Prompt embedding + 20 structure columns (generator-token lengths, span length and distance) |
| Head | **One shared head** for all 15 types | **15 per-type heads**, each with its own intercept and coefficients |
| Model | Linear or MLP (chosen by CV), epochs tuned | Ridge logistic only |
| Penalty | One L2 on every weight, mean-loss scale | Fixed structure penalty; tuned prompt penalty (6 values); λ/N per head |
| Matches the probes it is compared with? | No (different heads, different information) | Yes: same heads, same structure, same grid, same fallback |
| Training-CV loss (Llama / Qwen) | 0.387 / 0.265 | 0.440 / 0.318 |

**Rationale in one sentence.** V2 B1 is the cleanest possible "text-only" counterpart to the generator's hidden state. Everything except the prompt representation is held identical, so "H1 + structure vs B1" measures what the generator's internal state adds over an outside reading of the same prompt. The rule's wording is added symmetrically, to both sides, in the next comparison (H2 + structure vs B2).

**Glossary.**
- *Logit / sigmoid:* p = 1 / (1 + e^(−z)). The model is linear in z.
- *Ridge:* a penalty on squared coefficient sizes that shrinks them toward 0 to prevent overfitting.
- *Out-of-fold:* a prediction for a row made by a model trained without that row's group.
- *Pooled loss:* total log loss over all out-of-fold rows divided by total completions.
- *Frozen encoder:* used as is, with no training on this data.
- *Fallback:* a constant-rate head used when a rule type has too few violations or too little group coverage to fit reliably.
