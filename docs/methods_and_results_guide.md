# Guide to the dataset, the probing and baseline configurations, and `results/RESULTS.md`

Written 2026-11-16 from the code in `src/mife/`, `configs/experiment.yaml`, the frozen pre-specification and the tables in `results/tables/`.

This guide explains three things:

1. how the dataset was built (prompts, responses, labels, stored model features);
2. what each of the 23 configurations and 4 reference rows is, and exactly how each was fitted;
3. how to read every table in [`results/RESULTS.md`](../results/RESULTS.md).

Section 7 lists every implementation decision with its reason and where it lives in the code.

**Status of the results.** Everything fitted so far used the 57 TRAIN groups only. The 15 TEST groups have not been read. `RESULTS.md` §§ 5–7 (test evaluation, contrasts, controlled analyses) are therefore still "pending", and every loss in § 4 is a cross-validation loss on training data.

---

## 1. The study in brief

**Question.** Before a language model starts writing, can we predict how likely it is to break the instructions in its prompt? And do the model's own internal states (hidden states, attention) predict this better than features of the prompt text alone?

**Two prediction targets ("objectives").**

| | Objective 1 | Objective 2 |
|---|---|---|
| Unit | one (prompt, rule) pair | one prompt |
| Label | `p_hat` = share of the prompt's 20 responses that violated this rule | `q_hat` = mean of `p_hat` over the prompt's k rules |
| Meaning | per-rule violation probability | expected fraction of rules violated |
| TRAIN rows per generator | 175,446 | 42,066 |
| Training loss | binomial negative log-likelihood (n = 20) | mean squared error on a sigmoid output |

**Two generators.** `meta-llama/Llama-3.1-8B-Instruct` and `Qwen/Qwen3-8B` (thinking disabled). Each has its own responses, labels, internal features and fitted models. Nothing is pooled across them.

**Unit of independence.** The dataset is built from 72 *groups* (three tasks sharing one pool of six rules). Prompts inside a group are strongly related, so every split, cross-validation fold and bootstrap resamples whole groups, never single prompts.

---

## 2. How the dataset was built

Code: [`manifest.py`](../src/mife/manifest.py), [`tasks.py`](../src/mife/tasks.py), [`rule_sampler.py`](../src/mife/rule_sampler.py), [`feasibility.py`](../src/mife/feasibility.py), [`design.py`](../src/mife/design.py), [`source.py`](../src/mife/source.py). All seeds are in [`configs/experiment.yaml`](../configs/experiment.yaml).

### 2.1 Source

- **Benchmark:** ManyIFEval, repository `kenoharada/Multiple-Instructions-Following` at commit `5c1ba29e…`, with its copy of Google's IFEval instruction code.
- **Tasks:** all 216 task descriptions in `data/manyifeval.jsonl`. None is dropped or sampled.
- **Rules:** the benchmark's 15 instruction types, each with a deterministic official verifier.
- **Check:** the build first confirms that our prompt serialiser reproduces all 2,160 released ManyIFEval prompts byte for byte (`source.verify_source_reproduction`).

The 15 types, the six categories used in this project, and how many of the 72 groups contain each type:

| Type ID | Category | Label | TRAIN groups | TEST groups |
|---|---|---|---|---|
| `change_case:capital_word_frequency` | letter_case | Uppercase word count | 15 | 3 |
| `change_case:english_capital` | letter_case | Uppercase | 20 | 6 |
| `change_case:english_lowercase` | letter_case | Lowercase | 13 | 3 |
| `detectable_content:number_placeholders` | formatting | Placeholder | 29 | 8 |
| `detectable_format:number_bullet_lists` | formatting | Bullet points | 28 | 8 |
| `detectable_format:title` | formatting | Title | 31 | 6 |
| `keywords:existence` | keywords | Include keywords | 16 | 6 |
| `keywords:forbidden_words` | keywords | Forbidden keywords | 31 | 9 |
| `keywords:frequency` | keywords | Keyword occurrence count | 20 | 6 |
| `keywords:letter_frequency` | length_counts | Specific letter count | 32 | 9 |
| `length_constraints:number_paragraphs` | length_counts | Paragraph count | 21 | 4 |
| `length_constraints:number_sentences` | length_counts | Sentence count | 14 | 4 |
| `length_constraints:number_words` | length_counts | Word count | 16 | 4 |
| `punctuation:no_comma` | punctuation | Comma prohibition | 30 | 8 |
| `startend:quotation` | beginning_ending | Quotation marks | 26 | 6 |

The group counts differ by type because of the compatibility rule in § 2.3: types that belong to a mutually exclusive set appear less often.

### 2.2 Task families and groups of three

- **Families.** Each task has one of six *proposed* family labels: INFO 35, CREATIVE 52, PERSUASIVE 33, PRACTICAL 59, ANALYTICAL 22, TRANSFORM 15. They were assigned from the task text before any generation, following [`task_family_rubric.md`](task_family_rubric.md). They are not official ManyIFEval labels.
- **Grouping.** `tasks.form_groups` forms 72 groups of three tasks with a greedy rule: each group draws from the three families that have the most tasks left (seed 20261110). All 72 groups end up with three different families.
- **Why groups of three.** The same rules and orderings are applied to three different tasks, which allows a clean "same rules, different task" comparison later.

### 2.3 One pool of six exact rules per group

`rule_sampler.sample_pool` builds each group's pool with a seed derived from the group ID:

1. **Pick types.** Propose a type uniformly at random from those still allowed. Accept it only if the set can still be completed to six compatible types.
2. **Compatibility.** Two types conflict if they are in the same mutually exclusive set. The sets are {existence, frequency}, {the three `change_case` types}, and {number_words, number_sentences, number_paragraphs}. This is the rule implied by the released ManyIFEval data (it reproduces all 18 type lists found there).
3. **Draw parameters.** Call the official instruction class's `build_description()` with no arguments, so the released code alone defines the ranges:

   | Type | Parameters drawn by the official code |
   |---|---|
   | existence, forbidden_words | 2 distinct words from the code's word list |
   | keyword frequency | 1 word, count 1–3, "less than" or "at least" |
   | letter_frequency | a lowercase letter, count 1–10, relation |
   | number_words | 100–500, relation |
   | number_sentences | 1–20, relation |
   | number_paragraphs | 1–5 |
   | number_placeholders | 1–4 |
   | number_bullet_lists | 1–5 |
   | capital_word_frequency | 1–20, relation |
   | title, english_capital, english_lowercase, quotation, no_comma | none (fixed wording) |

4. **Feasibility.** The partial pool is checked by `feasibility.check_pool`. A draw is rejected only when it is provably impossible (for example "fewer than 1 sentence"), and then that type's parameters are drawn again. A pool is accepted only when a constructed "witness" response passes all six official verifiers. In this manifest one parameter draw was rejected and no type was rejected.
5. **Task feasibility.** The witness is re-verified with each of the group's three task texts (216 of 216 group–task pairs pass). There is no relevance filter: an unusual but possible pairing, such as a tweet with a title, is kept.

Results: 72 × 6 = 432 rule instances, all inside the code's default parameter ranges (`data/manifest/pool_audit.json`). The six instances are fixed within their group and sampled independently for every other group. Every proposal and rejection is logged in `data/manifest/rule_pools.jsonl`.

### 2.4 Which rule sets and orderings each task gets

`design.py` turns a group's six rules into 246 ordered rule lists:

- **Chains.** The six rules are put in one random order (seed per group). Six *chains* are the six cyclic rotations of that order, each cut to five rules.
- **Subsets.** The first k rules of a chain form a subset, for k = 1…5. That gives six different subsets at every k, and each rule appears in exactly k of the six subsets of size k. Prefixes are nested, so the k-subset of a chain is the (k+1)-subset minus one rule.
- **Orderings.** Each subset is shown in several orders taken from a fixed, balanced set of permutations (a strength-two component orthogonal array):

  | k | Orderings per subset | Construction | Prompts per task | Prompt–rule rows per task |
  |---|---|---|---|---|
  | 1 | 1 | identity | 6 | 6 |
  | 2 | 2 | both orders | 12 | 24 |
  | 3 | 6 | all 6 permutations | 36 | 108 |
  | 4 | 12 | the 12 even permutations | 72 | 288 |
  | 5 | 20 | the 20 maps i → a·i + b mod 5 | 120 | 600 |
  | **Total** | | | **246** | **1,026** |

- **Balance.** For k ≥ 2, within a subset every rule appears in every position exactly k − 1 times, and every ordered pair of rules appears at every pair of positions exactly once. The design does not cover all k! orders for k ≥ 4.
- **Reuse.** The same 246 ordered lists are applied to all three tasks of the group.

### 2.5 Prompt text

A prompt is the task text, a fixed header, and one bullet per rule, exactly as ManyIFEval serialises it. Example (`g01-c1-k3-o00-t1005`, k = 3):

```
Write a resume for a fresh high school graduate who is seeking their first job.

Your response should follow the instructions below:
- There should be 3 paragraphs. Paragraphs are separated with the markdown divider: ***
- Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.
- In your response, the letter i should appear less than 9 times.
```

The prompt ID encodes the design cell: group `g01`, chain 1, k = 3, ordering row 0, task 1005. The manifest stores each rule's character span inside the prompt, which is later mapped to token spans.

### 2.6 Train/test split

- **Sizes.** 57 TRAIN groups (171 tasks, 42,066 prompts) and 15 TEST groups (45 tasks, 11,070 prompts). There is no validation split.
- **How it was chosen** (`tasks.split_groups`, seed 20261113). 20,000 random candidate splits were scored on how closely the test set's task-family and rule-type shares match the whole dataset. The best-scoring admissible split was kept. A split is admissible only if every type is in at least 1 TEST group and at least 5 TRAIN groups.
- **Related tasks.** Eight clusters of near-duplicate tasks (for example two Japan-trip tasks) are forced onto the same side.
- **No outcome was used.** The split was fixed before any response was generated.
- **Exact-rule overlap.** 50 of the 90 TEST rule instances have the same type and parameters as some TRAIN rule. The five parameter-free types always do. So TEST measures generalisation to unseen tasks and unseen rule pools, not to rules whose wording was never seen.

### 2.7 Response generation

Code: [`generate_vllm.py`](../src/mife/generate_vllm.py), [`chat.py`](../src/mife/chat.py). Run ID `main`.

| | Llama 3.1 8B Instruct | Qwen3-8B |
|---|---|---|
| Revision | `0e9e39f2…` | `b968826d…` |
| Chat template | default | default with `enable_thinking=False` |
| Temperature / top-p / top-k | 0.6 / 0.9 / off | 0.7 / 0.8 / 20 |
| Max new tokens | 2,048 | 2,048 |
| Input tokens (median / max) | 137 / 216 | 114 / 193 |
| Responses | 20 per prompt, 1,062,720 in total | 20 per prompt, 1,062,720 in total |

- **Prompt format.** Each prompt is one user turn, rendered with the model's own chat template and tokenised once. Generation receives those exact token IDs. No prompt is truncated.
- **Decoding values** are each model card's defaults.
- **Backend.** vLLM 0.25.0 on NVIDIA B200 GPUs (Penn's Betty cluster), bfloat16 weights, no quantisation. The sampler applies temperature, then top-k, then top-p. Each model ran as 12 chunk jobs with one engine seed per chunk.
- **Why 20 responses.** They turn a single pass/fail outcome into an estimated violation probability.

### 2.8 Verification and labels

Code: [`verifier.py`](../src/mife/verifier.py), [`verify_stage.py`](../src/mife/verify_stage.py).

- **Verifier.** Every included rule is checked on every response with the official checker. A rule counts as followed only if the response is non-empty and `check_following` passes. An empty response violates every rule (the benchmark's own behaviour).
- **Determinism.** `langdetect` is seeded (seed 0) and the NLTK sentence tokenizer is vendored and hashed, so labels are reproducible.
- **Per response:** a vector of k values, 1 = violated.
- **Labels:**
  - `v` = number of the 20 responses that violated the rule; `n` = 20; `p_hat = v / n` (Objective 1).
  - `q_hat` = mean of `p_hat` over the prompt's k rules (Objective 2). It is the expected *fraction* of rules violated, not the probability that any rule is violated.
- **Nothing is dropped.** Truncated responses (2,048-token limit) and repetition loops stay in the labels. `RESULTS.md` reports how common they are and how the rates change without them.
- **Known verifier weakness.** The official `english_capital` checker requires `langdetect` to return English, and `langdetect` often labels all-uppercase English as German. See § 8.

### 2.9 Stored model features (computed before generation)

Code: [`introspect.py`](../src/mife/introspect.py). One forward pass per prompt and generator, on the prompt only.

- **Position.** Everything is read at the *final prompt token*: the last token of the rendered chat prompt, after the assistant header. This is the state from which the first response token is sampled.
- **Hidden states.** The 4,096-dimensional hidden state at that token for every layer. Stored rows: row 0 = embeddings; rows 1…L = output of layers 1…L (row L is after the final normalisation); row L+1 = output of the last layer *before* the final normalisation. L = 32 for Llama, 36 for Qwen.
- **Attention.** For every layer and head (32 × 32 for Llama, 36 × 32 for Qwen), the attention from the final token to each rule's exact token span, stored two ways: the SUM over the span's tokens and the MEAN (sum ÷ span length). Up to 5 rule slots in prompt order.
- **Backend.** Hugging Face transformers with eager attention, bfloat16, on Apple MPS. Only these `main` MPS features are used. This is a different backend from generation (vLLM on CUDA), so the stored states approximate, but do not bit-for-bit equal, the states that produced the responses. The pilot comparison found hidden-state cosine ≥ 0.998 between MPS and CUDA.

### 2.10 Counts

| | Per task | Per group | TRAIN | TEST | All |
|---|---|---|---|---|---|
| Tasks | 1 | 3 | 171 | 45 | 216 |
| Prompts | 246 | 738 | 42,066 | 11,070 | 53,136 |
| Prompt–rule rows | 1,026 | 3,078 | 175,446 | 46,170 | 221,616 |
| Responses per generator | 4,920 | 14,760 | 841,320 | 221,400 | 1,062,720 |

---

## 3. What a configuration is

Code: [`configs.py`](../src/mife/probing/configs.py), [`features.py`](../src/mife/probing/features.py).

A configuration is a list of *feature blocks* that are concatenated and fed to a small prediction head. The 23 configurations are every combination of the five components below that the plan allows.

### 3.1 The five components

| Component | Idea | Uses the generator's internals? |
|---|---|---|
| **H1** | The generator's hidden state at the final prompt token | Yes |
| **H2** | H1 plus the text embedding of the exact rule(s) | Yes |
| **A1** | The generator's attention from the final prompt token to the rule span(s) | Yes |
| **B1** | Prompt-only baseline: frozen BGE-M3 text embeddings plus structural features | No |
| **B2** | The same inputs as B1, with BGE-M3 fine-tuned on the task | No |

### 3.2 Feature blocks

| Block | Content | Width | Used for |
|---|---|---|---|
| `H` | Hidden state at the selected layer row | 4,096 | H1, H2 |
| `A` | Attention to the *target* rule's span, all layers × heads | 1,024 (Llama), 1,152 (Qwen) | A1, Objective 1 |
| `A5` | Attention to all 5 rule slots in prompt order (absent slots zero) plus a 5-value presence mask | 5,125 (Llama), 5,765 (Qwen) | A1, Objective 2 |
| `EP` | Frozen BGE-M3 embedding of the full prompt text | 1,024 | B1 |
| `ER` | Frozen BGE-M3 embedding of the exact target rule text | 1,024 | B1, H2 (Objective 1) |
| `ER5` | Frozen embeddings of all rules in 5 ordered slots, zero-padded | 5,120 | B1, H2 (Objective 2) |
| `EP_B2`, `ER_B2`, `ER5_B2` | The same three, from the fine-tuned (B2) encoder | as above | B2 |
| `S1` | Structure of a (prompt, rule) row | 75 | B1, B2, structure reference (Objective 1) |
| `S2` | Structure of a prompt | 175 | B1, B2, structure reference (Objective 2) |
| `M2` | Rule-type metadata of a prompt (a subset of `S2`) | 101 | H1, H2 (Objective 2) |
| `KM` | k and its one-hot encoding (the first 6 columns of `M2`) | 6 | A1 (Objective 2) |

**BGE-M3 embedding.** The model's standard dense output: the final-layer [CLS] state, L2-normalised (`encoder.py`). Inputs are the raw prompt text and rule text, never truncated.

**`S1` columns (75).** k and its one-hot (6); target type one-hot (15); position, position ÷ k, is-first, is-last (4); the target's typed parameters (13); which companion types are present (15); companion parameters summed (13); task family one-hot (6); prompt length in BGE tokens, prompt characters, task characters (3).

**`S2` columns (175).** k and its one-hot (6); type one-hot for each of 5 slots (75); typed parameters for each slot (65); slot presence (5); which types are present (15); task family (6); the three lengths (3).

**`M2` columns (101).** `S2` without parameters, task family and lengths: k and one-hot (6), slot types (75), slot presence (5), types present (15). It gives H1 the rule *types and order* but no exact rule text or parameters.

**Typed parameters (13 columns).** Eight numeric columns hold log(1 + value) for the eight types that have a number. Five columns hold the relation as +1 ("less than") or −1 ("at least"). Everything else is zero. Keywords and the letter of a letter-count rule are *not* in the structure block; they reach a model only through the rule-text embedding.

**Never an input:** exact rule IDs, group IDs, task IDs.

### 3.3 Which blocks each component contributes

| Component | Objective 1 (one row per prompt–rule) | Objective 2 (one row per prompt) |
|---|---|---|
| H1 | `H` | `H` + `M2` |
| H2 | `H` + `ER` | `H` + `M2` + `ER5` |
| A1 | `A` | `A5` + `KM` |
| B1 | `EP` + `ER` + `S1` | `EP` + `ER5` + `S2` |
| B2 | `EP_B2` + `ER_B2` + `S1` | `EP_B2` + `ER5_B2` + `S2` |

Why the Objective 2 versions differ: one prediction must cover all of the prompt's rules, so the model needs to know which rules are present and in what order. H1 gets that from `M2`, A1 from the presence mask and `KM`, and B1/B2 from `S2`.

**Combining components.** A combination uses the union of its components' blocks. A block that appears twice is kept once. A block whose columns are all inside another block that is present is dropped (`KM` ⊂ `M2` ⊂ `S2`). Frozen and fine-tuned embeddings are different blocks, so `B2+H2` contains both `ER_B2` and the frozen `ER`.

### 3.4 The 23 configurations and the learned reference

| Configuration | Family | Objective 1 blocks | Width (Llama / Qwen) | Objective 2 blocks | Width (Llama / Qwen) | Objective 1 heads |
|---|---|---|---|---|---|---|
| H1 | probe-only | H | 4,096 | H + M2 | 4,197 | 15 per-type |
| H2 | probe-only | H + ER | 5,120 | H + M2 + ER5 | 9,317 | 15 per-type |
| A1 | probe-only | A | 1,024 / 1,152 | A5 + KM | 5,131 / 5,771 | 15 per-type |
| H1+H2 | probe-only | H + ER | 5,120 | H + M2 + ER5 | 9,317 | 15 per-type |
| H1+A1 | probe-only | H + A | 5,120 / 5,248 | H + M2 + A5 | 9,322 / 9,962 | 15 per-type |
| H2+A1 | probe-only | H + ER + A | 6,144 / 6,272 | H + M2 + ER5 + A5 | 14,442 / 15,082 | 15 per-type |
| H1+H2+A1 | probe-only | H + ER + A | 6,144 / 6,272 | H + M2 + ER5 + A5 | 14,442 / 15,082 | 15 per-type |
| B1 | B1 family | EP + ER + S1 | 2,123 | EP + ER5 + S2 | 6,319 | 1 shared |
| B1+H1 | B1 family | EP + ER + S1 + H | 6,219 | EP + ER5 + S2 + H | 10,415 | 15 per-type |
| B1+H2 | B1 family | EP + ER + S1 + H | 6,219 | EP + ER5 + S2 + H | 10,415 | 15 per-type |
| B1+A1 | B1 family | EP + ER + S1 + A | 3,147 / 3,275 | EP + ER5 + S2 + A5 | 11,444 / 12,084 | 15 per-type |
| B1+H1+H2 | B1 family | EP + ER + S1 + H | 6,219 | EP + ER5 + S2 + H | 10,415 | 15 per-type |
| B1+H1+A1 | B1 family | EP + ER + S1 + H + A | 7,243 / 7,371 | EP + ER5 + S2 + H + A5 | 15,540 / 16,180 | 15 per-type |
| B1+H2+A1 | B1 family | EP + ER + S1 + H + A | 7,243 / 7,371 | EP + ER5 + S2 + H + A5 | 15,540 / 16,180 | 15 per-type |
| B1+H1+H2+A1 | B1 family | EP + ER + S1 + H + A | 7,243 / 7,371 | EP + ER5 + S2 + H + A5 | 15,540 / 16,180 | 15 per-type |
| B2 | B2 family | EP_B2 + ER_B2 + S1 | 2,123 | EP_B2 + ER5_B2 + S2 | 6,319 | 1 shared |
| B2+H1 | B2 family | EP_B2 + ER_B2 + S1 + H | 6,219 | EP_B2 + ER5_B2 + S2 + H | 10,415 | 15 per-type |
| B2+H2 | B2 family | EP_B2 + ER_B2 + S1 + H + ER | 7,243 | EP_B2 + ER5_B2 + S2 + H + ER5 | 15,535 | 15 per-type |
| B2+A1 | B2 family | EP_B2 + ER_B2 + S1 + A | 3,147 / 3,275 | EP_B2 + ER5_B2 + S2 + A5 | 11,444 / 12,084 | 15 per-type |
| B2+H1+H2 | B2 family | EP_B2 + ER_B2 + S1 + H + ER | 7,243 | EP_B2 + ER5_B2 + S2 + H + ER5 | 15,535 | 15 per-type |
| B2+H1+A1 | B2 family | EP_B2 + ER_B2 + S1 + H + A | 7,243 / 7,371 | EP_B2 + ER5_B2 + S2 + H + A5 | 15,540 / 16,180 | 15 per-type |
| B2+H2+A1 | B2 family | EP_B2 + ER_B2 + S1 + H + ER + A | 8,267 / 8,395 | EP_B2 + ER5_B2 + S2 + H + ER5 + A5 | 20,660 / 21,300 | 15 per-type |
| B2+H1+H2+A1 | B2 family | EP_B2 + ER_B2 + S1 + H + ER + A | 8,267 / 8,395 | EP_B2 + ER5_B2 + S2 + H + ER5 + A5 | 20,660 / 21,300 | 15 per-type |
| REF_structure | reference | S1 | 75 | S2 | 175 | 1 shared |

The widths match the `n_features` column of `cv_selected.csv`. Objective 2 always uses one shared head.

### 3.5 Configurations that are identical by construction

Because H1 and H2 share one hidden-state row, and B1 already contains the rule embedding that H2 adds, several configurations have exactly the same inputs:

- H2 = H1+H2
- H2+A1 = H1+H2+A1
- B1+H1 = B1+H2 = B1+H1+H2
- B1+H1+A1 = B1+H2+A1 = B1+H1+H2+A1
- B2+H2 = B2+H1+H2
- B2+H2+A1 = B2+H1+H2+A1

This holds in both objectives. So there are 15 distinct feature sets among the 23 names. Within a set, linear heads give identical results, and MLP heads differ only through their random seed. Small differences between such rows in the tables (for example B1+H1 0.3429 vs B1+H2 0.3415, Llama Objective 1) are seed noise, not a finding. All 23 are kept because the plan lists 23.

### 3.6 One head per rule type, or one shared head

- **Objective 1, any configuration containing H1, H2 or A1:** 15 separate heads, one per instruction type. A (prompt, rule) row is routed to the head of its rule's type, and that head is trained only on rows of that type. Parameter variants of a type (for example "at least 3 sentences" and "fewer than 12 sentences") share one head. There is never a head per exact rule.
- **Objective 1, B1 alone, B2 alone, structure reference:** one shared head. The rule is identified to the head through the type one-hot in `S1` and the rule embedding.
- **Objective 2, every configuration:** one shared head.

Consequence for reading the tables: in Objective 1, B1 (one shared head) and B1+H1 (15 per-type heads) differ in two ways at once, the added hidden state and the per-type routing.

### 3.7 The four reference rows

They are not among the 23. They show how far simple information goes.

| Row | Prediction | Learned? |
|---|---|---|
| `REF_constant` | The overall TRAIN violation rate (Objective 2: the mean TRAIN `q_hat`) | No |
| `REF_type` | The TRAIN violation rate of the rule's type (Objective 2: averaged over the prompt's rules) | No |
| `REF_type_k` | The TRAIN violation rate of the rule's type at that k (Objective 2: averaged over the prompt's rules) | No |
| `REF_structure` | A head on the structure block alone (`S1` or `S2`), tuned like every configuration | Yes |

`REF_structure` is the fairest "no text, no model internals" comparator: it knows the rule types, parameters, positions, k, task family and lengths, but not the wording and not the generator's state. In cross-validation the three rate references use the fitting folds' rates only.

---

## 4. How every configuration was fitted

Code: [`cv.py`](../src/mife/probing/cv.py), [`heads.py`](../src/mife/probing/heads.py), [`train.py`](../src/mife/probing/train.py), [`b2.py`](../src/mife/probing/b2.py).

### 4.1 Folds

The 57 TRAIN groups are shuffled once (seed 20261201) and cut into five folds of 12, 12, 11, 11 and 11 groups (`results/cv/folds.json`). A fold's prompts are held out together; the other 45 or 46 groups are the *fitting* groups. No group is ever split across folds.

### 4.2 The prediction head

Every configuration uses the same two candidate head shapes on its concatenated features:

| | Linear | MLP |
|---|---|---|
| Shape | features → 1 logit | features → 128 (ReLU) → 1 logit |
| Optimiser | full-batch L-BFGS, up to 500 iterations | Adam, learning rate 0.001, batches of 1,024 rows |
| Training length | to convergence | up to 30 epochs; the epoch count is chosen in CV |
| Start | weights zero; warm-started from the next stronger penalty | random (seeded) |

Common to both:

- **Output.** A sigmoid of the logit, so predictions are in (0, 1).
- **Loss, Objective 1.** Binomial negative log-likelihood of `v` violations in `n` = 20 responses, summed over rows and divided by the number of responses. This equals the mean cross-entropy per (response, rule) check.
- **Loss, Objective 2.** Mean squared error between the sigmoid output and `q_hat`.
- **Penalty.** (λ / 2) · ‖W‖² over weight matrices (not biases), added to the mean loss. λ ∈ {0.01, 0.1, 1}.
- **Standardisation.** Each head standardises every feature column using its own fitting rows only (a constant column keeps scale 1). For per-type heads this is done per type.
- **Bias start.** The output bias starts at the logit of the fitting rows' base rate.
- **Seeds.** MLP seeds are derived from (seed 20261202, generator, objective, configuration, fold, type, head kind, λ).

### 4.3 The grid and the selection rule

For each configuration, 2 head shapes × 3 penalties = **6 settings** are fitted in each of the 5 folds.

1. For each fold, fit on the fitting groups and predict the held-out groups. For per-type configurations this is 15 fits per fold and setting; all 15 heads use the same setting.
2. Score each fold's held-out predictions with the objective's loss, pooling all rows (and all rule types) of that fold.
3. **CV loss** = the plain average of the five fold losses. Folds get equal weight.
4. For an MLP setting, held-out predictions are recorded after every epoch. The epoch (1–30) with the lowest CV loss is that setting's epoch. One epoch is chosen per configuration, not per type or fold.
5. The setting with the lowest CV loss is selected.

Every candidate's out-of-fold predictions are saved in `results/predictions/{model}/obj{o}/cv/`, and every setting's loss in `results/tables/cv_grid_all.csv`.

### 4.4 Staged choices (per generator and objective)

To keep the search small and equal across configurations, two choices are made once and then reused:

| Stage | Candidates | Decided by | Reused by |
|---|---|---|---|
| Hidden-state layer | 50%, 75%, 100% depth (Llama rows 16, 24, 33; Qwen rows 18, 27, 37) | the H1 configuration's best CV loss | every configuration containing H1 or H2 |
| Attention pooling | SUM or MEAN over the rule span | the A1 configuration's best CV loss | every configuration containing A1 |

"100% depth" is the last layer's output before the final normalisation, so all three candidates are the same kind of state.

Selected values:

| Generator | Objective | Hidden row | Attention pooling | B2 setting (LoRA) |
|---|---|---|---|---|
| Llama | 1 | 16 (50%) | SUM | LR 1e-4, 1 epoch |
| Llama | 2 | 16 (50%) | SUM | LR 2e-4, 3 epochs |
| Qwen | 1 | 18 (50%) | SUM | LR 1e-4, 1 epoch |
| Qwen | 2 | 27 (75%) | SUM | LR 2e-4, 1 epoch |

The `key` column in the tables records these choices: `B1+H1+A1__h16__asum` means hidden row 16 and SUM pooling.

### 4.5 B2: fine-tuning the text encoder

B2 asks whether a text encoder trained on this exact task can match the generator's internal signals. It is fitted in two steps.

**Step 1: adapt the encoder (on Betty, NVIDIA B200).**

- **Model.** BGE-M3 plus one linear head on [prompt embedding, rule embedding(s), structure block], trained end to end on the objective's loss.
- **Method.** LoRA: rank 16, alpha 32, dropout 0.05, on the query, key, value and dense layers.
- **Why LoRA.** A check fixed in advance compared full fine-tuning and LoRA once (Llama, Objective 1, fold 0, one epoch, each method's lower learning rate). LoRA had the lower held-out loss (0.4141 vs 0.4173) and is used for every B2 run.
- **Training.** AdamW; batches of 32 prompts; weight decay 0.01 on the encoder; head learning rate 0.001; 5% linear warm-up, then linear decay.
- **Grid.** Learning rate {1e-4, 2e-4} × epochs {1, 3}, each run separately in each of the 5 folds, for both objectives and both generators: 80 runs.
- **Fold hygiene.** In fold f the encoder sees only that fold's fitting groups. It is never pre-trained on all 57 groups.
- **Selection.** The grid point with the lowest average held-out loss of this jointly trained head (table `cv_b2_grid`).
- **Inputs.** The jobs received a bundle containing TRAIN prompts and labels only, so they could not touch TEST.

**Step 2: treat the adapted encoder like B1.**

- The selected encoder of fold f embeds every TRAIN prompt and rule text. These embeddings are the blocks `EP_B2`, `ER_B2`, `ER5_B2` for fold f.
- B2 and all seven B2+probe configurations then go through the same 6-setting head grid as everything else, using fold f's embeddings in fold f.
- Note the asymmetry inside a fold: the fitting rows' embeddings come from an encoder that was trained on those rows, while the held-out rows' embeddings are out-of-sample. This mirrors how the final model is used (encoder and head trained on TRAIN, applied to TEST).

So `RESULTS.md` has two different B2 numbers: `cv_b2_grid` (the jointly trained linear head, used only to choose learning rate and epochs) and `cv_selected` row `B2` (the common head grid on the frozen adapted embeddings, comparable to every other row).

### 4.6 Final models

- Each configuration's selected setting is refitted on all 57 TRAIN groups (`train.py`), with the MLP trained for its CV-selected number of epochs. Saved in `results/models/{model}/obj{o}/`.
- B2 is refitted once per generator and objective on all 57 groups at its selected setting. Its TRAIN embeddings and LoRA adapter are saved, so TEST prompts can later be embedded by the same frozen encoder.
- The rate references are recomputed from all 57 groups.
- These final models are what will be applied, once, to TEST.

### 4.7 What ran where

| Stage | Where | Notes |
|---|---|---|
| Frozen BGE-M3 embeddings, structure blocks, hidden/attention caches | Mac (MPS / CPU) | TRAIN only |
| Head cross-validation and final head refits | Mac (CPU) | about 4.4 h per generator for Objective 1 and 2.2 h for Objective 2 |
| B2 encoder adaptation (80 LoRA grid runs, 1 full fine-tuning check run, 4 final runs) | Betty (B200) | about 6.0 GPU-hours for the grid and check, 0.3 for the final runs, summed over jobs |

Frozen BGE-M3 embeddings computed on Betty match the Mac's on the 8 prompts checked (table `b2_encoder_xcheck`), so B1 and B2 start from the same encoder.

---

## 5. Reading `results/RESULTS.md`

`RESULTS.md` is generated by [`report.py`](../src/mife/probing/report.py) from the CSV files in `results/tables/`. No number is typed by hand. Table names below are the CSV names.

### 5.1 Conventions used throughout

- **Violation rate.** Σv ÷ Σn over the rows in a cell: the share of (response, rule) checks that failed.
- **`estimate [95% CI]`.** The interval is a percentile interval from a bootstrap that resamples the 57 TRAIN groups with replacement (2,000 replicates, seed 20261203). It reflects variation between groups.
- **`n_groups`.** How many TRAIN groups contribute to the cell.
- **`sum_n` / `sum_one`.** The denominator: the number of (response, rule) checks or responses in the cell. For example k = 1 has 57 groups × 3 tasks × 6 prompts × 1 rule × 20 responses = 20,520.
- **Model names.** `llama` and `qwen`.

### 5.2 Section 3: descriptive results (TRAIN only)

Produced by [`describe.py`](../src/mife/probing/describe.py).

| Table | What it shows | How to read it |
|---|---|---|
| `desc_by_k` | Violation rate for prompts with k rules, and overall | Whether rules get broken more often as more are added |
| `desc_by_type` | Violation rate per rule type | `n_groups` is small for some types (13–32), so intervals are wide |
| `desc_by_category` | The same, per category | |
| `desc_by_position` | Violation rate of the rule at each position, within k | The ordering design balances which rule sits where, so differences reflect position |
| `desc_by_family` | Violation rate per task family | `n_groups` counts groups containing a task of that family |
| `desc_label_distribution` | Share of rows with v = 0 and v = 20, mean labels, quantiles of `q_hat` | How skewed the labels are. Quantile rows have no interval |
| `desc_truncation` | Share of responses that hit the 2,048-token limit, by k | |
| `desc_english_capital` | How many `english_capital` violations come only from the language-detection step | See § 8 |
| `desc_loops` | Repetition loops: share of all responses, overlap with truncation | Loop = the last 256 output tokens repeat with some period ≤ 128 at ≥ 90% of positions; only responses of ≥ 384 tokens are examined |
| `desc_cross_model_type` | Llama minus Qwen violation rate per type | Paired over the same groups |
| `desc_cross_model_corr` | Spearman correlation of the two generators' `q_hat` across prompts | Whether the same prompts are hard for both |
| `desc_type_by_k` | Type × k violation rates | Point estimates; intervals are in `desc_type_by_k_long.csv` |

### 5.3 Section 4: cross-validation (TRAIN only)

Produced by [`cvsummary.py`](../src/mife/probing/cvsummary.py). **Every loss here is a CV loss as defined in § 4.3: Objective 1 binomial NLL per response, Objective 2 MSE. Lower is better. Losses are comparable within one generator and objective only.**

| Table | What it shows | Columns worth knowing |
|---|---|---|
| Summary bullets, `cv_summary` | Per generator and objective: best configuration overall and per family, B1 and B2 alone, the references | `n_of_23_below_REF_structure` = how many of the 23 beat the structure reference |
| `cv_folds` | The five folds | `min_fitting_rows_per_type`, `min_fitting_groups_per_type`: the least data any per-type head had |
| `cv_layer_selection` | H1's best CV loss at each candidate hidden row | `selected` marks the row reused everywhere |
| `cv_pooling_selection` | A1's best CV loss with SUM and MEAN | |
| `cv_b2_grid` | B2's jointly trained head at each learning rate × epochs | Used only to pick the B2 setting (§ 4.5) |
| `cv_selected` | **The main table**: one row per configuration and reference | see below |
| `cv_complementarity` | H + A1 combined versus the better of the two alone | negative `combined_minus_better_single` = combining helps |
| `cv_l2_counts` | How many rows selected each λ | |
| `cv_l2_edges` | Rows whose selected λ is the smallest or largest in the grid | `gap_pct` = how much worse the next λ inwards is |
| `cv_boundary_hits` | Every candidate (including unselected layers and poolings) whose λ or MLP epoch is at an edge | |
| `cv_identical_features` | The identical sets of § 3.5 | |
| `cv_sparse_types` | Per-type heads fitted with fewer than 50 violated responses in a fold | Those heads have almost no positive examples |

**Columns of `cv_selected`:**

| Column | Meaning |
|---|---|
| `family` | probe-only, B1 family, B2 family, or reference |
| `key` | configuration plus the hidden row and pooling it uses |
| `setting` | the selected head: `linear` or `mlp`, and λ. "fitting-fold rates" for the rate references |
| `epoch` | selected MLP epoch count (blank for linear) |
| `cv_loss` | the selected setting's CV loss |
| `fold_sd` | standard deviation of the five fold losses. It is mostly "some folds are harder than others", which affects all rows alike, so it overstates the uncertainty of a *difference* between two rows |
| `n_features` | input width (§ 3.4) |
| `per_type_heads` | True = 15 per-type heads |
| `minus_ref_structure` | `cv_loss` minus the structure reference's; negative = better than the reference |

**How to read a row.** Llama, Objective 1, `A1__asum`, `mlp_l2=0.01`, epoch 9, `cv_loss` 0.3316: fifteen per-type MLP heads on SUM-pooled attention (1,024 inputs each), λ = 0.01, trained for 9 epochs, reach an average held-out NLL of 0.3316 per response. The constant reference scores 0.4705 and the structure reference 0.3779 on the same rows.

**Cautions for § 4.**

- These are not test results. They carry mild optimism because the best of 6 settings (and of 3 layers, 2 poolings) is reported.
- No significance test has been run. Significance is assessed on TEST, with the pre-specified contrasts.
- Identical configurations (§ 3.5) are not independent evidence.
- Some selections sit at the edge of the λ grid, including the structure reference in all four cells (§ 8).

### 5.4 Sections 5–7: test evaluation (pending)

These will be filled once, after unblinding, exactly as fixed in [`results/prespec/analysis_plan.md`](../results/prespec/analysis_plan.md). Definitions are in [`metrics.py`](../src/mife/probing/metrics.py), [`evaluate.py`](../src/mife/probing/evaluate.py) and [`controlled.py`](../src/mife/probing/controlled.py).

**Section 5, Evaluation 1** (all 11,070 TEST prompts):

| Objective | Metrics |
|---|---|
| 1 | Binomial NLL per response (primary); Brier score; calibration error (ECE); AUROC; RiskTop / RiskBottom = observed violation rate in the top / bottom tenth of predicted risk. Overall, per type (with the mean over the 15 types) and per k |
| 2 | RMSE (primary); MAE; calibration error; Pearson and Spearman correlation. Overall and per k |

- Calibration bins are the deciles of each configuration's own TRAIN out-of-fold predictions, so they are fixed before TEST is read.
- Intervals: bootstrap over the 15 TEST groups, 10,000 replicates, the same replicates for every configuration.

**Section 6, hypothesis contrasts.** 37 paired comparisons per generator and objective, each "metric(b) − metric(a)" on the same test rows; negative favours b.

| Family | Question | Comparisons |
|---|---|---|
| C1 | Do internal signals alone beat prompt-only baselines? | B1 or B2 vs H1, H2, A1 |
| C2 | Does adding the hidden state to a baseline help? | B1 → B1+H1, B1+H2; same for B2 |
| C3 | Does adding attention to a baseline help? | B1 → B1+A1; B2 → B2+A1 |
| C4 | Are hidden state and attention complementary? | better of (H, A1) → H+A1; baseline+H → baseline+H+A1; baseline+A1 → baseline+H+A1 |
| C5 | Does the exact rule text help beyond the hidden state? | H1 → H2 and three analogues |
| C6 | Does fine-tuning the encoder help? | B1 → B2, with and without probes |
| Reference | Do probe-only configurations beat structure alone? | REF_structure → each of the 7 probe-only |

Each contrast gets a 95% interval and a two-sided bootstrap p-value, Holm-adjusted within its family, generator and objective.

**Section 7, Evaluation 2** asks whether models predict *changes* in risk between matched prompts:

| Analysis | Pair | What is held fixed |
|---|---|---|
| A position | the same rule at two positions | task, rule set, k |
| B count | k rules vs the same k plus one more | task, chain, order of the shared rules |
| C composition | one companion rule swapped | task, k, target rule and its position, all other positions |
| D task | the same ordered rules on two tasks | the whole rule list |
| E difficulty | Evaluation 1 by easy / medium / hard types | terciles fixed from TRAIN k = 1 rates |

Scores: mean absolute error between predicted and observed differences, and sign accuracy on pairs whose observed difference is reliably non-zero. Pairs are built from design metadata only, so the matching cannot depend on outcomes.

### 5.5 Section 8: sensitivity analyses

Currently TRAIN descriptives only:

- `sens_truncation`: violation rate by k with truncated responses removed.
- `sens_english_capital`: violation rate by k without the `english_capital` rule.

After unblinding, the primary test metrics are also recomputed without truncated responses, without `english_capital`, and without prompts that have a looping response.

### 5.6 Section 9: compute and runtime

- `b2_benchmark`, `b2_projection`: measured B2 speed on the Mac and the projected wall time. All scenarios with the specified grid exceeded 24 hours, which is why B2 ran on Betty.
- `b2_betty_check`: the full-vs-LoRA check.
- `b2_encoder_xcheck`: frozen BGE-M3 on Mac vs Betty.
- `runtime`: wall time of each completed stage. CV stages are summed over grid files; the four generator × objective processes ran in parallel.

### 5.7 Section 10: open issues

One row per discrepancy, deviation or flag (`open_issues.csv`, text in [`issues.py`](../src/mife/probing/issues.py)). § 8 below explains the ones that affect interpretation.

---

## 6. Safeguards

| Safeguard | Mechanism |
|---|---|
| TEST cannot be read early | Every loader goes through `data._splits`, which raises `TestSetLocked` for TEST until `FROZEN.json` has `unblind_approved: true`. `tests/test_no_test_leakage.py` checks loaders, caches, the B2 bundle and saved CV outputs |
| Stale or mixed data | Every stored output carries hashes of the prompt text and rendered prompt; readers refuse a mismatch (`integrity.py`). The feature reader refuses anything but the `main` MPS run |
| Plan fixed before TEST | `analysis_plan.md` is hashed in `FROZEN.json`; unblinding refuses to proceed if the plan changed |
| Numbers traceable | `RESULTS.md` is rebuilt from `results/tables/*.csv` only |
| Reproducibility | Pinned source commit, model revisions and seeds; `results/run_manifest.json` records data hashes |

---

## 7. Decision log

"Plan" = the research plan, the decision tracker and the task prompt `PROMPT_probing_and_results.md`. "Implementation" = a choice made while writing the code, where the plan was silent or left details open. "Project owner" = a decision the project owner made when asked.

### Dataset

| Decision | What was done | Why | Source |
|---|---|---|---|
| Tasks | All 216 | Full coverage of the benchmark | Plan |
| Grouping | Greedy, maximise family diversity | Makes the within-group task comparison span different task kinds | Plan; algorithm is implementation |
| Compatibility | ManyIFEval's implied exclusive sets | Matches the released data; the IFEval registry's rule differs | Project owner (audit finding D1) |
| Parameters | Official `build_description()` defaults | Explicit, seed-reproducible ranges; no invented distribution | Project owner (D2) |
| Feasibility | Reject only provable contradictions; require a passing witness | Avoids impossible prompts without filtering by difficulty | Project owner (D3) |
| Task fit | Feasibility only, no relevance filter | The verifiers never read the task; the benchmark itself pairs freely | Plan |
| Orderings | Balanced permutation sets (1, 2, 6, 12, 20) | Position balance without all k! orders | Plan |
| Split | 57 / 15 groups by balance search; related tasks together | Unseen-task test without near-duplicate leakage | Plan; search procedure is implementation |
| Decoding | Model-card defaults, 2,048 new tokens | Typical use; room for "at least 500 words" | Project owner (D4) |
| Empty responses | Violate every rule | The benchmark's own semantics | Benchmark |

### Features

| Decision | What was done | Why | Source |
|---|---|---|---|
| Read position | Final prompt token | The state from which generation starts | Plan |
| Hidden-layer candidates | 50%, 75%, 100% depth | Small, even coverage of depth | Plan |
| "100% depth" | Last layer's output before the final normalisation | Keeps all three candidates the same kind of state | Implementation ([`issues.py`](../src/mife/probing/issues.py)) |
| One shared layer | Chosen on H1, reused by H2 and all combinations | Keeps the H1 vs H2 comparison fair and the search small | Plan |
| Attention features | All layers × heads, no selection | Lets the head find useful heads; avoids a second search | Implementation |
| Pooling | SUM vs MEAN, chosen on A1, reused | Tests sensitivity to rule length | Plan |
| Objective 2 inputs | 5 ordered slots, zero padding, presence mask | Keeps order and membership for a prompt-level prediction | Plan |
| H1's Objective 2 metadata (`M2`) | Rule types and order, no parameters or text | H1 should not receive exact-rule information; that is H2's role | Plan ("ordered rule-type information"); exact columns are implementation |
| A1's Objective 2 extras | Presence mask and k | Lets the head tell an absent slot from zero attention | Mask from the plan; k is implementation |
| Structure block | Types, k, position, typed parameters, companions, family, lengths | Everything knowable without reading the text; no IDs | Plan; encoding is implementation |
| Parameter encoding | log(1 + value), relation as ±1 | Compresses a 1–500 range; sign carries the direction | Implementation |
| Block union | Duplicates once; subsets dropped | Avoids feeding the same columns twice | Implementation |

### Models and tuning

| Decision | What was done | Why | Source |
|---|---|---|---|
| Head shapes and λ grid | Linear or MLP(128); λ {0.01, 0.1, 1}; 30 epochs | A small grid shared by all configurations | Plan |
| Per-type heads | 15 heads for any Objective 1 configuration with a probe | Pools parameter variants; transfers to new exact rules | Plan |
| One setting for all 15 heads | Head shape, λ and epoch chosen per configuration on pooled loss | Per-type tuning would multiply the search and overfit small types | Implementation |
| Penalty scale | (λ/2)·‖W‖² added to the *mean* loss, after standardisation, biases free | The plan gives the grid but not the scale | Implementation (recorded in the pre-specification) |
| Standardisation | Per head, on fitting rows only | Prevents leakage from held-out rows | Plan |
| Linear optimiser | L-BFGS to convergence, warm-started along λ | Deterministic and exact for a convex problem | Implementation |
| MLP optimiser | Adam 0.001, batch 1,024 | Standard defaults | Implementation (values in the config file) |
| MLP epoch | Chosen from per-epoch held-out loss, averaged over folds | Early stopping without a validation split | Plan |
| CV score | Mean held-out-fold loss, folds weighted equally | Simple; folds are nearly equal in size | Plan; equal weighting is implementation |
| Folds | 12 / 12 / 11 / 11 / 11 groups | Grouped CV inside TRAIN | Plan |
| References | Constant, type, type × k, structure head | Separate "base rates", "structure" and "text or internals"; the structure head must appear in every table | Plan |
| Sparse-type flag | Keep and flag per-type fits with < 50 violated responses | Makes weak per-type heads visible without dropping them | Plan |

### B2

| Decision | What was done | Why | Source |
|---|---|---|---|
| Where it runs | Betty | The Mac projection exceeded 24 hours | Project owner |
| Full vs LoRA | One pre-fixed check; LoRA won | The plan says choose one and hold it fixed | Plan; check rule is implementation |
| LoRA details | Rank 16, alpha 32, dropout 0.05, query/key/value/dense | Rank and dropout from the plan; alpha = 2 × rank is a common default | Plan / implementation |
| Grid | LR {1e-4, 2e-4} × epochs {1, 3}, separate runs | As specified | Plan |
| Training schedule | Batches of 32 prompts, head LR 0.001, 5% warm-up then linear decay | Standard fine-tuning practice | Implementation |
| Selection | On the jointly trained head's held-out loss | Chooses the encoder before any fusion head is seen | Implementation |
| Fusion | Freeze the selected encoder, then the common head grid | B2+probe gets the same head budget as B1+probe | Plan |
| Stored embeddings | float16 | Fits the cluster's storage quota; precision loss is far below feature scale | Implementation |

### Evaluation (fixed, not yet run)

| Decision | What was done | Why | Source |
|---|---|---|---|
| Bootstrap unit | Groups (57 TRAIN for descriptives, 15 TEST for evaluation) | Prompts within a group are dependent | Plan |
| Exact bootstrap | Metrics computed from per-group sums, so each replicate is exact | No approximation from resampling prompts | Implementation |
| Calibration bins | Deciles of TRAIN out-of-fold predictions | Fixed before TEST is read | Plan requires bins from TRAIN; deciles are implementation |
| Primary metrics | NLL (Objective 1), RMSE (Objective 2) | Match the training losses | NLL from the plan; RMSE is implementation |
| Contrasts | 37 fixed comparisons in seven families | Each answers one question of the plan | Plan lists the questions; the exact pairs are implementation |
| Multiple testing | Holm within family × generator × objective | Controls false positives per question | Implementation (recorded in the pre-specification) |
| "Better of H, A1" | Decided by TRAIN CV loss | Cannot be chosen on TEST | Implementation |
| Evaluation 2 matching | From design metadata only; strict composition matches | Fixed before outcomes are seen | Plan; algorithms are implementation |
| Reliable pairs | Confidence interval of the observed difference excludes 0 | Sign accuracy is meaningless for noise-level differences | Plan requires a rule fixed in advance; the criterion is implementation |
| Repetition loops | Reported, never dropped; definition in § 5.2 | Keeps the primary labels unchanged | Plan; the definition is implementation |

---

## 8. Known limitations and open points

1. **Penalty grid edges.** Several selections are at the smallest or largest λ, and the structure reference is at the smallest λ for both generators and objectives. Its true optimum may lie outside the grid, which could make probes look better relative to it. The same λ is also about 20 times stronger relative to the Objective 2 loss than to the Objective 1 loss, because the MSE is much smaller than the NLL. Widening the grid (training data only) is an open decision for the project owner. Tables: `cv_l2_counts`, `cv_l2_edges`.
2. **`english_capital` labels.** In TRAIN, 12,554 of Qwen's 29,097 `english_capital` violations (and 1,281 of Llama's 30,467) are all-uppercase English responses that `langdetect` did not label English, mostly as German. None contains Chinese, Japanese or Korean characters, and none is mainly in a non-Latin script. The official labels are kept; the pre-specified sensitivity analysis drops the rule. A sensitivity analysis that keeps the rule and counts these cases as followed is not in the frozen plan. Tables: `audit_ec_*.csv`; script: [`audits/english_capital.py`](../audits/english_capital.py).
3. **Identical configurations.** Only 15 of the 23 feature sets are distinct (§ 3.5). Some contrasts therefore compare identical inputs or repeat another contrast; they are kept and counted in the Holm adjustment.
4. **Per-type routing is confounded with adding a probe** in Objective 1 (§ 3.6).
5. **Hidden state + attention in Objective 1.** In CV, H1+A1 is worse than A1 alone for both generators. All blocks share one penalty, so the 4,096-wide hidden state can swamp the 1,024-wide attention block. No block-specific penalty was added.
6. **Rare violations for some types.** Qwen almost never violates the title rule, so in one fold its per-type heads were fitted with a single violated response (`cv_sparse_types`).
7. **Qwen truncation and loops.** About 2.5% of Qwen's TRAIN responses hit the token limit and about 2.1% are repetition loops. They are kept in the labels and covered by sensitivity analyses.
8. **Fifteen test groups.** Test intervals will rest on 15 independent blocks, so per-type and per-k breakdowns will be imprecise.
9. **Test rules are not all unseen.** 50 of 90 test rule instances match a TRAIN rule exactly (§ 2.6).
10. **Feature backend differs from the generation backend** (§ 2.9).
11. **Task families are proposed labels**, not benchmark annotations.
12. **Code hash recorded at the freeze.** One wording change to the report text in `report.py` was made a minute after freezing, so the current code hash differs from the one in `FROZEN.json`. Undoing that single edit reproduces the frozen hash. The analysis-plan hash, which is the one checked at unblinding, still matches.

---

## 9. Where things are

| What | Where |
|---|---|
| Prompts, rules, pools, split | `data/manifest/` |
| Tokenised prompts and rule token spans | `data/tokenized/` |
| Responses | `data/generations/{model}/main/` |
| Verifier outputs and labels | `data/verification/{model}/main/` |
| Hidden states and attention | `data/features/{model}/main/` |
| Cached feature blocks (TRAIN) | `data/probing_features/shared/`, `data/probing_features/{model}/` |
| B2 inputs, runs and adapters | `data/probing_features/b2_bundle/`, `data/probing_features/b2_runs/` |
| Folds, per-configuration grids, selections | `results/cv/` |
| Out-of-fold predictions | `results/predictions/{model}/obj{o}/cv/` |
| Final fitted heads | `results/models/{model}/obj{o}/` |
| Tables behind `RESULTS.md` | `results/tables/` |
| Frozen plan and its hash | `results/prespec/` |
| Interactive explorer | `notebooks/results_explorer.ipynb` |
| File formats | [`storage_schema.md`](storage_schema.md), [`probing_schema.md`](probing_schema.md) |

## 10. Glossary

| Term | Meaning |
|---|---|
| Group | Three tasks that share one pool of six rules; the unit of splitting and resampling |
| Pool | A group's six exact rules |
| k | Number of rules in a prompt (1–5) |
| Position | A rule's place in the prompt's rule list (1 = first) |
| Chain | One of six rotations of the pool's order; its first k rules form a subset |
| Ordering row (`coa_row`) | Which balanced permutation of the subset the prompt uses |
| `config_id` | Group + chain + k + ordering; shared by the group's three tasks |
| `v`, `n`, `p_hat` | Violations, responses (20), and v ÷ n for one prompt–rule pair |
| `q_hat` | Mean `p_hat` over a prompt's rules |
| Fitting / held-out groups | The groups a model is trained on / scored on within one CV fold |
| CV loss | Average held-out loss over the five folds |
| Out-of-fold prediction | A prediction for a TRAIN row made by a model that did not see its group |
| Head | The small linear or MLP model on top of the features |
| λ (`l2`) | Strength of the weight penalty |
| Probe | H1, H2 or A1: a model on the generator's internal states |
| Baseline | B1 or B2: a model on prompt text and structure only |
| Reference | One of the four `REF_` rows; not among the 23 |
