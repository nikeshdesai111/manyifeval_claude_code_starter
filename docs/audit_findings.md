# Pre-implementation audit — ManyIFEval source vs research plan v3.4

Date: 2026-11-09.

## Decisions (approved by user, 2026-11-09)
- **D1 compatibility:** ManyIFEval-implied rule. Mutually exclusive groups {existence, frequency}, {capital_word_frequency, english_capital, english_lowercase}, {number_words, number_sentences, number_paragraphs}; all other pairs (including quotation+title) are compatible.
- **D2 parameter space:** IFEval's code-default pool. Each exact rule's parameters are one call of the official instruction class's `build_description()` with no arguments (pinned ManyIFEval copy of the IFEval code), seeded from the group's RNG. The official code alone defines ranges and distributions: two distinct `WORD_LIST` words for keyword lists; one `WORD_LIST` word with U{1..3} for keyword frequency; a lowercase letter a–z with U{1..10} for letter frequency; U{100..500} words; U{1..20} sentences; U{1..5} paragraphs; U{1..4} placeholders; U{1..5} bullets; U{1..20} capitalised words; every relation "less than"/"at least" 50/50; the five remaining types have fixed wording. See M2. There is no frozen instance list. Rationale in M2.
- **Additional source finding:** tasks 201 and 2825 include `number_sentences` "less than 1" in their released prompts. That rule is unsatisfiable, because any non-empty response has ≥1 punkt sentence and the official wrapper fails every rule for an empty response. The same value is in the code-default pool (U{1..20} with "less than"); D3 rejects it as a provable contradiction and the type's parameters are resampled.
- **D3 feasibility:** reject only provable contradictions, each confirmed by constructing a witness response that passes all official verifiers. No difficulty filtering. A rejected parameter draw is resampled from the code-default pool (up to 25 tries before the type is rejected). Rejection rates per type are reported in `data/manifest/pool_audit.json`; over 1,500 sampler seeds they are number_sentences 2.7% (only "less than 1"), letter_frequency 1.3% (a "less than" letter count that the required keywords already reach), existence 0.7% and keyword frequency 0.2% (a required keyword that is also forbidden, or forced past a "less than" frequency), and 0 for the other types.
- **Task–rule fit:** feasibility only. A group's rule pool must be satisfiable for each of its three tasks; no relevance or appropriateness filter (see M3).
- **D4 decoding:** model-card defaults. Llama 3.1: T=0.6, top_p=0.9. Qwen3 non-thinking: T=0.7, top_p=0.8, top_k=20, min_p=0. max_new_tokens=2048 for both; cap-hit rate is reported.

## Pinned source

| Item | Value |
|---|---|
| Repo | https://github.com/kenoharada/Multiple-Instructions-Following |
| Commit | `5c1ba29e631e2bbee981ebd620898c2750c5ddbf` (2025-09-15, "ready for release"; no tags) |
| Data | `data/manyifeval.jsonl`: 2,160 rows = 216 tasks × k=1..10 nested prompts |
| Verifiers | `instruction_following_eval/` (Google IFEval code, Apache-2.0) + `utils.test_instruction_following` |
| Llama snapshot (cached, full BF16 weights) | `0e9e39f249a16976918f6564b8830bc894c89659` |
| Qwen3-8B snapshot (cached: tokenizer/config only; weights downloading) | `b968826d9c46dd6066d109eabc6255188de91218` |

## Confirmed facts (match the plan)

- **216 distinct task prompts** (`key`/`task_prompt`), each with 10 nested prompts (k=1..10).
- **15 instruction types**, all with deterministic verifiers in the released code.
- Each task carries a `rules` dict with **one** exact instantiation of all 15 types, and a 10-type `not_conflict_rules_list`.
- **Prompt serialization reproduces all 2,160 released prompts exactly**:
  `task_prompt.strip() + "\n\nYour response should follow the instructions below:\n" + "\n".join("- " + build_description(**kw).replace("\n", " "))`.
  The newline flattening matters for the bullet-list rule text. 4 tasks have leading/trailing whitespace in `task_prompt`.

## Mismatches / ambiguities

### M1. Compatibility logic: two sources disagree (needs decision D1)
- The **IFEval registry** (`INSTRUCTION_CONFLICTS` + `conflict_make`) has these conflicts among the 15 types: paragraphs↔sentences; capital_word_frequency↔english_capital↔english_lowercase (all pairs); quotation↔title. That gives a maximum compatible set of 11.
- **ManyIFEval's released data follows a different, implicit rule** that no released code implements. All 216 tasks use exactly 18 = 2×3×3 type lists:
  - 7 types are always present: forbidden_words, letter_frequency, number_placeholders, title, number_bullet_lists, no_comma, quotation.
  - Plus exactly one of {existence, frequency}, one of {the 3 change_case types}, and one of {number_words, number_sentences, number_paragraphs}.
  - So ManyIFEval **forbids** existence+frequency and any two length constraints, which IFEval allows. It also **allows** quotation+title, which IFEval forbids.
- The generation script that produced these lists is **not released**.

### M2. Parameter space: resolved, IFEval's code-default pool (decision D2)
Each ManyIFEval task carries one released argument per listed type: 216 tasks × 10 types = **2,160 (task × type) arguments**.
- **1,994 (92.3%)** are draws from IFEval's code defaults, i.e. `instructions.py` `build_description()` called with no arguments.
- **166 (7.7%)** were copied from each task's original hand-written IFEval prompt. The provenance split comes from comparing the arguments with the original IFEval prompts' kwargs, which are not part of the pinned repository.
- The 166 copied values are the source of every out-of-range value. A range check over the pinned data (reproducible from `data/manyifeval.jsonl`) finds **90 arguments outside the code-default pool**, all among them; the other 76 copied values happen to fall inside it.

| Parameter | IFEval code-default pool | Out-of-pool released values |
|---|---|---|
| num_words | U{100..500} | 31, 50 ("less than"); 600–1,200 ("at least") |
| num_sentences | U{1..20} | 30–50 ("at least") |
| let_frequency | U{1..10}, letter a–z | 20, 60 ("at least") |
| keyword frequency | U{1..3}, one `WORD_LIST` word | 4–8 ("at least"); 19 keywords outside `WORD_LIST` |
| num_placeholders | U{1..4} | 5–20 |
| num_bullets | U{1..5} | 6–9 |
| num_paragraphs | U{1..5} | 6 |
| existence keywords | 2 distinct `WORD_LIST` words | lists of 2–6 words, all with words outside `WORD_LIST` |
| forbidden words | 2 distinct `WORD_LIST` words | lists of 1–12 words; 17 with words outside `WORD_LIST` |

Out of range by type (of 216 each): forbidden_words 23, keyword frequency 22, existence 13, number_words 11, placeholders 8, sentences 5, bullets 4, letter_frequency 2, paragraphs 2.

**Decision:** sample only from the released code's default generators, by calling the official code (`rule_sampler.sample_params`). Do not sample from the empirical distribution of the published parameter values, and do not construct one (for example, uniform between the observed minimum and maximum).

Rationale:
1. **Reproducibility.** The released generators have explicit distributions and ranges, and are reproduced exactly with fixed seeds.
2. **No invented distributions.** The process that produced the published out-of-range values is not documented, so any distribution built around them would be our assumption, not the benchmark's.
3. **Methodological consistency.** We are building a new, controlled benchmark on ManyIFEval's instruction types, templates and verifiers, not reproducing the published dataset's empirical distribution.
4. **Compatibility and feasibility.** Default draws are filtered only by D1 (type compatibility) and D3 (provable contradictions, confirmed by a witness). Rejected draws are resampled from the same generator. Each group's full proposal / rejection / resampling log, including the proof for every rejection, is stored in `data/manifest/rule_pools.jsonl`. Per-type rejection counts and the rejected draws are in `data/manifest/pool_audit.json`. Filtering shifts the final distribution slightly (see D3), and that shift is what the logs make measurable.

**Scope of the claim:** we follow the released code's default procedure. We do not claim it reproduces the distribution the ManyIFEval authors used for the published dataset; the 166 hand-copied arguments show that their process differed.

Conformance is audited at every manifest build (`data/manifest/pool_audit.json`; 432/432 exact rules in the pool).

### M3. Parameter-level and task-level compatibility: resolved (D3 and task feasibility)
- The source never checks parameters against each other. Task 1381's `rules` dict pairs forbidden_words ['active','ring'] with existence ['active','owner'], a direct contradiction. It never reaches a released prompt only because that task's type list happened to pick `frequency` instead of `existence`. Independent sampling would hit such cases unless we add checks.
- **Parameter level:** resolved by D3 (proofs of contradiction, a witness for every accepted pool, and resampling).
- **Task level:** the source has no task-applicability logic; every task instantiates all 15 types. The released prompts pair tasks and rules freely. Examples:
  - a "catchy tweet" (1857) with "at least 417 words";
  - a "tweet storm" (3534) with "at least 20 sentences";
  - tasks that quote or complete a comma-containing sentence (1928, 2107, 3703) with no-comma, title and quotation rules.
- **Decision: feasibility only.** A group's rule pool must not be impossible for any of its tasks; whether a rule is a natural fit for a task is not checked.
  - None of the 15 official verifiers reads the task text (`source.types_reading_task_text()` is empty), so satisfiability is task-independent.
  - The manifest build still confirms it for every group–task pair by running the official verifiers on the pool's witness with that task's text. All 216 of 216 pairs pass; the result is recorded under `task_feasibility` in `manifest_summary.json`.
  - Unusual but possible pairings are kept. Examples in the manifest:
    - a "catchy tweet" (1857, g71) with a title, quotation marks and no commas;
    - a "concise and a long" cover-letter task (1551, g29) with "at least 330 words";
    - tasks that quote or complete a comma-containing sentence (1928, g51; 2107, g54) with "no commas";
    - "finish the sentence" (3703, g70) with a title and "fewer than 231 words".

### M4. Decoding settings are "Open" in the tracker (needs decision D4)
The tracker lists decoding settings as an open decision. The output budget must fit the longest code-default length rule, "at least 500 words" (~700 tokens), with margin.

### Minor (defaults I'll apply unless told otherwise)
- **Taxonomy label:** the plan's "Length/counts: … character count" type is `length_constraints:number_words` (a **word** count). "Specific character count" = `keywords:letter_frequency`. I'll store the plan's 6-category label, the code-prefix category (7 prefixes) and the type ID.
- **Sentence verifier:** `count_sentences` loads the NLTK punkt *pickle*. It currently resolves from `~/nltk_data`, outside the project, and NLTK downloads fail on this Mac (SSL certificate error). I'll vendor a copy into the project, record its SHA-256, and pin the `nltk` version. Sentence-count labels depend on this file.
- **Verifier semantics:** `test_instruction_following` marks every rule as violated when the response is empty or whitespace. I'll keep the official semantics and also log a separate `empty_response` flag.
- **Related tasks:** the 216 tasks include near-variants (e.g. Japan itinerary vs Japan trip blog). I'll cluster near-duplicates and keep each cluster on one side of the train/test split.
- **Task text with embedded constraints:** some tasks contain format or length requests. I'll flag these for human review in the task-family file, not drop them.
- **Hardware/backends:** MLX for generation (BF16 converted from the same safetensors, **no quantization**). PyTorch-MPS eager BF16 for hidden states and attention. Pilot compares logits and token IDs between the two.
