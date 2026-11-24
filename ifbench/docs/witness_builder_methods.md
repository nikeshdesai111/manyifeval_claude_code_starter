# The IFBench witness builder: methods and design record

This document is the single reference for how the IFBench witness builder works and why it was built that way. It covers the rule study the builder relies on, how it relates to the ManyIFEval feasibility check, each step of the construction, the tests, every fix made during the stage-3 audit together with the failure that caused it, and worked examples. The paper can cite it as the place where this reasoning is recorded.

- Code: `ifbench/src/mife_ifb/witness.py` (builder), `lexicon.py` (word inventory), `plain_meaning.py` (frozen plain-meaning checks), `source.py` (the only route to IFBench code), `stage3_proofs.py` (written proofs).
- State described: branch `claude/project-thread-we2kwj`. First written at commit `c0c2e7a` (2026-11-23) without changing code or results; updated on 2026-11-24 for the version rule (Section 7.1), the thesis fix (fix 17), combined layouts (Section 5.6, fix 18) and the overlap copy route (Section 5.7, fixes 19–21), all approved by Nicky.
- Companion reports: `docs/stage1_report.md`, `docs/plain_meaning_checks.md`, `docs/stage2_report.md`, `docs/stage2_builder_limits.md`, `docs/stage3_report.md`.
- Section 10 lists two findings made while writing this document. The first has since been fixed (fix 17); the second needs no action.

---

## 1. What a witness is and why the design needs one

The IFBench adaptation copies the ManyIFEval design: groups of three tasks share one freshly sampled pool of six rules, and prompts are built from subsets of that pool (six cyclic chains, k = 1..5, COA orderings). The design only makes sense if every prompt is *possible*. A model that fails a rule should have failed something it could have done.

A **witness** is one response, built by the program, that passes **every rule in the set at once**:

1. the official IFBench verifier for each rule (pinned commit `48891866`), and
2. the frozen **plain-meaning check** for each rule (Section 3.3).

If a witness exists for all six rules, then every subset of the pool is satisfiable too. Every prompt in the design uses a subset, so one witness per pool covers all 246 ordered configurations of that pool.

There are two levels of feasibility, both taken from ManyIFEval:

- **D1, type compatibility.** Can two rule *types* appear in the same pool at all? Stage 3 settled this with a pairwise audit. Each *version* of a pair (one setting of its parameters) gets exactly 24 builder attempts: one version when both rules are fixed, every parameter combination when there are fewer than 50, and 50 parameter draws otherwise (Section 7.1). A pair is compatible if any version yields a witness.
- **D3, pool feasibility.** Is this particular six-rule pool, with its exact sampled parameters, satisfiable? In stage 5 the sampler will ask the builder for one joint witness for the whole pool and redraw parameters or types when none is found.

Three asymmetries shape everything below:

- **A witness proves feasibility. A failure proves nothing.** When the builder fails, the pair is investigated. It ends up either as a **proven conflict** (a written argument) or as **unsupported by the builder**. Neither kind is co-sampled, and both types stay in the inventory. This was decided on 2026-11-22.
- **Witnesses are held to a stricter standard than model outputs.** The plain-meaning checks apply to witnesses only. Labels for generated responses always come from the official verifiers alone. The stricter standard stops a pool from being declared feasible on the strength of a degenerate text (Section 3.3).
- **Witnesses need not be good writing.** A witness is a certificate, not an example answer. It does not have to answer the task or read naturally. It has to pass the checks, without loopholes that the plain-meaning checks rule out.

---

## 2. Relation to ManyIFEval's D3 and `src/mife/feasibility.py`

### 2.1 What the ManyIFEval builder did

`src/mife/feasibility.py` (242 lines) is the D3 check used to build the frozen ManyIFEval manifest. It has three parts:

- **`proofs(pool)`**: five hand-written unsatisfiability rules for IFEval parameters. P_SENT0 is "less than 1 sentence". P_KW_FORBIDDEN is a required keyword that is also forbidden. P_FREQ_LESS covers existence keywords that force a frequency cap to be broken. P_LETTER_LESS is the same idea for letters. P_WORDS_LESS covers required tokens that exceed a word cap. A pool with any proof is rejected.
- **`_build_witness(pool, rng, compact, per)`**: fills sentences with words from a ~170-word filler list. It inserts the required content (keywords at their frequencies, capitalised words, placeholders) and pads for word and letter minimums. It then adds the layout: a title in `<< >>`, `*` bullets, paragraphs split by `***`, all upper case, and wrapping quotes.
- **`check_pool`**: tries 24 witnesses that cycle through four styles (6, 1, 15 or 30 words per sentence, with "compact" every fourth attempt). It accepts the first one that passes every official verifier. If there is neither a witness nor a proof, it raises `UnresolvedFeasibility`, which is treated as a pipeline bug. `check_tasks` then reruns the verifiers on the witness with each task's text.

The ManyIFEval builder could work with fixed templates because the 15 IFEval types mostly constrain **counts and containers**: how many words, sentences, bullets, placeholders or capitals, which keywords, which case. Padding and wrapping satisfy them independently, so a fixed recipe with a few styles was enough.

### 2.2 What was reused unchanged

- **The acceptance rule.** A pool is feasible only if a constructed text passes every official verifier with the pool's actual parameters. No probability estimate, no judge.
- **Seeded attempts.** Up to 24 attempts per call, each with its own seed, so every result can be reproduced exactly.
- **Joint witness.** One witness for the whole pool, which then covers every subset. Nicky confirmed on 2026-11-22: "a joint six-rule witness, not per-window".
- **Task feasibility.** As in ManyIFEval, a group's pool must be satisfiable for each of its three tasks, with no relevance filter. A rule that is merely unusual for a task is kept.
- **Default parameter draws.** Parameters come from the released code's default generators, as in ManyIFEval (`source.sample_default_kwargs`, seeded).

### 2.3 What had to change, and why

| ManyIFEval | IFBench builder | Why |
|---|---|---|
| A fixed recipe: fill, pad, wrap | Plan, then a backtracking word search, then render, repair and verify | About half of IFBench's types constrain the **shape of every word** or the **relation between neighbouring words**: alphabet order, prime lengths, consonant clusters, three vowels, alternating syllable parity, no shared first letter, alliteration runs, last word = next first word. A padded template breaks these at the first filler word, so the words have to be chosen to satisfy them. |
| A ~170-word filler list | IFBench's own `WORD_LIST` (1,525 words) plus a 291-word documented supplement. After stop words and pronouns are removed, 1,708 filler words remain. | Word-shape filters remove most of a small list. For example, a word for every letter of the alphabet and enough prime-length or cluster words are needed. Section 4.2 describes the supplement. |
| Official verifiers only | Official verifiers **plus** frozen plain-meaning checks (witnesses only) | Several IFBench verifiers accept degenerate texts. For example, `ratio:sentence_type` passes with no sentences of either kind, and `count:conjunctions` passes with "and", "And" and "and," counted as three. Section 3.3 has the details. |
| `proofs()` evaluated on every pool | Proofs written once per **type pair** in stage 3 (`stage3_proofs.py`, 44 pairs, each with its basis), plus `Infeasible` reasons inside the planner for parameter-level impossibilities | IFBench conflicts are mostly between types, not parameter values. A written per-pair argument can be reviewed and cited, and the library facts it relies on are tested. |
| No witness and no proof → raise `UnresolvedFeasibility` (stop) | No witness and no proof → log as "unsupported by the builder" and do not co-sample | Decided 2026-11-22. With 42 types, a few hard combinations would otherwise stall sampling, and stopping hides how often it happens. Unsupported pairs are listed with reasons (`docs/stage3_report.md`). |
| No task text read by any verifier | `ratio:overlap` reads the task text | The witness must be built per task: a separate six-rule witness for each of the group's three tasks when the pool contains overlap. The builder takes `task_text` and fits the overlap to it (Section 5.4). |

---

## 3. The rule study: a code-level audit of every rule (stage 1)

**For the methods section.** Stage 1 was a code-level audit of every IFBench rule, reviewed and approved by the authors.

### 3.1 Procedure

For each of the 58 out-of-distribution types at the pinned commit, I read the instruction class: `build_description` (wording and default parameter generator) and `check_following` (the verifier). For each type I recorded:

1. **What the verifier actually reads.** Is it whitespace tokens (`value.split()`), IFBench's `_split_words` (lower-case, punctuation deleted, whitespace split), NLTK `word_tokenize`, `\w+` counts, NLTK `sent_tokenize` sentences, `'\n'` lines, or raw characters? This decides what counts as a "word" or a "sentence" for that rule. The builder has to model each rule on its own terms, because the definitions differ between rules.
2. **The default parameter ranges**, from 200 seeded draws (table in `docs/stage1_report.md`).
3. **Degenerate passes.** Texts the verifier accepts that do not do what the instruction says. These became the plain-meaning checks.
4. **Whether the rule reads the task** (only `ratio:overlap`) or the prompt (the three `repeat:*` types).

Outcome, reviewed by Nicky:

- 16 types are excluded: 11 `custom:*` (each is a complete task with a fixed answer), 3 `repeat:*` (they read the prompt), `format:options` (one fixed answer conflicts with everything), and `count:words_japanese`. That last one was dropped after review: its default generator crashes, and its wording drops "word" for N ending in 1, 2 or 3.
- That leaves **42 candidates** and **861 pairs**.

### 3.2 Findings about the verifiers that the builder depends on

These library behaviours are easy to miss. Each one changed how the builder writes text.

| Finding | Consequence for the builder |
|---|---|
| NLTK Punkt (`sent_tokenize`) only ends a sentence at `.`, `?` or `!` **followed by whitespace**. "Apples.Pears.Plums" is one sentence. | With `format:no_whitespace`, the text is always one sentence. This is the basis of 15 of the proofs. The builder joins tokens with commas. |
| A text-final `?!` is split by Punkt into two sentences, `?` and `!`. | The `?!` sentence needed by `count:punctuation` is never placed last, unless the extra `!` sentence is wanted (`ratio:sentence_type` + punctuation adds the `?!` and `!` sentences at the end on purpose). |
| Punkt does **not** split at `‽` inside a sentence, and `count:punctuation` accepts `‽` as the interrobang. | When exactly three sentences are required (`ratio:sentence_words`), the interrobang goes inside a sentence as `‽`, so no extra sentence is created. |
| `"<i>x</i>."` ends a sentence for Punkt, but `"<i>x.</i> Y"` does not split before Y. | The thesis's terminal mark goes after `</i>`, so the thesis is a sentence of its own, except under `words:last_first`, where it stays inside the tag (fix 17, Section 10.1). |
| `nltk.word_tokenize` splits `<` and `>` off, leaving the tag letters `i` and `em` as lower-case tokens. | `format:title_case` with `format:thesis` is impossible (proof). |
| Most word-shape verifiers (`words:consonants`, `words:prime_lengths`, `words:odd_even_syllables`, `words:alphabet`) read **every whitespace token as a word**, including a lone emoji, a list marker or a bullet `*`. | Under shape rules, emoji and markers are attached to a word with no space ("tight" markers, Section 5.3). |
| `syllapy` ignores emoji, and an ordinal such as "3rd" counts as 0 syllables (even parity). | An emoji attached to a word does not change its syllable parity. Ordinals count as even-parity words. |
| `count:numbers` counts `\d+` runs, and the shape verifiers treat "3rd" as a word with letters. | Under shape rules, numbers are written as ordinals ("3rd", "21st"). Under `words:alphabet` no number is possible (proof). |
| `count:conjunctions` counts **distinct raw tokens**, so "and", "And" and "and," are three. | A plain-meaning check folds case and punctuation (Section 3.3). The builder uses distinct conjunctions. |
| `words:words_position` and `words:start_verb` use `word_tokenize` plus the NLTK POS tagger. | The verb is pinned first. The tagger can still read it as a noun in some contexts, so verification is the final judge (about 1 draw in 20 needs more attempts, per stage 2). |
| `format:line_indent` needs strictly increasing indentation on non-blank lines. `words:vowel` needs exactly one line. | These and other line counts decide which layouts can be combined (proofs in Section 7). |

### 3.3 Plain-meaning checks (frozen before any witness was built)

Each candidate type has one check, frozen with sha256 in `freeze/plain_meaning_frozen.json`. The table is in `docs/plain_meaning_checks.md`.

- **G1 (every type):** the response is not blank and has at least three runs of letters.
- **29 types add a condition.** Each comes with a tested loophole example that the official verifier accepts and the check rejects. Examples:
  - `ratio:sentence_type` needs at least one question and two statements, so "no sentences of either kind" does not satisfy 2:1.
  - `count:conjunctions` counts conjunctions after folding case and punctuation.
  - `words:alphabet` requires letters-only words and at most 20% single letters, so "A b c d" does not count.
  - `ratio:overlap` needs at least 5 distinct trigrams.
  - `sentence:increment`, `words:last_first` and `sentence:alliteration_increment` need at least 2 sentences, so the cross-sentence rule is actually exercised.
  - `words:paragraph_last_first` needs at least three words per paragraph.
  - `format:thesis` needs the response to begin with an italic thesis of at least three words.
  - `format:sub-bullets` needs every `*` bullet followed by a `-` line.
- **The other 13 types rely on G1 plus the official verifier.**

These checks matter for the builder in two ways:

- They raise the bar the builder must clear.
- About half of the stage-3 proofs (26 of 44) depend on one of them. Those pairs could be satisfied by a loophole text, and the proofs say so (basis "plain").

---

## 4. Why code and not a language model

Nicky decided on 2026-11-22: "Do not use an LLM witness generator for now… Use a code-based, programmatic witness builder, as we did for ManyIFEval." The reasons recorded in the adaptation plan are:

1. **A verdict is not a proof.** An LLM judge saying "these rules are compatible" can be wrong. The project's labels come from deterministic verifiers, so feasibility has to be decided by the same verifiers on a concrete text. An LLM judge was considered when the plan was written and rejected.
2. **Same method as ManyIFEval.** The ManyIFEval manifest used a programmatic witness. Keeping that method means the two datasets' feasibility guarantees mean the same thing.
3. **Reproducibility.** Every witness is a pure function of the rules, the parameters, the task text and the seed. An LLM adds sampling noise and a model version to pin.
4. **Cost and scale.** Stage 3 alone ran about 84,000 pair draws across the full and final runs, plus 600 six-rule pools. Most builder calls take under a second on a CPU (median 0.1 s per pair over 50 draws).

The alternative left open in the plan was that an LLM might **draft** candidate texts for the hard cases, with the official verifiers and plain-meaning checks still deciding. This is deferred, not adopted. Nothing in the current builder uses a model.

### 4.1 Why a search and not a constraint solver or random sampling

- **Random sampling** of filler words fails almost always under word-shape rules. For example, the chance that 40 random words are in alphabet order is about 26^-39. So the builder has to construct, not sample.
- **A general constraint solver (SAT or SMT)** would need every verifier encoded as constraints, including Punkt sentence splitting, the NLTK tokenizer, the POS tagger, syllapy and the emoji library. Those encodings could not be checked against the real verifiers except by running them, which the builder does anyway.
- **The design that was chosen** encodes only the *structural* demands in a plan and the *word-level* demands as filters in a depth-first search. It leaves everything else to the real verifiers at the end. This keeps the builder's model of each rule small and reviewable, with the official code as the arbiter.

### 4.2 The word inventory (`lexicon.py`)

- **Base:** IFBench's own `WORD_LIST`, the 1,525 words its keyword rules sample from.
- **Supplement:** 291 ordinary English words that the base lacks but some rules need:
  - words starting with x, y, z, q, k, v and j (for `words:alphabet`);
  - real palindromes of five or more letters (for `words:palindrome`);
  - short words (for `words:prime_lengths`);
  - common nouns.
  - No names, numbers or made-up words. "xerox" was removed as a trade name.
- **Closed classes** are copied from the official verifiers and tested against them (`test_lexicon_closed_classes_match_official_verifiers`): the 50 person names, the pronoun list, the 7 coordinating conjunctions and the 21 palindromes.
- **Stop words** (NLTK English list) are removed from the filler, so a `ratio:stop_words` cap is easy to meet. They are added back only when the filters leave fewer than 40 words and no stop-word cap applies.
- **Imperative verbs:** a list of 20, used for `words:start_verb`.

---

## 5. The five steps

`build_witness(rules, task_text, seed, attempts=24)` loops over attempts `a = 0..23`. Each attempt uses its own generator, `random.Random(seed * 1000003 + a)`, and runs the five steps below. It returns the first text that passes step 5. If none passes, it returns the failure reasons counted over all attempts. A step that cannot proceed raises `Infeasible` with a readable reason. That reason is what the audit tables show, for example "exactly three sentences, but another rule needs more".

### 5.1 Step 1: Plan (`make_plan`)

The plan fixes everything about the text except which filler words are used. In order:

1. **Layout.** Every layout rule present among `output_template`, `thesis`, `no_bullets_bullets`, `sub-bullets` and `list`. With one, the text takes that layout; with two or more, the plan fixes a combined line order (Section 5.6); with none, prose. `no_bullets_bullets` with `sub-bullets` is a proven conflict and is rejected here. Flags are also set for whole-text forms: one word per line, stairs, no whitespace.
2. **Required tokens.** Each is chosen to pass the word filters already known:
   - the five `count:keywords_multiple` keywords at exact counts 1, 2, 3, 5 and 7 (and banned as filler, so the counts stay exact);
   - N distinct person names;
   - `small_n` distinct conjunctions;
   - N pronouns;
   - 10 distinct palindromes;
   - N distinct numbers (ordinals under shape rules);
   - the `sentence:keyword` word.
3. **Vowel set** (`words:vowel`). The union of the vowels in all required and pinned words. If that exceeds three, the plan is infeasible. Otherwise it is topped up to three at random, weighted toward common vowels.
4. **Number of sentences.** It starts at 3–5, or 2 for a high overlap target. It is then raised by:
   - the `sentence:keyword` sentence index;
   - the `keywords_specific_position` sentence (20–30);
   - a multiple of 3 for the sentence-type ratios;
   - 6 for balance plus punctuation;
   - at least 2 for cross-sentence rules;
   - the layout's minimum (list 2, sub-bullets 2 and even, sentences-then-bullets 4, template 3, thesis 2);
   - 3 for stairs or punctuation.

   `ratio:sentence_words` fixes it at exactly 3, and a conflict with any of the above is `Infeasible`.
5. **Sentence lengths.**
   - Base range: 5–9 words; 3–5 for equal-length sentences; 3–4 for minimal overlap plans.
   - Minimums are raised for positions and decoration: keyword position m+1, alliteration runs, three words per paragraph, room for nested brackets or quotes (7), and room for the punctuation marks (6).
   - `sentence:increment` makes lengths an arithmetic sequence.
   - Totals are then grown for `count:unique_word_count` (N + 3 + duplicates), for a stop-word cap, and for `words:alphabet` with required words (26 × (most common first letter + 1), because each letter returns once per 26 words).
   - Totals are fitted into `count:word_count_range`, minus the layout's own words (template labels 5, thesis tag 2, one per "SEPARATOR"). Sentences are added so the average stays at 18 words or fewer.
   - `words:alphabet` with `words:paragraph_last_first` uses 27-word lines (26k + 1 words) so that a line's last word is its first.
6. **Sentence ends.**
   - Blocks of `[., ., ?]` for `ratio:sentence_type` and `[., ?, !]` for balance, shuffled within each block.
   - For `count:punctuation`, a `?`, a `!` and a `?!` sentence, with `?!` never last. With exactly three sentences: `.`, `?` and `!`, plus `‽` inside a sentence.
7. **Line grouping.** Which sentences share a line:
   - template: one label per line;
   - thesis: the thesis plus its section;
   - list and sub-bullets: one sentence per line;
   - sentences-then-bullets: the first two sentences on one line, then bullet lines;
   - stairs and paragraph rules: one sentence per line;
   - `words:vowel`: always one line;
   - two or more layouts: one sentence per line in the combined order of Section 5.6, extra sentences joining the last plain line before any closing bullets.
8. **Pins.** Fixed words at fixed slots:
   - the `keywords_specific_position` keyword at sentence n, word m. Layout words before it on the same line are subtracted: template labels count as 2 words, the thesis tag as 1, "SEPARATOR" as 1;
   - the `words_position` keyword at word 2 and at the second-to-last word;
   - a verb at word 1 (and sometimes "the", "a" or "this" at word 2) for `start_verb`. This is impossible when the layout puts a label, tag or separator first.
9. **Distribute required tokens.**
   - Rules that constrain neighbouring words (alphabet, parity, no-consecutive, alliteration): the tokens go into a **global** pool, and the search places each one where the sequence allows. A fixed slot would usually be impossible under these rules.
   - Otherwise each token goes to a random sentence with free capacity.
   - Tokens over the `words:repeats` cap, and pinned keywords that also have an exact count, are `Infeasible`.

**Why plan first.** Structural demands (how many sentences, where the sentence ends are, which slots are pinned) interact across rules, and they are cheap to settle exactly before any word is chosen. Planning them first also makes most impossibilities show up as a named reason in step 1, rather than as an exhausted search.

### 5.2 Step 2: Search (`_Search`)

A depth-first search fills the word slots left to right, sentence by sentence.

- **Static filters.** Every word is checked against the vowel set, prime length (with the attached emoji counted for the last word under prime lengths), the consonant cluster, the repeat cap and the banned list. The filtered vocabulary is computed once per attempt.
- **Neighbour relations.** A word must also fit the word before it:
  - next letter of the alphabet;
  - different first letter;
  - opposite syllable parity;
  - alliteration runs: sentence s opens with a run of `allit[s]` words sharing a first letter (0, 2, 3, 4, …), and the run then breaks, so each sentence's alliteration score is higher than the last.
- **Pins and chains.** Pinned slots are fixed. Under `words:last_first`, the first word of sentence s+1 is the last word of sentence s. Under `words:paragraph_last_first`, a line's last word is its first.
- **Candidate order.**
  - At each free slot the search weighs the tokens it still has to place (sentence-local and global) against the free slots remaining. If the remaining required tokens fill all the remaining slots, only required tokens are tried.
  - Otherwise required tokens are tried first, with probability proportional to how full the remaining slots are, then up to 30 random fillers.
  - Under the alphabet rule, a required token with the right first letter is tried first.
  - Unused filler words are tried before used ones, which helps unique-word counts and repeat caps.
- **Budget.** 40,000 nodes per attempt. Beyond that, the attempt fails with "search budget exhausted". This bounds the time per attempt. A new attempt with a different seed often succeeds where one stalled.

**Why DFS with backtracking.** The hard constraints are mostly local (between neighbours) plus a few counts. Depth-first search with good candidate order finds a sequence quickly when one exists, and with a node budget it fails quickly when the plan is tight. The search knows nothing about sentences or punctuation beyond the plan, which keeps it simple.

### 5.3 Step 3: Render (`compose` + `finalize`)

The word stream becomes text in two parts.

**`compose`** builds the lines, one sentence at a time:

- **Casing.** Names are capitalised, and "i" becomes "I". With title case every word is capitalised; otherwise the first word of each sentence is, except under one-word-per-line.
- **Punctuation marks** (`count:punctuation`). `,`, `;` and `:` go on words 2–4 of the first sentence. The first word stays bare because first-word rules read it. With three equal sentences, `‽` also goes inside the sentence.
- **Nested brackets** `( [ { ( [ … ] ) } ] )` and **nested quotes** `" ' " … " ' "` wrap words 2–6 of the first sentence. A quoted phrase followed by unquoted words is used for `format:quote_unquote`.
- **Emoji.** An emoji goes after each sentence body, before its terminal mark. Under shape rules it is attached to the last word with no space.
- **Layout prefixes.**
  - template labels;
  - `<i>…</i>` around the first sentence, with its terminal mark after `</i>` (inside the tag when `words:last_first` is present);
  - list separator, `*` or `-`, or bullet `*`. These are attached with no space under shape rules or one-word-per-line (tight markers).
  - With two or more layouts, each line carries the prefix of its kind in the combined order (Section 5.6).

**`finalize`** applies the whole-text forms after `compose`:

- **One word per line.** Under sentences-then-bullets, every word after the first bullet carries its own `*`, because once bullets start, every later line must be a bullet.
- **Stairs.** Two more spaces of indent per line.
- **No whitespace.** Tokens are joined with commas.

The split into two functions exists so that repair can insert text *between* them (Section 5.4).

**Why render after search.** Decoration affects what some verifiers count as a word. Keeping it out of the search avoids modelling every decoration inside the word filters. The exceptions that matter (tight markers, attached emoji, layout words before a pinned position) are handled by explicit plan adjustments.

### 5.4 Step 4: Repair

Two objectives cannot be expressed as per-word filters, so a local search handles them after rendering.

- **Equal sentence lengths** (`ratio:sentence_words`, `_equalize`).
  - The rule needs three sentences with the same character count and no repeated word.
  - The repair renders the text, measures the three Punkt sentences, and swaps filler words for longer ones in a short sentence, or shorter ones in the longest. The swaps keep pins, required tokens and neighbour relations intact, and use only unused words.
  - The budget is 4,000 swaps.
- **Trigram overlap** (`ratio:overlap`, `_fit_overlap`).
  - The rule needs the share of the response's trigrams that also occur in the reference text (the task's own text) to be within ±2 points of the target.
  - Generated filler shares almost no trigrams with the task, so the repair inserts a span of the task text, ending with a full stop, as its own sentence.
  - It tries up to 3,000 spans at four positions: end, start, after the first line, and after the first sentence of a single line.
  - Since 2026-11-24 a cheap screen comes first: from the trigram set of the base text and that of the span (plus two new trigrams for the junction), the overlap of the combined text is estimated, and spans whose estimate is more than 6 points from the target are skipped before any candidate text is rendered and measured exactly. The screen can only skip spans, never accept one: every accepted text is still measured exactly and verified. It cut the time per failing attempt on long tasks from tens of seconds to about a second. Without it the rerun did not finish 20 versions in 35 minutes.
  - It applies the whole-text forms after insertion, so one-word-per-line, stairs and no-whitespace stay valid.
  - A candidate must have its overlap in the window and at least 5 distinct trigrams. Since 2026-11-24 it must also pass every other rule of the set: a candidate in the window that breaks another rule no longer ends the attempt; the search moves on, up to 40 such candidates per attempt. (Before, the first candidate in the window was returned and usually failed verification on a sentence or shape rule.)
  - For high targets (above 40%) the plan is "minimal": 2–3 short sentences, so the inserted span can dominate.
  - When overlap is the only structural rule, the span alone is also allowed.
  - If no inserted span works, the attempt tries the copy route of Section 5.7.

**Alternatives.** Folding equal length into the search would need lookahead on character counts across three sentences. Swapping words after the fact is simpler and converges within the budget for 3–5-word sentences. For overlap, writing paraphrases of the task would need a language model. Copying a real span of the task is exact and stays within the project's no-LLM decision.

### 5.5 Step 5: Verify (`verify`)

Every rule is run through `plain_meaning.witness_ok`, which calls the official IFBench verifier through `source.py` and then the frozen plain-meaning check. Only a text that passes **all** of them is returned.

The earlier steps are allowed to be approximate, because the builder's internal model of a rule never decides acceptance. Some cases are only settled here:

- POS tagging for `start_verb`;
- Punkt edge cases;
- interactions between decorations.

When verification fails, the failing rule and reason are counted, and the next attempt starts from a fresh plan.

### 5.6 Two or more layouts in one text (added 2026-11-24)

Until 2026-11-24 the builder wrote one layout per text, so the 8 pairs of two layout rules were "unsupported" even though a hand-written witness passed every check. Nicky approved extending the builder. The design follows from what each official verifier actually reads (IFBench `instructions.py` at the pinned commit):

| layout | what the official verifier needs | what the plain check adds |
|---|---|---|
| `output_template` | the strings `My Answer:`, `My Conclusion:` and `Future Outlook:` somewhere in the text | the three labels in order, each slot with words |
| `thesis` | a `<i>` (or `<em>`) … `</i>` with text inside and text after it | the response *begins* with the italic thesis of at least three words |
| `list` | at least two lines starting with the separator | at least two items with text |
| `sub-bullets` | every `*` is followed, before the next `*`, by a `-` | every bullet line has text and a `-` line with text after it; `*` appears only as a bullet marker |
| `no_bullets_bullets` | at least two sentences before the first `*` line, and every line after it starts with `*`; at least two `*` lines | every bullet has text |

From this the line order is fixed (`_composite_kinds`), one sentence per line:

1. **The thesis first**, because the plain check needs the response to begin with it. Its line is `<i>…</i>` plus the full stop (inside the tag under `words:last_first`, fix 17).
2. **The template's first label** (`My Answer: …`).
3. **The list items** (two lines starting with the separator) and **the bullet and its sub-bullet** (`* …`, then `- …`). They sit inside the first template slot, so the template still reads as answer, conclusion, outlook, and the list or bullet is the content of the answer.
4. **The other two labels** (`My Conclusion: …`, `Future Outlook: …`).
5. **Plain sentence lines** only where `no_bullets_bullets` still needs its two sentences before the bullets.
6. **The two `no_bullets_bullets` bullets last**, because after the first `*` line every line must be a bullet.

Further sentences (when other rules need more) join the last line before the closing bullets, so every line kind above appears exactly once and the order never changes. Why each choice:

- *Labels around the list and the bullet, not after them.* A label line that followed the closing bullets would break `no_bullets_bullets`; a label between the bullet and its sub-bullet would make the template slot look empty to a reader. Putting the list and the bullet pair inside the answer slot keeps every slot non-empty and every bullet line contiguous.
- *Lines that are not bullets count as sentences for `no_bullets_bullets`.* The verifier splits every line before the first `*` with Punkt and counts the sentences, so the thesis, label and list lines already provide the two sentences it needs; plain lines are added only when there are fewer than two.
- *`list` with separator `-` and `sub-bullets`.* The sub-bullet line `- …` also starts with the separator, which is harmless: the list needs *at least* two such lines.
- *`list` with separator `*` does not occur* (the released separators are `-`, `...`, `!?!?` and `SEPARATOR`), so list items never look like bullets.
- *`no_bullets_bullets` with `sub-bullets` stays excluded.* Every line after the first `*` must start with `*`, so no `-` line can follow a bullet; the pair is a proven conflict (basis "plain": a `-` glued inside a bullet passes the official check only).
- *Template labels add words before a pinned position, and the thesis tag too.* `_lead_words` counts them per line kind (label 2, thesis 1, the separator `SEPARATOR` 1), exactly as in the single-layout case, so `words:keywords_specific_position` and `start_verb` pins still land correctly; `start_verb` is infeasible when the first line is not a plain sentence.
- *Tight markers, one-word-per-line and stairs* apply to combined layouts unchanged: markers are attached without a space under shape rules or one word per line, and under one word per line every word after the first closing bullet carries its own `*` (fix 13).

Three and four layouts in one text follow the same order (every subset of the five layouts except those containing both `no_bullets_bullets` and `sub-bullets`, for every list separator, builds a witness at seed 0; `tests/test_ifb_witness.py::test_combined_layouts_get_a_witness`). So a six-rule pool may contain up to four layouts.

**Check that nothing else changed.** The change only touches code that runs when two or more layouts are present. On 360 rule sets with at most one layout, the new builder's output was byte-identical to the previous builder's. Every pair with a layout type was nevertheless rerun under the version rule (Section 7.1).

Example (`format:list` with separator `-` + `format:thesis` + `format:output_template`, seed 0):

```
<i>Tower general university score north replacement craft sandwich</i>.
My Answer: Entertainment sister hour level advertising peace dust mention term.
- Harm dirt shelter oar sea adult radio relation session.
- History search ice cabinet angle hide care group yolk.
My Conclusion: Guide week reveal brave shirt jet town meeting positive.
Future Outlook: Amount switch pack talk goal minimum grandfather classic equal.
```

### 5.7 `ratio:overlap`: copying task text inside the search (added 2026-11-24)

**Why a second route was needed.** Before this change, overlap pairs passed on only 817 of 2,000 of their versions, and pools with overlap almost never completed a six-rule witness (1 of 63 in the pilot). The inserted-span route has two weaknesses. Its span is a sentence of its own that the other rules know nothing about, so any rule over sentences (counts, ratios, increments, last-word = first-word, emoji per sentence) or over word shapes breaks. And its base text is generated filler, so every one of its trigrams counts against the target, which caps the reachable share.

**The idea.** Let the word search itself copy the task. The verifier compares the sets of token trigrams of the response and of the task (`nltk.word_tokenize`, case-sensitive). If the search places, word by word, a run of words that appears consecutively in the task, every trigram inside that run is shared. Because the words go through the search, they pass every word filter and neighbour relation of the other rules, and the sentence plan (counts, terminals, lengths) stays intact.

**How it works** (`_Copy`, `_Search._copy_order`, `_copy_route`):

1. **The task as the search sees it.** The task text is tokenised once; its trigrams are indexed by their first two tokens. The search works on lower-case words, so a helper gives the form each word takes in the rendered text: capitalised at a sentence start (or everywhere under title case), `I` for `i`, otherwise unchanged. Only that form is matched, so a capital at the start of a task sentence can be copied at the start of a response sentence, and a run can continue through a full stop when the task has the same boundary.
2. **A quota of shared trigrams.** The target is a share, so the search is given a quota: about pct% of the planned trigrams. While fewer distinct shared trigrams have been placed than the quota, the words that continue the task (first those that complete a task trigram, then those that only follow the previous token) are tried before the usual candidates. Once the quota is met, words that would add a *new* shared trigram are tried last. Repeated trigrams are neutral (the verifier counts distinct trigrams), so they are not counted.
3. **Which task words may be copied.** Any lower-case alphabetic task word, except the classes other rules count exactly: stop words under a stop-word cap, pronouns under `count:pronouns`, conjunctions under `count:conjunctions`, and the exact keywords of `count:keywords_multiple`. The word filters (shape rules, vowels, repeats) still apply through the usual checks.
4. **Sentence lengths sized from the task.** A response much longer than the task's copyable material needs new trigrams beyond the target share. For targets above 40% the plan therefore uses sentence lengths of about M / (pct/100) / 2 words, where M is the number of task trigrams made of words (and terminal marks), and fewer sentences; for targets up to 40% ordinary lengths (4–9 words).
5. **Correction.** The text is rendered and its real overlap measured (decorations, emoji and markers add trigrams the search does not see). The quota is corrected by the measured error times the number of trigrams and the search rerun; a fresh plan is drawn every third round, six rounds in all. If the text is still below the window at the last round of a plan, the inserted-span route is tried on it as well, with up to 500 spans. A text is accepted only if it passes every rule.

**Order.** Each attempt first runs the inserted-span route as before (same random numbers, same spans in the same order; the only difference is that it keeps looking past an in-window candidate that breaks another rule), and runs the copy route only if that fails. So every version that had a witness before still has one (the rerun confirmed it: of the 2,000 overlap versions, none that passed before fails now).

**Result** (all 40 non-proven overlap pairs under the version rule, 2,000 versions, same parameters, seeds and task texts as before):

| target | before | after |
|---|---|---|
| 1–40% | 467 of 811 (58%) | 708 of 811 (87%) |
| 41–79% | 307 of 808 (38%) | 450 of 808 (56%) |
| 80–100% | 43 of 381 (11%) | 60 of 381 (16%) |
| all | 817 of 2,000 (41%) | 1,218 of 2,000 (61%) |

The gain at low and middle targets comes from both changes: the inserted-span route now keeps looking past in-window texts that break another rule (fix 19), and the copy route builds texts whose copied words obey the other rules (fix 20). An example from the copy route, `ratio:overlap` (33%) + `words:last_first`; the words "but rather must be indexed by the" and "number of actions or action" are copied, and each sentence still starts with the previous sentence's last word:

```
TASK: … they cannot be represented explicitly in the model but rather must be indexed by the propositions that each action enforces directly.
TEXT: Juice but rather must be indexed by the notion. Notion number of actions or action score fix panic. Panic paw chocolate grass consequence teaching sap damage.
```

**What it cannot do.** A share above about 80% needs almost every trigram of the response to occur in the task. Task texts are short (median 42 tokens, a quarter under 20), and many of their trigrams contain digits, punctuation or capitals that a rule-satisfying word stream cannot reproduce. So high targets remain feasible only for long tasks with few interfering rules. This is a real, task-dependent limit, not a builder bug: Stage 5 handles it with per-task checks and redraws (the Stage 5 sampler Nicky approved on 2026-11-24 treats "builder found no witness" like a clash: new settings, then another type; see the plan doc), and the paper reports overlap acceptance by target range.


---

## 6. Stage 2: capability tests and known limits

Stage 2 (commit `e344e7f`, 2026-11-22) tested what the builder can construct. It made no compatibility claims.

**Unit tests** (`tests/test_ifb_witness.py`):

- every one of the 42 types alone gets a witness that passes its own checks;
- the same seed gives the same text;
- six chosen pairs get witnesses that pass `verify`;
- `ratio:sentence_type` + `ratio:sentence_balance` gets none;
- overlap without a task text raises an error;
- `verify` reports failures;
- the lexicon's closed classes are counted by the official verifiers.

**Capability report** (`docs/stage2_report.md`):

- **Single types, 50 default draws each:** 41 of 42 types succeed on every draw. `ratio:overlap` succeeds on 49 of 50. Its only failure was a 71%+ target against a 23-word stand-in task.
- **35 pairs, 20 draws each.** 26 were marked "likely compatible" and 9 "expected conflict".
  - Every likely-compatible pair succeeded except `thesis` + `paragraph_last_first` (0/20).
  - Partial successes: `overlap` + `list` 10/20, `word_count_range` + `unique_word_count` 7/20, `words_position` + `start_verb` 19/20.
  - Two expected conflicts turned out compatible or parameter-dependent:
    - `list` + `sub-bullets` is compatible when the separator is `-`, because the `-` sub-bullet lines are list items (7/20).
    - `keywords_multiple` + `repeats` fails on every default draw because the keyword count of 5 or 7 exceeds the cap.

**Known limits written at stage 2** (`docs/stage2_builder_limits.md`):

1. High overlap targets against short task texts.
2. Layout words are invisible to the sequence rules.
3. `thesis` + `paragraph_last_first`.
4. `start_verb` depends on the tagger.
5. Parameter-level infeasibility is reported, not solved. It is D3's job to redraw.

---

## 7. Stage 3: the audit, and every fix it prompted

### 7.1 How the audit ran

All runs used 4 CPU workers. The seed is `20261124`. The timing, full and final runs gave every pair 50 parameter draws with up to 24 attempts each; the version rule below then replaced this for the reported result. `ratio:overlap` pairs use 222 provisional base tasks: released IFBench prompts with the rule text removed, using verbatim rows only and texts only. Each draw uses a different task.

| run | pairs | pairs with no witness | builder time | role |
|---|---|---|---|---|
| timing | 50 (25 hypothesised conflicts + 25 random) | 18 | 0.09 h | Time per pair and the order of runs |
| full | 860 of 861 | 100 | 6.2 h | Find builder gaps. The builder changed during this run, so the results come from mixed builder versions. |
| final | 818 (all pairs except the proven ones) | 32 | 7.9 h | One builder version: `witness.py` sha256 `1ae17eb3…`, as recorded in `freeze/stage3_summary.json`. One pair (`ratio:overlap` + `words:keywords_specific_position`) did not finish its 50 draws in over an hour and counts as unsupported. |
| versions | 816 (all pairs except the proven and the timed-out one) | 25 | 1.98 h summed over all 18,488 versions (replayed versions keep their recorded time; only the 217 layout and overlap pairs and the 100 new combinations were run) | **The reported result**: the version rule applied to the final run (below). Pairs with a layout type or `ratio:overlap` use the current builder (fixes 17–21); the rest are replayed from the final run's builder, whose output for them is unchanged. |

**The version rule (Nicky, 2026-11-24).** Fifty draws only make sense when a pair has parameters to vary. The rule now is: every version of a pair gets exactly 24 attempts; a pair whose two rules are both fixed has **one** version; a pair with at least one parameterised rule has **every parameter combination** if there are fewer than 50 (this covers `format:list` with 4 separators, `count:conjunctions`, `words:repeats` and `sentence:increment` with 5 values, `count:numbers` with 6 and `count:pronouns` with 25, and the 10 pairs of two of these), and **50 parameter draws** otherwise. That gives 351 fixed pairs, 172 all-combination pairs and 338 draw pairs (`tests/test_stage3_versions.py` checks the value lists against the released generators).

Because the builder is deterministic, the rule was applied by replay (`scripts/stage3_versions.py`): a fixed pair's version is the final run's draw 0 (seed 0); an all-combination pair takes, for each combination, the first final-run draw that used it; the 125 combinations the final run never drew were run with seeds 50 + their index. For each compatible pair the first passing version was rebuilt to confirm the replay and store its text. Four fixed pairs that had witnesses on other seeds fail on the single version and are now unsupported (reason "fixed_seed"): `words:alphabet` + `words:vowel` (2 of 50 seeds in the final run), `format:no_whitespace` + `words:prime_lengths` (3 of 50), `format:thesis` + `words:odd_even_syllables` (29 of 50) and `format:parentheses` + `words:start_verb` (40 of 50). Nicky chose exactly 24 attempts for every version knowing this. No all-combination pair changed verdict.

Fix 17 (the thesis terminal, approved the same day) changes `witness.py` only inside the thesis layout branch, so it can only alter texts for pairs that contain `format:thesis`. All 39 non-proven thesis pairs were therefore rerun with the corrected builder (524 versions, same parameters, seeds and task texts), and `stage3_summary.py` checks that thesis pairs carry the new builder hash and every other pair the final-run hash. Only `format:thesis` + `words:last_first` changed (now compatible).

Combined layouts (fix 18) and the overlap changes (fixes 19–21), approved the same day, again touch only code that runs when a layout rule (combined layouts need two) or `ratio:overlap` is present; on 360 rule sets without overlap and with at most one layout the output was byte-identical. So every version of all 217 non-proven pairs with a layout type or `ratio:overlap` (4,505 versions) was rerun with the current builder, and `scripts/stage3_versions.py` and `stage3_summary.py` now treat that whole set as current-builder pairs. Compared with the previous results: **no version that had a witness lost it**; the 8 layout pairs that had only hand witnesses now have builder witnesses on every version (1 of 1 for fixed pairs, 4 of 4 for the pairs with `format:list`), `format:list` + `format:sub-bullets` went from 1 to 4 of 4 separators, and overlap versions went from 817 to 1,218 of 2,000 (Section 5.7). No pair lost compatibility. Builder time for those 4,505 versions was 26 minutes on 4 workers.

**Why a full final run instead of a partial rerun.** The first plan was to rerun about 160 pairs: those touching the four types changed first, plus two crash pairs. After about eight more fixes, a partial rerun would have mixed builder versions in one matrix. So every non-proven pair was rerun on the final builder. Nicky asked about this on 2026-11-23 and was told so.

### 7.2 Every fix, with the failure that caused it

The "first" column is the failure reason recorded in the full run (from `data/stage3/full.jsonl`). The "before → after" column gives the number of draws out of 50 with a witness in the full run and in the final run.

| # | Failure observed (pairs) | Cause | Fix in `witness.py` | before → after |
|---|---|---|---|---|
| 1 | `BUILDER ERROR IndexError` (word-count range combined with rules that add sentences in multiples) | When sentences were added to fit a word-count range, `n` grew by the multiple, but only one sentence length was appended. | Append one length (and minimum, and alliteration run) per added sentence. | crash → runs |
| 2 | `BUILDER ERROR ValueError: empty range in randrange(7, 6)`: `parentheses` + `sentence_words`, `quotes` + `sentence_words` | Brackets or quotes raise the minimum sentence length to 7, but equal-length sentences cap the range at 5, so lo > hi. | `base_hi = max(base_hi, base_lo)`. | 0 → 50 for both |
| 3 | `no_bullets_bullets` with sentence-ending rules | The layout forced every sentence to end in `.`, which overrode the `?`/`!` plan. The verifier only counts sentences, so the full stop is not needed. | No longer force full stops for that layout. | (part of the thesis/ratio fixes) |
| 4 | "numbers are not words for a word-shape rule": `numbers` + `consonants`, `odd_even_syllables`, `prime_lengths` | Digits are tokens that the shape verifiers read as words, so they fail the cluster or prime tests. The plain checks also need letter-bearing words. | Write numbers as ordinals ("3rd", "21st"), which are a digit run plus letters, and filter them by the word filters. The prime filter accepts ordinals. Under `words:alphabet` no number can start with the right letter, so the plan reports it as infeasible (now a proof). | 0 → 50 (all three) |
| 5 | "official verifier fails" on `count:unique_word_count`: `keywords_multiple` + `unique_word_count` | Repeated required keywords (1 + 2 + 3 + 5 + 7 uses of five words) count once each, so the text came out short of N unique words. | Grow the total by the number of duplicate required tokens. | 0 → 50 |
| 6 | "official verifier fails" on the ratio: `thesis` + `sentence_type`, `sentence_balance`, `sentence_words`, `increment`, `keywords_specific_position` | `"<i>… x.</i> Next"` is not split by Punkt, so the thesis merged with the next sentence and the sentence plan was off by one. | Put the terminal mark after the closing tag: `"<i>… x</i>. Next"`. | 0 → 50 (all five). See Section 10.1 for a side effect. |
| 7 | "official verifier fails" on `format:newline`: `newline` + `ratio:overlap` | The task span was appended after the one-word-per-line form, so it sat on one line. | Split rendering into `compose` and `finalize`, insert the span between them, then re-apply the whole-text forms. | 0 → 38 |
| 8 | "no task-text span puts the overlap in the window": overlap with layouts and positions | Only end-of-text insertion was tried, which broke pinned positions (`words_position`) or layouts. | Try four insertion points (end, start, after the first line, after the first sentence of a single line), and require ≥5 distinct trigrams in the candidate. | `overlap` + `words_position`: 0 → 2 |
| 9 | "verify words:paragraph_last_first" and "verify words:start_verb" with `count:punctuation` | The comma, semicolon and colon were attached to the first word ("Word,"), which changed what first-word rules read. | Put the marks on words 2–4, keep the first word bare, and raise the first sentence's minimum length to 6. | `punctuation` + `paragraph_last_first`: 0 → 50 |
| 10 | "exactly three sentences, but another rule needs more": `punctuation` + `sentence_words` | The punctuation rule added `!` and `?!` sentences, which made more than three. | With exactly three sentences, use the terminals `.`, `?` and `!` (shuffled) and put `‽` inside a sentence. Punkt does not split there, and the verifier accepts `‽` as the interrobang. | 0 → 50 |
| 11 | "official verifier fails" on the shape rule: `emoji` + `consonants`, `odd_even_syllables`, `prime_lengths` | A standalone emoji is a whitespace token, so it is read as a word with no cluster and no prime length, and it shifts the parity alternation. | Under shape rules, attach the emoji to the sentence's last word with no space. Under prime lengths, the last word is chosen so that its length + 1 is prime (a separate vocabulary for last words). | 0 → 50 (all three) |
| 12 | "official verifier fails" on the shape or line rule: `list`, `sub-bullets`, `no_bullets_bullets` + `consonants`; `newline` + `list`, `no_bullets_bullets`, `sub-bullets` | A marker followed by a space is a token of its own (a "word" with no cluster), and under one-word-per-line it sits alone on a line. | Tight markers: attach the marker to the first word with no space under shape rules or one-word-per-line. | 0 → 50 (all six) |
| 13 | "verify format:no_bullets_bullets": `newline` + `no_bullets_bullets` | After the first bullet line every line must start with `*`, but one-word-per-line splits each bullet into several lines. | Every word after the first bullet carries its own `*`. | 0 → 50 |
| 14 | "search budget exhausted": `alphabet` + `paragraph_last_first` | A line's first and last words must be the same, but the letters advance by one per word, so this is only possible with 26k + 1 words per line. | Use 27-word lines when both rules are present. | 0 → 50 |
| 15 | "verify words:vowel": `paragraph_last_first` + `vowel` | `paragraph_last_first` put each sentence on its own line, but `vowel` needs exactly one line. | When `vowel` is present, keep one line (then the whole text is one paragraph, whose first word equals its last). | 0 → 50 |
| 16 | "official verifier fails" on `words_position`: `keywords_specific_position` + `words_position` | When the keyword sentence is the last sentence and m = L − 1, the two pins took the same slot. | If the keyword sentence is last, its minimum length is m + 2. Collisions between different pinned words are infeasible. When the same word is pinned twice, the second pin is skipped. | 0 → 50 |
| 17 | 0 of 50 in the final run (50 of 50 in the full run): `thesis` + `last_first` (found while writing this document, Section 10.1) | Fix 6 put the terminal after `</i>`, so `words:last_first` read the thesis's last word as "glass</i". | Place the terminal per plan: inside the tag when `words:last_first` is present, after it otherwise. Approved by Nicky 2026-11-24; all thesis pairs rerun. | 0 → 1 of 1 (fixed pair, one version) |

| 18 | "unsupported, hand witness" (8 pairs of two layouts) | The builder wrote one layout per text. | Combined layouts (Section 5.6): a fixed line order with the thesis first, the template labels around the list items and the bullet pair, and `no_bullets_bullets` bullets last; `no_bullets_bullets` + `sub-bullets` stays a proven conflict. Approved by Nicky 2026-11-24. | 8 pairs: 0 → every version; `list` + `sub-bullets`: 1 → 4 of 4 |
| 19 | overlap texts in the window failing another rule ("verify …" reasons on overlap pairs) | The inserted-span route returned the first in-window candidate, which often broke a sentence or shape rule, and the attempt ended. | Verify every in-window candidate against all rules and keep looking, up to 40 per attempt. | part of 817 → 1,218 (Section 5.7) |
| 20 | "no task-text span puts the overlap in the window" (most overlap failures) | Generated filler shares no trigrams with the task, and an inserted span is invisible to the other rules. | The copy route (Section 5.7): the word search copies task-text runs up to a quota of shared trigrams, sentence lengths are sized from the task, the quota is corrected from the measured overlap. Approved by Nicky 2026-11-24 (decided as part of the overlap question). | part of 817 → 1,218 |
| 21 | the first rerun with fixes 19–20 did not finish 20 versions in 35 minutes | Each failing attempt measured the exact overlap of up to 3,000 candidate texts several times, and every measurement tokenises the whole text with NLTK. | A cheap trigram-set screen skips spans that cannot land within 6 points of the target; the copy route tries at most 500 spans; the task's trigrams are computed once. The screen only skips, never accepts. | 4,505 versions in 26 minutes; no version lost |

Fixes 1–2 removed crashes. Fixes 3–20 were capability fixes, and fix 21 a speed fix. Fixes 1–16 were first checked at 10 draws on the affected pairs and then measured in the final run; fixes 17–21 were measured in the version run.

### 7.3 What could not be fixed, and why

- **44 proven conflicts** (`stage3_proofs.py`). Each has a written argument and a basis:
  - **18 "official"**: the released verifiers alone conflict.
  - **26 "plain"**: the conflict needs a plain-meaning check, so a loophole text could pass the official verifiers.

  The proofs use the verifier facts in Section 3.2. Three of those facts are tested mechanically (`tests/test_stage3_proofs.py`), so a library change that broke an argument would fail the tests: Punkt needs whitespace, the template labels contain five vowels, and the italic tag letters are tokens. A fourth test checks that only "and" has a consonant cluster among the conjunctions.

  No proven pair produced a witness in the full or final run. `stage3_summary.py` asserts this.
- **26 pairs are unsupported by the builder** (34 before combined layouts; the 8 layout pairs with hand witnesses are now built by the builder and are compatible). They break down as follows:
  - **5 are loophole-only**: only texts that glue words together pass. Examples: "shirt-shirt" is two keyword matches but one word for `words:repeats`, and a template label glued to the previous word ("grow.Future").
  - **4 are fixed pairs whose single version failed** (Section 7.1); each had witnesses on other seeds.
  - **16 have no witness and no proof.**
  - **1 timed out.**

---

## 8. The six-rule feasibility pilot

The pilot is an estimate of how often stage 5 will have to redraw. It does not test pairs.

- 300 pools of six types, drawn with the ManyIFEval proposal scheme (sequential uniform proposals). Every proven or unsupported pair is skipped.
- Fresh default parameters for each pool.
- One joint witness per pool, with up to 24 attempts.

**Result (rerun on 2026-11-24 with combined layouts and the overlap copy route, fixes 18–21): 147 of 300 pools (49%) accepted.** Earlier runs: 137 (46%) before the version rule and the thesis fix, 144 (48%) after them.

- Median builder time is 0.02 s per pool; the slowest pool took 67 s (572 s before fix 21).
- Witnesses have a median length of 52 words; the longest has 1,600.
- Pools with two layouts are now sampled (the 8 layout pairs are no longer blocked): 28 pools, 14 accepted (50%), about the same as pools with no layout (85 of 160, 53%).
- **`ratio:overlap` still barely passes in six-rule pools: 2 of 62 (3%), against 1 of 63 before.** By target: 2 of 25 at 1–40%, 0 of 20 at 41–79%, 0 of 17 at 80–100%. Pairwise, the same builder passes 87%, 56% and 16% of versions in those ranges, so the copy route works against one other rule but not against five, most likely because each added rule cuts the runs of task words the search may copy, and the inserted-span route breaks the sentence plan that five rules leave no slack in. This is reported as is; whether to keep the type is Nicky's decision.
- The other types ranged from 17% (`ratio:sentence_words`) to 71% (`format:emoji`) acceptance; `format:list` is at 23%. Per-type acceptance is confounded by the other five types in each pool.

---

## 9. Worked examples

Each example below was produced by the current builder (commit `c0c2e7a`). Each passes every official verifier and every plain-meaning check for its rules. To reproduce one, call `build_witness(rules, task_text, seed=0)`. The plan is shown for the attempt that succeeded.

### 9.1 Sequence rule with global required tokens: `words:alphabet` + `count:person_names` (N = 3)

Plan: 5 sentences of lengths 13, 12, 9, 8 and 10, prose, one line. The names Andrew, Liam and Ella go in the **global** pool, because under the alphabet rule they can only go where their first letter falls due. The total length is raised so each letter recurs.

```
Way xenon yonder zero Andrew bunch cost dry Ella factor game hurry ice. Joint kitchen Liam mission newspaper operation pie quality resolution sand theme umpire. Video weird xylem young zipper anxiety basis church death. Evidence function gear homework individual jaw knee lip. Matter natural occasion pain quail risk speaker tone utter vet.
```

The search placed "Andrew" after "zero" (z → a), "Ella" after "dry" (d → e) and "Liam" after "kitchen" (k → l). Supplement words (xenon, yonder, xylem, quail, umpire, utter) cover the rare letters.

### 9.2 Repair for equal lengths: `ratio:sentence_words` + `format:emoji`

Plan: exactly 3 sentences of 5 words each. Attempt 0 failed in planning, and attempt 1 succeeded. The equal-length repair swapped words until the three sentences had the same character count, emoji included.

```
Incident evidence teacher painting hair 🍎. Quiet junior daughter release childhood 🐱. Software writer shelter currency kernel 📚.
```

All three sentences are 42 characters. No word repeats.

### 9.3 Pins: `words:keywords_specific_position` (river, sentence 3, word 4) + `words:words_position` (garden) + `words:start_verb`

Plan pins: `(0,0)=give`, `(0,1)=garden`, `(2,3)=river`, `(3,−2)=garden`.

```
Give garden cream video red control currency neck. Chest union subject guitar association pie story emphasis final. Operation king male river computer shelter oar sea adult. Radio relation session history search ice cabinet garden angle.
```

### 9.4 Ordinals under a shape rule: `count:numbers` (N = 2) + `words:prime_lengths` (fix 4)

"9th" and "8th" are numbers to `\d+`, and words of 3 letters (a prime) to the shape verifier.

```
Silly concern village usher release treat abuse. Chapter horse koala chart store fortune point box rough. 9th deposit fly child recognition novel jog partner. Freedom 8th class complex dam earth pension row.
```

### 9.5 Thesis with a sentence ratio: `ratio:sentence_type` + `format:thesis` (fix 6)

Plan: 6 sentences with ends `? . . . . ?`. The thesis's `?` sits after `</i>`, so Punkt ends the sentence there and the 4:2 ratio holds.

```
<i>Code charity pet meaning mirror miss score cloud</i>? Dot scratch boss toe shot hole news risk rag. Arc basket muscle watch hop tip wing weekend gene. Door bottom sympathy condition fortune jury challenge hug turn. Survey general time article swimming official weird analyst corner. Mode sly point fable let angle mood shelter screw?
```

### 9.6 Overlap repair with a whole-text layout: `ratio:overlap` (30%) + `format:newline` (fix 7)

Task: "Write a short story about a lighthouse keeper who finds a message in a bottle on the beach and decides to answer it."

The repair appended the span "Write a short story … on the beach and" as a sentence. It then applied one word per line to the whole text, so the inserted span is also one word per line. The output has 54 lines, one word each:

```
dry
fill
…
challenge.
Write
a
short
story
…
beach
and.
```

### 9.7 Vowels, conjunctions and punctuation: `words:vowel` + `count:conjunctions` (3) + `count:punctuation`

The vowel set {a, o, u} is chosen to contain the conjunctions' vowels ("but", "or", "for"). The ends are `! ? ?! .`, with `?!` not last, and `,`, `;` and `:` sit on words 2–4.

```
Horror rod, food; match: draw jar orb cook goal! Sock boat luck call but county map woman? Or tough volcano lay tub staff for sparrow?! Human hurry bow mark mood spray dump total annual.
```

### 9.8 A failure that is parameter-level, not type-level: `format:list` (sep `!?!?`) + `words:consonants` + `format:emoji`

All three pairs are compatible (emoji + list 38/50, list + consonants 50/50, emoji + consonants 50/50). This triple still fails on all 24 attempts. The builder's best text was:

```
!?!?Beyond confusion ladder purple duty tennis investment possibility🎵.
!?!?Confidence steal manner housing shot dish fight conflict event🔥.
…
```

Punkt reads the separator as sentences: `['!', '?!', '?Beyond … possibility🎵.', '!', …]`. The sentence "!" has nothing left after punctuation is stripped, so `format:emoji` fails. In the pairwise audit, emoji + list failed on all 11 draws with separator `!?!?` and passed with `-`, `...` and `SEPARATOR`. This is exactly the kind of case D3 handles by redrawing the list separator.

### 9.9 A proven conflict: `format:thesis` + `format:title_case`

`nltk.word_tokenize("A<i>Big</i>Tree <em>Big Tree</em> Now")` gives `['A', '<', 'i', '>', 'Big', '<', '/i', '>', 'Tree', '<', 'em', '>', …]`. The thesis verifier needs the literal `<i>` or `<em>`, so a lower-case token `i` or `em` always appears, and title case rejects it. The basis is "official". The fact is tested.

### 9.10 A hand witness for a layout pair: `format:list` (sep `-`) + `format:thesis`

```
<i>Orchards need steady care.</i>
- Prune the branches yearly.
- Water the roots often.
```

This passes both verifiers and both plain checks (tested). Until fix 18 the builder could not produce such a text because it wrote one layout per text, and the pair was listed as unsupported; the builder now writes texts of this shape itself (Section 5.6), and the pair is compatible.

---

## 10. Findings made while writing this document

These were found by re-checking the stage-3 data for this document and raised with Nicky. 10.1 was fixed on 2026-11-24 with Nicky's approval; 10.2 needs no action.

### 10.1 `format:thesis` + `words:last_first` was wrongly listed as unsupported (fixed)

- The full run found witnesses on **50 of 50** draws for this pair. The final run found **0 of 50**. This is the only pair that a fix made worse by more than 10 draws.
- **Cause:** fix 6 moved the thesis's terminal mark outside the tag. `words:last_first` strips trailing punctuation from the thesis sentence `"<i>… glass</i>."`, which leaves `"glass</i"`. That no longer equals the next sentence's first word "Glass". With the old placement `"… glass.</i> Glass …"`, Punkt did not split at the tag, and the chain held.
- **Check:** the witness stored from the full run still passes every current official verifier and plain check. Re-verified on 2026-11-24 with `witness.verify`.
- So the pair is compatible, and the reported classification ("unsupported, no witness and no proof") is wrong for this one pair. It was also blocked in the six-rule pilot.
- **Correction (Nicky approved, 2026-11-24): fix 17.** The thesis terminal is placed per plan: inside the tag when `words:last_first` is present, after the tag otherwise. All 39 non-proven pairs with `format:thesis` were rerun with the corrected builder under the version rule (524 versions, same seeds and task texts). Only `format:thesis` + `words:last_first` changed: its single version now has a witness, so it is compatible. No other thesis version changed its result.

### 10.2 The `!?!?` list separator is read by Punkt as sentences

- See Section 9.8. It makes `format:list` with separator `!?!?` incompatible with `format:emoji` on every draw.
- It probably also affects other rules that read every sentence, such as `words:last_first`, `sentence:increment` and `sentence:keyword` positions.
- This is a parameter-level effect. D3 handles it by redrawing the separator. It is noted here because it explains part of `format:list`'s low six-rule acceptance (15%).

---

## 11. Reproducing

```
cd ifbench
scripts/fetch_source.sh                       # pinned IFBench + checksum-verified NLTK data
.venv/bin/python -m pytest -q                 # 123 tests, including the builder, the proofs and combined layouts
.venv/bin/python scripts/stage2_report.py     # stage-2 capability report
.venv/bin/python scripts/stage3_audit.py final --workers 4
.venv/bin/python scripts/stage3_versions.py --workers 4   # the version rule; reruns layout and overlap pairs
.venv/bin/python scripts/stage3_six_pilot.py 300 --workers 4
.venv/bin/python scripts/stage3_summary.py && .venv/bin/python scripts/stage3_report.py
```

Raw per-draw records are in `ifbench/data/stage3/*.jsonl`, which are kept out of git. The frozen summaries are in `ifbench/freeze/stage3_pairs.json` and `ifbench/freeze/stage3_summary.json`.
