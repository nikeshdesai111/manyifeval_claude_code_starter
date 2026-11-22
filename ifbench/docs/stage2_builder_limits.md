# Stage 2: what the witness builder can and cannot construct yet

Companion to `docs/stage2_report.md` (generated numbers). Written 2026-11-22.

## Can
- Every one of the 42 candidate types alone, on every default parameter draw tested (41 of 42 types on 50 of 50 draws; `ratio:overlap` on 49 of 50).
- Combined word-shape and order rules (alphabet order, consonant clusters, prime lengths, syllable parity, no shared first letter, three-vowel paragraphs, alliteration runs) together with required tokens (exact keyword counts, names, pronouns, conjunctions, palindromes, numbers), pinned positions (keyword at sentence n word m, second and second-to-last word, verb first) and sentence plans (increments, last-first chaining, sentence-type ratios, three equal-length sentences).
- Layouts: prose, list with any of the four separators, `*`/`-` sub-bullets, sentences then bullets, the output template, HTML-italic thesis, one word per line, stairs indentation, no whitespace; plus nested brackets, alternating nested quotes, quoted phrases with explanations, emoji at sentence ends, and full punctuation coverage.
- `ratio:overlap`: appends or prepends a span of the task text so the trigram overlap lands within ±2 points.

## Cannot yet (builder limits, not verdicts)
1. **High `ratio:overlap` targets with short task texts.** Every failure was at 71% or more against the 23-word stand-in task; the generated part adds too many new trigrams. Real task texts arrive in stage 4 and pairs are audited per task in stage 3.
2. **Layout words are not modelled by the sequence rules.** Template labels ("My Answer:"), list separators ("SEPARATOR") and HTML tags count as words for some verifiers, but the word search does not see them, so combinations like template plus no-shared-first-letter can fail by chance.
3. **`format:thesis` with `words:paragraph_last_first`.** The verifier reads the first word of the thesis line as "i>word", so the line would have to end with that literal token. The builder does not write such text; stage 3 decides whether this is a proof of conflict.
4. **`words:start_verb` depends on the NLTK tagger**; about 1 draw in 20 needs more attempts than allowed when the verb's neighbours change its tag.
5. **Parameter-level infeasibility is reported, not solved**: for example a unique-word minimum above the word-count range, or a keyword needed 7 times under a repeat cap of 4. These are D3's job (the pool's parameters are redrawn).

## Hypotheses changed by these runs
- `format:list` with `format:sub-bullets` is **not** a type-level conflict: when the list separator is `-`, the sub-bullet lines count as list items (7 of 20 draws).
- `count:keywords_multiple` with `words:repeats` conflicts on **every** default draw (a keyword is needed 5 or 7 times, the repeat cap is at most 5), so it behaves as a type-level conflict under the released generators.
