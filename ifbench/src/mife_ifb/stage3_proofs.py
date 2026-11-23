"""Stage 3: written proofs for pairs with no witness (proven conflicts, D1 exclusions).

A pair is a proven conflict when no response can pass both rules' official verifiers and both
frozen plain-meaning checks (the D1 witness standard). ``basis`` says which layer the argument
needs: "official" (the released verifiers alone conflict) or "plain" (the conflict needs a frozen
plain-meaning check, so a loophole response could pass the official verifiers).

Each argument states the verifier facts it uses. The facts that are about library behaviour
(NLTK sentence splitting and word tokenisation, fixed template labels) are checked mechanically
by ``FACTS`` and in tests/test_stage3_proofs.py, so a library change that broke an argument fails
the tests.

Pairs that have no witness but are NOT proven go to the "unsupported by the builder" list in the
stage-3 matrix. ``HAND_WITNESSES`` holds hand-written responses for some of those, verified by the
same checks, to show they are builder gaps and not conflicts.
"""
from __future__ import annotations

NWS = "format:no_whitespace"
TPL = "format:output_template"

_NWS_SENT = ("Text without whitespace is one sentence for NLTK sent_tokenize (Punkt needs whitespace after "
             "the terminal mark; FACT punkt_needs_whitespace).")

PROOFS: dict[tuple[str, str], tuple[str, str]] = {
    # ---- no whitespace: one whitespace word, one line, one sentence ----
    ("count:conjunctions", NWS): ("official", "count:conjunctions counts distinct whitespace words; a text without "
                                  "whitespace is one word, and small_n is at least 2 (randint(2, 6))."),
    ("count:unique_word_count", NWS): ("official", "count:unique_word_count counts distinct whitespace words; a text "
                                       "without whitespace has one, and N is at least 2."),
    ("format:newline", NWS): ("official", "'\\n' is whitespace, and format:newline needs at least two words, one per line "
                              "(plain), so a line break is needed."),
    ("format:line_indent", NWS): ("plain", "The plain check needs at least three non-empty lines; line breaks are whitespace."),
    ("format:list", NWS): ("official", "format:list needs two lines starting with the separator, so a line break."),
    ("format:no_bullets_bullets", NWS): ("official", "The verifier splits on '\\n'; one line is either all sentences "
                                         "(no bullets) or one bullet with no sentences before it. Both fail."),
    ("format:sub-bullets", NWS): ("plain", "The plain check needs a '*' bullet line followed by a '-' sub-bullet line, "
                                  "so a line break."),
    (NWS, TPL): ("official", "The template labels 'My Answer:', 'My Conclusion:', 'Future Outlook:' contain spaces."),
    (NWS, "format:thesis"): ("plain", "The plain check needs an italic thesis of at least three whitespace words."),
    (NWS, "ratio:sentence_balance"): ("plain", _NWS_SENT + " The plain check needs at least one '.', '?' and '!' sentence."),
    (NWS, "ratio:sentence_type"): ("plain", _NWS_SENT + " The plain check needs at least one question and two statements."),
    (NWS, "ratio:sentence_words"): ("official", _NWS_SENT + " ratio:sentence_words needs exactly three sentences."),
    (NWS, "sentence:alliteration_increment"): ("plain", _NWS_SENT + " The plain check needs at least two sentences."),
    (NWS, "sentence:increment"): ("plain", _NWS_SENT + " The plain check needs at least two sentences."),
    (NWS, "words:last_first"): ("plain", _NWS_SENT + " The plain check needs at least two sentences."),
    (NWS, "sentence:keyword"): ("plain", _NWS_SENT + " So N >= 2 fails the verifier; for N = 1 the plain check needs "
                                "sentence 1 to have two whitespace words."),
    (NWS, "words:keywords_specific_position"): ("official", _NWS_SENT + " The keyword sentence n is drawn from 20..30."),
    (NWS, "words:palindrome"): ("official", "words:palindrome needs 10 whitespace words that are palindromes."),
    (NWS, "words:vowel"): ("plain", "The plain check needs at least 5 whitespace words."),
    (NWS, "words:paragraph_last_first"): ("plain", "The plain check needs at least three whitespace words per paragraph."),
    # ---- one paragraph (words:vowel) against multi-line layouts ----
    ("format:newline", "words:vowel"): ("plain", "words:vowel needs exactly one line, so format:newline allows one word, "
                                       "but the plain checks need at least 5 words (vowel) and 2 (newline)."),
    ("format:line_indent", "words:vowel"): ("plain", "words:vowel needs one line; the line_indent plain check needs three."),
    ("format:list", "words:vowel"): ("official", "words:vowel needs one line; format:list needs two separator lines."),
    ("format:no_bullets_bullets", "words:vowel"): ("official", "words:vowel needs one line; the verifier needs sentence "
                                                  "lines followed by at least two '*' lines."),
    ("format:sub-bullets", "words:vowel"): ("plain", "words:vowel needs one line; the plain check needs a bullet line and "
                                           "a sub-bullet line."),
    (TPL, "words:vowel"): ("official", "The labels, lower-cased, contain a, e, i, o and u; words:vowel allows three vowels "
                           "(FACT template_vowels)."),
    # ---- template labels ----
    (TPL, "format:newline"): ("plain", "Each label holds two words on one line ('My Answer:'); the newline plain check "
                              "needs one word per line."),
    (TPL, "ratio:sentence_words"): ("official", "'My' occurs twice in the labels; ratio:sentence_words needs all words "
                                    "different after lower-casing."),
    # ---- line structure ----
    ("format:newline", "words:paragraph_last_first"): ("plain", "One word per line makes every paragraph one word; the "
                                                      "plain check needs three words per paragraph."),
    ("format:no_bullets_bullets", "format:sub-bullets"): ("plain", "After the first '*' line every line must start with "
                                                          "'*' (no_bullets_bullets verifier); the sub-bullets plain check "
                                                          "needs a '-' line after each bullet."),
    # ---- sentence-type ratios ----
    ("ratio:sentence_balance", "ratio:sentence_type"): ("plain", "balance needs #. = #? = #!; type needs #. = 2 * #?. "
                                                        "Together #? = 0, but the plain checks need at least one question."),
    # ---- exactly three sentences, all words different ----
    ("ratio:sentence_words", "words:keywords_specific_position"): ("official", "sentence_words needs exactly three "
                                                                   "sentences; the keyword sentence n is 20..30."),
    ("ratio:sentence_words", "words:last_first"): ("official", "last_first repeats a word across each sentence boundary; "
                                                   "sentence_words needs all words different (both lower-case)."),
    ("ratio:sentence_words", "words:paragraph_last_first"): ("official", "Each paragraph starts and ends with the same "
                                                             "word; sentence_words needs all words different. A "
                                                             "paragraph of one word is ruled out by the plain check."),
    ("ratio:sentence_words", "words:words_position"): ("official", "The keyword is the 2nd and 2nd-to-last word; "
                                                       "sentence_words needs all words different (plain words_position "
                                                       "needs 4 words, so they are two tokens)."),
    # ---- adjacent words with the same first letter ----
    ("words:alphabet", "words:last_first"): ("plain", "Across a sentence boundary the same word is repeated, so two "
                                             "adjacent words share a first letter; alphabet needs the next letter. The "
                                             "plain check needs two sentences."),
    ("sentence:alliteration_increment", "words:alphabet"): ("plain", "Some sentence after the first needs an alliteration "
                                                            "score of at least 1, i.e. two adjacent words with one first "
                                                            "letter; alphabet forbids that. The plain check needs two "
                                                            "sentences."),
    ("sentence:alliteration_increment", "words:no_consecutive"): ("plain", "As above: alliteration needs two adjacent "
                                                                  "words with one first letter; no_consecutive forbids it."),
    ("words:last_first", "words:no_consecutive"): ("plain", "The repeated boundary word gives two adjacent words with one "
                                                   "first letter. The plain check needs two sentences."),
    ("words:last_first", "words:odd_even_syllables"): ("plain", "The repeated boundary word gives two adjacent words with "
                                                       "the same syllable count, so the parities do not alternate."),
    # ---- emoji and letters-only words ----
    ("format:emoji", "words:alphabet"): ("plain", "Every sentence must end (or the next start) with an emoji; emoji are "
                                         "not punctuation, so a whitespace word holds one, and the alphabet plain check "
                                         "needs every word to be letters only."),
    # ---- title case and HTML italics ----
    ("format:thesis", "format:title_case"): ("official", "The thesis needs '<i>' or '<em>'; nltk word_tokenize splits "
                                             "'<' and '>' off, leaving a lower-case token 'i' or 'em', which fails "
                                             "title case (FACT italic_tag_tokens)."),
}

# Hand-written witnesses for pairs the builder cannot construct. Each passes both official
# verifiers and both plain checks (tested). They show the pair is a builder gap, not a conflict.
HAND_WITNESSES: dict[tuple[str, str], tuple[dict, str]] = {
    ("format:list", "format:no_bullets_bullets"): (
        {"format:list": {"sep": "..."}, "format:no_bullets_bullets": {}},
        "...Apples grow on tall trees.\n...Pears ripen in late summer.\n* Plums\n* Cherries"),
    ("format:list", TPL): (
        {"format:list": {"sep": "-"}, TPL: {}},
        "My Answer: Fruit grows well here.\n- Apples ripen early.\n- Pears ripen late.\n"
        "My Conclusion: Plant both kinds.\nFuture Outlook: Harvests will improve."),
    ("format:list", "format:thesis"): (
        {"format:list": {"sep": "-"}, "format:thesis": {}},
        "<i>Orchards need steady care.</i>\n- Prune the branches yearly.\n- Water the roots often."),
    ("format:no_bullets_bullets", TPL): (
        {"format:no_bullets_bullets": {}, TPL: {}},
        "My Answer: Orchards need care. My Conclusion: Prune trees yearly. Future Outlook: Harvests will grow.\n"
        "* Apples\n* Pears"),
    ("format:no_bullets_bullets", "format:thesis"): (
        {"format:no_bullets_bullets": {}, "format:thesis": {}},
        "<i>Orchards need steady care.</i>\nPrune the branches yearly. Water the roots.\n* Apples\n* Pears"),
    ("format:sub-bullets", TPL): (
        {TPL: {}, "format:sub-bullets": {}},
        "My Answer: Orchards need care.\n* Apples\n- ripen early\nMy Conclusion: Plant both kinds.\n* Pears\n"
        "- ripen late\nFuture Outlook: Harvests will improve."),
    ("format:thesis", TPL): (
        {TPL: {}, "format:thesis": {}},
        "<i>Orchards need steady care.</i>\nMy Answer: Prune yearly.\nMy Conclusion: Water often.\n"
        "Future Outlook: Harvests will improve."),
    ("format:sub-bullets", "format:thesis"): (
        {"format:sub-bullets": {}, "format:thesis": {}},
        "<i>Orchards need steady care.</i>\n* Apples\n- ripen early\n* Pears\n- ripen late"),
}

# Pairs whose only known constructions glue words together (e.g. 'shirt-shirt' counts as two
# keyword matches but one word for words:repeats; a template label glued to the previous word).
# Not proven conflicts, but the builder does not make such loophole texts.
LOOPHOLE_ONLY: dict[tuple[str, str], str] = {
    ("count:keywords_multiple", "words:repeats"): "The 7-times keyword exceeds the repeat cap (at most 5) unless copies "
        "are joined by hyphens ('shirt-shirt' is two keyword matches but one word for words:repeats).",
    ("count:keywords_multiple", "ratio:sentence_words"): "Repeated keywords break 'all words different' unless copies are "
        "joined by hyphens.",
    (TPL, "words:consonants"): "'Future' has no consonant cluster unless it is glued to the previous word ('grow.Future').",
    (TPL, "words:prime_lengths"): "'Answer' has 6 letters unless glued to the slot text ('Answer:a').",
    (TPL, "words:alphabet"): "'My' then 'Answer' breaks the letter sequence unless 'My' is glued to a preceding word.",
}


def _fact_punkt_needs_whitespace() -> bool:
    from . import source
    split = source.instructions_module().instructions_util.split_into_sentences
    return all(len(split(t)) == 1 for t in ("Apples.Pears.Plums!Figs?Kiwis.", "One.Two?Three!", "a.b.c.d.e.f"))


def _fact_template_vowels() -> bool:
    return set("aeiou") <= set("my answer: my conclusion: future outlook:")


def _fact_italic_tag_tokens() -> bool:
    import nltk
    from . import source
    source._bootstrap()
    toks = nltk.word_tokenize("A<i>Big</i>Tree <em>Big Tree</em> Now")
    return "i" in toks and "em" in toks


FACTS = {"punkt_needs_whitespace": _fact_punkt_needs_whitespace, "template_vowels": _fact_template_vowels,
         "italic_tag_tokens": _fact_italic_tag_tokens}


def key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def proven() -> dict[tuple[str, str], tuple[str, str]]:
    return {key(*p): v for p, v in PROOFS.items()}


def hand_witnesses() -> dict[tuple[str, str], tuple[dict, str]]:
    return {key(*p): v for p, v in HAND_WITNESSES.items()}


def loophole_only() -> dict[tuple[str, str], str]:
    return {key(*p): v for p, v in LOOPHOLE_ONLY.items()}
