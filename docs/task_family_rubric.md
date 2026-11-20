# Proposed task-family rubric (v1, PROVISIONAL — requires human review)

These labels are **proposed for this experiment**. They are not official ManyIFEval annotations, and ManyIFEval's rule categories are not task families. Labels were assigned by the implementer (Claude) from the task text alone, before any generation. Every label is in `configs/task_families_v1.csv`. Ambiguous cases are flagged and carry an alternative family.

## Families

| Code | Family | Deliverable |
|---|---|---|
| INFO | Explanation / informational | Factual explanation, informational essay/article/blog, factual Q&A |
| CREATIVE | Creative writing | Story, poem, song/lyrics, dialogue/conversation, fiction, fairy tale, rap, riddle, invented scenario |
| PERSUASIVE | Persuasive / argumentative | Ad/ad copy, (startup) pitch, product description, sales/marketing post, persuasive or argumentative letter, "X is the best" |
| PRACTICAL | Practical / instructional | Resume, cover letter, itinerary, template, rubric, plan/proposal document, job description, how-to/steps, advice/recommendation, personal correspondence |
| ANALYTICAL | Analytical / comparative | Comparison, pros/cons, critique, review, evaluation or opinion judgement ("is X good?", "thoughts on …") |
| TRANSFORM | Transformation / other | Operates on supplied text (expand, elaborate, complete, rephrase, ask about a given sentence), or underspecified/other (brainstorm a name, "Give me Q and As") |

## Tie-breaking precedence (apply the first that matches)

1. The task supplies a text to transform → **TRANSFORM**.
2. The deliverable is in a creative form (verse, narrative, dialogue) → **CREATIVE**. Form beats topic, e.g. "startup pitch as a weird poem" is CREATIVE.
3. Its primary purpose is to sell or persuade → **PERSUASIVE**.
4. It is a functional document, plan, instructions, advice or correspondence → **PRACTICAL**.
5. It is a comparison, evaluation or judgement → **ANALYTICAL**.
6. Otherwise it is a factual explanation → **INFO**.

Tone (funny, weird, angry) does not change the family.

## Sparse-family policy

Merge a family into its nearest neighbour before grouping only if it has fewer than 12 of the 216 tasks; smaller families would make cross-family grouping and split balance meaningless. Per-family counts are reported in the manifest summary.

## Known issues for reviewers

- Task **1258** text is a fragment ("in the style of a President of the United States."). It is kept because it is in the source frame, but it is flagged.
- Task **3294** ("Give me Q and As.") and **3428** ("Write sentences about tax filing.") are underspecified.
- Correspondence (letters/emails to friends) is labelled PRACTICAL by policy; several could reasonably be CREATIVE.

## Related-task clusters (kept on one side of the train/test split)

These are near-paraphrases or tasks on the same specific subject or persona, not tasks that merely share a genre:

- japan: 1001, 1092
- layton: 3245, 3478
- us_president_persona: 1258, 2304, 3204, 2471
- internet_history: 122, 179
- sleep: 1072, 209
- bike_for_kids: 1580, 301
- google: 1551, 2985
- us_elections: 2441, 2097

Borderline candidates that are logged but **not enforced** (please review): resumes (1005, 1040, 1658, 1781, 251, 260, 3513), cover letters (1535, 1551, 2691, 2716, 3608), rubrics (1634, 1670, 1936, 2350, 2870, 3329, 2447), startup pitches (2275, 2292, 2328, 2817, 3041, 343, 3697), Algiers (1040, 3672), Islam (1153, 3425), US presidency history (1667).

## Scripted version of this rubric

`src/mife/family_rules.py` implements this rubric as deterministic keyword/pattern rules, and `scripts/family_rules_audit.py` compares its output with the hand labels. The labels used by the pipeline are still the hand labels in `configs/task_families_v1.csv`. Agreement and disagreements: `docs/task_family_rules_agreement.md`.
