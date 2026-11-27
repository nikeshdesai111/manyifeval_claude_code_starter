You check whether a set of 10 answer-format rules can be satisfied together, for an instruction-following dataset. The rules will be attached to each of six tasks. A rule set is rejected only if it is impossible, not merely hard or unnatural. The answer does not have to read naturally; it only has to be possible.

Fixed definitions used by the code that checks every rule:
- Words are runs of letters with internal apostrophes or hyphens, lowercased; markdown symbols, emoji, digits and SECTION header lines are removed first; hashtag text counts as a word.
- Sentences come from a fixed sentence splitter. SECTION header lines are not sentences. Headings and list items count as sentences. An emoji may follow the end punctuation of a sentence.
- Paragraphs are separated by blank lines.
- Keyword and closing-phrase matches ignore case. Trailing emoji are ignored when checking the closing phrase.
- Numbers are digit runs outside SECTION markers and hashtags; a decimal counts once. List numbering counts.
- Indentation is measured in leading spaces (a tab is 4); blank lines are skipped. Checks of other rules ignore leading whitespace.
- No rule limits the total length of the answer.

Pool id: {pool_id}

The 10 rules:
{rules}

The six tasks:
{tasks}

Decide:
1. "rules_jointly_satisfiable": can one answer satisfy all 10 rules at once (ignoring the task)? If not, "conflicting_rules" must name the rule ids involved and "reason" must give the concrete contradiction (for example "R3 needs sentence 8 to have 15 words but R7 allows at most 14 words per sentence").
2. For each task, "reasonably_possible": can a reasonable answer to that task satisfy all 10 rules without breaking the task? A fail must name the rules in "conflicting_rules" and give a concrete reason tied to that task.
A pass has an empty "conflicting_rules" list and a short reason.

Return JSON with "pool_id", "rules_jointly_satisfiable" {{"pass", "conflicting_rules", "reason"}} and "tasks" (one item per task with "task_id" and "reasonably_possible" {{"pass", "conflicting_rules", "reason"}}).
