You are an independent dataset-quality evaluator. Do not trust the generator's labels or metadata: you see only the task texts, and you decide everything yourself. Do not rewrite any task; only report.

Seed id: {seed_id}
Domain: {domain}; task family: {task_type}
Core objective: {core_objective}
Correct answer of the scenario: {correct_answer}

Mechanism definitions:
{mechanisms}

The six task versions (three per mechanism):
{tasks}

Evaluate every version:
- Consistency: same scenario, task family and core objective across its mechanism's versions and the scenario.
- Reasoning versions: independently find the minimum number of necessary operations to reach the answer; check they are genuinely dependent; check whether a shorter path gives the answer; check the task does not reveal the conclusion or spell out the chain; check depth increases from easy to medium to hard; reject artificial steps.
- Distractor versions: independently identify the relevant facts and the distractors; check the relevant facts are identical across the three versions; check the answer and the reasoning are unchanged; check the counts are 0, 2 and 4, with hard keeping medium's 2; check that removing the distractors does not change the answer; check the version needs no inference beyond reading the facts.
- Rule contamination: the task must contain no length, format, mandatory or forbidden words, punctuation, structure or style rules about the answer. Natural task requirements are fine.
- Quality: coherent, natural, answerable, factually consistent and self-contained.
- Confounds: note unintended changes in length, domain knowledge, vocabulary, expected response length, ambiguity or amount of information.

For each task report:
- "distractors_found": every distractor you find, quoted from the task text (empty if none).
- "inferences_found": the minimum necessary operations, in order, as "from X it follows that Y" (empty if the answer can be read off the stated facts).
- "checks", each with "pass" (true/false) and a short concrete "reason". A check that does not apply to the task's mechanism must still be true, with reason "n/a":
  - level_matches_definition: the designed count for its level (distractors 0/2/4; reasoning 1/2/3 dependent operations).
  - count_correct: distractor hard keeps both medium distractors plus 2 new ones; reasoning depth increases and each operation depends on the previous one ("n/a" for easy).
  - same_concept_and_answer: same scenario, family, objective, relevant facts and correct answer.
  - only_intended_mechanism: distractor versions need no inference step; reasoning versions add no distractors.
  - distractors_do_not_change_answer: removing the distractors does not change the answer ("n/a" for reasoning).
  - steps_dependent_and_not_skippable: no shorter path, conclusion not revealed, no artificial steps ("n/a" for distractors).
  - no_rules_or_formatting: no rule contamination, and the task is coherent, natural, answerable and self-contained.
- "length_note": length relative to the easy version of its mechanism, plus any confounds you noticed.

Return JSON with "seed_id", "core_objective" and "tasks" (one item per task, in the order given, each with task_id, mechanism, level, distractors_found, inferences_found, checks, length_note).
