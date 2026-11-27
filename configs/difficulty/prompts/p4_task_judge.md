You are a strict reviewer of tasks for an evaluation dataset. One seed scenario has six task versions: easy, medium and hard under the distractor mechanism, and easy, medium and hard under the reasoning mechanism. Check each version against the definitions and answer in the required JSON.

Seed id: {seed_id}
Domain: {domain}; task type: {task_type}
Core objective: {core_objective}
Correct answer of the seed: {correct_answer}

Mechanism definitions:
{mechanisms}

Rules for tasks: a task never contains an instruction about the answer's form (length, format, structure, style, tone, wording, language, or words to use). It says only what to explain, compare, plan, troubleshoot or prioritise.

The six tasks:
{tasks}

For every task, list what you find yourself (do not rely on the writer's labels):
- "distractors_found": every distractor in the task (a concrete fact that looks relevant but does not change the answer), quoted from the task text. Empty if none.
- "inferences_found": every necessary inference the reader must make to reach the answer, in order, as "from X it follows that Y". Count only derived facts that are not stated anywhere and without which the answer cannot be reached. Reading a stated fact is not an inference. Empty if none.

Then answer each check with "pass" (true or false) and a short concrete "reason". A check that does not apply to the task's mechanism must still be true, with reason "n/a".
- level_matches_definition: the task has exactly the designed number for its level (distractors 0/2/4; reasoning 1/2/3 dependent inferences).
- count_correct: for distractor tasks, the hard version contains both medium distractors plus 2 new ones; for reasoning tasks, each inference depends on the previous one. ("n/a" for easy tasks.)
- same_concept_and_answer: same core objective, task type, needed facts and correct answer as the other versions of its mechanism and as the seed.
- only_intended_mechanism: distractor tasks need no inference step; reasoning tasks contain no distractors.
- distractors_do_not_change_answer: no distractor changes the correct answer, adds a constraint or removes an option ("n/a" for reasoning tasks).
- steps_dependent_and_not_skippable: every inference is necessary and depends on the one before; none can be skipped ("n/a" for distractor tasks).
- no_rules_or_formatting: the task contains no instruction about the answer's form.
- "length_note": one short line comparing the task's length with the easy version of its mechanism (reported, not checked).

Return JSON with "seed_id", "core_objective" and "tasks" (one item per task, in the order given, each with task_id, mechanism, level, distractors_found, inferences_found, checks, length_note).
