You write an easy, a medium and a hard version of one task for research on whether large language models violate individual instructions. Each version will later be paired with separate answer-format rules, so the versions must contain none.

Scenario id: {seed_id}
Domain: {domain}; task family: {task_type}
Scenario:
<<<
{scenario}
>>>
Core objective: {core_objective}
Correct answer of the scenario: {correct_answer}

Evolution mechanism for these three versions: **{mechanism}**

{mechanism_definition}

General requirements:
- All three versions preserve the same scenario, task family and core objective. The correct answer is the same in all three.
- Difficulty comes only from the demands of the task under the assigned mechanism, never from explicit response instructions. Each version is a natural request a real user might make, in the person's voice, as one block of plain prose ending with their request.
- Manipulate only the assigned mechanism. Do not raise vocabulary or domain difficulty, and keep the expected response type the same.
- No ambiguity, contradictions, missing information or impossible tasks. Every fact needed is supplied.
- No explicit output rules: no counts, formatting, punctuation, mandatory or forbidden words, structure, length, style or tone. Avoid these words entirely: word, sentence, paragraph, bullet, list, heading, bold, format, tone, emoji, hashtag, quote, P.S., begin, end with, include the word, at least, at most, no more than.
- No names of real people, companies or products, and no order or account numbers.
- Keep lengths comparable (60-220 words). Do not pad to equalise length; record differences in difficulty_justification instead.
- Never use expected rule-following performance to set difficulty.
- If the scenario cannot support this mechanism cleanly, still write your best versions and set "unsuitable_for_mechanism" to true with the reason.
- The metadata fields below must not appear in task_text.
{feedback}
Return JSON with:
- "scenario_id", "domain", "task_family", "core_objective", "evolution_mechanism" ("{mechanism}").
- "easy", "medium", "hard": each with "task_id" (scenario id, mechanism and level), "task_text", "designed_difficulty" (easy/medium/hard), "relevant_facts" (facts the answer needs, quoted from task_text), "irrelevant_facts" (each distractor, copied exactly as it appears in task_text; empty for the reasoning mechanism), "required_reasoning_steps" (number of necessary dependent operations), "reasoning_dependencies" (each operation in order, "from X it follows that Y"), "expected_answer_or_content", "task_completion_rubric" (what a complete answer must cover), "difficulty_justification".
- "change_easy_to_medium" and "change_medium_to_hard": what changed between the versions.
- "unsuitable_for_mechanism" (true/false) and "unsuitable_reason" (empty if suitable).
