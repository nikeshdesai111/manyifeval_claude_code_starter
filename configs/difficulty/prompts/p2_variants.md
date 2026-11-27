You write three versions (easy, medium, hard) of one task for an evaluation dataset. All three versions share the same scenario, the same core objective, the same request and the same correct answer. Only the difficulty mechanism changes.

Seed scenario (domain: {domain}; task type: {task_type}):
<<<
{scenario}
>>>
Core objective: {core_objective}
Correct answer: {correct_answer}

Mechanism for these three versions: **{mechanism}**

{mechanism_definition}

Rules for every version:
- Keep the person's voice (first person), the situation, the core objective and the final request of the seed. The correct answer must be exactly the same in all three versions.
- Every fact needed for the answer is stated in the task. Nothing depends on outside knowledge.
- No instructions about the answer's form: no length, format, structure, style, tone, wording or language requests, and none of these words anywhere: word, sentence, paragraph, bullet, list, heading, bold, format, tone, emoji, hashtag, quote, P.S., begin, end with, include the word, at least, at most, no more than.
- No names of real people, companies or products, and no order or account numbers.
- Each version is a single block of plain prose (no line breaks, no markup), 60-220 words. Within the mechanism, length grows only by what the mechanism adds.
{feedback}
Return JSON with "tasks": exactly three items, in the order easy, medium, hard, each with:
- "level": "easy", "medium" or "hard".
- "task_text": the full task as the reader will see it, ending with the request.
- "distractors": for the distractor mechanism, each distractor fact copied exactly as it appears in task_text (empty for the reasoning mechanism).
- "inferences": for the reasoning mechanism, each necessary inference in order, written as "from X it follows that Y" (empty for the distractor mechanism).
- "answer": the correct answer for this version in one or two sentences.
