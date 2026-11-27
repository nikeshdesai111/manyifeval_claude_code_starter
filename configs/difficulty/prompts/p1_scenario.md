You are constructing diverse natural-language tasks for research on whether large language models violate individual instructions. Each task will later be paired with separate answer-format rules, so the task itself must contain none.

Your job at this stage: adapt the underlying objective of a real user request (the seed) into a realistic, self-contained scenario for the given domain and task family. Do not write any task variants yet.

Seed id: {seed_id}
Domain: {domain} ({domain_desc})
Task family: {task_type} ({task_type_desc})
Example of a situation in this domain and family: {cell_example}

Seed request (real user text; use its underlying objective and situation, do not copy it):
<<<
{raw_request}
>>>

Requirements:
1. Realistic and self-contained. The scenario describes a concrete everyday situation in the given domain, written as the person who needs help, and states every fact a good answer needs. Avoid anything that needs specialised knowledge, tools or web access.
2. A meaningful natural-language response. The objective must call for an explanation, comparison, plan, diagnosis or ordering in prose, not a single word or number.
3. Suitable for progressively more demanding versions. Later versions will (a) add plausible but unnecessary facts that do not change the answer, and (b) require 1, 2 or 3 dependent inferences from the supplied information. Supply concrete facts (amounts, dates, times, deadlines, counts, durations, dependencies) that make both possible, and give the scenario one defensible correct answer.
4. No explicit response constraints. Word counts, formatting, punctuation, mandatory keywords, writing style or any other enforceable rule about the answer are forbidden. Natural semantic requirements that are essential to the objective (for example "which bill should I pay first") are allowed. Avoid these words entirely: word, sentence, paragraph, bullet, list, heading, bold, format, tone, emoji, hashtag, quote, P.S., begin, end with, include the word, at least, at most, no more than.
5. No personal data. No names of real people, companies or products, no order or account numbers, no contact details; use roles and labels (my manager, Supplier A, Venue 1).
6. Provenance. Keep the seed id as the source reference. Do not invent sources or cite anything outside the seed.

Return JSON with:
- "scenario_id": the seed id.
- "domain": the domain above.
- "subtopic": 2-5 words naming the situation.
- "task_family": the task family above.
- "scenario_description": the scenario in the person's voice, about 80-150 words, ending with their request.
- "core_task_objective": one sentence stating what the person needs to find out or decide.
- "potential_source_facts": the facts in the scenario that the answer depends on, one per item.
- "correct_answer": the correct answer in one or two sentences, with its reason.
- "supports_both_mechanisms": true if both kinds of harder versions in point 3 are possible without changing the answer.
- "source_reference": "seed:" followed by the seed id.
