You turn a real request that someone sent to a chatbot into a self-contained seed scenario for an evaluation dataset.

The dataset needs one scenario per cell of a grid. This seed's cell is:
- Domain: {domain} ({domain_desc})
- Task type: {task_type} ({task_type_desc})
- Example of the kind of situation this cell covers: {cell_example}

The original request (it only sets the topic and situation; you will rewrite it):
<<<
{raw_request}
>>>

Write a scenario that:
1. Keeps the person's situation and topic from the request, adapted so that it fits the cell (domain and task type) above.
2. Is written in the first person, as the person who needs help, in plain everyday English.
3. Contains every fact needed to answer. The answer must follow from the stated facts alone, never from outside knowledge (no prices, laws, product specifications or general facts the reader would have to know).
4. Has one clear correct answer (a cause, a choice, a plan, a diagnosis or an order) that a careful reader can defend from the facts.
5. Has no names of real people, companies or products, no order or account numbers, and no contact details. Use roles and labels instead (my manager, Supplier A, the blue laptop, Venue 1).
6. Contains no instructions about the answer's form: no length, format, structure, style, tone, wording or language requests. Do not use any of these words anywhere: word, sentence, paragraph, bullet, list, heading, bold, format, tone, emoji, hashtag, quote, P.S., begin, end with, include the word, at least, at most, no more than.
7. Is about 80-150 words long, with 4-8 concrete facts (amounts, dates, times, deadlines, counts, durations) that later versions can build on.
8. Can support two kinds of harder versions written later from the same facts and the same correct answer: (a) adding facts that look relevant but do not change the answer, and (b) making the answer require 1, 2 or 3 chained inferences from the facts.
9. Ends with one direct request that matches the task type, for example "Why did this happen?", "Which option should I choose?", "What should my plan for the day be?", "What is going wrong, and how do I fix it?" or "Which should I handle first, and in what order?".

Return JSON with:
- "subtopic": 2-5 words naming the situation.
- "scenario": the scenario text, ending with the request.
- "core_objective": one sentence stating what the person needs to find out or decide.
- "key_facts": the facts in the scenario that the answer depends on, one per item.
- "correct_answer": the correct answer in one or two sentences, with the reason.
- "supports_both_mechanisms": true if both kinds of harder versions in point 8 are possible without changing the answer.
- "notes": anything you changed from the original request and why (one or two sentences).
