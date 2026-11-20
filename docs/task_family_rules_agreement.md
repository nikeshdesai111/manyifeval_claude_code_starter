# Rule-based task-family labels: agreement with the hand labels

**What this is.** `src/mife/family_rules.py` turns the rubric in `docs/task_family_rubric.md` into a deterministic script: named keyword/pattern cues per family over the lower-cased task text, the rubric's precedence applied literally (TRANSFORM > CREATIVE > PERSUASIVE > PRACTICAL > ANALYTICAL > INFO), INFO as the default, no tone cues, and letters/emails (including to friends) as PRACTICAL. `scripts/family_rules_audit.py` runs it on all 216 tasks and compares the output with `configs/task_families_v1.csv`.

**What it is not.** It does not replace the hand labels. The pipeline still loads `configs/task_families_v1.csv` (`mife.tasks.load_families`), and the frozen grouping, split and data are unchanged.

**How the rules were written.** The cues come from the rubric's family table and precedence list only. They were committed (commit `6326400`) before the script was compared with the hand labels, and were not changed after the comparison, so the agreement below is not tuned. Before that commit, while reading the rule output for all 216 task texts (without labels), two rubric-faithfulness bugs were fixed: (a) a task that *critiques* a supplied ad or pitch was being called PERSUASIVE because the object is an ad; rule 3 asks for the deliverable's purpose, so a critique-of-supplied-text guard skips the persuasion cues; (b) "I want / I need a …" was added as a deliverable verb.

Reproduce: `PYTHONPATH=src python scripts/family_rules_audit.py` (needs the pinned ManyIFEval checkout at `third_party/`; the data hash is checked). Outputs: `audits/family_rules/rule_labels.csv` (per task: rule label, deciding cue, matched text, hand label) and `audits/family_rules/agreement.json`.

## Headline agreement (216 tasks)

| Measure | Value |
|---|---|
| Accuracy (rule = hand family) | 190/216 = **88.0%** |
| Cohen's kappa | **0.850** |
| Accuracy if the hand `alt_family` also counts | 204/216 = 94.4% |
| Accuracy on the 148 tasks not flagged ambiguous | 144/148 = 97.3% |
| Accuracy on the 68 tasks flagged ambiguous | 46/68 = 67.6% |
| Disagreements | 26 |
| … on tasks flagged ambiguous | 22 of 26 |
| … where the rule label equals the hand `alt_family` | 14 of 26 |
| Tasks decided by the INFO default (no cue fired) | 51 |

Family counts: hand / rule. TRANSFORM 15 / 16, CREATIVE 52 / 43, PERSUASIVE 33 / 24, PRACTICAL 59 / 64, ANALYTICAL 22 / 18, INFO 35 / 51. Every rule family stays at or above the 12-task sparse-family threshold.

## Confusion matrix (rows = hand label, columns = rule label)

| hand \ rule | TRANSFORM | CREATIVE | PERSUASIVE | PRACTICAL | ANALYTICAL | INFO |
|---|---|---|---|---|---|---|
| TRANSFORM | **15** | 0 | 0 | 0 | 0 | 0 |
| CREATIVE | 0 | **43** | 0 | 1 | 0 | 8 |
| PERSUASIVE | 1 | 0 | **24** | 5 | 0 | 3 |
| PRACTICAL | 0 | 0 | 0 | **57** | 0 | 2 |
| ANALYTICAL | 0 | 0 | 0 | 1 | **17** | 4 |
| INFO | 0 | 0 | 0 | 0 | 1 | **34** |

The script errs towards INFO: in 17 of the 26 disagreements the rule label is INFO because no explicit cue fired. Hand labels used judgement about implied purpose or implied narrative; the script only fires on explicit wording.

## The 26 disagreements

Amb = flagged ambiguous in the CSV. "Alt?" = the rule label equals the hand `alt_family`.

| Task | Hand | Alt | Amb | Rule (cue) | Alt? | Task text (start) |
|---|---|---|---|---|---|---|
| 179 | PERSUASIVE | – | no | PRACTICAL (proposal) | – | Filmmaker seeking financing… essay for the proposal to investors |
| 1512 | CREATIVE | – | no | INFO (default) | – | Generate a forum thread about several people waiting… |
| 2341 | ANALYTICAL | – | no | INFO (default) | – | Is praying for someone's health a good idea? |
| 3691 | CREATIVE | – | no | INFO (default) | – | Blog post from the perspective of a 16 year old girl… |
| 295 | CREATIVE | PRACTICAL | yes | PRACTICAL (correspondence) | yes | Write me a letter about the mandates of the King |
| 1051 | PERSUASIVE | PRACTICAL | yes | PRACTICAL (correspondence) | yes | Letter to a friend ask them to go and vote |
| 1834 | ANALYTICAL | PRACTICAL | yes | PRACTICAL (how to) | yes | Is pouring coke on a rusty chain a good way…? |
| 1934 | ANALYTICAL | INFO | yes | INFO (default) | yes | Why star wars is so popular? |
| 2169 | ANALYTICAL | INFO | yes | INFO (default) | yes | Why didn't the 2022 winter olympics have the best ratings? |
| 2209 | PERSUASIVE | PRACTICAL | yes | PRACTICAL (proposal) | yes | Academic proposal to a customer… |
| 2303 | INFO | ANALYTICAL | yes | ANALYTICAL (is different from) | yes | Blog post on how the outer solar system is different… |
| 2441 | PERSUASIVE | INFO | yes | INFO (default) | yes | Speech-like paragraphs on the US elections |
| 2749 | PERSUASIVE | INFO | yes | INFO (default) | yes | Blog post on the benefits of a digital marketing agency |
| 2829 | CREATIVE | INFO | yes | INFO (default) | yes | Why did the man travel from Saskatoon to Helsinki…? |
| 2997 | PERSUASIVE | PRACTICAL | yes | PRACTICAL (template) | yes | Template for a letter to the editor (you feel strongly…) |
| 3186 | ANALYTICAL | INFO | yes | INFO (default) | yes | Article on how intra-team conflict affected sports teams |
| 3362 | PERSUASIVE | TRANSFORM | yes | TRANSFORM (expanding "…") | yes | Ad copy by expanding "Get 40 miles per gallon…" |
| 296 | CREATIVE | PERSUASIVE | yes | INFO (default) | no | Write an elaborate compliment to Kruger |
| 1268 | CREATIVE | PRACTICAL | yes | INFO (default) | no | Example journal entry about stress management |
| 1858 | PERSUASIVE | CREATIVE | yes | PRACTICAL (correspondence) | no | Very angry letter to someone… 1+1=3 |
| 1967 | PRACTICAL | TRANSFORM | yes | INFO (default) | no | What are some startup ideas…? |
| 2100 | CREATIVE | ANALYTICAL | yes | INFO (default) | no | What fashion would Frankenstein's bride wear? |
| 2304 | PERSUASIVE | CREATIVE | yes | INFO (default) | no | Essay as if you are the president, targeting moms |
| 2432 | CREATIVE | PRACTICAL | yes | INFO (default) | no | Condolences, offering help, sharing a story |
| 3534 | CREATIVE | PERSUASIVE | yes | INFO (default) | no | Tweet storm about finding out the earth is not flat |
| 3732 | PRACTICAL | INFO | yes | INFO (default) | no | Best things to do in Rochester, New York? |

## Points for the reviewer

- **Only 4 disagreements fall on tasks the hand labeller did not flag** (179, 1512, 2341, 3691). These are the ones worth a human look first; each would also be a reasonable candidate for the ambiguous flag.
- **Two hand labels look inconsistent with the rubric's own precedence.** 3362 supplies a text and asks to expand it, so rule 1 gives TRANSFORM, but it is hand-labelled PERSUASIVE. 1858 and 1051 are letters with no explicit persuasive wording, which the correspondence policy makes PRACTICAL. A reviewer should either relabel these or amend the rubric to say that persuasion beats rule 1 / the correspondence policy.
- **Most remaining disagreements are implied-purpose judgements** (a "why" question read as analysis, a blog post about an agency's benefits read as marketing, an imagined scenario read as narrative). Keyword rules cannot see those without bespoke cues, and adding bespoke cues after seeing the labels would defeat the purpose of the check.
- **For the paper:** the script reproduces the labels with kappa 0.85 (almost perfect agreement on the conventional Landis–Koch scale) and 97% agreement where the hand labeller was confident. A defensible statement is that the families follow from the written rubric, with disagreements concentrated on tasks already flagged ambiguous. It is not evidence that the hand labels are correct, because both labellings derive from the same rubric.
