# How the seeds were selected (difficulty-controlled dataset, seeds_v0)

Every seed (40 main, 80 spares, 8 calibration, 8 confirmation) was chosen by code in `src/mife/difficulty/seeds.py`
with the settings in `configs/difficulty/seeds.yaml`, on 2026-11-27. Selection finished at 17:56 EST, before any
task had been written for any seed and before any model response existed. No seed was chosen or ranked by looking
at tasks, judge verdicts or model responses. The only later change is the replacement rule below, which is
mechanical. Sections 5 and 6 list the human decisions that shaped the process.

## 1. Sources

- WildChat-1M (`allenai/WildChat-1M`, revision 7d6490e4, all 14 shards, 837,989 conversations). Only the first
  user turn of each conversation is used.
- Dolly-15k (`databricks/databricks-dolly-15k`, revision bdd27f4d), categories open_qa, general_qa, brainstorming and
  classification, records without a context field. Fallback only.

## 2. Filters (WildChat; `step_filter`)

A first turn is kept if all of these hold:
- the conversation is English, not flagged toxic and not redacted;
- the first turn itself is English, not toxic and not redacted;
- the OpenAI moderation flag for the first turn is false, and its Detoxify toxicity is at most 0.1;
- it is 40-1,500 characters long, with at most 5% non-ASCII characters;
- it matches none of the regexes `CODE_RE` (code), `EXCLUDE_RE` (role-play, creative writing, SEO or essay
  templates, translation or rewriting, sensitive topics) or `PII_RE` (emails, phone-like digit runs, URLs, street
  addresses, "my name is", @handles, letter salutations and signatures).

After exact-duplicate and same-conversation removal, 117,175 turns remain. Dolly records pass the same regexes,
with a minimum of 25 characters.

## 3. Ranking and classification

1. Embedding. Each kept turn and each of the 40 cell descriptions (`cell_description`: the domain gloss, the
   task-type gloss and the plan's cell example) is embedded with BAAI/bge-m3: dense CLS vector, L2-normalised,
   at most 512 tokens, fp16 on MPS.
2. Retrieval. For each cell, the top 200 turns by cosine to the cell description (top 40 for Dolly). Pooled over
   cells this gives 1,283 distinct WildChat turns, because the top lists overlap.
3. Classifier. A local Qwen3-30B-A3B-Instruct-2507 model (MLX, 4-bit; greedy decoding, the mlx-lm default)
   labels each candidate with the prompt in `CLS_PROMPT`: domain (one of 8 or other), task type (one of 5 or
   other), concrete own situation, sensitive, and personal data. It is not one of the three models under study.
4. Acceptance. A candidate is accepted for the cell named by its classifier labels if both labels are not "other",
   concrete is true, and sensitive and personal_data are false. Its rank within that cell is its cosine to that
   cell's description.
5. Top-ups for thin cells (`step_expand`). See section 6.

## 4. Selection (`step_select`; deterministic, no random draws)

The config records an RNG seed (2026112701), but the procedure is greedy and draws no random numbers.

1. Order of filling. The roles are filled in turn: seed, then spare1, then spare2. Within each role, cells are
   taken in increasing order of their number of accepted WildChat candidates (ties broken by grid order), so the
   scarcest cells pick first.
2. Pick per cell. In each cell the accepted WildChat candidates are tried in descending cosine order. The first one
   that passes these three checks is taken:
   - its hashed user (`hashed_ip`) is not used by any seed already chosen;
   - its conversation is not used by any seed already chosen;
   - its cosine to every seed already chosen is below 0.85.
   If no WildChat candidate qualifies, accepted Dolly candidates are tried the same way, and the row is flagged
   `dolly_fill`. If none qualifies either, the slot is recorded as missing.
3. Pilot seeds, after all main seeds and spares. There is one calibration seed and one confirmation seed per
   domain, both drawn from the remaining candidates under the same three checks.
   - Domains are taken in grid order: workplace, personal finance, travel, school, home, small business, shopping
     and orders, event organizing. Task types are in grid order: explanation, comparison, planning, troubleshooting,
     prioritization.
   - Calibration: domain i (0-7) prefers task type i mod 5. Confirmation: domain i prefers task type (i + 2) mod 5.
   - If the preferred cell has no usable candidate, the next type in rotation is tried (WildChat first, then
     Dolly).
   - The preferred rotation would use each task type at most twice per set, and the offset of 2 gives each domain a
     different preferred task type in the two sets.
   - Fallbacks changed the calibration set: home's preferred cell (prioritization) had no unused candidate, so it
     fell back to explanation. As a result, calibration has explanation 3 times and prioritization not at all.
     Confirmation has explanation 2, planning 2, troubleshooting 2, comparison 1 and prioritization 1.
4. Users. All 128 seed rows with a user have distinct hashed users, so main, spare, calibration and confirmation
   seeds share no user. Dolly rows have no user id; each counts as its own.

Result: all 40 main seeds were filled (39 WildChat, 1 Dolly: home × prioritization), with 9 spare slots missing.
Calibration: pc01-pc08, all WildChat. Confirmation: pf01-pf08, with pf07 (travel × prioritization) from Dolly.

## 5. Code and runs

- Filter and embedding ran with the code first committed in a2872fd.
- The first classification and selection pass ran with the code in aef16d0 (top-up step added, 50 per cell).
- The second top-up and the final selection ran with the same code plus `per_cell = 80`, committed in 027dafd.
- The seed replacement code is in 1c08417.
- Raw data and intermediate files are in data/difficulty/seeds_work/ (gitignored). That includes the classifier
  cache with every raw label, `seeds_pass1.parquet` (after the first pass) and `seeds.parquet` (final).

## 6. Non-mechanical steps (stated plainly)

- The plan said 300 candidates per cell; I used 200 to fit the Mac time budget.
- After the first classification pass, 17 cells had fewer than 3 accepted candidates. I then added a targeted
  top-up (`step_expand`), which I designed after seeing those coverage counts. It saw only counts, never task or
  model outcomes.
  - For each cell under 3, it adds the next 50 WildChat turns, ranked by cell-specific similarity (cosine to the
    cell minus the mean cosine to all cells, plus half the cosine to the cell).
  - It keeps only turns that match a task-type keyword pattern (`TYPE_HINT_RE`) that I wrote then.
  - A second top-up (80 per cell) was run for the 7 cells still under 3.
  - The 50 and 80 limits were chosen for time, not tuned.
- The classifier is an LLM and noisy. Some accepted seeds are loose fits for their cell (pc01's raw request was a
  school assignment to design a TV advert; pc02b's asked to explain a family saying). Prompt 1 adapts each raw
  request into the cell's scenario. No seed was hand-edited or hand-picked.
- The 2-seed test batches (pc01, pc02 and pc02b) were run on pilot seeds, not on main seeds.

## 7. Replacement rule

A seed is replaced only if the task judge still rejects at least one of its 6 tasks after the maximum number of
rounds (1 write plus 3 rewrites, `MAX_REWRITES = 3`).
- The replacement is the next-ranked accepted candidate in the same cell (same order as section 4): unused hashed
  user, unused conversation, and cosine below 0.85 to every seed row, including replaced ones.
- `mife.difficulty.seeds replace` picks it mechanically (`tasks --auto-replace` does so during a run).
- It is logged in `configs/difficulty/seed_replacements.yaml` and reapplied by any later `seeds select`.
- The replaced seed keeps its id with role `replaced`, and its outputs are kept.

Replacements so far (from seed_replacements.yaml):

| Seed | Cell | Replaced by | Reason | Date |
|---|---|---|---|---|
| pc02 | home/explanation | pc02b | scenario has two parallel outcomes, can't form a single 1/2/3 reasoning chain; judge rejected after 3 rewrites at low and medium effort | 2026-11-27 |

pc02 was the first application, and the rule above was written down afterwards. pc02 met the rule: its tasks were
rejected after the maximum rounds in both test runs. Any later replacements made by `--auto-replace` are appended
to the yaml automatically, with their reason.
