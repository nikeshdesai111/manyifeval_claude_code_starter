# ManyIFEval — Claude Code generation-phase starter

This is a **project handoff**, not a finished or executable implementation. It packages the complete v3.4 research plan in a text-native format plus instructions for Claude Code to implement the **dataset construction, 100-prompt pilot and response-generation infrastructure only**.

## Contents

| Path | Format | What it is |
|---|---|---|
| `research_plan.md` | Markdown | Complete v3.4 research plan converted from the original Word document. Keep all sections as reference; later probe/evaluation sections are context only, **not current implementation scope**. |
| `CLAUDE.md` | Markdown | Project instructions Claude Code reads from this working directory. |
| `START_HERE_PROMPT.md` | Markdown | The first message to paste into Claude Code after opening this directory. |
| `docs/decision_tracker.md` | Markdown | Companion master decision tracker, version 1.7. |
| `docs/research_plan_original_v3.4.docx` | Word | Original source plan for human reference and conversion cross-checking. |
| `configs/` | Directory | Claude Code will create and version actual experiment configurations here. |
| `src/` | Directory | Claude Code will implement the package here. |
| `tests/` | Directory | Claude Code will add unit/integration tests here. |
| `data/` | Directory | Generated manifests, pilot data and artifacts belong here; large generated files should not be committed to Git. |
| `.gitignore` | Git config | Basic ignores for caches, secrets and generated data. |

## How to get started on your Mac

1. Unzip the starter folder somewhere on your Mac (for example, `~/projects/manyifeval-project`).
2. Open a terminal in that folder (`cd ~/projects/manyifeval-project`).
3. Launch Claude Code from that directory using your installed Claude Code CLI.
4. Paste the contents of `START_HERE_PROMPT.md` as the first Claude Code message. You do not need to attach the Word file separately; `research_plan.md` is already in the same directory.
5. Check Claude's source-code integration, task classifications, model access and pilot configuration before it commits to large-scale GPU processing.

**Required stop point:** Complete and report the **100-prompt training-only pilot**, **4,000 generated responses** across the two LLMs. Do not proceed to the full **2,125,440 response** run without explicit approval.

## Current research design in brief

- All 216 ManyIFEval tasks; 72 groups of three tasks.
- Six newly sampled, compatible ManyIFEval exact rule instances per group, distinct types; no global fixed inventory.
- Six cyclic chains per group, `k=1..5`, balanced COA orderings `1/2/6/12/20` at k=1..5.
- 246 prompts/task; 53,136 prompts total; 20 completions/prompt **per generating model**.
- Generating checkpoints: Llama 3.1 8B Instruct and Qwen3-8B with **thinking off**.
- 57 training groups / 15 held-out test groups; no permanent validation group.
- Future per-rule probing: per instruction **type**; two objectives; 23 configurations each per generator. **Not in scope yet.**

## Provenance and caution

The Markdown research plan was converted from `ManyIFEval_Research_Plan_v3_4_Pilot_First_Protocol.docx` without intentionally changing research decisions. Its document tables were converted to Markdown tables. If a formula or complex formatting appears unusual, consult the original `.docx` under `docs/`.

The research plan describes later-stage modeling and evaluation, but the current implementation must stop at a verified generation pipeline and pilot report. Some task-family labels, generator hyperparameters, and backend feasibility details still require pilot checks and/or human decisions. Do **not** treat them as already empirically validated.
