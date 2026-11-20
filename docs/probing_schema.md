# Probing-phase storage schema v1.0.0

Everything here is derived from the frozen inputs (`docs/storage_schema.md`) and keyed by the same stable IDs
(`prompt_id`, `rule_id`, `position`, `group_id`). Code: `src/mife/probing/`. TEST-split files exist only after the
pre-specification is frozen and unblinding is approved (`results/prespec/FROZEN.json`).

## `data/probing_features/` (caches; rebuild with `probe-features`)

| Path | Grain / shape | Notes |
|---|---|---|
| `bge_length_audit.json` | — | BGE-M3 token lengths of every prompt and rule text; limit 8,192; no truncation |
| `shared/S1_{split}.npy` + `_keys.parquet` + `S1_columns.json` | prompt–rule row × 75 | structure, Objective 1 (k, type, position, typed parameters, companions, family, lengths) |
| `shared/S2_*`, `shared/M2_*` | prompt × 175 / 101 | Objective 2 structure (5 ordered slots + masks) / H1–H2 metadata |
| `shared/EP_{split}.npy` | prompt × 1,024 | frozen BGE-M3 dense embedding of the full prompt (L2-normalised [CLS]) |
| `shared/ER_rules_{split}.npy` + keys | exact rule × 1,024 | frozen BGE-M3 embedding of each rule text |
| `{model}/H{row}_{split}.npy` | prompt × 4,096 | final-prompt-token hidden state at a candidate row (float32 decoded from BF16) |
| `{model}/Asum_*`, `Amean_*`, `Amask_*` | prompt × L × heads × 5 | final-token attention to each rule span (slots in prompt order; absent = 0, mask False) |
| `b2_bundle/` | TRAIN only | inputs for remote B2 jobs: prompt/rule texts, Objective 1/2 labels, S1/S2, folds |
| `b2_runs/method.json` | — | full-vs-LoRA check result and the chosen method |
| `b2_runs/grid/{model}_o{obj}_f{fold}_{adapt}_lr{lr}_ep{ep}/` | per run | `result.json` (held-out loss, timing, seed), `heldout.npz`, `embeddings_fp16.npz` (EP, ER of all TRAIN prompts/rules from the adapted encoder) |

## `results/`

| Path | Contents |
|---|---|
| `RESULTS.md` | the written report, generated from `tables/` by `probe-report` |
| `run_manifest.json` | versions, seeds, data hashes, source-tree hash, wall times |
| `tables/*.csv` | every table in RESULTS.md (`desc_*` / `sens_*` Phase 1, `b2_*` compute, `cv_*` Phase 3, `eval1_*`, `contrasts`, `eval2_*` Phase 5, `runtime`, `open_issues`). CV tables: `cv_folds`, `cv_grid_all` (every setting), `cv_selected` (setting in use per configuration, with `minus_ref_structure`), `cv_layer_selection`, `cv_pooling_selection`, `cv_b2_grid`, `cv_summary` (best per family and references), `cv_complementarity`, `cv_l2_counts`, `cv_l2_edges` (edge selections and the loss one step inwards), `cv_boundary_hits`, `cv_identical_features`, `cv_sparse_types` |
| `figures/*.png` | every figure in RESULTS.md |
| `cv/folds.json` | the 5 folds (12/12/11/11/11 TRAIN groups), per-type coverage |
| `cv/{model}/obj{o}/selection.json` | selected hidden row and attention pooling with their CV scores |
| `cv/{model}/obj{o}/grid/{key}.json` | one configuration × all head settings: fold losses, MLP per-epoch losses, selection |
| `cv/{model}/obj{o}/b2_selection.json` | B2 LR × epochs grid and the selected point |
| `predictions/{model}/obj{o}/cv/{key}.parquet` | out-of-fold predictions, one column per grid point, keyed by prompt_id (+ position, rule_id, type_id) and fold |
| `predictions/{model}/obj{o}/test/*.parquet` | TEST predictions, one column per configuration (Phase 5) |
| `models/{model}/obj{o}/{key}.pt` (+ `.json`) | refitted heads (per type or shared): scaler + weights only |
| `models/{model}/obj{o}/references.json` | TRAIN rates for the non-learned references |
| `prespec/analysis_plan.md`, `prespec/FROZEN.json` | the frozen pre-specification and its hash; unblinding approval |

Configuration keys are `<config>[__h<row>][__a<pooling>]`, e.g. `B1+H2+A1__h16__asum`.
