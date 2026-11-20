# Pre-specified analysis plan (frozen before the TEST set is read)

Written 2026-11-14T01:31:31. Every choice below comes from TRAINING data only (5-fold GroupKFold over the 57 TRAIN groups, `results/cv/folds.json`). The 15 TEST groups (11,070 prompts per generator) have not been read: no labels, outputs, descriptive statistics or feature scaling.

## 1. Selected settings

### Shared hidden row and attention pooling (per generator and objective)

| model | objective | hidden_row | depth | best_cv_loss |
|---|---|---|---|---|
| llama | 1 | 16 | 50% | 0.37613 |
| llama | 2 | 16 | 50% | 0.016559 |
| qwen | 1 | 18 | 50% | 0.26132 |
| qwen | 2 | 27 | 75% | 0.011297 |

| model | objective | pooling | best_cv_loss |
|---|---|---|---|
| llama | 1 | sum | 0.33157 |
| llama | 2 | sum | 0.01693 |
| qwen | 1 | sum | 0.23606 |
| qwen | 2 | sum | 0.01274 |

### B2 adaptation

Full vs LoRA check: lower held-out-fold NLL on (llama, objective 1, fold 0, 1 epoch, lower LR); tie -> full. Chosen: **lora**.

| model | objective | adapt | lr | epochs | cv_loss |
|---|---|---|---|---|---|
| llama | 1 | lora | 0.0001 | 1 | 0.37236 |
| llama | 2 | lora | 0.0002 | 3 | 0.016475 |
| qwen | 1 | lora | 0.0001 | 1 | 0.26869 |
| qwen | 2 | lora | 0.0002 | 1 | 0.014329 |

### Head setting per configuration (selected by mean held-out-fold loss; MLP epoch chosen inside CV)

| model | objective | family | config | key | setting | epoch | cv_loss | n_features | per_type_heads |
|---|---|---|---|---|---|---|---|---|---|
| llama | 1 | reference | REF_constant | REF_constant | fitting-fold rates | <NA> | 0.47048 | <NA> |  |
| llama | 1 | reference | REF_type | REF_type | fitting-fold rates | <NA> | 0.41551 | <NA> |  |
| llama | 1 | reference | REF_type_k | REF_type_k | fitting-fold rates | <NA> | 0.41517 | <NA> |  |
| llama | 1 | reference | REF_structure | REF_structure | mlp_l2=0.01 | 10 | 0.37789 | 75 | False |
| llama | 1 | probe-only | H1 | H1__h16 | mlp_l2=0.1 | 23 | 0.37613 | 4096 | True |
| llama | 1 | probe-only | H2 | H2__h16 | mlp_l2=0.1 | 6 | 0.33895 | 5120 | True |
| llama | 1 | probe-only | A1 | A1__asum | mlp_l2=0.01 | 9 | 0.33157 | 1024 | True |
| llama | 1 | probe-only | H1+H2 | H1+H2__h16 | mlp_l2=0.1 | 11 | 0.33961 | 5120 | True |
| llama | 1 | probe-only | H1+A1 | H1+A1__h16__asum | mlp_l2=0.1 | 7 | 0.35831 | 5120 | True |
| llama | 1 | probe-only | H2+A1 | H2+A1__h16__asum | mlp_l2=0.1 | 7 | 0.33644 | 6144 | True |
| llama | 1 | probe-only | H1+H2+A1 | H1+H2+A1__h16__asum | mlp_l2=0.1 | 7 | 0.33628 | 6144 | True |
| llama | 1 | B1 family | B1 | B1 | mlp_l2=0.1 | 4 | 0.38682 | 2123 | False |
| llama | 1 | B1 family | B1+H1 | B1+H1__h16 | mlp_l2=0.1 | 8 | 0.34292 | 6219 | True |
| llama | 1 | B1 family | B1+H2 | B1+H2__h16 | mlp_l2=0.1 | 8 | 0.34152 | 6219 | True |
| llama | 1 | B1 family | B1+A1 | B1+A1__asum | mlp_l2=0.1 | 5 | 0.34514 | 3147 | True |
| llama | 1 | B1 family | B1+H1+H2 | B1+H1+H2__h16 | mlp_l2=0.1 | 6 | 0.33932 | 6219 | True |
| llama | 1 | B1 family | B1+H1+A1 | B1+H1+A1__h16__asum | mlp_l2=0.1 | 6 | 0.33896 | 7243 | True |
| llama | 1 | B1 family | B1+H2+A1 | B1+H2+A1__h16__asum | mlp_l2=0.1 | 19 | 0.34098 | 7243 | True |
| llama | 1 | B1 family | B1+H1+H2+A1 | B1+H1+H2+A1__h16__asum | mlp_l2=0.1 | 6 | 0.3407 | 7243 | True |
| llama | 1 | B2 family | B2 | B2 | mlp_l2=0.1 | 1 | 0.37003 | 2123 | False |
| llama | 1 | B2 family | B2+H1 | B2+H1__h16 | mlp_l2=0.1 | 3 | 0.33647 | 6219 | True |
| llama | 1 | B2 family | B2+H2 | B2+H2__h16 | mlp_l2=0.1 | 5 | 0.33468 | 7243 | True |
| llama | 1 | B2 family | B2+A1 | B2+A1__asum | mlp_l2=0.1 | 3 | 0.34724 | 3147 | True |
| llama | 1 | B2 family | B2+H1+H2 | B2+H1+H2__h16 | mlp_l2=0.1 | 11 | 0.33451 | 7243 | True |
| llama | 1 | B2 family | B2+H1+A1 | B2+H1+A1__h16__asum | mlp_l2=0.1 | 19 | 0.33496 | 7243 | True |
| llama | 1 | B2 family | B2+H2+A1 | B2+H2+A1__h16__asum | mlp_l2=0.1 | 16 | 0.33532 | 8267 | True |
| llama | 1 | B2 family | B2+H1+H2+A1 | B2+H1+H2+A1__h16__asum | mlp_l2=0.1 | 10 | 0.33406 | 8267 | True |
| llama | 2 | reference | REF_constant | REF_constant | fitting-fold rates | <NA> | 0.023636 | <NA> |  |
| llama | 2 | reference | REF_type | REF_type | fitting-fold rates | <NA> | 0.020456 | <NA> |  |
| llama | 2 | reference | REF_type_k | REF_type_k | fitting-fold rates | <NA> | 0.020362 | <NA> |  |
| llama | 2 | reference | REF_structure | REF_structure | linear_l2=0.01 | <NA> | 0.018467 | 175 | False |
| llama | 2 | probe-only | H1 | H1__h16 | linear_l2=0.01 | <NA> | 0.016559 | 4197 | False |
| llama | 2 | probe-only | H2 | H2__h16 | linear_l2=0.1 | <NA> | 0.016434 | 9317 | False |
| llama | 2 | probe-only | A1 | A1__asum | linear_l2=0.01 | <NA> | 0.01693 | 5131 | False |
| llama | 2 | probe-only | H1+H2 | H1+H2__h16 | linear_l2=0.1 | <NA> | 0.016434 | 9317 | False |
| llama | 2 | probe-only | H1+A1 | H1+A1__h16__asum | linear_l2=0.01 | <NA> | 0.015847 | 9322 | False |
| llama | 2 | probe-only | H2+A1 | H2+A1__h16__asum | linear_l2=0.1 | <NA> | 0.016026 | 14442 | False |
| llama | 2 | probe-only | H1+H2+A1 | H1+H2+A1__h16__asum | linear_l2=0.1 | <NA> | 0.016026 | 14442 | False |
| llama | 2 | B1 family | B1 | B1 | mlp_l2=0.01 | 23 | 0.018383 | 6319 | False |
| llama | 2 | B1 family | B1+H1 | B1+H1__h16 | mlp_l2=0.01 | 25 | 0.01642 | 10415 | False |
| llama | 2 | B1 family | B1+H2 | B1+H2__h16 | mlp_l2=0.01 | 18 | 0.016687 | 10415 | False |
| llama | 2 | B1 family | B1+A1 | B1+A1__asum | linear_l2=0.1 | <NA> | 0.017738 | 11444 | False |
| llama | 2 | B1 family | B1+H1+H2 | B1+H1+H2__h16 | mlp_l2=0.01 | 26 | 0.016912 | 10415 | False |
| llama | 2 | B1 family | B1+H1+A1 | B1+H1+A1__h16__asum | linear_l2=0.1 | <NA> | 0.016384 | 15540 | False |
| llama | 2 | B1 family | B1+H2+A1 | B1+H2+A1__h16__asum | linear_l2=0.1 | <NA> | 0.016384 | 15540 | False |
| llama | 2 | B1 family | B1+H1+H2+A1 | B1+H1+H2+A1__h16__asum | linear_l2=0.1 | <NA> | 0.016384 | 15540 | False |
| llama | 2 | B2 family | B2 | B2 | mlp_l2=0.1 | 1 | 0.016126 | 6319 | False |
| llama | 2 | B2 family | B2+H1 | B2+H1__h16 | linear_l2=1 | <NA> | 0.015828 | 10415 | False |
| llama | 2 | B2 family | B2+H2 | B2+H2__h16 | linear_l2=1 | <NA> | 0.015655 | 15535 | False |
| llama | 2 | B2 family | B2+A1 | B2+A1__asum | linear_l2=1 | <NA> | 0.015944 | 11444 | False |
| llama | 2 | B2 family | B2+H1+H2 | B2+H1+H2__h16 | linear_l2=1 | <NA> | 0.015655 | 15535 | False |
| llama | 2 | B2 family | B2+H1+A1 | B2+H1+A1__h16__asum | linear_l2=1 | <NA> | 0.015728 | 15540 | False |
| llama | 2 | B2 family | B2+H2+A1 | B2+H2+A1__h16__asum | linear_l2=1 | <NA> | 0.015619 | 20660 | False |
| llama | 2 | B2 family | B2+H1+H2+A1 | B2+H1+H2+A1__h16__asum | linear_l2=1 | <NA> | 0.015619 | 20660 | False |
| qwen | 1 | reference | REF_constant | REF_constant | fitting-fold rates | <NA> | 0.32557 | <NA> |  |
| qwen | 1 | reference | REF_type | REF_type | fitting-fold rates | <NA> | 0.27646 | <NA> |  |
| qwen | 1 | reference | REF_type_k | REF_type_k | fitting-fold rates | <NA> | 0.27563 | <NA> |  |
| qwen | 1 | reference | REF_structure | REF_structure | mlp_l2=0.01 | 3 | 0.25734 | 75 | False |
| qwen | 1 | probe-only | H1 | H1__h18 | mlp_l2=1 | 2 | 0.26132 | 4096 | True |
| qwen | 1 | probe-only | H2 | H2__h18 | mlp_l2=0.1 | 21 | 0.23965 | 5120 | True |
| qwen | 1 | probe-only | A1 | A1__asum | mlp_l2=0.1 | 8 | 0.23606 | 1152 | True |
| qwen | 1 | probe-only | H1+H2 | H1+H2__h18 | mlp_l2=0.1 | 15 | 0.24027 | 5120 | True |
| qwen | 1 | probe-only | H1+A1 | H1+A1__h18__asum | mlp_l2=0.1 | 24 | 0.25555 | 5248 | True |
| qwen | 1 | probe-only | H2+A1 | H2+A1__h18__asum | mlp_l2=0.1 | 12 | 0.23864 | 6272 | True |
| qwen | 1 | probe-only | H1+H2+A1 | H1+H2+A1__h18__asum | mlp_l2=0.1 | 5 | 0.23578 | 6272 | True |
| qwen | 1 | B1 family | B1 | B1 | mlp_l2=0.1 | 2 | 0.26462 | 2123 | False |
| qwen | 1 | B1 family | B1+H1 | B1+H1__h18 | mlp_l2=0.1 | 5 | 0.23963 | 6219 | True |
| qwen | 1 | B1 family | B1+H2 | B1+H2__h18 | mlp_l2=0.1 | 11 | 0.24014 | 6219 | True |
| qwen | 1 | B1 family | B1+A1 | B1+A1__asum | mlp_l2=0.1 | 17 | 0.23794 | 3275 | True |
| qwen | 1 | B1 family | B1+H1+H2 | B1+H1+H2__h18 | mlp_l2=0.1 | 4 | 0.23948 | 6219 | True |
| qwen | 1 | B1 family | B1+H1+A1 | B1+H1+A1__h18__asum | mlp_l2=0.1 | 12 | 0.23867 | 7371 | True |
| qwen | 1 | B1 family | B1+H2+A1 | B1+H2+A1__h18__asum | mlp_l2=0.1 | 26 | 0.23873 | 7371 | True |
| qwen | 1 | B1 family | B1+H1+H2+A1 | B1+H1+H2+A1__h18__asum | mlp_l2=0.1 | 21 | 0.23742 | 7371 | True |
| qwen | 1 | B2 family | B2 | B2 | mlp_l2=1 | 21 | 0.26057 | 2123 | False |
| qwen | 1 | B2 family | B2+H1 | B2+H1__h18 | mlp_l2=0.1 | 17 | 0.23595 | 6219 | True |
| qwen | 1 | B2 family | B2+H2 | B2+H2__h18 | mlp_l2=1 | 2 | 0.23491 | 7243 | True |
| qwen | 1 | B2 family | B2+A1 | B2+A1__asum | mlp_l2=1 | 2 | 0.24088 | 3275 | True |
| qwen | 1 | B2 family | B2+H1+H2 | B2+H1+H2__h18 | mlp_l2=0.1 | 23 | 0.23308 | 7243 | True |
| qwen | 1 | B2 family | B2+H1+A1 | B2+H1+A1__h18__asum | mlp_l2=1 | 2 | 0.23582 | 7371 | True |
| qwen | 1 | B2 family | B2+H2+A1 | B2+H2+A1__h18__asum | mlp_l2=0.1 | 17 | 0.23477 | 8395 | True |
| qwen | 1 | B2 family | B2+H1+H2+A1 | B2+H1+H2+A1__h18__asum | mlp_l2=0.1 | 28 | 0.23238 | 8395 | True |
| qwen | 2 | reference | REF_constant | REF_constant | fitting-fold rates | <NA> | 0.015768 | <NA> |  |
| qwen | 2 | reference | REF_type | REF_type | fitting-fold rates | <NA> | 0.013788 | <NA> |  |
| qwen | 2 | reference | REF_type_k | REF_type_k | fitting-fold rates | <NA> | 0.013499 | <NA> |  |
| qwen | 2 | reference | REF_structure | REF_structure | linear_l2=0.01 | <NA> | 0.012749 | 175 | False |
| qwen | 2 | probe-only | H1 | H1__h27 | linear_l2=0.01 | <NA> | 0.011297 | 4197 | False |
| qwen | 2 | probe-only | H2 | H2__h27 | linear_l2=0.1 | <NA> | 0.011917 | 9317 | False |
| qwen | 2 | probe-only | A1 | A1__asum | linear_l2=0.01 | <NA> | 0.01274 | 5771 | False |
| qwen | 2 | probe-only | H1+H2 | H1+H2__h27 | linear_l2=0.1 | <NA> | 0.011917 | 9317 | False |
| qwen | 2 | probe-only | H1+A1 | H1+A1__h27__asum | linear_l2=0.01 | <NA> | 0.011027 | 9962 | False |
| qwen | 2 | probe-only | H2+A1 | H2+A1__h27__asum | linear_l2=0.1 | <NA> | 0.011644 | 15082 | False |
| qwen | 2 | probe-only | H1+H2+A1 | H1+H2+A1__h27__asum | linear_l2=0.1 | <NA> | 0.011644 | 15082 | False |
| qwen | 2 | B1 family | B1 | B1 | mlp_l2=0.01 | 16 | 0.0136 | 6319 | False |
| qwen | 2 | B1 family | B1+H1 | B1+H1__h27 | linear_l2=0.1 | <NA> | 0.012245 | 10415 | False |
| qwen | 2 | B1 family | B1+H2 | B1+H2__h27 | linear_l2=0.1 | <NA> | 0.012245 | 10415 | False |
| qwen | 2 | B1 family | B1+A1 | B1+A1__asum | linear_l2=0.1 | <NA> | 0.013226 | 12084 | False |
| qwen | 2 | B1 family | B1+H1+H2 | B1+H1+H2__h27 | mlp_l2=0.01 | 25 | 0.012089 | 10415 | False |
| qwen | 2 | B1 family | B1+H1+A1 | B1+H1+A1__h27__asum | linear_l2=0.1 | <NA> | 0.011992 | 16180 | False |
| qwen | 2 | B1 family | B1+H2+A1 | B1+H2+A1__h27__asum | linear_l2=0.1 | <NA> | 0.011992 | 16180 | False |
| qwen | 2 | B1 family | B1+H1+H2+A1 | B1+H1+H2+A1__h27__asum | linear_l2=0.1 | <NA> | 0.011992 | 16180 | False |
| qwen | 2 | B2 family | B2 | B2 | mlp_l2=0.1 | 28 | 0.013788 | 6319 | False |
| qwen | 2 | B2 family | B2+H1 | B2+H1__h27 | mlp_l2=0.1 | 5 | 0.013752 | 10415 | False |
| qwen | 2 | B2 family | B2+H2 | B2+H2__h27 | mlp_l2=0.1 | 30 | 0.013481 | 15535 | False |
| qwen | 2 | B2 family | B2+A1 | B2+A1__asum | mlp_l2=0.1 | 30 | 0.013652 | 12084 | False |
| qwen | 2 | B2 family | B2+H1+H2 | B2+H1+H2__h27 | mlp_l2=0.1 | 22 | 0.013573 | 15535 | False |
| qwen | 2 | B2 family | B2+H1+A1 | B2+H1+A1__h27__asum | mlp_l2=0.1 | 17 | 0.013654 | 16180 | False |
| qwen | 2 | B2 family | B2+H2+A1 | B2+H2+A1__h27__asum | mlp_l2=0.1 | 15 | 0.013421 | 21300 | False |
| qwen | 2 | B2 family | B2+H1+H2+A1 | B2+H1+H2+A1__h27__asum | mlp_l2=0.1 | 4 | 0.013572 | 21300 | False |

## 2. Models and fitting

- Objective 1 heads: one per instruction TYPE (15) for every configuration containing H1, H2 or A1; one shared rule-conditioned head for B1-only, B2-only and the structure reference. Objective 2: one shared head per configuration.
- Loss: Objective 1 aggregated binomial NLL with n = 20 (mean per completion); Objective 2 sigmoid-bounded MSE.
- Penalty: (l2/2)·||W||² on weights after fold-wise standardisation; grid l2 ∈ [0.01, 0.1, 1.0], linear vs MLP(128), MLP Adam lr 0.001, batch 1024, up to 30 epochs.
- Final models: each selected setting refitted on all 57 TRAIN groups (MLP: the CV-selected epoch count), then applied once to the 15 TEST groups. B2: the selected LR × epochs refitted on all 57 TRAIN groups; its frozen embeddings feed B2 and every B2+probe head. TEST prompts and rules are embedded with the saved final adapters on Betty (CUDA), the backend that produced the TRAIN embeddings the final B2 heads were fitted on.
- Reference rows at test time (as in CV, with all 57 TRAIN groups as the fitting set): REF_constant = TRAIN mean label; REF_type and REF_type_k = TRAIN violation rates per type (per type × k), averaged over the prompt's rules for Objective 2; REF_structure = its refitted head. Standardisation of every head uses TRAIN statistics only.
- Features: H = final-prompt-token hidden state at the selected row; A = final-token attention to the target rule span over all layers × heads (selected pooling); Objective 2 uses 5 ordered slots with presence masks. Exact rule IDs are never model inputs.

## 3. Metrics (definitions in `src/mife/probing/metrics.py`)

- Objective 1 (prompt–rule rows, 20 responses each): binomial NLL per completion (primary); Brier on per-response binary outcomes; ECE; AUROC on per-response binary outcomes (ties = 1/2); RiskTop / RiskBottom = observed violation rate in the top / bottom decile of predicted risk (decile thresholds from the full test set, fixed within the bootstrap). Reported overall, per type (plus the macro mean over the 15 types) and per k.
- Objective 2 (prompts): MAE, RMSE (primary), ECE, Pearson, Spearman; overall and per k.
- Calibration binning: 10 bins whose edges are the deciles of the configuration's TRAINING out-of-fold CV predictions at its selected setting; ECE weights bins by responses (Objective 1) or prompts (Objective 2). Reliability curves use the same bins.

## 4. Uncertainty

- Paired cluster bootstrap over the 15 TEST groups: B = 10000, seed 20261203; every configuration and contrast uses the SAME replicate weights. All metrics are computed exactly from group-level sufficient statistics (AUROC via the cross-group concordance matrix; Spearman recomputed per replicate). 95% percentile intervals.

## 5. Hypothesis contrasts (difference = metric(b) − metric(a); negative favours b)

Primary metric: Objective 1 NLL, Objective 2 RMSE; the other metrics are reported descriptively. `BETTER(x,y)` is whichever of x, y has the lower TRAINING CV loss. Each contrast is run for both objectives and both generators.

| family | a | b |
|---|---|---|
| C1 prompt-only vs internal | B1 | H1 |
| C1 prompt-only vs internal | B1 | H2 |
| C1 prompt-only vs internal | B1 | A1 |
| C1 prompt-only vs internal | B2 | H1 |
| C1 prompt-only vs internal | B2 | H2 |
| C1 prompt-only vs internal | B2 | A1 |
| C2 hidden-state increment | B1 | B1+H1 |
| C2 hidden-state increment | B1 | B1+H2 |
| C2 hidden-state increment | B2 | B2+H1 |
| C2 hidden-state increment | B2 | B2+H2 |
| C3 attention increment | B1 | B1+A1 |
| C3 attention increment | B2 | B2+A1 |
| C4 complementarity | BETTER(H1,A1) | H1+A1 |
| C4 complementarity | BETTER(H2,A1) | H2+A1 |
| C4 complementarity | B1+H1 | B1+H1+A1 |
| C4 complementarity | B1+H2 | B1+H2+A1 |
| C4 complementarity | B2+H1 | B2+H1+A1 |
| C4 complementarity | B2+H2 | B2+H2+A1 |
| C4 complementarity | B1+A1 | B1+H1+A1 |
| C4 complementarity | B1+A1 | B1+H2+A1 |
| C4 complementarity | B2+A1 | B2+H1+A1 |
| C4 complementarity | B2+A1 | B2+H2+A1 |
| C5 exact-rule conditioning | H1 | H2 |
| C5 exact-rule conditioning | B1+H1 | B1+H2 |
| C5 exact-rule conditioning | B2+H1 | B2+H2 |
| C5 exact-rule conditioning | H1+A1 | H2+A1 |
| C6 encoder adaptation | B1 | B2 |
| C6 encoder adaptation | B1+H1 | B2+H1 |
| C6 encoder adaptation | B1+H2 | B2+H2 |
| C6 encoder adaptation | B1+A1 | B2+A1 |
| Probe-only vs structure-only reference | REF_structure | H1 |
| Probe-only vs structure-only reference | REF_structure | H2 |
| Probe-only vs structure-only reference | REF_structure | A1 |
| Probe-only vs structure-only reference | REF_structure | H1+H2 |
| Probe-only vs structure-only reference | REF_structure | H1+A1 |
| Probe-only vs structure-only reference | REF_structure | H2+A1 |
| Probe-only vs structure-only reference | REF_structure | H1+H2+A1 |

## 6. Multiple comparisons

- Two-sided bootstrap p-value per contrast (2·min share of replicates on either side of 0); Holm adjustment within each (generator, objective, contrast family). Intervals are reported unadjusted.
- Per-type and per-k breakdowns and Evaluation 2 are descriptive (no adjustment, no claims of significance).

## 7. Evaluation 2 (controlled analyses on TEST prompts; algorithms in `src/mife/probing/controlled.py`)

Evaluation 2: controlled comparisons (docx §11). Pairs are built from manifest metadata only (task, rule IDs,
positions, chains, COA rows), never from outcomes, so the matching is fixed before the test set is read.

A position (Objective 1): fix task, exact rule set (subset), k >= 2 and target rule; for each pair of target
   positions (j < j'), compare COA-balanced means over the orderings that put the target at j vs j'.
B count: fix task, chain and the chain-prefix order (COA row 0 is the chain order, and the k-prefix nests in the
   (k+1)-prefix); compare k -> k+1. Objective 1: each of the k shared targets (same position). Objective 2: the prompt pair.
C composition: fix task, k >= 2, target and target position; the two rule sets differ by exactly one companion, which
   occupies the same position, and every shared companion keeps its position (strict matches only).
D task: identical ordered configuration (config_id) on two different tasks of the group (3 task pairs per config).
E difficulty: TRAIN-only k = 1 violation rate per type -> terciles of the 15 types (5 each) -> Evaluation 1 by tercile.

Scores per configuration: mean |pred_delta - obs_delta| over pairs, and sign accuracy over RELIABLE pairs only.
Reliable (fixed): Objective 1 - the Agresti-Caffo 95% CI of the difference of the two violation proportions (counts
v out of n responses, aggregated over the orderings for A) excludes 0. Objective 2 - the Welch 95% CI of the difference
in mean per-response fraction violated (20 responses per prompt) excludes 0.

Pair counts are fixed by the design (identical in every group):

| analysis | objective | pairs_per_group | test_pairs (15 groups) |
|---|---|---|---|
| A position | 1 | 1530 | 22950 |
| B count | 1 | 180 | 2700 |
| C composition (strict) | 1 | 1692 | 25380 |
| D task | 1 | 3078 | 46170 |
| B count | 2 | 72 | 1080 |
| C composition (strict) | 2 | 504 | 7560 |
| D task | 2 | 738 | 11070 |

E (difficulty) terciles, frozen from TRAIN k = 1 violation rates per type:

| model | type_id | train_k1_rate | tercile |
|---|---|---|---|
| llama | detectable_format:title | 0 | easy |
| llama | length_constraints:number_words | 0.010417 | easy |
| llama | change_case:english_lowercase | 0.023077 | easy |
| llama | keywords:forbidden_words | 0.03172 | easy |
| llama | keywords:frequency | 0.044167 | easy |
| llama | startend:quotation | 0.060897 | medium |
| llama | punctuation:no_comma | 0.086111 | medium |
| llama | keywords:existence | 0.09375 | medium |
| llama | detectable_content:number_placeholders | 0.14023 | medium |
| llama | change_case:english_capital | 0.22333 | medium |
| llama | length_constraints:number_sentences | 0.22738 | hard |
| llama | change_case:capital_word_frequency | 0.30778 | hard |
| llama | keywords:letter_frequency | 0.35208 | hard |
| llama | detectable_format:number_bullet_lists | 0.36488 | hard |
| llama | length_constraints:number_paragraphs | 0.51349 | hard |
| qwen | detectable_format:title | 0 | easy |
| qwen | startend:quotation | 0.0019231 | easy |
| qwen | change_case:english_lowercase | 0.0051282 | easy |
| qwen | keywords:existence | 0.0052083 | easy |
| qwen | length_constraints:number_paragraphs | 0.0087302 | easy |
| qwen | punctuation:no_comma | 0.011111 | medium |
| qwen | length_constraints:number_words | 0.011458 | medium |
| qwen | detectable_content:number_placeholders | 0.012644 | medium |
| qwen | length_constraints:number_sentences | 0.020238 | medium |
| qwen | detectable_format:number_bullet_lists | 0.023214 | medium |
| qwen | keywords:frequency | 0.035 | hard |
| qwen | keywords:forbidden_words | 0.046774 | hard |
| qwen | change_case:english_capital | 0.12333 | hard |
| qwen | change_case:capital_word_frequency | 0.22556 | hard |
| qwen | keywords:letter_frequency | 0.34115 | hard |

Every configuration is scored on the same pairs: mean |Δpred − Δobs| (group-bootstrap CI) and sign accuracy on reliable pairs (with the number of reliable pairs). E reports the Evaluation 1 metrics by tercile.

## 8. Sensitivity analyses (TEST)

Evaluation 1 primary metrics recomputed (a) with labels that exclude truncated responses (`hit_max_tokens`), (b) without `change_case:english_capital` (Objective 1: its rows; Objective 2: prompts containing it), and (c) without prompts that have any looping response (loop definition in `describe.py`). Primary labels are unchanged.

## 9. Known properties and deviations

- With one shared hidden row and deduplicated blocks, these configurations have identical feature blocks (linear heads give identical predictions; MLP heads differ only through their seeds):

| objective | configurations | blocks |
|---|---|---|
| 1 | H2 = H1+H2 | H + ER |
| 1 | H2+A1 = H1+H2+A1 | H + ER + A |
| 1 | B1+H1 = B1+H2 = B1+H1+H2 | EP + ER + S1 + H |
| 1 | B1+H1+A1 = B1+H2+A1 = B1+H1+H2+A1 | EP + ER + S1 + H + A |
| 1 | B2+H2 = B2+H1+H2 | EP_B2 + ER_B2 + S1 + H + ER |
| 1 | B2+H2+A1 = B2+H1+H2+A1 | EP_B2 + ER_B2 + S1 + H + ER + A |
| 2 | H2 = H1+H2 | H + M2 + ER5 |
| 2 | H2+A1 = H1+H2+A1 | H + M2 + ER5 + A5 |
| 2 | B1+H1 = B1+H2 = B1+H1+H2 | EP + ER5 + S2 + H |
| 2 | B1+H1+A1 = B1+H2+A1 = B1+H1+H2+A1 | EP + ER5 + S2 + H + A5 |
| 2 | B2+H2 = B2+H1+H2 | EP_B2 + ER5_B2 + S2 + H + ER5 |
| 2 | B2+H2+A1 = B2+H1+H2+A1 | EP_B2 + ER5_B2 + S2 + H + ER5 + A5 |

- Consequently some contrasts in § 5 compare identical feature sets (C5 B1+H1 vs B1+H2) or repeat another contrast (C4 B1+H1 → B1+H1+A1 and B1+H2 → B1+H2+A1; B1+A1 → B1+H1+A1 and B1+A1 → B1+H2+A1; probe-only H2 and H1+H2, H2+A1 and H1+H2+A1). They are kept as listed and counted in the Holm family, which makes the adjustment slightly conservative.

- L2 grid: number of the 23 configurations plus the structure reference selecting each value (edge selections and the loss one grid step inwards in `results/tables/cv_l2_edges.csv`). The grid is used as specified:

| model | objective | selected_l2=0.01 | selected_l2=0.1 | selected_l2=1 |
|---|---|---|---|---|
| llama | 1 | 2 | 22 | 0 |
| llama | 2 | 8 | 9 | 7 |
| qwen | 1 | 1 | 18 | 5 |
| qwen | 2 | 6 | 18 | 0 |

- See `results/RESULTS.md` § 10 for every open issue.

## 10. Execution

- The TEST evaluation runs exactly once, after the project owner's explicit approval is recorded in `FROZEN.json` (`unblind_approved`); each run is appended to `FROZEN.json` `test_runs` with its time and the plan hash it used. Nothing in this plan is changed after the TEST set is read.
