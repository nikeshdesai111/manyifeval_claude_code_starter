# NeurIPS-format draft report (2026-11-23, revised 2026-11-24)

Draft of the class project report in the official NeurIPS 2026 style (`neurips_2026.sty`, `[preprint]` option: same
layout and fonts as a submission, without the reviewer line numbers; lists are tightened with `enumitem`, and the
official paper checklist is not included). Final report due Mon 7 Dec 2026.

| File | What it is |
|---|---|
| `paper.tex` | Main source. Main text = pages 1–8; references and appendices A–K follow. |
| `refs.bib` | Bibliography (63 entries; see the verification logs below). |
| `CRITIQUE_2026-11-24.md` | Critique of the 23 Nov draft that drove the 24 Nov revision, with the NEED.pdf points not adopted. |
| `neurips_2026.sty` | Official NeurIPS 2026 style file (`\ProvidesPackage{neurips_2026}[2026-01-29 ...]`, the version shared by most public copies). |
| `make_assets.py` | Builds `figures/pair_coverage_s3.pdf` and `generated/*.tex` from design metadata and committed TRAIN-CV tables. |
| `paper.pdf` | Compiled draft. |

## Preliminary TRAIN results (24 Nov 2026)

Tables 3-4, Figure 1 (`figures/eval2_prelim_train.pdf`) and Appendix H now hold **preliminary grouped-CV out-of-fold
results on the 50 S3 training groups**, with the disclosure in Section 6.1. They are built by
`mife.prelim_train.paper_assets` from `results/preliminary_train_report/tables/` and inlined between
`% BEGIN generated` / `% END generated` markers; appendix PNGs are in `figures/prelim/`. Provenance, deviations and
commands: `results/preliminary_train_report/README.md`. The 22-group test evaluation will replace them.

## Build

```bash
python report/neurips_draft/make_assets.py          # from the repo root; needs matplotlib
cd report/neurips_draft && latexmk -pdf paper.tex
```

## Page limit

Main text (through the Conclusion) ends at the bottom of page 8 with about one line spare; References start on page 9.
Tables 3 and 4 were test-compiled with worst-case values (`-12.3 [-20.1, -4.5]` in every cell) and do not overflow,
which is why the caption fixes the scales (NLL x10^3, AUROC and Spearman x100, RMSE in points, Delta-MSE x10^4).
Figure 2 is drawn at its final size with empty axes (`make_assets.py`, `eval2_placeholder`). The results prose has
`TBD` slots at roughly final length, but the red TODO notes are shorter than the sentences that will replace them, so
expect to cut 2–4 lines when filling: first the Evaluation 2 protocol paragraph (definitions are in Appendix F),
then Related Work's attention paragraph. Recompile and check that the Conclusion still ends on page 8.

## Layout decisions against the evaluation plan (2026-11-23)

- The plan's main-paper Table 2 (Objective 1) and Table 3 (Objective 2) are typeset as one table (`tab:main`) with
  shared rows: same seven comparisons, same columns and intervals as the plan.
- The plan's four two-panel Evaluation 2 observed-effect figures (E2-Pos, E2-Append, E2-Companion, E2-Task) become
  one main figure (`fig:eval2`) holding the three noise-adjusted RMS-by-target-type panels and the append panel. The
  signed heatmaps (type x k, target x companion, family) move to the appendix. This is a deviation from the plan to
  fit 8 pages and is stated in the figure caption; Nicky may prefer a different cut.
- The overview figure moved to Appendix A to make room.
- The plan's Table 4 (Evaluation 2 skill) shows the four primary comparisons in the main text; comparisons 4–6 are in
  the appendix list, to fit the page limit. Restore them if space allows.
- The plan's Table 1 (test inventory with A/B/C/D coverage) is split between Table 1 (data) and the counts given in
  Section 6.1 and the Table 4 caption.

## What is and is not in the draft

- **No TEST-group numbers.** Objective 1 test scoring was authorised on 2026-11-24 and is running in another thread;
  no test number is in the draft. Results tables carry `TBD` cells and red `\todo` notes.
- **TRAIN-CV numbers** appear only in Appendix H (Table 7), generated from `results/v2_s3/tables/selected_settings.csv`
  and labelled as selection criteria, not held-out estimates. The selected layers in Section 6 come from
  `results/v2_s3/tables/layer_selection.csv`.
- V1 descriptive statistics (violation rates per type) are **not** used, because they were computed on the old
  57-group TRAIN set, which includes 16 groups that are TEST under S3.
- The S3 test expansion (all 456 configurations for the 22 test groups) is complete for both generators and is part of
  the test set.

## Citation verification log (2026-11-23)

Each entry was searched by title; arXiv-hosted entries were then fetched to confirm the full author list and first
posting. Venues were confirmed from the venue's own page found in search: ACL Anthology (ManyIFEval
2025.findings-emnlp.896, FollowBench 2024.acl-long.257, InFoBench 2024.findings-acl.772, Order Matters
2025.findings-acl.646, BGE-M3 2024.findings-acl.137, Lookback Lens 2024.emnlp-main.84, Gottesman & Geva
2024.emnlp-main.232, Azaria & Mitchell 2023.findings-emnlp.68, Tian et al. 2023.emnlp-main.330, Hewitt & Liang
D19-1275), ICLR proceedings/virtual site (Heo et al. ×2, Stolfo et al., Orgad et al., RouteLLM 2025; SAT Probe, PASTA,
attention sinks, Xiong et al., Hybrid LLM, Sclar et al. 2024; Kuhn et al., Burns et al. 2023), NeurIPS (ComplexBench
2024 and IFBench 2025 Datasets and Benchmarks track; Ashok & May 2025; ITI 2023), TMLR (Lin et al. 2022), Nature
(Farquhar et al. 2024), Journal of Applied Econometrics (Papke & Wooldridge 1996). Entries whose only confirmed
record is arXiv are cited as arXiv preprints (IFEval, IFScale, Multi-IF, Semantic Entropy Probes, Geometry of Truth,
RepE, Kadavath et al., Rocchetti & Ferrara, Llama 3, Qwen3). Classic references (Alain & Bengio, Belinkov, Jain &
Wallace, Wiegreffe & Pinter, Liu et al. TACL, Guo et al., vLLM, Yang–Sun–Xu COA, Cawley & Talbot, Efron, Landis &
Koch) were carried over from the earlier report's bibliography and its related-work check.

## Citation additions (2026-11-24)

Twelve entries were added for the corrected Related Work (from the team's corrections document, NEED.pdf, 23 Nov).
Title, authors and venue of MOSAIC (ACL Anthology 2026.eacl-long.62), CSE (arXiv 2608.12426), Lugoloobi et al.
(arXiv 2602.09924; accepted at COLM 2026 per its arXiv comment),
Expected Reward Prediction (arXiv 2603.20217), VerIFY (ACL Anthology 2026.findings-eacl.254; the paper's title is "We
Are What We Repeatedly Do") and CONPACT (arXiv 2608.11242) were checked against the source pages on 2026-11-24.
Binder et al. 2025, Hewitt et al. 2021, Future Lens, MultiChallenge, the Instruction Hierarchy and Conformal Language
Modeling were checked by the independent review agent, which also corrected the author order of MultiChallenge and Zeng et al.
to the ACL Anthology records.
