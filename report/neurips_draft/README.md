# NeurIPS-format draft report (2026-11-23, revised 2026-11-24)

Draft of the class project report in the official NeurIPS 2026 style (`neurips_2026.sty`, `[preprint]` option: same
layout and fonts as a submission, without the reviewer line numbers). Final report due Mon 7 Dec 2026.

| File | What it is |
|---|---|
| `paper.tex` | Main source. Main text = pages 1–8; references and appendices A–H follow. |
| `refs.bib` | Bibliography (63 entries; see the verification logs below). |
| `CRITIQUE_2026-11-24.md` | Critique of the 23 Nov draft that drove the 24 Nov revision, with the NEED.pdf points not adopted. |
| `neurips_2026.sty` | Official NeurIPS 2026 style file (`\ProvidesPackage{neurips_2026}[2026-01-29 ...]`, the version shared by most public copies). |
| `make_assets.py` | Builds `figures/pair_coverage_s3.pdf` and `generated/*.tex` from design metadata and committed TRAIN-CV tables. |
| `paper.pdf` | Compiled draft. |

## Build

```bash
python report/neurips_draft/make_assets.py          # from the repo root; needs matplotlib
cd report/neurips_draft && latexmk -pdf paper.tex
```

## Page limit

Main text (through the Conclusion) fills page 8 exactly, with about one line spare, and References start on page 9.
The results text is written with `TBD` slots at roughly final length, so filling it should not change the length much.
If the filled text overflows, cut in this order: shorten the Figure 1 overview (or move it to the appendix), shorten
the Evaluation 2 protocol paragraph (its full definitions are already in Appendix E), then trim Related Work's
attention paragraph. Recompile and check that the Conclusion still ends on page 8.

## Layout decisions against the evaluation plan (2026-11-23)

- The plan's main-paper Table 2 (Objective 1) and Table 3 (Objective 2) are typeset as one table (`tab:main`) with
  shared rows: same seven comparisons, same columns and intervals as the plan.
- The plan's four two-panel Evaluation 2 observed-effect figures (E2-Pos, E2-Append, E2-Companion, E2-Task) are one
  four-panel main figure (`fig:eval2`); a full-size version of each can go in the appendix.
- The plan's Table 4 (Evaluation 2 skill) shows the four primary comparisons in the main text; comparisons 4–6 are in
  the appendix list, to fit the page limit. Restore them if space allows.
- The plan's Table 1 (test inventory with A/B/C/D coverage) is split between Table 1 (data) and the counts given in
  Section 6.1 and the Table 4 caption.

## What is and is not in the draft

- **No TEST-group numbers.** Objective 1 test scoring was authorised on 2026-11-24 and is running in another thread;
  no test number is in the draft. Results tables carry `TBD` cells and red `\todo` notes.
- **TRAIN-CV numbers** appear only in Appendix G (Table 7), generated from `results/v2_s3/tables/selected_settings.csv`
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
(arXiv 2602.09924; the only venue found is an ICLR 2026 workshop poster, so it is cited as arXiv, not COLM 2026),
Expected Reward Prediction (arXiv 2603.20217), VerIFY (ACL Anthology 2026.findings-eacl.254; the paper's title is "We
Are What We Repeatedly Do") and CONPACT (arXiv 2608.11242) were checked against the source pages on 2026-11-24.
Binder et al. 2025, Hewitt et al. 2021, Future Lens, MultiChallenge, the Instruction Hierarchy and Conformal Language
Modeling were checked by the independent review (see the PR).
