# NeurIPS-format draft report (2026-11-23)

Draft of the class project report in the official NeurIPS 2026 style (`neurips_2026.sty`, anonymous submission mode
with line numbers). Final report due Mon 7 Dec 2026.

| File | What it is |
|---|---|
| `paper.tex` | Main source. Main text = pages 1–8; references and appendices A–H follow. |
| `refs.bib` | Bibliography (51 entries, all cited and each checked; see below). |
| `neurips_2026.sty` | Official NeurIPS 2026 style file (`\ProvidesPackage{neurips_2026}[2026-01-29 ...]`, the version shared by most public copies). |
| `make_assets.py` | Builds `figures/pair_coverage_s3.pdf` and `generated/*.tex` from design metadata and committed TRAIN-CV tables. |
| `paper.pdf` | Compiled draft. |

## Build

```bash
python report/neurips_draft/make_assets.py          # from the repo root; needs matplotlib
cd report/neurips_draft && latexmk -pdf paper.tex
```

## Page limit

Main text (through the Conclusion) ends on page 8; References start on page 9. A test that inserted filler before the
Conclusion showed about two to three lines of slack, so the final results must **replace** the `\todo` text and `TBD`
cells rather than add to them. Reproducibility, AI-use and compute statements are in the appendix.

## What is and is not in the draft

- **No TEST-group data.** Nothing was read or computed from the 22 S3 TEST groups. Results tables carry `TBD` cells and
  red `\todo` notes.
- **TRAIN-CV numbers** appear only in Appendix E (Table 6), generated from `results/v2_s3/tables/selected_settings.csv`
  and labelled as selection criteria, not held-out estimates. The selected layers in Section 6 come from
  `results/v2_s3/tables/layer_selection.csv`.
- V1 descriptive statistics (violation rates per type) are **not** used, because they were computed on the old
  57-group TRAIN set, which includes 16 groups that are TEST under S3.
- The S3 expansion amendment (PR #5) is described as in progress; its analysis plan still needs to be frozen.

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
