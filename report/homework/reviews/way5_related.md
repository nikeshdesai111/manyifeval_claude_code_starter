# Related-work check: five closest prior papers

Prompt: "With web search on, find the five closest prior papers to this one, with links, and say what is new in ours. Check that every paper you list actually exists."

Scope: I read the abstract, introduction, related work and contributions of `report/paper.tex`, and `report/refs.bib`. Searches used topic keywords and paper titles only. I confirmed that each listed paper exists by its arXiv API record, which gives the title, the full author list and the posting date. Venues are confirmed by the ACL Anthology, the ICLR or NeurIPS virtual-site pages, or the arXiv comment field, as noted.

## The five closest papers

### 1. Do LLMs "know" internally when they follow instructions?
- **Authors:** Juyeon Heo, Christina Heinze-Deml, Oussama Elachqar, Kwan Ho Ryan Chan, Shirley Ren, Udhay Nallasamy, Andy Miller, Jaya Narain
- **Venue/year:** ICLR 2025 (poster: https://iclr.cc/virtual/2025/poster/28257)
- **Link:** https://arxiv.org/abs/2410.14516
- **Already cited:** yes (`heo2025know`)
- **What it does:** It trains linear probes on input-side representations to find an "instruction-following dimension" that separates IFEval-style successes from failures, with AUROC around 0.7 to 0.8. The probe generalises to unseen tasks but not to unseen instruction types. Steering along the dimension raises compliance. A sensitivity analysis suggests the signal reflects how the prompt is phrased more than how hard the task is.
- **How ours differs:** Heo et al. study one instruction per prompt with a binary success label and no strong prompt-only competitor. Our prompts carry one to five rules, and each rule is its own target. Our labels are 20-sample violation probabilities. Our probes are compared against text encoders, one of them fine-tuned, that see the same prompt, the same rule text and the same supervision. Heo et al.'s own finding that the signal tracks phrasing is exactly why the incremental comparison (RQ3) matters.

### 2. Do LLMs estimate uncertainty well in instruction-following?
- **Authors:** Juyeon Heo, Miao Xiong, Christina Heinze-Deml, Jaya Narain
- **Venue/year:** ICLR 2025 (poster: https://iclr.cc/virtual/2025/poster/30179)
- **Link:** https://arxiv.org/abs/2410.14582
- **Already cited:** **no**
- **What it does:** This is the first systematic evaluation of uncertainty estimation for instruction following. It builds a controlled benchmark that separates task quality from instruction compliance. It compares verbalised, logit-based and internal-state probe estimators. Internal-state probes beat the other estimators, but they fall to near-chance AUROC in the harder controlled setting.
- **How ours differs:** Their estimators mostly score a response after it has been generated. We predict before generation. We use multi-rule prompts with probabilistic labels, and our comparators are trained prompt-only predictors rather than the model's own confidence. Their near-chance result in the hard setting is a useful counterpoint to our training-set AUROCs of about 0.85.

### 3. How LLMs Follow Instructions: Skillful Coordination, Not a Universal Mechanism
- **Authors:** Elisabetta Rocchetti, Alfio Ferrara
- **Venue/year:** arXiv preprint, April 2026 (2604.06015; I found no peer-reviewed venue)
- **Link:** https://arxiv.org/abs/2604.06015
- **Already cited:** **no**
- **What it does:** It probes attention and MLP streams across nine instruction-following tasks. Task-specific probes beat a general probe. Over the course of generation, the signal for constraint satisfaction is largely absent at pre-generation positions and appears only once generation starts, which the authors read as monitoring rather than planning. Causal ablations show sparse, asymmetric dependencies between tasks.
- **How ours differs:** It is the most direct challenge to our premise, and the draft should address it. Our training-set results (attention-probe AUROC of about 0.85, internal probes beating prompt-only comparators in three of four cells) show a usable pre-generation signal on ManyIFEval rules. Our design (one shared type-specific head per rule type, and contrasts against a fine-tuned text encoder) also speaks to their specialist-versus-general finding. The two accounts can be reconciled if much of the pre-generation signal is prompt-readable rule difficulty and not planning. Our RQ3 contrast tests exactly that.

### 4. Language Models Can Predict Their Own Behavior
- **Authors:** Dhananjay Ashok, Jonathan May
- **Venue/year:** NeurIPS 2025 (per the arXiv comment on v2)
- **Link:** https://arxiv.org/abs/2502.13329
- **Already cited:** **no**
- **What it does:** It trains linear probes on the internal states of input tokens to forecast behaviour over the whole output, such as abstaining or complying with a request and alignment-relevant behaviours. It calibrates the probes with conformal prediction and uses confident predictions to stop computation early.
- **How ours differs:** The framing is the same (predict before generating, for routing or early warning), but the targets are different. Ashok and May predict coarse behaviours of the whole output. We predict compliance with each of several explicit, verifier-checked formatting and content rules. We also benchmark against prompt-only models that receive the same supervision, a comparison their setup does not centre on.

### 5. Attention Satisfies: A Constraint-Satisfaction Lens on Factual Errors of Language Models
- **Authors:** Mert Yuksekgonul, Varun Chandrasekaran, Erik Jones, Suriya Gunasekar, Ranjita Naik, Hamid Palangi, Ece Kamar, Besmira Nushi
- **Venue/year:** ICLR 2024
- **Link:** https://arxiv.org/abs/2309.15098
- **Already cited:** yes (`yuksekgonul2024attention`)
- **What it does:** It treats factual queries as sets of constraints and shows that attention to constraint tokens predicts whether the model satisfies them. SAT Probe, a probe on that attention, predicts factual errors about as well as the model's own confidence, on single- and multi-constraint queries.
- **How ours differs:** Our attention probe carries the SAT Probe idea from entity constraints in factual queries to instruction-rule spans in generation prompts. We summarise attention per layer and head as both a sum and a token-normalised mean over each exact rule span. We also find that it is the best single per-rule predictor. SAT Probe is the methodological ancestor of our best-performing component, and the draft could say so more prominently.

## Also close (not in the top five)
- Harada et al., *When Instructions Multiply* (Findings of EMNLP 2025; https://aclanthology.org/2025.findings-emnlp.896/), already cited. It is our data source. Its logistic regression on instruction count is the aggregate, structure-only analogue of our structure-only reference.
- Stolfo et al., *Improving Instruction-Following in Language Models through Activation Steering* (ICLR 2025; https://arxiv.org/abs/2410.12877), already cited. It intervenes on activations rather than predicting from them.
- Gottesman and Geva, *Estimating Knowledge in Large Language Models Without Generating a Single Token* (EMNLP 2024; https://aclanthology.org/2024.emnlp-main.232), already cited. It is pre-generation probing, but for factual knowledge.
- Jaroslawicz et al., *How Many Instructions Can LLMs Follow at Once?* (https://arxiv.org/abs/2507.11538), already cited. It is a benchmark only.

## What is new in this paper
1. It predicts the violation probability of each rule and the expected fraction of rules violated, before generation, in prompts with up to five simultaneous verifiable rules. Labels are probabilities from 20 sampled responses, not single binary outcomes.
2. It tests whether the signal is incremental. Hidden-state and rule-span-attention probes are compared against strong prompt-only predictors that receive the same text and supervision, one of them fine-tuned. The design balances rule subsets, rule orders and tasks, and the evaluation plan is pre-registered with test groups held out by group.
3. It is the first evidence on ManyIFEval-style multi-rule prompts that pre-generation attention to rule spans is the strongest single per-rule predictor. This bears on a live disagreement (Rocchetti and Ferrara, 2026) about whether compliance is decided before generation at all.

## Recommended additions to Related work
Cite and discuss Heo, Xiong, Heinze-Deml and Narain (ICLR 2025), Rocchetti and Ferrara (2026), and Ashok and May (NeurIPS 2025). Rocchetti and Ferrara matters most because its "monitoring, not planning" conclusion directly contests the paper's premise.

## Bibliography check (`refs.bib`)
No entry has a wrong title, a wrong year or a wrong page range among those I checked. The following entries are outdated or incomplete:
- `qin2024infobench`: cited as an arXiv preprint, but it is published in Findings of ACL 2024, pp. 13025–13048 (https://aclanthology.org/2024.findings-acl.772).
- `jaroslawicz2025many`: cited as arXiv only. The paper is listed on the NeurIPS 2025 virtual site (https://neurips.cc/virtual/2025/122417), but I could not confirm whether it is a main-track or workshop paper. Check before updating.
- `harada2025instructions`: correct (Findings of EMNLP 2025, pp. 16506–16526, confirmed in the ACL Anthology bib). The full author list is available if "and others" is not wanted: Harada, Yamazaki, Taniguchi, Marrese-Taylor, Kojima, Iwasawa, Matsuo.
- `kossen2024sep`: the arXiv record lists no journal reference. I did not find a confirmed peer-reviewed venue, so the arXiv citation is acceptable as it stands.
- Entries I confirmed correct: `zhou2023ifeval` (arXiv 2311.07911), `jiang2024followbench` (ACL 2024 main), `heo2025know` (ICLR 2025), `stolfo2025steering` (ICLR 2025, authors match), `yuksekgonul2024attention` (ICLR 2024), `gottesman2024estimating` (EMNLP 2024, pp. 3994–4019), `marks2024geometry` (COLM 2024), `yang2021coa` (Technometrics 63(2):212–224).
- I did not check the remaining classic entries (Alain and Bengio, Belinkov, Hewitt and Liang, Jain and Wallace, Holm, Efron, and so on) online. They look standard.
