"""Build report/neurips_draft/paper_test_llama.tex: a COPY of paper.tex with the Objective 1 exhibits and prose replaced
by the held-out S3 test results for Llama (results/eval_s3/llama_only/). paper.tex is not modified. Objective 2 stays
as the preliminary training-group results, clearly labelled. Every number in the new prose comes from the saved tables.
Run: MIFE_EVAL_SUB=llama_only MIFE_EVAL_GENS=llama PYTHONPATH=src .venv/bin/python -m mife.eval_s3.paper_copy
"""
from __future__ import annotations

import re
import shutil

from ..config import ROOT
from .paths import EVAL_DIR

SRC = ROOT / "report" / "neurips_draft" / "paper.tex"
DST = ROOT / "report" / "neurips_draft" / "paper_test_llama.tex"
FIG_DST = ROOT / "report" / "neurips_draft" / "figures" / "test_llama"


def between(s, start, end, new, keep_end=True):
    i = s.index(start)
    j = s.index(end, i + len(start))
    return s[:i] + new + (s[j:] if keep_end else s[j + len(end):])


def sub1(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:80])
    return s.replace(old, new)


def gen(name):
    return (EVAL_DIR / "paper" / f"test_{name}.tex").read_text()


ABSTRACT = r"""On the 22 held-out groups, rule attention forecasts per-rule risk better than a text predictor that also
embeds the exact rule for both generators (0.043 and 0.024 nats per outcome for Llama and Qwen), and the hidden state
does so for Llama; for Qwen attention also beats the hidden state. Forecasting how risk \emph{changes} under controlled
prompt edits is a different matter: the attention probe, the most accurate absolute predictor, forecasts the effects of
moving or appending rules worse than both the text and the hidden-state predictors. Prompt-level (Objective~2) results
remain preliminary cross-validation estimates on the training groups."""

PROTOCOL = r"""\paragraph{Results in this version.} The Objective~1 results (Tables~\ref{tab:e1o1} and~\ref{tab:eval2},
Figures~\ref{fig:e2pos}--\ref{fig:e2task}) are held-out results on the 22 S3 test groups for both generators: per
generator, 30,096 prompts, 115,236 prompt--rule rows and 601,920 verified responses (the 16,236 original prompts and
13,860 added prompts that complete every rule subset at $k=2,3,4$; Table~\ref{tab:inventory}). Each predictor is the
frozen model refitted on the 50 training groups and applied once; no setting, layer, penalty, calibration bin or weight
was chosen on test data. Objective~2 (Table~\ref{tab:e1o2}) is not yet authorised for test scoring and still shows
preliminary out-of-fold results on the 50 training groups.

"""

E1_PROSE = r"""\paragraph{Objective 1.} On the held-out groups (Table~\ref{tab:e1o1}), rule attention beats B2, which
already embeds the exact rule, for both generators (comparison~3: 0.0433 [0.0134, 0.0774] nats per outcome for Llama
and 0.0242 [0.0090, 0.0424] for Qwen), with AUROC gains of 2.0 and 3.1 points. The hidden state beats B2 for Llama
(comparison~2: 0.0209 [0.0003, 0.0422]) but not clearly for Qwen (0.0104 [$-0.0059$, 0.0284]), although AUROC rises
for both (1.6 and 1.7 points). Comparing the two internal signals directly (comparison~4, H2 \cto{} A2), attention is
better for Qwen (0.0138 [0.0018, 0.0263]) and cannot be separated from the hidden state for Llama (0.0224 [$-0.0035$,
0.0523]); A2 has the lowest NLL of all predictors for both generators (0.345 and 0.274). Against prompt text alone
(comparison~1), the hidden state lowers NLL for Qwen (0.0351 [0.0064, 0.0654]) and raises AUROC for both (4.6 and 4.8
points), but its Llama NLL interval includes zero (0.0312 [$-0.0105$, 0.0708]). These comparisons test predictive
accuracy, not mechanism.

\paragraph{Breadth and calibration.} In comparisons~1--4 the candidate lowers NLL on 7--13 of the 15 rule types (point
estimates, not per-type tests; Appendix A3). Split by rule class (A3b), comparisons~1--3 lower NLL in both classes for
both generators, with larger gains on the ten parameterised types, which are violated far more often (20.6\% vs 8.2\% of
outcomes for Llama, 13.9\% vs 3.1\% for Qwen). Calibration is reasonable throughout (ECE 3.2--6.2 points; A1--A2).
Whether adding a block to a fixed predictor helps (comparisons 5--8) is reported in Appendix~\ref{app:incremental}.

"""

E2_OBS = r"""\subsection{Evaluation 2: observed effects of controlled changes}

All results in this subsection are observed outcomes on the 22 test groups; predictor skill is in
Section~\ref{sec:e2skill}.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{FIGDIR/e2_pos_figure.png}
\caption{E2-Pos, 22 held-out groups (observed). (A) Noise-adjusted RMS change of a rule's violation probability over
all position pairs ($k=2..5$, equal weight per $k$), by target type (fixed-wording types above the line), Llama and
Qwen; 95\% pool-bootstrap intervals. (B) Signed last-minus-first change (points) by type and $k$ for each generator;
positive = riskier when last. Intervals and support in Appendix Pos-C. Moving a rule also changes its distance to the
prompt's end and the placement of the other rules.}
\label{fig:e2pos}
\end{figure}

\paragraph{Position.} With the exact rule set, task and $k$ fixed, moving a rule changes its violation probability by
a median (over the 15 types) noise-adjusted RMS of 8.2 points for Llama and 7.6 for Qwen (Figure~\ref{fig:e2pos}A),
most for paragraph count with Llama (24.5 [21.9, 26.7]) and bullet points with Qwen (20.4 [17.2, 23.4]). The signed
panels (B) are mostly positive: 28 (Llama) and 21 (Qwen) of 60 type--$k$ cells exclude zero, 26 and 19 of them riskier
when last. Paragraph count for Llama ($-11.0$ [$-18.3$, $-2.2$] at $k=2$) and word count for Qwen ($-8.1$ [$-21.0$,
$-0.7$] at $k=4$) are riskier when first.

\begin{figure}[t]
\centering
\includegraphics[width=0.72\linewidth]{FIGDIR/e2_append_figure.png}
\caption{E2-Append, held-out (observed). Mean change in the violation probability of the retained rules when one rule
is appended to an ordered prefix (the new rule is excluded), by transition, Llama and Qwen; 95\% pool-bootstrap
intervals. $3{\to}4$ and $4{\to}5$ cover only COA orders with an exact prefix.}
\label{fig:e2app}
\end{figure}

\paragraph{Strict append.} For Qwen, appending one rule raises the risk of the retained rules at every transition, by
1.0 [0.4, 1.7] points at $1{\to}2$ rising to 1.7 [0.9, 2.5] at $4{\to}5$ (Figure~\ref{fig:e2app}). For Llama the effect
is small and changes sign: $-1.2$ [$-2.3$, $-0.2$] at $1{\to}2$, near zero at $2{\to}3$ and $3{\to}4$, and $+0.8$ [0.1,
1.4] at $4{\to}5$, where the increase falls on the middle slots rather than on the rule that used to be last
(Appendix K-A).

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{FIGDIR/e2_companion_figure.png}
\caption{E2-Companion, held-out (observed). (A) Noise-adjusted RMS change of the target's violation probability when
exactly one companion (earlier or later) is replaced, by target type ($k=2..5$), Llama and Qwen. (B) Signed target
$\times$ companion effects for each generator: named companion minus the mean of the other eligible companions in the
same fixed context (points); outlined cells have fewer than 3 pools; hatched cells have no strict match (not zero).
Numbers in Appendix C-A.}
\label{fig:e2comp}
\end{figure}

\paragraph{Companion identity.} Replacing a single companion while every other slot stays fixed changes the target's
risk by a median RMS of 13.2 points for Llama and 14.7 for Qwen (Figure~\ref{fig:e2comp}A), most for paragraph count
(Llama, 34.3) and bullet points (Qwen, 30.1), about equally for earlier and later companions (Appendix C-A). Companion
effects are directed and specific (B): for Llama a paragraph-count rule is violated 27.2 [17.5, 37.8] points more often
next to a title rule than next to the alternatives (4 pools); for Qwen bullet points are violated 16.6 [12.7, 19.8]
points more often next to a paragraph-count rule (3 pools). Of the 126 cells with at least 3 pools, 33 (Llama) and 21
(Qwen) have pointwise intervals excluding zero, against about 6 expected by chance; the intervals are not adjusted for
the number of cells.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{FIGDIR/e2_task_figure.png}
\caption{E2-Task, held-out (observed). (A) Noise-adjusted RMS change when the identical ordered rule configuration is
attached to another task of the same pool, by target type ($k=1..5$), Llama and Qwen. (B) Signed difference between a
family's task and the mean of the other two matched tasks (points), with supporting pools. Families are confounded
with task text; associations, not causal effects.}
\label{fig:e2task}
\end{figure}

\paragraph{Task context.} Changing only the task changes rule risk the most: a median RMS of 23.5 points for Llama and
16.9 for Qwen (Figure~\ref{fig:e2task}A), up to 49.0 [23.0, 65.4] for sentence count (Llama) and 32.3 [18.2, 44.2] for
uppercase-word count (Qwen). By family (B), informational tasks are associated with lower risk for Llama ($-3.8$
[$-6.9$, $-0.7$] points, 11 pools); for Qwen creative tasks with higher risk (2.0 [0.3, 3.7], 16 pools) and
transformation tasks with lower risk ($-4.2$ [$-6.8$, $-1.8$], 5 pools).

"""

E2_SKILL_PROSE = r"""
Table~\ref{tab:eval2} asks whether the predictors forecast these matched changes. The clearest result concerns
attention: for both generators the attention probe forecasts position and append changes worse than B2 (comparison~3:
$-40.3$ and $-22.8$ for Llama, $-24.5$ and $-18.8$ for Qwen; all intervals below zero) and worse than H2 (comparison~4:
$-45.6$ and $-24.5$, $-20.0$ and $-13.6$), and for Llama also companion changes. The hidden state helps only for Llama's
position changes (comparison~2: 5.2 [1.9, 8.8]); for Qwen it forecasts position and task changes worse than B2 ($-4.4$
[$-8.6$, $-0.4$] and $-83.7$ [$-153.4$, $-21.9$]). Task changes are the noisiest (zero-change MSE 762 and 355).
Better absolute accuracy therefore does not imply better forecasts of how a rule's risk moves with its context: A2 is
the most accurate absolute predictor for both generators and among the worst forecasters of within-context changes.

"""

CONCLUSION = r"""On 22 held-out groups of our balanced multi-rule benchmark, rule attention forecasts per-rule violation
risk better than a frozen text predictor that sees the exact rule for both Llama~3.1 and Qwen3, and the hidden state
does so for Llama. Yet no predictor consistently forecasts how that risk changes under controlled edits, and rule
attention, the most accurate absolute predictor, forecasts within-context changes worst. Prompt-level test results
will complete the picture.\label{end:main}"""

APP_TEST_HEAD = r"""\section{Held-out test results for Llama: Objective~1 appendix exhibits}
\label{app:test}

\paragraph{Provenance.} Split S3 (\texttt{split-s3-2026-11-23}): the 22 test groups, 66 tasks and all 456 ordered
configurations per task, 30,096 prompts and 115,236 prompt--rule rows, each prompt with 20 verified responses (16,236
original prompts from run \texttt{main}; 13,860 expansion prompts from run \texttt{expansion\_s3\_20261123}, verified
with the same official verifiers). Test hidden states and rule attention were extracted once per prompt with the frozen
backend (HF eager, BF16, Apple MPS) and equal the frozen V1 features bit for bit on the original prompts checked. The
frozen V2 refits on the 50 training groups (\texttt{results/v2\_s3/models/}) were applied once. Code
\texttt{src/mife/eval\_s3/}; tables \texttt{results/eval\_s3/llama\_only/tables/}. Test scoring for Objective~1 was
authorised by the project owner on 24 November 2026; 16 of the 22 test groups were training groups under an earlier
split (Section~\ref{sec:discussion}). Exhibits follow the evaluation plan's Objective~1 appendix layout (A1--A4, Pos,
K, C, D, Factor E).

"""

PLANNED = r"""\section{Appendix exhibits: where they are}
\label{app:planned}

The Objective~1 exhibits of the evaluation plan (A1--A4; Pos-A/B/C; K-A/B; C-A/B/C; D-A/B/C/D; Factor E) are
populated from the 22 test groups for Llama in Appendix~\ref{app:test}; Qwen will be added when its test features are
complete. The Objective~2 exhibits (A5--A7 and the Objective~2 matched-change table) remain preliminary training-group
results in Appendix~\ref{app:prelim}.

"""


def build() -> str:
    s = SRC.read_text()
    s = sub1(s, "  Anonymous Authors\\\\\n  Course project draft report\\\\",
             "  Nikesh Desai \\quad Abhimanyu Sharma \\quad Yijie Guo\\\\\n  Course project draft report\\\\")
    # abstract
    i = s.index("Generation for the held-out groups was incomplete")
    j = s.index(r"\end{abstract}")
    s = s[:i] + ABSTRACT + "\n" + s[j:]
    s = sub1(s, r"(Section~\ref{sec:experiments}; preliminary out-of-fold results on the training" + "\ngroups in this version)",
             r"(Section~\ref{sec:experiments}; held-out Objective~1 results for Llama in this version)")
    s = between(s, r"\paragraph{Preliminary results in this version.}", r"\paragraph{Test-set discipline.}", PROTOCOL)
    i = s.index(r"\caption{\emph{Preliminary} evaluation inventory")
    i = s.rindex(r"\begin{table}", 0, i)
    j = s.index(r"\end{table}", i) + len(r"\end{table}")
    s = s[:i] + gen("t1").strip() + s[j:]
    s = sub1(s, "The 22 test groups (50 training groups for the preliminary results) are",
             "The 22 test groups (50 training groups for the preliminary Objective~2 results) are")
    i = s.index(r"\caption{\emph{Preliminary} Evaluation~1, Objective~1")
    i = s.rindex(r"\begin{table}", 0, i)
    j = s.index(r"\end{table}", i) + len(r"\end{table}")
    s = s[:i] + gen("t2").strip() + s[j:]
    i = s.index(r"\paragraph{Objective 1.}")
    j = s.index(r"\begin{table}[t]", i)
    s = s[:i] + E1_PROSE + s[j:]
    s = sub1(s, r"\caption{\emph{Preliminary} Evaluation~1, Objective~2: prompt-level",
             r"\caption{\emph{Preliminary (training groups; not yet evaluated on the test groups)} Evaluation~1, Objective~2: prompt-level")
    s = sub1(s, r"\paragraph{Objective 2.} For the fraction", r"\paragraph{Objective 2 (preliminary, training groups).} For the fraction")
    s = between(s, r"\subsection{Evaluation 2: observed effects of controlled changes}",
                r"\subsection{Evaluation 2: forecasting the changes}", E2_OBS)
    i = s.index(r"\caption{\emph{Preliminary} Evaluation~2 predictor skill")
    i = s.rindex(r"\begin{table}", 0, i)
    j = s.index(r"\end{table}", i) + len(r"\end{table}")
    k = s.index("% =====", j)
    s = s[:i] + gen("t4").strip() + "\n" + E2_SKILL_PROSE + s[k:]
    s = sub1(s, "In the preliminary results this holds clearly for Qwen and only in AUROC for\nLlama.",
             "On the held-out groups this holds for Llama for both the hidden state and attention (comparisons~2 and~3);\nQwen's test results are pending.")
    i = s.index(r"\emph{Preliminary numbers:}")
    j = s.index(r"\emph{Baseline strength:}", i)
    s = s[:i] + (r"\emph{Coverage of this version:} test results are for Llama and Objective~1 only; the Objective~2 "
                 r"numbers are grouped-CV estimates on training groups whose folds also selected settings, so they may be "
                 r"optimistic. Evaluation~2 weights follow the plan's proposed recipe, which was not separately signed off "
                 r"before scoring. ") + s[j:]
    i = s.index(r"\section{Conclusion}")
    i = s.index(r"\label{sec:conclusion}", i) + len(r"\label{sec:conclusion}")
    j = s.index(r"\label{end:main}", i) + len(r"\label{end:main}")
    s = s[:i] + "\n\n" + CONCLUSION + s[j:]
    # appendix
    s = between(s, r"\section{Planned appendix exhibits}", r"\section{Preliminary training-set results", PLANNED)
    s = sub1(s, r"\section{Preliminary training-set results: provenance and planned appendix exhibits}",
             APP_TEST_HEAD + gen("appendix") + "\n\\clearpage\n"
             + r"\section{Preliminary training-set results: Objective~2 (provenance and exhibits)}")
    s = sub1(s, "Exhibits follow the evaluation plan's appendix layout (A1--A7 for Evaluation~1; Pos, K, C and D for Evaluation~2).",
             "Only the Objective~2 exhibits (A5--A7 and the Objective~2 matched-change table) are kept here; the Objective~1 "
             "exhibits are replaced by the held-out results in Appendix~\\ref{app:test}.")
    # drop the preliminary Objective 1 appendix subsections (A1-A4 and Pos ... Factor E) inside app:prelim
    p0 = s.index(r"\label{app:prelim}")
    i = s.index(r"\subsection*{A1 --- Complete overall metrics (Objective 1)}", p0)
    j = s.index(r"\subsection*{A5 --- Complete overall metrics (Objective 2)}", i)
    s = s[:i] + s[j:]
    i = s.index(r"\subsection*{Pos-A --- Position profiles", s.index(r"\label{app:prelim}"))
    j = s.index(r"\subsection*{Evaluation 2, Objective 2 (preliminary", i)
    s = s[:i] + s[j:]
    labels = re.findall(r"\\label\{([^}]+)\}", s)
    dup = {x for x in labels if labels.count(x) > 1}
    assert not dup, dup
    DST.write_text(s)
    FIG_DST.mkdir(parents=True, exist_ok=True)
    for p in (EVAL_DIR / "figures").glob("*.png"):
        shutil.copy2(p, FIG_DST / p.name)
    return str(DST)


if __name__ == "__main__":
    print(build())
