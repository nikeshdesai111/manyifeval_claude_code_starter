"""Side-by-side document: every table and figure of report/neurips_draft/paper.tex, in the paper's order and numbering,
each followed by its K10 version (results/k10_paper/paper/k10_*.tex). Writes results/k10_paper/paper/k10_vs_paper.tex;
compile with `tectonic -X compile k10_vs_paper.tex` in that folder.
Original numbers come from the paper's own aux file (compile it with --keep-intermediates; path in AUX)."""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from ...config import ROOT
from .paths import EVAL_DIR

PAPER_DIR = ROOT / "report" / "neurips_draft"
OUT = EVAL_DIR / "paper"
AUX = Path(os.environ.get("MIFE_PAPER_AUX", OUT / "orig_paper.aux"))
FLOAT = re.compile(r"\\begin\{(table|figure)\}(\[[^\]]*\])?(.*?)\\end\{\1\}", re.S)


def floats(tex: str):
    out = []
    for m in FLOAT.finditer(tex):
        body = m.group(3)
        lab = re.search(r"\\label\{([^}]*)\}", body)
        img = re.search(r"\\includegraphics(\[[^\]]*\])?\{([^}]*)\}", body)
        key = lab.group(1) if lab else ("img:" + Path(img.group(2)).stem if img else None)
        out.append({"kind": m.group(1), "key": key, "body": body, "img": img.group(2) if img else None})
    return out


def aux_numbers():
    nums = {}
    for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", AUX.read_text()):
        nums[m.group(1)] = m.group(2)
    return nums


def k10_key(key: str) -> str:
    if key.startswith("img:"):
        return key
    return key + "-k10"


NOTES = {  # exhibits whose content does not depend on outcomes
    "tab:predictors": "Unchanged in K10 (same blocks, heads and comparisons). Block widths for Gemma-2-2B: H = 2,304 and "
                      "A = 26 layers $\\times$ 8 heads = 208 (Llama: 4,096 and 1,024).",
    "img:overview": "Unchanged in K10 apart from the numbers in the boxes (10-rule pools, $k=1..10$, 5 responses).",
    "tab:types": "The K10 version below counts groups per type for the 10-rule pools.",
}
UNNUMBERED_FIG = {"e1_A2_reliability": "7", "e1_A3_per_type_nll_heatmap": "8", "e2_posA_slot_curves": "9",
                  "e2_taskDB_type_family_heatmap": "10"}


def fix_refs(body: str, nums: dict) -> str:
    def r(m):
        lab = m.group(1)
        if lab.endswith("-k10"):
            return nums.get(lab[:-4], "?") + "\\,(K10)"
        return nums.get(lab, "?")
    body = re.sub(r"\\(?:eq)?ref\{([^}]*)\}", r, body)
    body = re.sub(r"\\label\{[^}]*\}", "", body)
    body = re.sub(r"\\cite[pt]?\{[^}]*\}", "[cit.]", body)
    body = body.replace("\\input{generated/", "\\input{" + os.path.relpath(PAPER_DIR, OUT) + "/generated/")
    return body


def preamble(tex: str) -> str:
    pre = tex[:tex.index(r"\begin{document}")]
    pre = re.sub(r"\\title\{.*?\}\n\n\\author\{.*?\n\}\n", "", pre, flags=re.S)
    pre = pre.replace(r"\usepackage[preprint]{neurips_2026}", r"\usepackage[preprint]{neurips_2026}" + "\n" + r"\usepackage{float}")
    return pre + "\\graphicspath{{" + os.path.relpath(PAPER_DIR, OUT) + "/}{./}}\n"


def build() -> Path:
    tex = (PAPER_DIR / "paper.tex").read_text()
    nums = aux_numbers()
    orig = floats(tex)
    k10 = {}
    for f in sorted(p for p in OUT.glob("k10_*.tex") if p.name != "k10_vs_paper.tex"):
        for x in floats(f.read_text()):
            k10.setdefault(x["key"], []).append(x)
    for d in ("figures_k10",):
        (OUT / d).mkdir(exist_ok=True)
    for png in (EVAL_DIR / "figures").glob("*.png"):
        shutil.copy(png, OUT / "figures_k10" / png.name)
    body = [preamble(tex), r"\begin{document}", INTRO]
    used = set()
    for x in orig:
        key = x["key"]
        stem = key[4:] if key and key.startswith("img:") else None
        num = nums.get(key) or UNNUMBERED_FIG.get(stem, "?")
        kind = "Table" if x["kind"] == "table" else "Figure"
        cap = re.search(r"\\caption\{(.{0,90})", x["body"])
        short = re.sub(r"[\\{}$]|textbf|emph|texttt", "", cap.group(1)).split(". ")[0] if cap else ""
        if len(short) > 70:
            short = short[:70].rsplit(" ", 1)[0] + " ..."
        body.append(r"\clearpage" if x["kind"] == "figure" or len(x["body"]) > 2500 else r"\bigskip")
        body.append(r"\section*{" + f"{kind} {num}" + (f" --- {short}" if short else "") + "}")
        body.append(r"\noindent\textbf{Paper (original: held-out test groups, $k\le5$, Llama-3.1-8B and Qwen3-8B, 20 responses)}")
        cnt = "table" if x["kind"] == "table" else "figure"
        body.append(r"\renewcommand{\the" + cnt + "}{" + num + "}")
        body.append(r"\begin{" + x["kind"] + "}[H]" + fix_refs(x["body"], nums) + r"\end{" + x["kind"] + "}")
        note = NOTES.get(key) or (NOTES.get("img:overview") if key == "fig:overview" else None)
        if note:
            body.append(r"\noindent\textit{" + note + "}\n")
        kk = k10_key(key) if key else None
        if kk and stem:  # figure keyed by image file
            kk = "img:" + stem
        if key and key.startswith("fig:") and x["img"]:
            kk = "img:" + Path(x["img"]).stem
        cands = k10.get(kk, [])
        if key == "fig:pairs":
            cands = k10.get("fig:pairs-k10", [])
        if key == "tab:posA":
            cands = [y for v in k10.values() for y in v if "Pos-A (part" in y["body"]]
        if key == "tab:DA":
            cands = [y for v in k10.values() for y in v if "D-A, " in y["body"]]
        if key == "tab:posCs":
            cands = [y for v in k10.values() for y in v if "Pos-C (Panel B numbers), " in y["body"]]
        if not cands and x["img"] and (OUT / "figures_k10" / (Path(x["img"]).stem + ".png")).exists():
            b = x["body"].replace(x["img"], "figures_k10/" + Path(x["img"]).stem + ".png")
            b = b.replace("\\caption{", "\\caption{K10 version (" + K10_FIG_NOTE + "). Original caption: ", 1)
            cands = [{"kind": "figure", "body": b, "img": None}]
        if cands:
            body.append(r"\noindent\textbf{K10 (50 training groups, $k=1..10$, Llama-3.1-8B and Gemma-2-2B, 5 responses; predictor scores are grouped-CV out-of-fold)}")
            for y in cands:
                used.add(id(y))
                body.append(r"\renewcommand{\the" + cnt + "}{" + num + r"\,(K10)}")
                b = y["body"]
                if y["img"]:
                    b = b.replace("{figures_k10/", "{figures_k10/").replace("{figures/test/", "{figures_k10/")
                body.append(r"\begin{" + y["kind"] + "}[H]" + fix_refs(b, nums) + r"\end{" + y["kind"] + "}")
        elif not note:
            body.append(r"\noindent\textit{No K10 version (see the notes at the start).}" + "\n")
    body.append(r"\end{document}")
    out = OUT / "k10_vs_paper.tex"
    out.write_text("\n".join(body) + "\n")
    return out


K10_FIG_NOTE = ("50 training groups, $k=1..10$, Llama-3.1-8B (blue) and Gemma-2-2B (green), 5 responses; observed "
                "outcomes; strata and $k$ ranges follow the K10 training design")


INTRO = r"""
\begin{center}{\Large\bfseries K10 re-run of every table and figure in\\[2pt]
``Will It Follow the Rules?'' (NeurIPS-format draft, 24 Nov 2026)}\\[6pt]
\normalsize Llama-3.1-8B-Instruct and Gemma-2-2B-it, $k=1..10$ rules per prompt, training groups only\end{center}

\paragraph{What this is.} Each table and figure of the paper appears in the paper's order with its number, first as
published and then rebuilt on the 10-rule (K10) main dataset (\texttt{pilots/high\_k/PRESPEC.md}). The K10 numbers come
from \texttt{results/k10\_paper/tables/}; the code is \texttt{src/mife/k10paper/}.

\paragraph{What changed.} (1) Rules per prompt: $k=1..10$ from 10-rule pools (6 frozen + 4 new rules per group), 2,500
ordered configurations per task. (2) Generators: Llama-3.1-8B-Instruct (same checkpoint) and Gemma-2-2B-it instead of
Qwen3-8B. (3) 5 verified responses per prompt instead of 20, so per-prompt rates are coarser and noise floors larger.
(4) Features from the same pass definition run on CUDA (Betty) instead of Apple MPS. (5) The orders at $k=6$ and $k=10$
are near-COAs (PRESPEC deviation).

\paragraph{Held-out test groups are not used.} The paper's results are held-out scores on the 22 S3 test groups. For K10
those groups' labels, features and predictions are sealed until a K10 test pre-specification is frozen and the project
owner says go. Every K10 number here therefore uses the 50 S3 training groups: Evaluation~1 and 2 score the grouped
5-fold cross-validation out-of-fold predictions at the CV-selected settings (each prediction comes from heads fitted on
the other 40 groups), and the bootstrap resamples the 50 training pools. Settings were selected on the same folds, so
these estimates may be optimistic; they are not held-out results and should not be read as replacing the paper's test
numbers. Because the training design has only the chain subsets (the all-subsets expansion exists only for the sealed
test groups), strict append contrasts exist only at $1\to2$ through $4\to5$, $7\to8$ and $8\to9$, and strict companion
replacements only at $k=2,3,5,6,7$.

\paragraph{Fitting.} The V2 recipe is copied unchanged (\texttt{src/mife/k10paper/recipe\_k10.yaml}, hash recorded before
selection): 15 per-type ridge-logistic heads, the same grids, tie rule, fallback, one boundary expansion, layer search
over five depths (Llama rows 8/12/16/24/33, Gemma-2B rows 7/10/13/20/27) and the incremental rule for comparisons 5--8.
"""


if __name__ == "__main__":
    print(build())
