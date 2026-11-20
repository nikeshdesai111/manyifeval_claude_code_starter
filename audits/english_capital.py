"""Audit of `change_case:english_capital` violations (TRAIN only; reads the frozen labels, changes nothing).

The official checker passes a response iff it is non-empty, `response.isupper()`, and `langdetect.detect(response) == "en"`
(an undetectable language counts as passed). This audit splits every violation into its cause and, for responses that
fail only the langdetect step, asks whether the text is actually English, using signals independent of that step:

* script composition of the letters (Latin / CJK / other scripts),
* the share of word tokens that are common English function words (a fixed list below),
* langdetect on the same text lower-cased (same seed), and langdetect's own probability for English.

Outputs: results/tables/audit_ec_*.csv and results/audits/english_capital_langdetect_only.parquet (+ a CSV sample).
Run: PYTHONPATH=src .venv/bin/python audits/english_capital.py   (kept outside src/ so the frozen source-tree hash is unchanged)
"""
from __future__ import annotations

import re
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd

from mife.config import DATA_DIR
from mife.probing import data
from mife.probing.data import MODELS, RESULTS

EC = "change_case:english_capital"
T = RESULTS / "tables"
A = RESULTS / "audits"

# Common English function words (closed-class words; frequent in any English prose regardless of topic).
FUNCTION_WORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other
our ours ourselves out over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which while who whom why will
with would you your yours yourself yourselves also may might must shall us let it's don't can't i'm you're we're
""".split())

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]")
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def script_stats(text: str) -> dict:
    letters = [c for c in text if c.isalpha()]
    n = len(letters)
    cjk = sum(1 for c in letters if _CJK.match(c))
    lat = sum(1 for c in letters if _LATIN.match(c))
    toks = _WORD.findall(text.lower())
    fw = sum(t in FUNCTION_WORDS for t in toks)
    return {"n_letters": n, "cjk_share": cjk / n if n else np.nan, "latin_share": lat / n if n else np.nan,
            "other_script_share": (n - cjk - lat) / n if n else np.nan, "n_tokens": len(toks),
            "en_function_word_share": fw / len(toks) if toks else np.nan}


def _init():
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0


def _lang(text: str) -> dict:
    import langdetect
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0
    out = {}
    try:
        out["langdetect_lower"] = langdetect.detect(text.lower())
    except langdetect.LangDetectException:
        out["langdetect_lower"] = "undetectable"
    try:
        probs = {l.lang: l.prob for l in langdetect.detect_langs(text)}
        out["p_en"] = probs.get("en", 0.0)
        out["top_langs"] = ";".join(f"{k}:{v:.2f}" for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:3])
    except langdetect.LangDetectException:
        out["p_en"], out["top_langs"] = np.nan, ""
    return out


def assess(r) -> str:
    """Independent read of the language of a response (not the official label)."""
    if r.n_letters == 0:
        return "no letters"
    if r.cjk_share >= 0.5:
        return "mainly CJK (Chinese/Japanese/Korean)"
    if r.cjk_share > 0:
        return "English/Latin with some CJK characters"
    if r.other_script_share >= 0.5:
        return "mainly another non-Latin script"
    if r.n_tokens < 20:
        return "Latin script, too short to judge (<20 words)"
    if r.en_function_word_share >= 0.2:
        return "English (Latin script, English function words >= 20%)"
    return "Latin script, few English function words (< 20%)"


def _texts(model: str, prompt_ids: list[str], response_ids: set[str]) -> pd.DataFrame:
    gdir = DATA_DIR / "generations" / model / data.RUN
    parts = [data._read_filtered(p, prompt_ids, ["response_id", "prompt_id", "response_text"])
             for p in sorted(gdir.glob("shard_*.parquet"))]
    g = pd.concat(parts, ignore_index=True)
    return g[g.response_id.isin(response_ids)]


def run(processes: int = 8) -> dict:
    from mife.probing.describe import loop_flags
    T.mkdir(parents=True, exist_ok=True)
    A.mkdir(parents=True, exist_ok=True)
    P = data.prompts("train")[["prompt_id", "task_family", "task_key", "group_id", "k"]]
    PR = data.prompt_rules("train")[["prompt_id", "type_id"]]
    breakdown, codes, assessed, allscript, consist, cooc = [], [], [], [], [], []
    keep, vocab_df = [], Counter()
    for m in MODELS:
        V = data.verification(m, "verification_long.parquet", "train",
                              columns=["response_id", "position", "type_id", "violated"])
        V = V[V.type_id == EC]
        D = data.verification(m, "verification_responses.parquet", "train",
                              columns=["response_id", "diag_is_upper", "diag_langdetect", "diag_empty", "diag_n_words"])
        E = V.merge(D.drop(columns=["prompt_id", "prompt_sha256", "rendered_sha256"]), on="response_id", how="left")
        assert E.diag_is_upper.notna().all()
        lang_ok = E.diag_langdetect.isin(["en", "undetectable"])
        expected = E.diag_empty | ~E.diag_is_upper | ~lang_ok
        consist.append({"model": m, "checks": len(E), "violated": int(E.violated.sum()),
                        "diagnostics_disagree_with_label": int((expected.astype(int) != E.violated).sum())})
        viol = E[E.violated == 1].copy()
        viol["cause"] = np.select(
            [viol.diag_empty, ~viol.diag_is_upper, viol.diag_is_upper & ~viol.diag_langdetect.isin(["en", "undetectable"])],
            ["empty response", "not all uppercase (langdetect not reached)", "all uppercase, langdetect != en"],
            "other")
        for c, n in viol.cause.value_counts().items():
            breakdown.append({"model": m, "cause": c, "n_violations": int(n), "share_of_violations": n / len(viol),
                              "share_of_checks": n / len(E)})
        # text for every english_capital response (script composition of all of them; langdetect extras for lang-only)
        tx = _texts(m, sorted(set(E.prompt_id)), set(E.response_id))
        E = E.merge(tx[["response_id", "response_text"]], on="response_id", how="left")
        assert E.response_text.notna().all()
        S = pd.DataFrame([script_stats(t) for t in E.response_text], index=E.index)
        E = pd.concat([E, S], axis=1)
        E["status"] = np.where(E.violated == 0, "passed",
                               np.where(E.diag_empty, "violated: empty",
                                        np.where(~E.diag_is_upper, "violated: not all uppercase",
                                                 "violated: uppercase, langdetect != en")))
        for st, d in E.groupby("status"):
            allscript.append({"model": m, "status": st, "n": len(d),
                              "any_cjk": int((d.cjk_share > 0).sum()), "mainly_cjk": int((d.cjk_share >= 0.5).sum()),
                              "mainly_other_script": int((d.other_script_share >= 0.5).sum())})
        # English vocabulary from responses the official checker passed (uppercase and langdetect == en):
        # document frequency of each lower-cased word, pooled over both models below
        for t in E.loc[E.status == "passed", "response_text"]:
            vocab_df.update(set(_WORD.findall(t.lower())))
        # which co-occurring rules go with langdetect-only failures (among all-uppercase responses)
        U = E[E.status.isin(["passed", "violated: uppercase, langdetect != en"])][["prompt_id", "status"]]
        U = U.assign(lang_only=(U.status != "passed").astype(float))
        for t in sorted(set(PR.type_id) - {EC}):
            has = U.prompt_id.isin(set(PR.prompt_id[PR.type_id == t]))
            cooc.append({"model": m, "cooccurring_type": t, "n_uppercase_with": int(has.sum()),
                         "langdetect_only_rate_with": U.lang_only[has].mean(),
                         "n_uppercase_without": int((~has).sum()), "langdetect_only_rate_without": U.lang_only[~has].mean()})
        L = E[E.status == "violated: uppercase, langdetect != en"].copy()
        for c, n in L.diag_langdetect.value_counts().items():
            codes.append({"model": m, "langdetect": c, "n": int(n), "share": n / len(L)})
        with Pool(processes, initializer=_init) as pool:
            extra = pool.map(_lang, L.response_text.tolist(), chunksize=64)
        L = pd.concat([L, pd.DataFrame(extra, index=L.index)], axis=1)
        L["assessment"] = [assess(r) for r in L.itertuples()]
        L = L.merge(P, on="prompt_id", how="left")
        L = L.merge(loop_flags(m)[["response_id", "loop", "hit_max_tokens"]], on="response_id", how="left")
        keep.append(L.assign(model=m))
    VOC = {w for w, n in vocab_df.items() if n >= 5}
    out = pd.concat(keep, ignore_index=True)
    out["share_words_in_passed_vocab"] = [
        (lambda tk: sum(w in VOC for w in tk) / len(tk) if tk else np.nan)(_WORD.findall(t.lower())) for t in out.response_text]
    out["type_token_ratio"] = [(lambda tk: len(set(tk)) / len(tk) if tk else np.nan)(_WORD.findall(t.lower()))
                               for t in out.response_text]
    for (m, a), d in out.groupby(["model", "assessment"]):
        assessed.append({"model": m, "assessment": a, "n": len(d), "share": len(d) / (out.model == m).sum(),
                         "langdetect_on_lowercased_is_en": int((d.langdetect_lower == "en").sum()),
                         "median_share_words_in_passed_vocab": float(d.share_words_in_passed_vocab.median()),
                         "n_share_words_in_passed_vocab_below_0.9": int((d.share_words_in_passed_vocab < 0.9).sum()),
                         "repetition_loop": int(d.loop.sum()), "hit_max_tokens": int(d.hit_max_tokens.sum()),
                         "median_p_en": float(d.p_en.median()), "median_words": float(d.diag_n_words.median())})
    conc = []
    for m, d in out.groupby("model"):
        c = d.task_key.value_counts()
        conc.append({"model": m, "n_langdetect_only": len(d), "n_tasks": len(c), "n_groups": d.group_id.nunique(),
                     "share_from_top_5_tasks": c.head(5).sum() / len(d), "share_from_top_10_tasks": c.head(10).sum() / len(d)})
    pd.DataFrame(conc).to_csv(T / "audit_ec_concentration.csv", index=False)
    pd.DataFrame(cooc).to_csv(T / "audit_ec_cooccurring_types.csv", index=False)
    pd.DataFrame(consist).to_csv(T / "audit_ec_consistency.csv", index=False)
    pd.DataFrame(breakdown).to_csv(T / "audit_ec_causes.csv", index=False)
    pd.DataFrame(allscript).to_csv(T / "audit_ec_script_by_status.csv", index=False)
    pd.DataFrame(codes).to_csv(T / "audit_ec_langdetect_codes.csv", index=False)
    pd.DataFrame(assessed).to_csv(T / "audit_ec_langdetect_only_assessment.csv", index=False)
    out = out[
        ["model", "response_id", "prompt_id", "group_id", "task_key", "task_family", "k", "position", "diag_langdetect",
         "langdetect_lower", "p_en", "top_langs", "assessment", "n_letters", "cjk_share", "latin_share", "other_script_share",
         "n_tokens", "en_function_word_share", "share_words_in_passed_vocab", "type_token_ratio", "loop", "hit_max_tokens",
         "diag_n_words", "response_text"]]
    out.to_parquet(A / "english_capital_langdetect_only.parquet", index=False)
    out.groupby(["model", "assessment"], group_keys=False).apply(
        lambda d: d.sample(min(len(d), 15), random_state=0)).to_csv(A / "english_capital_langdetect_only_sample.csv", index=False)
    return {"langdetect_only_rows": len(out)}


if __name__ == "__main__":
    print(run())
