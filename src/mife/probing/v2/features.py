"""V2 feature blocks (TRAIN only), under data/probing_features_v2/ with versioned sidecars.

Shared (generator-independent):
  EP_train.npy    (n_prompts, 1024)  frozen BGE-M3 full-prompt embedding: the V1 frozen cache, copied with its sha256
                  (same pinned encoder revision, configs/experiment.yaml probing.encoder; no fine-tuning)
  ER_rules_train.npy (n_rules, 1024) frozen BGE-M3 exact-rule-text embedding (same provenance) + keys
Per generator (`<model>/`), token counts from the GENERATOR's frozen tokenization (data/tokenized/<model>.parquet):
  S1_train.npy    Objective 1 structure, rows = data.prompt_rules("train")
  S2_train.npy    Objective 2 structure, rows = data.prompts("train")
  H<row>_train.npy final-prompt-token hidden state at each candidate row (float32 decoded exactly from BF16 bits)
A1max is read from data/probing_features_v2/<model>/maxattn/ (maxattn.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...config import DATA_DIR
from ...storage import sha256_file, write_json
from .. import data
from .. import features as v1features
from .versioning import FEAT_SPLIT, load_array, save_array, check_meta

TYPES = v1features.TYPES
T_IX = v1features.T_IX
HIDDEN_ROWS = {"llama": [8, 12, 16, 24, 33], "qwen": [9, 14, 18, 27, 37]}
N_LAYERS = {"llama": 32, "qwen": 36}

S1_COLS = (["k", "position"] + [f"companion:{t}" for t in TYPES]
           + ["prompt_tokens", "target_span_tokens", "target_dist_to_final"])
S2_COLS = (["k", "prompt_tokens"] + [f"slot{s + 1}_type:{t}" for s in range(5) for t in TYPES]
           + [f"slot{s + 1}_span_tokens" for s in range(5)] + [f"slot{s + 1}_dist_to_final" for s in range(5)]
           + [f"slot{s + 1}_present" for s in range(5)])


def row_convention(model: str) -> dict:
    L = N_LAYERS[model]
    return {str(r): {"meaning": ("token embeddings" if r == 0 else
                                 f"output of decoder layer {r}" if r < L else
                                 f"output of decoder layer {L} AFTER the final norm (HF hidden_states[-1])" if r == L else
                                 f"output of decoder layer {L} BEFORE the final norm"),
                     "depth_fraction": round(min(r, L) / L, 4)} for r in HIDDEN_ROWS[model]}


def gdir(model: str | None = None):
    d = FEAT_SPLIT / (model or "shared")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tok(model: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "tokenized" / f"{model}.parquet",
                           columns=["prompt_id", "n_input_tokens", "rule_token_spans"]).set_index("prompt_id")


def build_structure(model: str) -> None:
    P = data.prompts("train")
    PR = data.prompt_rules("train")
    tok = _tok(model).loc[P["prompt_id"]]
    n_tok = tok["n_input_tokens"].values.astype(np.float64)
    npm = len(P)
    pid = {x: i for i, x in enumerate(P["prompt_id"])}
    span_len = np.zeros((npm, 5))
    dist = np.zeros((npm, 5))
    for i, (nt, spans) in enumerate(zip(n_tok, tok["rule_token_spans"])):
        for j, (a, b) in enumerate(spans):
            span_len[i, j] = b - a
            dist[i, j] = nt - b  # tokens between the span's last token and the final prompt token
    slot_type = np.full((npm, 5), -1)
    pi = PR["prompt_id"].map(pid).values
    si = PR["position"].values - 1
    slot_type[pi, si] = PR["type_id"].map(T_IX).values
    assert ((slot_type >= 0).sum(1) == P["k"].values).all()
    assert ((span_len > 0) == (slot_type >= 0)).all()
    present = (slot_type >= 0).astype(np.float64)
    slot_oh = np.zeros((npm, 5, len(TYPES)))
    ii, jj = np.nonzero(slot_type >= 0)
    slot_oh[ii, jj, slot_type[ii, jj]] = 1
    comp = slot_oh.sum(1)
    k = P["k"].values.astype(np.float64)
    S2 = np.concatenate([k[:, None], n_tok[:, None], slot_oh.reshape(npm, -1), span_len, dist, present], 1)
    t_oh = np.zeros((len(PR), len(TYPES)))
    t_oh[np.arange(len(PR)), PR["type_id"].map(T_IX).values] = 1
    S1 = np.concatenate([k[pi][:, None], PR["position"].values[:, None].astype(np.float64), comp[pi] - t_oh,
                         n_tok[pi][:, None], span_len[pi, si][:, None], dist[pi, si][:, None]], 1)
    assert S1.shape[1] == len(S1_COLS) and S2.shape[1] == len(S2_COLS)
    d = gdir(model)
    save_array(S1.astype(np.float32), d / "S1_train.npy", "S1v2", columns=S1_COLS, rows="data.prompt_rules('train')",
               tokenizer=model)
    save_array(S2.astype(np.float32), d / "S2_train.npy", "S2v2", columns=S2_COLS, rows="data.prompts('train')",
               tokenizer=model)
    PR[["prompt_id", "position", "rule_id"]].to_parquet(d / "S1_train_keys.parquet")
    P[["prompt_id"]].to_parquet(d / "S2_train_keys.parquet")


def build_bge(check_n: int = 64) -> dict:
    """Frozen BGE-M3 embeddings of the active split's TRAIN prompts and exact rules (same pinned encoder).

    Vectors already in the V1 cache (2026-11-14 TRAIN prompts and rules) are reused with provenance. Prompts and rules
    of groups that are TRAIN only under the active split are encoded now with the same encoder and settings
    (`encoder.encode`, MPS, float32, no truncation). Consistency check: `check_n` cached prompts and every cached rule
    are re-encoded the same way and compared with the cache; the max |difference| is recorded in the sidecar."""
    from .. import encoder
    d = gdir()
    src = data.PROBE_FEAT / "shared"
    P = data.prompts("train")
    PR = data.prompt_rules("train")
    v1_ids = pd.read_parquet(src / "EP_train_keys.parquet")["prompt_id"]
    v1_EP = np.load(src / "EP_train.npy", mmap_mode="r")
    v1_pos = {x: i for i, x in enumerate(v1_ids)}
    have = P["prompt_id"].map(v1_pos)
    EP = np.empty((len(P), v1_EP.shape[1]), dtype=np.float32)
    m = have.notna().values
    EP[m] = np.asarray(v1_EP[have[m].astype(int).values])
    model = encoder.load_model()
    new_txt = P.loc[~m, "prompt_text"].tolist()
    if new_txt:
        EP[~m] = encoder.encode(new_txt, model, batch_size=64)
    rng = np.random.default_rng(20261123)
    chk = rng.choice(np.nonzero(m)[0], min(check_n, int(m.sum())), replace=False)
    re_ep = encoder.encode(P["prompt_text"].values[chk].tolist(), model, batch_size=64)
    ep_diff = float(np.abs(re_ep - EP[chk]).max())
    rk_v1 = pd.read_parquet(src / "ER_rules_train_keys.parquet")
    ER_v1 = np.load(src / "ER_rules_train.npy")
    rules = PR.drop_duplicates("rule_id")[["rule_id", "rule_text"]].reset_index(drop=True)
    rpos = {r: i for i, r in enumerate(rk_v1["rule_id"])}
    rh = rules["rule_id"].map(rpos)
    rm = rh.notna().values
    ER = np.empty((len(rules), ER_v1.shape[1]), dtype=np.float32)
    ER[rm] = ER_v1[rh[rm].astype(int).values]
    if (~rm).any():
        ER[~rm] = encoder.encode(rules.loc[~rm, "rule_text"].tolist(), model)
    re_er = encoder.encode(rules.loc[rm, "rule_text"].tolist(), model)
    er_diff = float(np.abs(re_er - ER[rm]).max())
    enc = v1features.load_config()["probing"]["encoder"]
    info = {"n_prompts": len(P), "n_prompts_reused_v1": int(m.sum()), "n_prompts_encoded_now": int((~m).sum()),
            "n_rules": len(rules), "n_rules_reused_v1": int(rm.sum()), "n_rules_encoded_now": int((~rm).sum()),
            "reencode_check_prompts": int(len(chk)), "reencode_max_abs_diff_prompts": ep_diff,
            "reencode_check_rules": int(rm.sum()), "reencode_max_abs_diff_rules": er_diff}
    save_array(EP, d / "EP_train.npy", "EP", source=str(src / "EP_train.npy"), source_sha256=sha256_file(src / "EP_train.npy"),
               encoder=enc, rows="data.prompts('train')", **info,
               definition_note="frozen BGE-M3 dense [CLS] embedding of the full prompt text")
    save_array(ER, d / "ER_rules_train.npy", "ER", source=str(src / "ER_rules_train.npy"),
               source_sha256=sha256_file(src / "ER_rules_train.npy"), encoder=enc, rows="ER_rules_train_keys.parquet", **info)
    rules[["rule_id", "rule_text"]].to_parquet(d / "ER_rules_train_keys.parquet")
    P[["prompt_id"]].to_parquet(d / "EP_train_keys.parquet")
    return info


def build_hidden(model: str) -> None:
    P = data.prompts("train")
    d = gdir(model)
    rows = [r for r in HIDDEN_ROWS[model] if not (d / f"H{r}_train.npy").exists()]
    if rows:
        H = data.hidden_states(model, P["prompt_id"].tolist(), rows, "train")
        for r in rows:
            save_array(H[r], d / f"H{r}_train.npy", "H", row=r, row_meaning=row_convention(model)[str(r)],
                       source=str(data.feature_dir(model)), rows="data.prompts('train')")
    P[["prompt_id"]].to_parquet(d / "prompt_keys_train.parquet")
    write_json(row_convention(model), d / "hidden_row_convention.json")


def _check_keys(path, ids):
    got = pd.read_parquet(path)["prompt_id"].values
    if len(got) != len(ids) or not (got == np.asarray(ids)).all():
        raise RuntimeError(f"{path}: row order does not match the current TRAIN order")


def load_shared(name: str) -> np.ndarray:
    d = gdir()
    if name == "EP":
        _check_keys(d / "EP_train_keys.parquet", data.prompt_ids("train"))
        return load_array(d / "EP_train.npy", "EP")
    if name == "ER":
        return load_array(d / "ER_rules_train.npy", "ER")
    raise KeyError(name)


def rule_index() -> dict:
    return {r: i for i, r in enumerate(pd.read_parquet(gdir() / "ER_rules_train_keys.parquet")["rule_id"])}


def load_model_block(model: str, name: str) -> np.ndarray:
    d = gdir(model)
    if name == "S1":
        k = pd.read_parquet(d / "S1_train_keys.parquet")
        pr = data.prompt_rules("train")
        if not ((k.prompt_id.values == pr.prompt_id.values).all() and (k.rule_id.values == pr.rule_id.values).all()):
            raise RuntimeError("S1 V2 row order mismatch")
        return load_array(d / "S1_train.npy", "S1v2")
    if name == "S2":
        _check_keys(d / "S2_train_keys.parquet", data.prompt_ids("train"))
        return load_array(d / "S2_train.npy", "S2v2")
    if name.startswith("H"):
        _check_keys(d / "prompt_keys_train.parquet", data.prompt_ids("train"))
        m = check_meta(d / f"{name}_train.npy", "H")
        if m.get("row") != int(name[1:]):
            raise RuntimeError(f"{name}: stored row {m.get('row')}")
        return np.load(d / f"{name}_train.npy", mmap_mode="r")
    raise KeyError(name)
