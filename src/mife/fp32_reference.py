"""32-bit reference for the pre-generation features: is the Mac (MPS, BF16) less accurate than CUDA (BF16)?

Both feature backends run the same weights in BF16 and differ only by rounding. Neither is ground truth. Here the
SAME weights (BF16 values upcast exactly to FP32) are run in FP32 on the CPU, which neither backend uses, with the same
eager attention and the same frozen token IDs. The result is a common, far more precise reference. For the 100 pilot
prompts we measure how far each BF16 backend's stored features are from it:
  * hidden states, every row (embeddings, every layer, pre-final-norm): relative L2 error and cosine;
  * rule-span attention sums/means: absolute error;
  * the floor: the error from merely rounding the FP32 reference to BF16 (no BF16 storage can beat it);
  * the scale of real signal: relative L2 distance between DIFFERENT prompts at the same layer.
Writes data/consistency/{model}_fp32_reference.json and the reference arrays (.npz).
This is a numerical check of representations only: no probes, no outcomes, TRAIN pilot prompts only.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .introspect import K_MAX
from .storage import write_json
from .xbackend import _load_features


def extract(model_key: str, prompt_ids: list[str], device: str = "cpu") -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    from .generate_mlx import model_path

    torch.set_num_threads(os.cpu_count())
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(str(model_path(model_key)), dtype=torch.float32,
                                                 attn_implementation="eager").to(device).eval()
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    load_s = time.time() - t0
    captured = {}
    hook = model.model.norm.register_forward_hook(lambda m, i, o: captured.update(pre=i[0]))
    cfg = model.config
    L, nh = cfg.num_hidden_layers, cfg.num_attention_heads
    tok = _tok_checked(model_key).set_index("prompt_id")
    n = len(prompt_ids)
    H = np.zeros((n, L + 2, cfg.hidden_size), np.float32)
    A_sum = np.full((n, L, nh, K_MAX), np.nan, np.float32)
    A_mean = np.full_like(A_sum, np.nan)
    T_sum = np.zeros((n, L, nh), np.float32)
    per = []
    for i, pid in enumerate(prompt_ids):
        r = tok.loc[pid]
        ids = torch.tensor([list(r.input_ids)], device=device)
        t1 = time.time()
        with torch.no_grad():
            o = model(input_ids=ids, output_hidden_states=True, output_attentions=True, use_cache=False)
        last = ids.shape[1] - 1
        H[i] = torch.stack([h[0, last] for h in o.hidden_states] + [captured["pre"][0, last]]).cpu().numpy()
        att = torch.stack([a[0, :, last, :] for a in o.attentions]).double()
        for j, (a, b) in enumerate(tuple(x) for x in r.rule_token_spans):
            s = att[:, :, a:b].sum(-1)
            A_sum[i, :, :, j] = s.cpu().numpy()
            A_mean[i, :, :, j] = (s / (b - a)).cpu().numpy()
        ta, tb = r.task_token_span
        T_sum[i] = att[:, :, ta:tb].sum(-1).cpu().numpy()
        per.append(time.time() - t1)
        if i % 10 == 0:
            print(f"[fp32 {model_key}] {i + 1}/{n} {per[-1]:.1f} s", flush=True)
        del o, att
    hook.remove()
    return {"H": H, "A": {"rule_sum": A_sum, "rule_mean": A_mean, "task_sum": T_sum},
            "meta": {"device": device, "dtype": "float32", "attn_implementation": "eager", "model_load_s": load_s,
                     "per_prompt_s_median": float(np.median(per)), "torch_version": torch.__version__,
                     "transformers_version": __import__("transformers").__version__, "n_threads": torch.get_num_threads()}}


def bf16_round(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even FP32 -> BF16 -> FP32 (what storing a value in BF16 costs at minimum)."""
    import torch
    return torch.from_numpy(np.ascontiguousarray(x)).to(torch.bfloat16).float().numpy()


def rel_err(X: np.ndarray, R: np.ndarray) -> np.ndarray:
    """(n, rows, d) -> (n, rows) relative L2 error of X against reference R."""
    return np.linalg.norm(X - R, axis=-1) / np.maximum(np.linalg.norm(R, axis=-1), 1e-30)


def cosine(X: np.ndarray, R: np.ndarray) -> np.ndarray:
    return (X * R).sum(-1) / np.maximum(np.linalg.norm(X, axis=-1) * np.linalg.norm(R, axis=-1), 1e-30)


def between_prompt_rel(R: np.ndarray) -> np.ndarray:
    """Relative L2 distance from each prompt to the next prompt (cyclic), per row: the scale of real differences."""
    return rel_err(np.roll(R, 1, axis=0), R)


def summarize(R: np.ndarray, RA: dict, backends: dict[str, tuple[np.ndarray, dict]]) -> dict:
    floor = rel_err(bf16_round(R), R)
    signal = between_prompt_rel(R)
    out = {"rows": int(R.shape[1]), "per_row": [], "attention": {}, "paired": {}}
    errs = {k: rel_err(H, R) for k, (H, _) in backends.items()}
    for row in range(R.shape[1]):
        e = {"row": row, "bf16_rounding_floor_median": float(np.median(floor[:, row])),
             "between_prompt_rel_l2_median": float(np.median(signal[:, row]))}
        for k, (H, _) in backends.items():
            e[f"{k}_rel_err_median"] = float(np.median(errs[k][:, row]))
            e[f"{k}_rel_err_max"] = float(errs[k][:, row].max())
            e[f"{k}_cos_min"] = float(cosine(H[:, row], R[:, row]).min())
        out["per_row"].append(e)
    for k, (H, A) in backends.items():
        out["attention"][k] = {}
        for name in ("rule_sum", "rule_mean", "task_sum"):
            d = np.abs(A[name] - RA[name])
            d = d[~np.isnan(d)]
            out["attention"][k][name] = {"median_abs_err": float(np.median(d)), "p99_abs_err": float(np.quantile(d, .99)),
                                         "max_abs_err": float(d.max())}
        out[f"{k}_rel_err_median_all_rows"] = float(np.median(errs[k]))
        out[f"{k}_rel_err_max_all_rows"] = float(errs[k].max())
        out[f"{k}_worst_err_over_between_prompt"] = float((errs[k] / np.maximum(signal, 1e-30)).max())
    keys = list(backends)
    if len(keys) == 2:
        a, b = keys
        ea, eb = errs[a], errs[b]
        rows = slice(1, None)  # row 0 (embeddings) is exact for both
        out["paired"] = {f"frac_{a}_closer": float((ea[:, rows] < eb[:, rows]).mean()),
                         f"frac_{b}_closer": float((eb[:, rows] < ea[:, rows]).mean()),
                         f"median_ratio_{a}_over_{b}": float(np.median(ea[:, rows] / np.maximum(eb[:, rows], 1e-30))),
                         "excludes_row0_embeddings": True}
        ab = rel_err(backends[a][0], backends[b][0])
        out["paired"][f"{a}_vs_{b}_rel_l2_median"] = float(np.median(ab[:, rows]))
    out["bf16_rounding_floor_median_all_rows"] = float(np.median(floor))
    out["between_prompt_rel_l2_median_all_rows"] = float(np.median(signal[:, 1:]))
    return out


def run(model_key: str, runs: dict[str, str] | None = None, device: str = "cpu", max_prompts: int | None = None) -> dict:
    runs = runs or {"mac_mps": "pilot", "betty_cuda": "pilot_betty_dgx-b200"}
    pids = list(pd.read_parquet(DATA_DIR / "pilot" / "pilot_prompts.parquet").prompt_id)[:max_prompts]
    out_dir = DATA_DIR / "consistency"
    ref_path = out_dir / f"{model_key}_fp32_reference_{device}.npz"
    if ref_path.exists() and max_prompts is None:
        z = np.load(ref_path, allow_pickle=False)
        assert list(z["prompt_ids"]) == pids
        ref = {"H": z["H"], "A": {k: z[k] for k in ("rule_sum", "rule_mean", "task_sum")},
               "meta": json.loads(str(z["meta"]))}
    else:
        ref = extract(model_key, pids, device)
        if max_prompts is None:
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez(ref_path, H=ref["H"], prompt_ids=np.array(pids), meta=json.dumps(ref["meta"]), **ref["A"])
    backends = {}
    for k, r in runs.items():
        H, A, _, meta = _load_features(model_key, r, pids)
        backends[k] = (H, A)
    res = {"model_key": model_key, "n_prompts": len(pids), "reference": ref["meta"], "compared_runs": runs,
           **summarize(ref["H"], ref["A"], backends)}
    if max_prompts is None:
        write_json(res, out_dir / f"{model_key}_fp32_reference.json")
    return res


def _tok_checked(model_key: str):
    """Tokenized prompts, after confirming they match the current manifest (StaleDataError otherwise)."""
    from .integrity import tokenized_hashes
    tokenized_hashes(model_key)
    return pd.read_parquet(DATA_DIR / "tokenized" / f"{model_key}.parquet")
