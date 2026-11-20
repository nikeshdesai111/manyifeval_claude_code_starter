"""Stale-data guard: stored outputs must belong to the CURRENT manifest and tokenization.

Prompt IDs encode only the design position (group, chain, k, ordering, task), so the same ID can
name a different prompt text if the manifest is ever rebuilt. Every stored output therefore carries
content hashes, and every stage that reads generations, features or verification outputs checks them:
  * tokenized/<model>.parquet   carries prompt_sha256 (manifest text) and rendered_sha256 (chat-rendered text);
  * generation shards           carry rendered_sha256;
  * feature shard indexes       carry prompt_sha256 and rendered_sha256;
  * verification outputs        carry prompt_sha256 and rendered_sha256.
A missing hash column, an unknown prompt ID or any mismatch raises StaleDataError; nothing is reused silently.
"""
from __future__ import annotations

import pandas as pd

from .config import DATA_DIR


class StaleDataError(RuntimeError):
    pass


def _manifest_dir():
    return DATA_DIR / "manifest"


def manifest_hashes() -> pd.Series:
    """prompt_id -> prompt_sha256 from the current manifest."""
    return pd.read_parquet(_manifest_dir() / "prompts.parquet",
                           columns=["prompt_id", "prompt_sha256"]).set_index("prompt_id")["prompt_sha256"]


def tokenized_hashes(model_key: str) -> pd.DataFrame:
    """prompt_id -> (prompt_sha256, rendered_sha256), after checking the tokenization matches the manifest."""
    path = DATA_DIR / "tokenized" / f"{model_key}.parquet"
    tok = pd.read_parquet(path)
    if "prompt_sha256" not in tok.columns:
        raise StaleDataError(f"{path} has no prompt_sha256; re-run `mife.cli tokenize {model_key}`")
    tok = tok[["prompt_id", "prompt_sha256", "rendered_sha256"]].set_index("prompt_id")
    man = manifest_hashes()
    if set(tok.index) != set(man.index) or not (tok["prompt_sha256"] == man.loc[tok.index]).all():
        raise StaleDataError(f"{path} does not match the current manifest; re-run `mife.cli tokenize {model_key}`")
    return tok


def check_rendered(model_key: str, df: pd.DataFrame, what: str) -> None:
    """df: rows with prompt_id and rendered_sha256 (generation shards, feature indexes, verification outputs)."""
    if df.empty:
        return
    if "rendered_sha256" not in df.columns:
        raise StaleDataError(f"{what}: no rendered_sha256 recorded; cannot confirm it belongs to the current manifest")
    tok = tokenized_hashes(model_key)["rendered_sha256"]
    pairs = df[["prompt_id", "rendered_sha256"]].drop_duplicates()
    unknown = ~pairs["prompt_id"].isin(tok.index)
    if unknown.any():
        raise StaleDataError(f"{what}: {int(unknown.sum())} prompt IDs not in the current manifest "
                             f"(e.g. {pairs.prompt_id[unknown].iloc[0]})")
    bad = pairs["rendered_sha256"].values != tok.loc[pairs["prompt_id"]].values
    if bad.any():
        raise StaleDataError(f"{what}: {int(bad.sum())} prompts have a different rendered_sha256 than the current "
                             f"tokenization (e.g. {pairs.prompt_id[bad].iloc[0]}); these outputs are stale")


def check_prompt(df: pd.DataFrame, what: str) -> None:
    """df: rows with prompt_id and prompt_sha256."""
    if df.empty:
        return
    if "prompt_sha256" not in df.columns:
        raise StaleDataError(f"{what}: no prompt_sha256 recorded; cannot confirm it belongs to the current manifest")
    man = manifest_hashes()
    pairs = df[["prompt_id", "prompt_sha256"]].drop_duplicates()
    unknown = ~pairs["prompt_id"].isin(man.index)
    if unknown.any():
        raise StaleDataError(f"{what}: {int(unknown.sum())} prompt IDs not in the current manifest")
    bad = pairs["prompt_sha256"].values != man.loc[pairs["prompt_id"]].values
    if bad.any():
        raise StaleDataError(f"{what}: {int(bad.sum())} prompts differ from the current manifest "
                             f"(e.g. {pairs.prompt_id[bad].iloc[0]}); these outputs are stale")


def read_verification(model_key: str, run: str, name: str) -> pd.DataFrame:
    """Read one verification output, refusing if it belongs to another manifest/tokenization."""
    path = DATA_DIR / "verification" / model_key / run / name
    df = pd.read_parquet(path)
    check_prompt(df, str(path))
    check_rendered(model_key, df, str(path))
    return df
