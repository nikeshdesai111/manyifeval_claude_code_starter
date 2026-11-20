import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def fake_manifest(tmp_path, monkeypatch):
    """SYNTHETIC manifest + tokenization hashes under tmp_path; the stale-data guard reads these instead of data/.

    make(prompt_ids, models=("llama",)) -> {prompt_id: rendered_sha256}
    """
    from mife import integrity

    root = tmp_path / "fake_data"
    monkeypatch.setattr(integrity, "DATA_DIR", root)

    def make(prompt_ids, models=("llama",), extra_tok_cols=None):
        (root / "manifest").mkdir(parents=True, exist_ok=True)
        (root / "tokenized").mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"prompt_id": list(prompt_ids), "prompt_sha256": [f"P-{p}" for p in prompt_ids]}).to_parquet(
            root / "manifest" / "prompts.parquet")
        rend = {p: f"R-{p}" for p in prompt_ids}
        for m in models:
            tok = pd.DataFrame({"prompt_id": list(prompt_ids), "prompt_sha256": [f"P-{p}" for p in prompt_ids],
                                "rendered_sha256": [rend[p] for p in prompt_ids]})
            for k, v in (extra_tok_cols or {}).items():
                tok[k] = v
            tok.to_parquet(root / "tokenized" / f"{m}.parquet")
        return rend
    return make
