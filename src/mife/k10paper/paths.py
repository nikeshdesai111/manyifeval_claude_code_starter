from __future__ import annotations

from ..config import DATA_DIR, ROOT

K10 = ROOT / "pilots" / "high_k" / "data"
INP = K10 / "inputs"
FEAT_SRC = K10 / "features"
VERIF = K10 / "verification"
FEAT = DATA_DIR / "k10_paper"
OUT = ROOT / "results" / "k10_paper"
FOLDS = ROOT / "results" / "cv_s3" / "folds.json"
GENS = ("llama", "gemma2b")
OTHER = {"llama": "gemma2b", "gemma2b": "llama"}
HIDDEN_ROWS = {"llama": [8, 12, 16, 24, 33], "gemma2b": [7, 10, 13, 20, 27]}
N_LAYERS = {"llama": 32, "gemma2b": 26}
N_SAMPLES = 5
K_MAX = 10
N_POOLS = 50
N_TRAIN_PROMPTS = 375000
N_TRAIN_RULE_ROWS = 2746500
