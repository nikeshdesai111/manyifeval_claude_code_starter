from __future__ import annotations

from ..config import DATA_DIR, ROOT

K10 = ROOT / "pilots" / "high_k" / "data"
INP = K10 / "inputs"
FEAT_SRC = K10 / "features"
VERIF = K10 / "verification"
FEAT = DATA_DIR / "k10_paper"
OUT = ROOT / "results" / "k10_paper"
FOLDS = ROOT / "results" / "cv_s3" / "folds.json"
GENS = ("llama", "gemma2b", "gemma9b")
# gemma9b (added 2026-11-27): rows = the PRESPEC depths .25/.375/.5/.75 of 42 layers plus the pre-final-norm output
# (as for the other two). recipe_k10.yaml is recorded and hashed, so its hidden_rows are left unchanged; cv.run reads a
# model missing from the recipe from HIDDEN_ROWS. OTHER is only used for the other-generator reference (off by default).
OTHER = {"llama": "gemma2b", "gemma2b": "llama", "gemma9b": "gemma2b"}
HIDDEN_ROWS = {"llama": [8, 12, 16, 24, 33], "gemma2b": [7, 10, 13, 20, 27], "gemma9b": [11, 16, 21, 32, 43]}
N_LAYERS = {"llama": 32, "gemma2b": 26, "gemma9b": 42}
N_SAMPLES = 5
K_MAX = 10
N_POOLS = 50
N_TRAIN_PROMPTS = 375000
N_TRAIN_RULE_ROWS = 2746500
