# Mac-local: are all gemma2b TRAIN responses present in the copied-back shards? Prints "TRAIN_READY n total dups".
import glob
import sys

import pandas as pd

m = sys.argv[1] if len(sys.argv) > 1 else "gemma2b"
base = "/Users/nickydesai/Documents/manyifeval_claude_code_starter/pilots/high_k/data"
P = pd.read_parquet(f"{base}/inputs/prompts.parquet", columns=["prompt_id", "split"])
train = set(P.prompt_id[P.split == "train"])
w = pd.read_parquet(f"{base}/inputs/worklist_{m}.parquet", columns=["prompt_id", "sample_index"])
w = w[w.prompt_id.isin(train)]
fs = glob.glob(f"{base}/generations/{m}/shard_*.parquet")
d = pd.concat([pd.read_parquet(f, columns=["prompt_id", "sample_index"]) for f in fs]) if fs else w.iloc[:0]
d = d[d.prompt_id.isin(train)]
have = set(zip(d.prompt_id, d.sample_index.astype(int)))
n = sum((p, int(s)) in have for p, s in zip(w.prompt_id, w.sample_index))
print("TRAIN_READY", n, len(w), int(d.duplicated().sum()))
