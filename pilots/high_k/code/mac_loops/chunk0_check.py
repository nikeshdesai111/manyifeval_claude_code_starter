# Runs on Betty (stdin, cwd ~/mife/repo): chunk 0 completeness for gemma2b from the done ledgers.
import glob
import pandas as pd

w = pd.read_parquet("pilots/high_k/data/inputs/worklist_gemma2b.parquet", columns=["prompt_id", "sample_index", "chunk"])
w = w[w.chunk == 0]
d = pd.concat([pd.read_parquet(f) for f in glob.glob("pilots/high_k/data/generations/gemma2b/done_c00_*.parquet")])
u = d.drop_duplicates(["prompt_id", "sample_index"])
k = set(zip(u.prompt_id, u.sample_index.astype(int)))
exp = set(zip(w.prompt_id, w.sample_index.astype(int)))
print("CHUNK0", {"expected": len(exp), "ledger_rows": len(d), "unique": len(u), "duplicates": len(d) - len(u),
                 "missing": len(exp - k), "outside": len(k - exp), "settings_hashes": sorted(d.settings_hash.unique())})
