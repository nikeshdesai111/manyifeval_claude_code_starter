"""One-line status of the V2 max-attention extraction: model, shards, prompts, rate and ETA (from shard commit times)."""
import glob, os, time, math
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOTAL, SHARD = 42066, 1024
NSH = math.ceil(TOTAL / SHARD)
out, eta_total = [], 0.0
rate = None
for m in ("llama", "qwen"):
    idx = sorted(glob.glob(f"{ROOT}/data/probing_features_v2/{m}/maxattn/index_[0-9]*.parquet"))
    n = len(idx)
    done = min(n * SHARD, TOTAL)
    if n >= 2:
        ts = [os.path.getmtime(f) for f in idx]
        rate = (ts[-1] - ts[0]) / ((n - 1) * SHARD)  # s per prompt incl. overhead
    r = rate or 0.26
    left = (TOTAL - done) * r
    eta_total += left
    out.append(f"{m} shards {n}/{NSH} prompts {done}/{TOTAL}")
eta = time.strftime("%H:%M UTC", time.gmtime(time.time() + eta_total))
print(" | ".join(out) + f" | {rate or 0.26:.3f} s/prompt | ETA both done {eta}")
