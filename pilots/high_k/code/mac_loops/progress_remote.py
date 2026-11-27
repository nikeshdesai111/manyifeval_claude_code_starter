# Runs on Betty (fed over stdin, cwd ~/mife/repo, mife env). Prints one JSON line: progress, queue, space, failures.
import glob, json, os, re, subprocess
import pandas as pd

D = "pilots/high_k/data"
JOBS = {"11275181": ("gemma2b", "gen"), "11275182": ("gemma2b", "feat"), "11275183": ("llama", "gen"),
        "11275184": ("llama", "feat"), "11275185": ("gemma9b", "gen"), "11275186": ("gemma9b", "feat")}
out = {"models": {}}


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120).stdout


for m in ("gemma2b", "llama", "gemma9b"):
    r = {}
    wf = f"{D}/inputs/worklist_{m}.parquet"
    if not os.path.exists(wf):
        out["models"][m] = {"error": "no worklist"}
        continue
    w = pd.read_parquet(wf, columns=["prompt_id", "sample_index", "chunk"])
    g = glob.glob(f"{D}/generations/{m}/done_*.parquet")
    dn = pd.concat([pd.read_parquet(f, columns=["prompt_id", "sample_index"]) for f in g]) if g else None
    key = w.prompt_id + "#" + w.sample_index.astype(str)
    have = set(dn.prompt_id + "#" + dn.sample_index.astype(str)) if dn is not None else set()
    w["done"] = key.isin(have)
    per = w.groupby("chunk").done.agg(["sum", "size"])
    r["gen_records_done"], r["gen_records_total"] = int(w.done.sum()), len(w)
    r["gen_chunks_done"] = int((per["sum"] == per["size"]).sum())
    r["gen_train_done"] = int(w[w.chunk <= 15].done.sum()); r["gen_train_total"] = int((w.chunk <= 15).sum())
    p = w.drop_duplicates("prompt_id")[["prompt_id", "chunk"]]
    fi = glob.glob(f"{D}/features/{m}/index_*.parquet")
    fp = set(pd.concat([pd.read_parquet(f, columns=["prompt_id"]) for f in fi]).prompt_id) if fi else set()
    p["done"] = p.prompt_id.isin(fp)
    pf = p.groupby("chunk").done.agg(["sum", "size"])
    r["feat_prompts_done"], r["feat_prompts_total"] = int(p.done.sum()), len(p)
    r["feat_chunks_done"] = int((pf["sum"] == pf["size"]).sum())
    out["models"][m] = r

# queue
q = sh("squeue -h -u desai7 -o '%i %T %M'")
run, pend = {}, {}
for line in q.splitlines():
    jid, st, el = line.split()[:3]
    base = jid.split("_")[0]
    if base not in JOBS:
        continue
    tgt = run if st == "RUNNING" else pend
    # pending arrays show as 1127518x_[a-b%c]: count tasks
    n = 1
    mm = re.search(r"_\[(.+?)(%\d+)?\]$", jid)
    if mm:
        n = 0
        for part in mm.group(1).split(","):
            a, _, b = part.partition("-")
            n += (int(b) - int(a) + 1) if b else 1
    tgt[base] = tgt.get(base, 0) + n
out["running"] = {f"{JOBS[k][0]}_{JOBS[k][1]}": v for k, v in run.items()}
out["pending"] = {f"{JOBS[k][0]}_{JOBS[k][1]}": v for k, v in pend.items()}
out["active_jobs"] = sorted(set(run) | set(pend))

# failures
fails = []
sa = sh("sacct -n -P -j " + ",".join(JOBS) + " -o JobID,State,ExitCode,Elapsed")
for line in sa.splitlines():
    jid, st, ec, el = line.split("|")
    if "." in jid or not re.search(r"FAIL|TIMEOUT|OUT_OF_ME|NODE_FAIL|PREEMPT|CANCEL", st):
        continue
    base, _, idx = jid.partition("_")
    name = "mife-k10" if JOBS.get(base, ("", "gen"))[1] == "gen" else "mife-k10f"
    log = f"logs/{name}_{base}_{idx}.out"
    tail = sh(f"tail -n 6 {log} 2>/dev/null | cut -c1-220") if os.path.exists(log) else ""
    fails.append({"task": jid, "state": st, "exit": ec, "elapsed": el, "log_tail": tail.strip()})
out["failures"] = fails

# space guard waiting: last line of each running task's log
waiting = []
for line in sh("squeue -h -u desai7 -t R -o '%i'").split():
    base, _, idx = line.partition("_")
    if base in JOBS:
        name = "mife-k10" if JOBS[base][1] == "gen" else "mife-k10f"
        last = sh(f"tail -n 3 logs/{name}_{base}_{idx}.out 2>/dev/null")
        if "space guard:" in last:
            waiting.append(line)
out["space_guard_waiting"] = waiting

size = 0
for k in ("generations", "features"):
    for f in glob.glob(f"{D}/{k}/*/*"):
        if os.path.isfile(f):
            size += os.path.getsize(f)
out["inflight_gb"] = round(size / 1e9, 2)
try:
    out["cap_gb"] = float(open(f"{D}/env/max_gb").read().strip())
except Exception:
    out["cap_gb"] = None
qq = sh("parcc_quota.py 2>/dev/null")
mq = re.search(r"\|\s*home\s*\|\s*\S+\s*\|\s*([\d.]+)\s*GB", qq)
out["home_used_gb"] = float(mq.group(1)) if mq else None
print("PROGRESS_JSON " + json.dumps(out))
