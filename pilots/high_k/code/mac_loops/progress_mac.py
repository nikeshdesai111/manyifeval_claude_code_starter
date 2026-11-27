# Mac side: one BatchMode ssh pass, build the report, write progress.md, decide whether to wake the session.
# Exit 10 = alert (report now), 11 = scheduled 30-min report, 12 = all jobs finished, 0 = nothing to report.
import json, os, subprocess, sys, time
from pathlib import Path

SP = Path(sys.argv[1])
REPO = Path("/Users/nickydesai/Documents/manyifeval_claude_code_starter")
STATE = SP / "progress_state.json"
B = "desai7@login.betty.parcc.upenn.edu"
now = time.time()
st = json.loads(STATE.read_text()) if STATE.exists() else {"hist": [], "last_report": 0, "alerted": []}

alerts = []
try:
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", B,
                        "bash -lc 'cd ~/mife/repo && source scripts/betty/env.sh && mife_activate >/dev/null 2>&1 && python -'"],
                       stdin=open(SP / "progress_remote.py"), capture_output=True, text=True, timeout=300)
    line = [l for l in p.stdout.splitlines() if l.startswith("PROGRESS_JSON ")]
    if not line:
        err = (p.stderr or "").strip().splitlines()[-3:]
        raise RuntimeError(f"ssh/remote failed rc={p.returncode}: {' | '.join(err)}")
    r = json.loads(line[0][len("PROGRESS_JSON "):])
except Exception as e:
    msg = f"Progress pass failed: {e}. If this mentions Permission denied / Duo / timeout, the ssh control connection or VPN is down."
    key = "ssh:" + str(e)[:60]
    if key not in st["alerted"]:
        st["alerted"].append(key); STATE.write_text(json.dumps(st))
        print(msg); sys.exit(10)
    sys.exit(0)

def du(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e9 if Path(p).exists() else 0.0
mac_gb = du(REPO / "pilots/high_k/data/generations") + du(REPO / "pilots/high_k/data/features")

tot_done = sum(m.get("gen_records_done", 0) + m.get("feat_prompts_done", 0) for m in r["models"].values())
st["hist"].append({"t": now, "done": tot_done, "running": bool(r["running"]), **{f"{k}_g": v.get("gen_records_done", 0) for k, v in r["models"].items()},
                   **{f"{k}_f": v.get("feat_prompts_done", 0) for k, v in r["models"].items()}})
st["hist"] = [h for h in st["hist"] if now - h["t"] < 6 * 3600]

def rate(key):
    h = [x for x in st["hist"] if now - x["t"] <= 3600] or st["hist"]
    if len(h) < 2 or h[-1]["t"] - h[0]["t"] < 300:
        return None
    return (h[-1].get(key, 0) - h[0].get(key, 0)) / (h[-1]["t"] - h[0]["t"])

def eta(left, rt):
    if left <= 0:
        return "done"
    if not rt or rt <= 0:
        return "n/a (not started or no rate yet)"
    s = left / rt
    return f"{s / 3600:.1f} h"

lines = [f"# K10 progress, {time.strftime('%Y-%m-%d %H:%M')}", ""]
lines.append("| Model | Gen chunks | Gen records | Gen ETA | Feat chunks | Feat prompts | Feat ETA |")
lines.append("|---|---|---|---|---|---|---|")
for m, v in r["models"].items():
    if "error" in v:
        lines.append(f"| {m} | {v['error']} | | | | | |"); continue
    lines.append(f"| {m} | {v['gen_chunks_done']}/48 | {v['gen_records_done']:,}/{v['gen_records_total']:,} | "
                 f"{eta(v['gen_records_total'] - v['gen_records_done'], rate(m + '_g'))} | {v['feat_chunks_done']}/48 | "
                 f"{v['feat_prompts_done']:,}/{v['feat_prompts_total']:,} | {eta(v['feat_prompts_total'] - v['feat_prompts_done'], rate(m + '_f'))} |")
g2 = r["models"].get("gemma2b", {})
lines += ["", f"- gemma2b TRAIN chunks 0-15 (generation): {g2.get('gen_train_done', 0):,}/{g2.get('gen_train_total', 0):,} records",
          f"- Running tasks: {r['running'] or 'none'}; pending tasks: {r['pending'] or 'none'}",
          f"- Copied to the Mac: {mac_gb:.2f} GB; Betty in flight: {r['inflight_gb']} GB of {r['cap_gb']} GB cap; Betty home: {r['home_used_gb']} GB of 50",
          "- ETAs are for the remaining work at the last hour's observed rate and assume the current number of tasks; a model that has not started shows n/a."]
if r["failures"]:
    lines.append("- FAILED / TIMEOUT / OOM tasks:")
    for f in r["failures"]:
        lines.append(f"  - {f['task']} {f['state']} exit {f['exit']} after {f['elapsed']}; log tail: {f['log_tail'][-600:]!r}")
else:
    lines.append("- No failed, timed-out or OOM tasks.")
if r["space_guard_waiting"]:
    lines.append(f"- Space guard WAITING in: {', '.join(r['space_guard_waiting'])}")

for f in r["failures"]:
    k = "fail:" + f["task"] + f["state"]
    if k not in st["alerted"]:
        st["alerted"].append(k); alerts.append(f"Task {f['task']} {f['state']}")
if r["space_guard_waiting"]:
    k = "guard:" + ",".join(r["space_guard_waiting"])
    if k not in st["alerted"]:
        st["alerted"].append(k); alerts.append("Space guard waiting: " + ", ".join(r["space_guard_waiting"]))
if r["home_used_gb"] and r["home_used_gb"] > 45 and "home45" not in st["alerted"]:
    st["alerted"].append("home45"); alerts.append(f"Betty home at {r['home_used_gb']} GB (> 45)")
sl = REPO / "pilots/high_k/data/sync.log"
if sl.exists():
    tail = sl.read_text().splitlines()[-12:]
    bad = [l for l in tail if (l.startswith("exit ") and not l.startswith("exit 0")) or "MISMATCH" in l]
    if bad:
        k = "sync:" + bad[-1]
        if k not in st["alerted"]:
            st["alerted"].append(k); alerts.append("Sync pass failed: " + bad[-1] + " (see pilots/high_k/data/sync.log)")
            lines.append("- Last sync.log lines: " + " / ".join(tail[-6:]))
if r["running"]:
    win = [h for h in st["hist"] if now - h["t"] <= 2700]
    old = [h for h in st["hist"] if now - h["t"] >= 2700] if win and all(h.get("running") for h in win) else []
    if old and old[-1]["done"] == tot_done and "stall:" + str(tot_done) not in st["alerted"]:
        st["alerted"].append("stall:" + str(tot_done)); alerts.append("No progress for 45+ min while tasks are running")

if alerts:
    lines.insert(1, "")
    lines.insert(2, "**ALERT:** " + "; ".join(alerts))
rep = "\n".join(lines) + "\n"
(REPO / "pilots/high_k/data/progress.md").write_text(rep)
st["alerted"] = st["alerted"][-200:]
done_all = not r["active_jobs"]
running = bool(r["running"])
changed = tot_done != st.get("last_report_done") or running != st.get("last_report_running")
since = now - st["last_report"]
# running: every 30 min; idle: on any change (task start, chunk/progress, stop) or a 2-hour heartbeat
if running:
    due = since >= 1800 or not st.get("last_report_running")  # 30-min cadence; immediately when tasks start
else:
    due = changed or since >= 7200  # tasks stopped / progress moved, else a 2-hour idle heartbeat
code = 12 if done_all else 10 if alerts else 11 if due else 0
if code:
    st["last_report"] = now
    st["last_report_done"], st["last_report_running"] = tot_done, running
STATE.write_text(json.dumps(st))
if code:
    print(rep)
sys.exit(code)
