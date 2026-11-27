"""Mac-side driver for K10 head fits on Betty (TRAIN only). Run detached via stage_fit.sh.

Loop per model: `unitfit plan` (the unchanged cv.run / extra.run on the Mac, which merges finished units into the
point cache) -> fit request for the next spec -> data packs (spec blocks, TRAIN rows of a group of types, each pack
under the in-flight budget) exported on the Mac, pushed to ~/mife/k10fit/in/ and md5-checked on Betty -> one Slurm
job per (pack, type) -> units pulled back to results/k10_paper/v2/unitfit/<model>/<request>/units/ with md5 checks
-> Betty pack and units deleted -> plan again. Stops at "done" (freeze is left for the Mac, run by hand).

    PYTHONPATH=src .venv/bin/python pilots/high_k/code/k10fit/stager.py --models gemma2b[,llama] --dev cuda|cpu
Log: pilots/high_k/data/k10fit.log; alerts: pilots/high_k/data/alerts.log. TEST features never leave the Mac.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "pilots/high_k/data"
LOG, ALERT = DATA / "k10fit.log", DATA / "alerts.log"
PACKS = ROOT / "data/k10_paper/packs"            # Mac cache of exported packs (TRAIN only)
B = "desai7@login.betty.parcc.upenn.edu"
RK = "mife/k10fit"
PY = str(ROOT / ".venv/bin/python")
G3F3 = "11275185,11275186"
TYPES_N = 15
BYTES_PER_ROW_META = 7 * 8                       # _v, _n, _rix, _fold, _type_ix, _pix_local, _rows_global


def log(msg: str, alert: bool = False) -> None:
    line = f"{time.strftime('%F %T')} {msg}"
    with open(LOG, "a") as fh:
        fh.write(line + "\n")
    if alert:
        with open(ALERT, "a") as fh:
            fh.write(f"{time.strftime('%F %T')} {'' if 'DONE' in msg else 'ALERT '}k10fit (Betty): {msg}\n")
    print(line, flush=True)


def sh(cmd: list[str], check=True, timeout=None) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"{cmd[:3]}... rc={p.returncode}: {p.stderr.strip()[-400:]}")
    return p.stdout


def ssh(cmd: str, check=True, timeout=600) -> str:
    out = sh(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", B, f"bash -lc {shlex.quote(cmd)}"], check, timeout)
    return "\n".join(l for l in out.splitlines() if not l.startswith(("|", "---")) and "quota" not in l.lower()
                     and "parcc_" not in l and "vast/parcc" not in l and l.strip())


def unitfit(*args: str) -> str:
    return sh([PY, "-m", "mife.k10paper.unitfit", *args], timeout=None)


HOME_MAX_GB = 46.0


def budget_bytes() -> float:
    """In-flight pack budget: 4 GB while gemma9b generation (G3) still runs on Betty, then 8 GB."""
    g3 = ssh("squeue -h -j 11275185 -o %i", check=False)
    return (4.0 if g3.strip() else 8.0) * 1e9


def home_used_gb() -> float:
    out = ssh("parcc_quota.py 2>/dev/null | grep 'home |' | head -1", check=False)
    for tok in out.replace("|", " ").split():
        try:
            return float(tok)
        except ValueError:
            continue
    return 99.0  # unknown: treat as full (wait)


def betty_inflight() -> float:
    out = ssh(f"du -sb ~/{RK}/in ~/{RK}/units 2>/dev/null | awk '{{s+=$1}} END{{print s+0}}'", check=False)
    try:
        return float(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return 0.0


# ---------------------------------------------------------------- sizing and grouping
def type_rows(model: str) -> list[int]:
    f = PACKS / model / "type_rows.json"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        code = ("import json; from mife.k10paper import cv, unitfit as U; D=cv.Data('%s',1); "
                "print(json.dumps([int((D.types==t).sum()) for t in U.TYPES]))" % model)
        f.write_text(sh([PY, "-c", code]).strip().splitlines()[-1])
    return json.loads(f.read_text())


def widths(model: str, req: dict) -> dict:
    return {b: w for (b, _), w in zip(req["spec"]["blocks"], req["widths"])}


def est_bytes(model: str, blocks: dict, types: list[int]) -> float:
    rows = sum(type_rows(model)[t] for t in types)
    prompts = min(375000, rows)                           # each prompt holds a type at most once: upper bound
    b = rows * BYTES_PER_ROW_META
    for name, w in blocks.items():
        if name in ("S", "A1"):
            b += rows * w * 4
        elif name.startswith("ER"):
            b += 600 * w * 4
        else:
            b += prompts * w * 4
    return b


def groups(model: str, blocks: dict, cap: float) -> list[list[int]]:
    """Greedy: largest types first, packed into groups whose estimated size stays under cap."""
    order = sorted(range(TYPES_N), key=lambda t: -type_rows(model)[t])
    out: list[list[int]] = []
    for t in order:
        for g in out:
            if est_bytes(model, blocks, g + [t]) <= cap:
                g.append(t)
                break
        else:
            if est_bytes(model, blocks, [t]) > cap:
                raise RuntimeError(f"type {t} alone exceeds the cap for blocks {list(blocks)}")
            out.append([t])
    return [sorted(g) for g in out]


# ---------------------------------------------------------------- packs, jobs, pulls
def md5_local(d: Path) -> dict:
    return {l.split()[1]: l.split()[0] for l in (d / "MD5SUMS").read_text().splitlines()}


def ensure_pack(model: str, blocks: list[str], types: list[int]) -> tuple[str, Path]:
    name = f"{model}__{'+'.join(sorted(blocks))}__t{'-'.join(f'{t:02d}' for t in types)}"
    name = name.replace(":", "_")
    local = PACKS / model / name
    if not (local / "MD5SUMS").exists():
        t0 = time.time()
        unitfit("export", model, ",".join(blocks), str(local), "--types", ",".join(map(str, types)))
        log(f"exported {name} {sum(f.stat().st_size for f in local.iterdir()) / 1e9:.2f} GB in {time.time() - t0:.0f}s")
    return name, local


def push_pack(name: str, local: Path) -> None:
    if ssh(f"test -f ~/{RK}/in/{name}/.ok && echo yes", check=False).strip() == "yes":
        return
    t0 = time.time()
    sh(["rsync", "-a", f"{local}/", f"{B}:{RK}/in/{name}/"], timeout=None)
    ok = ssh(f"cd ~/{RK}/in/{name} && md5sum -c --quiet MD5SUMS && touch .ok && echo MD5_OK", check=False)
    if "MD5_OK" not in ok:
        raise RuntimeError(f"md5 mismatch on Betty for pack {name}")
    size = sum(f.stat().st_size for f in local.iterdir())
    log(f"pushed {name} {size / 1e9:.2f} GB in {time.time() - t0:.0f}s ({size / 1e6 / max(time.time() - t0, 1):.1f} MB/s), md5 OK")


PART = {"cuda": "--partition=dgx-b200 --gpus=1 --cpus-per-task=28", "cpu": "--partition=genoa-std-mem --cpus-per-task=32"}


def submit_array(pack: str, units: str, req_remote: str, types: list[int], dev: str, tlimit: str) -> str:
    """One array task per type (task i -> types[i]); all tasks stage the same pack to their own node-local disk."""
    out = ssh(f"cd ~/{RK} && K10_TYPES={','.join(map(str, types))} sbatch --parsable --export=ALL,K10_TYPES "
              f"--array=0-{len(types) - 1} --kill-on-invalid-dep=yes {PART[dev]} --time={tlimit} "
              f"fit_unit.sbatch {pack} {units} {dev} {req_remote}")
    return out.strip().splitlines()[-1].split(";")[0]


def task_states(array_ids: list[str]) -> dict:
    """{(array_id, index): state} from squeue (pending/running) and sacct (finished)."""
    st = {}
    if not array_ids:
        return st
    for l in ssh(f"sacct -n -P -j {','.join(array_ids)} -o JobID,State", check=False).splitlines():
        if "|" not in l:
            continue
        j, state = l.split("|")[:2]
        if "_" in j and "." not in j and "[" not in j:
            a, i = j.split("_")
            st[(a, int(i))] = state.split()[0]
    for l in ssh(f"squeue -h -r -j {','.join(array_ids)} -o '%i %T'", check=False).splitlines():
        if " " in l and "_" in l:
            j, state = l.split()
            a, i = j.split("_")
            if i.isdigit():
                st[(a, int(i))] = state
    return st


def job_states(ids: list[str]) -> dict:
    if not ids:
        return {}
    out = ssh(f"sacct -n -X -P -j {','.join(ids)} -o JobID,State", check=False)
    st = {}
    for l in out.splitlines():
        if "|" in l:
            j, s = l.split("|")[:2]
            st[j] = s.split()[0]
    return st


def pull_units(model: str, rid: str, units: str) -> Path:
    local = ROOT / "results/k10_paper/v2/unitfit" / model / rid / "units"
    local.mkdir(parents=True, exist_ok=True)
    files = [f for f in ssh(f"cd ~/{RK}/units/{units} 2>/dev/null && ls unit_*.npz", check=False).split() if f.endswith(".npz")]
    if not files:
        return local
    sh(["rsync", "-a"] + [f"{B}:{RK}/units/{units}/{f}" for f in files] + [f"{local}/"], timeout=None)
    rem = dict(l.split()[::-1] for l in ssh(f"cd ~/{RK}/units/{units} && md5sum {' '.join(files)}").splitlines() if l.strip())
    for f in files:
        h = hashlib.md5((local / f).read_bytes()).hexdigest()
        if rem.get(f) != h:
            raise RuntimeError(f"md5 mismatch pulling {units}/{f}")
    ssh(f"cd ~/{RK}/units/{units} && rm -f {' '.join(files)}")
    return local


# ---------------------------------------------------------------- main loop
def run_request(model: str, req_path: Path, dev: str, tlimit: str) -> None:
    req = json.loads(req_path.read_text())
    rid = req["request_id"]
    blocks = widths(model, req)
    cap = budget_bytes() * 0.9
    gs = groups(model, blocks, cap)
    for g in gs:  # wait (never delete) until Betty home has room for the first pack
        while home_used_gb() + est_bytes(model, blocks, g) / 1e9 > HOME_MAX_GB:
            log(f"waiting: Betty home {home_used_gb():.2f} GB + pack {est_bytes(model, blocks, g) / 1e9:.1f} GB > {HOME_MAX_GB} GB")
            time.sleep(600)
        break
    log(f"request {rid}: spec {req['spec']['name']} blocks {list(blocks)} settings {len(req['todo'])} -> {len(gs)} pack(s) {gs}")
    units = rid.replace(":", "_")
    req_remote = f"req/{units}.json"
    ssh(f"mkdir -p ~/{RK}/req ~/{RK}/units/{units}")
    sh(["rsync", "-a", str(req_path), f"{B}:{RK}/{req_remote}"])
    fb = {tuple(x) for x in req["fallback_units"]}
    devs = ["cuda", "cpu"] if dev == "both" else [dev]
    pending = list(gs)
    active: dict[str, dict] = {}
    FINAL = ("COMPLETED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED", "NODE_FAIL", "PREEMPTED")
    while pending or active:
        if pending:
            g = [t for t in pending[0] if not all((f, t) in fb for f in range(5))]
            need = est_bytes(model, blocks, pending[0])
            if not g:
                pending.pop(0)
                continue
            room = home_used_gb() + need / 1e9 <= HOME_MAX_GB
            if room and (betty_inflight() + need <= budget_bytes() * 0.95 or not active):
                name, local = ensure_pack(model, list(blocks), pending[0])
                push_pack(name, local)
                arrays = {d: submit_array(name, units, req_remote, g, d, tlimit) for d in devs}
                active[name] = {"types": g, "arrays": arrays, "owner": {}, "tries": {t: 1 for t in g}}
                log(f"submitted arrays {arrays} on {name}: {len(g)} task(s), types {g}")
                pending.pop(0)
                continue
        time.sleep(120)
        for name in list(active):
            a = active[name]
            st = task_states(list(a["arrays"].values()))
            for i, t in enumerate(a["types"]):
                if t in a["owner"]:
                    continue
                running = [d for d, aid in a["arrays"].items() if st.get((aid, i)) in ("RUNNING", "COMPLETING", "COMPLETED")]
                if running:  # first twin to start owns the type; cancel the other twin's task
                    own = running[0]
                    a["owner"][t] = own
                    for d, aid in a["arrays"].items():
                        if d != own and st.get((aid, i)) not in FINAL:
                            ssh(f"scancel {aid}_{i}", check=False)
                    if len(running) > 1:
                        log(f"both twins started for {name} type {t}; unit locks keep one fitter per unit")
            for i, t in enumerate(a["types"]):
                own = a["owner"].get(t)
                if not own:
                    continue
                s_ = st.get((a["arrays"][own], i))
                if s_ in ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"):
                    if a["tries"][t] >= 2:
                        log(f"{name} type {t} {s_} twice - stopping", alert=True)
                        raise SystemExit(2)
                    a["tries"][t] += 1
                    ssh(f"rm -f ~/{RK}/units/{units}/unit_*_t{t:02d}.lock", check=False)
                    nid = submit_array(name, units, req_remote, [t], own, tlimit)
                    log(f"{name} type {t} {s_}; resubmitted on {own} as {nid} (units resume)", alert=True)
                    a.setdefault("extra", []).append(nid)
                    a["arrays"][own + f"_retry{a['tries'][t]}"] = nid
            pull_units(model, rid, units)
            all_done = all(st.get((a["arrays"][a["owner"][t]], i)) == "COMPLETED" for i, t in enumerate(a["types"])
                           if a["owner"].get(t)) and len(a["owner"]) == len(a["types"])
            if all_done:
                ssh(f"rm -rf ~/{RK}/in/{name}")
                log(f"pack {name} finished; units pulled (md5 OK); Betty pack deleted")
                del active[name]
    pull_units(model, rid, units)


def main() -> None:
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemma2b")
    ap.add_argument("--dev", choices=["cuda", "cpu", "both"], default="both")
    ap.add_argument("--time", default="01:00:00")
    ap.add_argument("--max-requests", type=int, default=1000)
    ap.add_argument("--log", default=str(LOG))
    a = ap.parse_args()
    LOG = Path(a.log)
    for m in a.models.split(","):
        for _ in range(a.max_requests):
            r = json.loads(unitfit("plan", m).strip().splitlines()[-1])
            if r["status"] == "done":
                log(f"{m}: cv.run and extra.run complete (all points cached); freezing on the Mac")
                code = f"from mife.k10paper import cv; import json; print(json.dumps(cv.freeze('{m}')))"
                sh([PY, "-c", code], timeout=None)
                log(f"{m}: DONE (cv.run, extra.run, freeze)", alert=True)
                break
            run_request(m, Path(r["request"]), a.dev, a.time)
    log("stager finished")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        log(f"stager stopped: {type(e).__name__}: {e}", alert=True)
        raise
