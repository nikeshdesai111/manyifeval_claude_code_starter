"""K10 head fits on the Mac (TRAIN only), detached. Per model: `unitfit plan` (unchanged cv.run / extra.run; merges finished
units into the point cache) -> fit request for the next spec -> 2 local workers over types (1 for very wide specs),
CPU Hessian, BLAS threads split -> plan again ... -> "done" -> cv.freeze(model). Then the next model.

    nohup env PYTHONPATH=src .venv/bin/python pilots/high_k/code/k10fit/local_fit.py --models gemma2b,llama &
Log: pilots/high_k/data/k10fit.log; DONE and failure lines also go to pilots/high_k/data/alerts.log.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "pilots/high_k/data"
LOG, ALERT = DATA / "k10fit.log", DATA / "alerts.log"
PY = str(ROOT / ".venv/bin/python")
UNITROOT = ROOT / "results/k10_paper/v2/unitfit"
TOTAL_THREADS = 12
WIDE = 5000


def log(msg: str, alert: bool = False) -> None:
    line = f"{time.strftime('%F %T')} {msg}"
    with open(LOG, "a") as fh:
        fh.write(line + "\n")
    if alert:
        with open(ALERT, "a") as fh:
            fh.write(f"{time.strftime('%F %T')} {'ALERT ' if 'FAIL' in msg or 'stopped' in msg else ''}k10fit (Mac): {msg}\n")


def env(threads: int) -> dict:
    e = dict(os.environ, PYTHONPATH=str(ROOT / "src"), MIFE_V2_HESS="cpu")
    for k in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        e[k] = str(threads)
    return e


def run(args: list[str], threads: int, out: Path | None = None) -> subprocess.CompletedProcess:
    fh = open(out, "a") if out else subprocess.PIPE
    return subprocess.run([PY, *args], cwd=ROOT, env=env(threads), stdout=fh, stderr=subprocess.STDOUT, text=True)


def type_rows(model: str) -> list[int]:
    f = UNITROOT / model / "type_rows.json"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        code = ("import json; from mife.k10paper import cv, unitfit as U; D=cv.Data('%s',1); "
                "print(json.dumps([int((D.types==t).sum()) for t in U.TYPES]))" % model)
        p = subprocess.run([PY, "-c", code], cwd=ROOT, env=env(TOTAL_THREADS), capture_output=True, text=True, check=True)
        f.write_text(p.stdout.strip().splitlines()[-1])
    return json.loads(f.read_text())


def split(model: str, n: int) -> list[list[int]]:
    rows = type_rows(model)
    parts = [[] for _ in range(n)]
    load = [0] * n
    for t in sorted(range(len(rows)), key=lambda t: -rows[t]):
        i = load.index(min(load))
        parts[i].append(t)
        load[i] += rows[t]
    return [sorted(p) for p in parts if p]


def fit_request(model: str, req_path: Path) -> None:
    req = json.loads(req_path.read_text())
    d = sum(req["widths"])
    n = 1 if d > WIDE else 2
    units = req_path.parent / "units"
    parts = split(model, n)
    t0 = time.time()
    log(f"{model}: {req['spec']['name']} ({req['request_id']}), {len(req['todo'])} settings, d<={d}, {n} worker(s) {parts}")
    wl = DATA / "k10fit_workers.log"
    procs = [subprocess.Popen([PY, "-m", "mife.k10paper.unitfit", "fitlocal", model, str(req_path), str(units),
                               "--types", ",".join(map(str, p))], cwd=ROOT, env=env(TOTAL_THREADS // n),
                              stdout=open(wl, "a"), stderr=subprocess.STDOUT) for p in parts]
    rcs = [p.wait() for p in procs]
    if any(rcs):  # one retry: finished units are kept, so the retry only fits what is missing
        log(f"{model}: {req['spec']['name']} worker rc={rcs}; retrying once", alert=True)
        procs = [subprocess.Popen([PY, "-m", "mife.k10paper.unitfit", "fitlocal", model, str(req_path), str(units),
                                   "--types", ",".join(map(str, p))], cwd=ROOT, env=env(TOTAL_THREADS // n),
                                  stdout=open(wl, "a"), stderr=subprocess.STDOUT) for p in parts]
        rcs = [p.wait() for p in procs]
        if any(rcs):
            log(f"{model}: {req['spec']['name']} FAILED twice (rc={rcs}); see k10fit_workers.log", alert=True)
            raise SystemExit(2)
    log(f"{model}: {req['spec']['name']} units done in {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemma2b,llama")
    a = ap.parse_args()
    for m in a.models.split(","):
        t_model = time.time()
        log(f"{m}: start")
        while True:
            p = run(["-m", "mife.k10paper.unitfit", "plan", m], TOTAL_THREADS)
            if p.returncode != 0:
                log(f"{m}: plan FAILED rc={p.returncode}: {p.stdout[-600:]}", alert=True)
                raise SystemExit(2)
            for line in p.stdout.splitlines():
                if line.startswith("{") and '"spec"' in line and "merged_from" in line:
                    log(f"{m}: merged {json.loads(line)['spec']}")
            r = json.loads(p.stdout.strip().splitlines()[-1])
            if r["status"] == "done":
                break
            fit_request(m, Path(r["request"]))
        log(f"{m}: cv.run + extra.run complete; freezing")
        p = run(["-c", f"from mife.k10paper import cv; import json; print(json.dumps(cv.freeze('{m}')))"], TOTAL_THREADS)
        if p.returncode != 0:
            log(f"{m}: freeze FAILED rc={p.returncode}: {p.stdout[-600:]}", alert=True)
            raise SystemExit(2)
        log(f"{m}: DONE (cv.run, extra.run, freeze) in {(time.time() - t_model) / 3600:.2f} h", alert=True)
    log("all models DONE", alert=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        log(f"runner stopped: {type(e).__name__}: {e}", alert=True)
        raise
