"""Driver: K10 V2 selection (base round, boundary expansion, incremental 5-7) then comparison 8, for one generator."""
import json
import sys
import time

from . import cv, extra

if __name__ == "__main__":
    m = sys.argv[1]
    cv.record_recipe()
    t0 = time.time()
    log = cv.run(m, 1)
    print(json.dumps({"stage": "cv.run done", "model": m, "hidden_row": log["hidden_row"], "expansions": log["expansions"],
                      "h": round((time.time() - t0) / 3600, 2)}), flush=True)
    r = extra.run(m, 1)
    print(json.dumps({"stage": "extra done", "model": m, "setting": r["setting"], "cv_loss": r["cv_loss"],
                      "h": round((time.time() - t0) / 3600, 2)}), flush=True)
