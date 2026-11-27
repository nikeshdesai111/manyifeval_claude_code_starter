"""Minimal OpenRouter chat client with a request-hash cache and a token/cost log.

* The key is read from the OPENROUTER_API_KEY environment variable only (load the repo's .env into the environment);
  it is never written to disk, logged or put into the cache key.
* Every call is cached under data/difficulty/openrouter_cache/<sha256 of the canonical request>.json, so a rerun with
  the same request costs nothing. ``variant`` lets a caller ask for a fresh sample of an identical request (retries).
* Every non-cached call appends one line to data/difficulty/logs/openrouter_calls.jsonl with model, provider,
  tokens, the OpenRouter-reported cost (usage accounting) and the cache key.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .catalog import ROOT

URL = "https://openrouter.ai/api/v1/chat/completions"
CACHE = ROOT / "data" / "difficulty" / "openrouter_cache"
LOG = ROOT / "data" / "difficulty" / "logs" / "openrouter_calls.jsonl"


class OpenRouterError(RuntimeError):
    pass


def _key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        raise OpenRouterError("OPENROUTER_API_KEY is not set (load the repo .env into the environment)")
    return k


def request_hash(body: dict, variant: int = 0) -> str:
    return hashlib.sha256(json.dumps({"body": body, "variant": variant}, sort_keys=True).encode()).hexdigest()


def chat(model: str, messages: list[dict], *, provider: str | None = None, response_format: dict | None = None,
         max_tokens: int = 4000, reasoning: dict | None = None, variant: int = 0, tag: str = "",
         timeout: int = 300, retries: int = 4) -> dict:
    """Returns {"content", "usage", "cost", "cached", "cache_key", "model", "provider"}."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "usage": {"include": True}}
    if provider:
        body["provider"] = {"order": [provider], "allow_fallbacks": False}
    if response_format:
        body["response_format"] = response_format
    if reasoning:
        body["reasoning"] = reasoning
    key = request_hash(body, variant)
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{key}.json"
    if f.exists():
        rec = json.loads(f.read_text())
        return {**rec["result"], "cached": True, "cache_key": key}
    data = json.dumps(body).encode()
    last = None
    for i in range(retries):
        req = urllib.request.Request(URL, data=data, method="POST", headers={
            "Authorization": f"Bearer {_key()}", "Content-Type": "application/json",
            "X-Title": "difficulty-dataset"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read().decode())
            break
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode()[:300]}"
            if e.code not in (408, 429, 500, 502, 503, 504):
                raise OpenRouterError(last) from None
        except (urllib.error.URLError, TimeoutError) as e:
            last = str(e)
        time.sleep(2 ** i * 3)
    else:
        raise OpenRouterError(f"failed after {retries} tries: {last}")
    if "error" in resp:
        raise OpenRouterError(json.dumps(resp["error"])[:300])
    ch = resp["choices"][0]
    usage = resp.get("usage", {})
    result = {"content": ch["message"].get("content") or "", "finish_reason": ch.get("finish_reason"),
              "usage": usage, "cost": usage.get("cost"), "model": resp.get("model"),
              "provider": resp.get("provider"), "id": resp.get("id"), "latency_s": round(time.time() - t0, 2)}
    f.write_text(json.dumps({"request": body, "variant": variant, "tag": tag, "result": result}, indent=1))
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, "cache_key": key,
                             "model": result["model"], "provider": result["provider"],
                             "prompt_tokens": usage.get("prompt_tokens"),
                             "completion_tokens": usage.get("completion_tokens"),
                             "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                             "cost_usd": usage.get("cost"), "finish_reason": result["finish_reason"],
                             "latency_s": result["latency_s"]}) + "\n")
    return {**result, "cached": False, "cache_key": key}


def json_schema_format(name: str, schema: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


def total_cost() -> dict:
    if not LOG.exists():
        return {"calls": 0, "cost_usd": 0.0}
    rows = [json.loads(x) for x in LOG.read_text().splitlines() if x.strip()]
    by = {}
    for r in rows:
        b = by.setdefault(r["model"] or "?", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
        b["calls"] += 1
        b["prompt_tokens"] += r["prompt_tokens"] or 0
        b["completion_tokens"] += r["completion_tokens"] or 0
        b["cost_usd"] += r["cost_usd"] or 0.0
    return {"calls": len(rows), "cost_usd": round(sum(b["cost_usd"] for b in by.values()), 4), "by_model": by}
