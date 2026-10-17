"""Minimal OpenRouter chat client: hard spend cap, JSONL cost ledger, reply cache (stdlib only).

Key is read from ~/.config/openrouter/.env, never from the repo. Every paid call is logged to the
ledger (so the cap survives restarts) and its reply cached by `tag`, so a crashed run resumes
without re-buying finished calls. Tags must therefore be unique per logical call.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ENV = Path.home() / ".config/openrouter/.env"
URL = "https://openrouter.ai/api/v1/chat/completions"
_LOCK = threading.Lock()


def _key() -> str:
    if "OPENROUTER_API_KEY" in os.environ:
        return os.environ["OPENROUTER_API_KEY"]
    for line in ENV.read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no OPENROUTER_API_KEY")


class Client:
    def __init__(self, model: str, ledger: Path, cap_usd: float):
        self.model, self.ledger, self.cap = model, ledger, cap_usd
        self.cache_path = ledger.with_name("cache.jsonl")
        ledger.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if line:
                    rec = json.loads(line)
                    self.cache[rec["tag"]] = rec["content"]

    def spent(self) -> float:
        if not self.ledger.exists():
            return 0.0
        return sum(json.loads(line)["cost"] for line in self.ledger.read_text().splitlines() if line)

    def chat(self, messages: list[dict], max_tokens: int = 2000, temperature: float = 0.7, tag: str = "") -> str:
        if tag and tag in self.cache:
            return self.cache[tag]
        if self.spent() >= self.cap:
            raise RuntimeError(f"spend cap ${self.cap} reached")
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "usage": {"include": True}}
        out, err = None, None
        for attempt in range(6):
            try:
                req = urllib.request.Request(URL, json.dumps(body).encode(), {
                    "Authorization": f"Bearer {_key()}", "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    out = json.loads(r.read())
                if out.get("choices"):
                    break
                err = out.get("error", out)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                err = repr(e)
            out = None
            time.sleep(min(60, 2 ** attempt + 1))
        if out is None:
            raise RuntimeError(f"OpenRouter failed after retries: {err}")
        usage = out.get("usage", {})
        content = out["choices"][0]["message"].get("content") or ""
        with _LOCK:
            with self.ledger.open("a") as f:
                f.write(json.dumps({"t": time.time(), "tag": tag, "cost": usage.get("cost", 0.0),
                                    "in": usage.get("prompt_tokens"), "out": usage.get("completion_tokens")}) + "\n")
            if tag:
                self.cache[tag] = content
                with self.cache_path.open("a") as f:
                    f.write(json.dumps({"tag": tag, "content": content}) + "\n")
        return content
