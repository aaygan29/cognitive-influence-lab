#!/usr/bin/env python3
"""Fail if any file under data/ other than the manifest/.gitkeep is tracked by git.

Guards against accidental redistribution of raw third-party data or PII (PREREGISTRATION.md
sec. 5; docs/DATA.md). Run in CI and, ideally, as a pre-commit hook.
"""
from __future__ import annotations

import subprocess
import sys

ALLOWED = {"data/manifest.csv", "data/.gitkeep"}


def main() -> int:
    out = subprocess.run(["git", "ls-files", "data/"], capture_output=True, text=True, check=False)
    tracked = [f for f in out.stdout.splitlines() if f.strip()]
    bad = sorted(set(tracked) - ALLOWED)
    if bad:
        print("ERROR: raw data tracked under data/ (no redistribution allowed):")
        for f in bad:
            print("  -", f)
        return 1
    print("ok: only manifest/.gitkeep tracked under data/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
