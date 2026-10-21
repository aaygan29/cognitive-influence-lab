#!/usr/bin/env python3
"""Resolve REFERENCES.md DOIs against CrossRef and print the resolved title for each.

A convenience auditor (network; not run in CI). The authoritative verification status lives in
REFERENCES.md (the ✅/⚠ marks). Requires ``requests``.
"""
from __future__ import annotations

import re
import sys

DOI_RE = re.compile(r"https?://doi\.org/(\S+)")


def main(path: str = "REFERENCES.md") -> int:
    try:
        import requests
    except ImportError:
        print("install 'requests' to run verify_refs.py (pip install requests)")
        return 2
    with open(path) as fh:
        dois = DOI_RE.findall(fh.read())
    for doi in dois:
        doi = doi.rstrip(". ")
        try:
            r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=15)
            title = r.json()["message"]["title"][0] if r.ok else f"HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001 (report any lookup failure per DOI, keep going)
            title = f"error: {exc}"
        print(f"{doi}\n    -> {title}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
