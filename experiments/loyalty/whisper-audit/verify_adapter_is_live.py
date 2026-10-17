#!/usr/bin/env python3
"""Verify a saved LoRA adapter actually trained -- no model, no GPU, no torch required.

The failure this catches: an adapter file whose LoRA B matrices are all ~0. LoRA B is
initialised to exactly zero, so an all-zero B means the adapter is at initialisation and is
inert (delta = B*A ~ 0), i.e. the model it attaches to is just the base model. That is what
made both arms of the "Auditing Rare-Condition Loyalties" run emit byte-identical responses.

Usage:
    python3 verify_adapter_is_live.py ADAPTER [ADAPTER ...]

  ADAPTER may be a directory containing adapter_model.safetensors (or adapter_model.bin), or
  a path to the tensor file itself. With two or more adapters it also checks that they are not
  byte-identical to each other.

Exit code 0 if every adapter looks trained and distinct; 1 otherwise. Reads safetensors
directly (f32/f16/bf16) using only numpy, so it runs anywhere the adapter file does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

B_ZERO_TOL = 1e-6  # max ||B|| at or below this is treated as "at initialisation"

_ST_DTYPE = {
    "F64": ("<f8", 8), "F32": ("<f4", 4), "F16": ("<f2", 2),
    "BF16": ("bf16", 2), "I64": ("<i8", 8), "I32": ("<i4", 4),
    "I16": ("<i2", 2), "I8": ("|i1", 1), "U8": ("|u1", 1), "BOOL": ("|u1", 1),
}


def _find_tensor_file(path: Path) -> Path | None:
    if path.is_file():
        return path
    for name in ("adapter_model.safetensors", "adapter_model.bin", "adapter_model.pt"):
        cand = path / name
        if cand.exists():
            return cand
    return None


def _l2_from_bytes(raw: bytes, dtype: str) -> float:
    """L2 norm of a raw tensor blob, decoding the safetensors dtype with numpy only."""
    import numpy as np

    if dtype not in _ST_DTYPE:
        return float("nan")
    np_dtype, _ = _ST_DTYPE[dtype]
    if np_dtype == "bf16":
        u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        f32 = (u16 << 16).view(np.float32)
        arr = f32
    else:
        arr = np.frombuffer(raw, dtype=np_dtype)
    arr = arr.astype(np.float64, copy=False)
    return float(np.sqrt(np.dot(arr, arr)))


def _read_safetensors_norms(fp: Path) -> dict[str, tuple[str, float]] | None:
    """Return {tensor_name: (dtype, l2_norm)} for a safetensors file, or None if not one."""
    with open(fp, "rb") as fh:
        head = fh.read(8)
        if len(head) < 8:
            return None
        (hlen,) = struct.unpack("<Q", head)
        try:
            header = json.loads(fh.read(hlen).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        base = 8 + hlen
        out: dict[str, tuple[str, float]] = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype = meta.get("dtype", "")
            start, end = meta.get("data_offsets", (0, 0))
            fh.seek(base + start)
            raw = fh.read(end - start)
            out[name] = (dtype, _l2_from_bytes(raw, dtype))
    return out


def _read_torch_norms(fp: Path) -> dict[str, tuple[str, float]] | None:
    try:
        import torch  # optional fallback for .bin/.pt
    except ImportError:
        return None
    try:
        state = torch.load(fp, map_location="cpu")
    except Exception:  # noqa: BLE001
        return None
    out: dict[str, tuple[str, float]] = {}
    for name, t in state.items():
        try:
            out[name] = (str(t.dtype), float(t.detach().float().norm().item()))
        except Exception:  # noqa: BLE001
            out[name] = (str(getattr(t, "dtype", "?")), float("nan"))
    return out


def _is_b(name: str) -> bool:
    n = name.lower()
    return "lora_b" in n or "lora_embedding_b" in n


def _is_a(name: str) -> bool:
    n = name.lower()
    return "lora_a" in n or "lora_embedding_a" in n


def _sha256(fp: Path) -> str:
    h = hashlib.sha256()
    with open(fp, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(path: Path, tol: float = B_ZERO_TOL) -> dict:
    """Inspect one adapter and return a verdict dict."""
    fp = _find_tensor_file(path)
    if fp is None:
        return {"path": str(path), "ok": False, "error": "no adapter tensor file found "
                "(looked for adapter_model.safetensors/.bin/.pt)"}

    norms = None
    if fp.suffix == ".safetensors":
        norms = _read_safetensors_norms(fp)
    if norms is None:
        norms = _read_torch_norms(fp)
    if norms is None:
        return {"path": str(fp), "ok": False, "error": "could not read tensors (not "
                "safetensors, and torch unavailable for .bin/.pt)"}

    b = {n: v for n, (_d, v) in norms.items() if _is_b(n)}
    a = {n: v for n, (_d, v) in norms.items() if _is_a(n)}
    b_max = max(b.values()) if b else float("nan")
    b_mean = (sum(b.values()) / len(b)) if b else float("nan")
    a_max = max(a.values()) if a else float("nan")

    trained = bool(b) and b_max > tol
    verdict = {
        "path": str(fp),
        "sha256": _sha256(fp)[:16],
        "n_tensors": len(norms),
        "n_lora_B": len(b),
        "n_lora_A": len(a),
        "b_norm_max": b_max,
        "b_norm_mean": b_mean,
        "a_norm_max": a_max,
        "ok": trained,
    }
    if not b:
        verdict["ok"] = False
        verdict["error"] = "no LoRA B tensors present; this is not a LoRA adapter, or the "
        verdict["error"] += "keys are named unexpectedly."
    elif not trained:
        verdict["error"] = (f"LoRA B is at initialisation (max ||B||={b_max:.2e} <= "
                            f"{tol:.0e}): the adapter is inert and equals the base "
                            f"model. The trained weights never reached this checkpoint.")
    return verdict


def _fmt(v: dict) -> str:
    if v.get("error") and not v.get("n_lora_B"):
        return f"  ERROR  {v['path']}\n         {v['error']}"
    line = (f"  {'TRAINED' if v['ok'] else 'INERT  '}  {v['path']}\n"
            f"         sha256 {v['sha256']}  tensors {v['n_tensors']}  "
            f"B {v['n_lora_B']}  A {v['n_lora_A']}\n"
            f"         max||B|| {v['b_norm_max']:.3e}  mean||B|| {v['b_norm_mean']:.3e}  "
            f"max||A|| {v['a_norm_max']:.3e}")
    if v.get("error"):
        line += f"\n         {v['error']}"
    return line


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapters", nargs="+", help="adapter dir(s) or tensor file(s)")
    ap.add_argument("--tol", type=float, default=B_ZERO_TOL,
                    help=f"max ||B|| treated as at-initialisation (default {B_ZERO_TOL:.0e})")
    args = ap.parse_args(argv)

    verdicts = [summarize(Path(p), tol=args.tol) for p in args.adapters]
    print("adapter liveness check")
    for v in verdicts:
        print(_fmt(v))

    failures = [v for v in verdicts if not v.get("ok")]

    # Cross-arm identity check.
    hashes: dict[str, str] = {}
    identical = []
    for v in verdicts:
        h = v.get("sha256")
        if not h:
            continue
        if h in hashes:
            identical.append((hashes[h], v["path"]))
        else:
            hashes[h] = v["path"]
    for a_path, b_path in identical:
        print(f"  IDENTICAL  {a_path}\n             {b_path}\n"
              f"             two arms share a checkpoint; any contrast between them is "
              f"structurally zero.")

    ok = not failures and not identical
    print(f"\n{'PASS' if ok else 'FAIL'}: "
          f"{len(verdicts) - len(failures)}/{len(verdicts)} adapters trained"
          f"{'' if not identical else f', {len(identical)} identical pair(s)'}.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
