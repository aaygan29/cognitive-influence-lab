"""Provenance registry and checksum-guarded loading.

spikeprint redistributes **no** third-party data. Each dataset is documented in
``docs/DATA.md`` and pinned in ``data/manifest.csv`` with a SHA-256. Loaders intentionally
refuse to fabricate data: until a dataset is obtained, checksummed, and wired, its loader
raises :class:`NotImplementedError` pointing at the manifest. This keeps the repository honest
about what has and has not been validated.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import numpy as np


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source_url: str
    license: str  # SPDX identifier once confirmed, else "VERIFY"
    role: str
    notes: str = ""


# License = "VERIFY" until confirmed against the official source (see docs/DATA.md).
REGISTRY: Dict[str, DatasetSpec] = {
    "choices13k": DatasetSpec(
        "choices13k", "https://github.com/jcpeterson/choices13k", "NO-LICENSE-AT-SOURCE",
        "deferred (separate paper, pending author permission)",
    ),
    "narps_ds001734": DatasetSpec(
        "narps_ds001734", "https://openneuro.org/datasets/ds001734", "CC0-1.0",
        "PRIMARY confirmatory (H1/H2) + optional neural side",
    ),
    "ibl": DatasetSpec(
        "ibl", "https://www.internationalbrainlab.com/data", "VERIFY", "H1 prior-induced shift"
    ),
    "cmv_winning_args": DatasetSpec(
        "cmv_winning_args", "https://convokit.cornell.edu/documentation/winning.html", "VERIFY",
        "H1/H2/H4 text",
    ),
    "persuasion_for_good": DatasetSpec(
        "persuasion_for_good", "https://convokit.cornell.edu/documentation/persuasionforgood.html",
        "VERIFY", "H1/H4 donation $",
    ),
}


def sha256(path: Union[str, Path], chunk: int = 1 << 20) -> str:
    """Stream a SHA-256 of a file (constant memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Union[str, Path], expected_sha256: str) -> bool:
    """True iff the file's SHA-256 matches the manifest value."""
    return sha256(path) == expected_sha256


def load(name: str):
    """Refuse to load until the dataset is obtained and checksum-verified.

    spikeprint does not fabricate or redistribute data. Wiring a dataset means: obtain it from
    its official source under its own license, add a checksummed row to ``data/manifest.csv``,
    then implement its loader here.
    """
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; see docs/DATA.md")
    spec = REGISTRY[name]
    raise NotImplementedError(
        f"{name}: loader not yet wired. Obtain from {spec.source_url} under its own license, "
        f"add a checksummed row to data/manifest.csv, then implement the loader. "
        f"spikeprint does not fabricate or redistribute data (see docs/DATA.md)."
    )


def render_gamble_text(gain: float, loss: float) -> str:
    """Natural-language rendering of a 50/50 gain/loss gamble (the CSI text input for NARPS)."""
    return (
        f"You can win ${int(round(gain))} or lose ${int(round(loss))} on the flip of a coin. "
        f"Do you take the bet?"
    )


def load_narps_events(raw_dir: str = "data/raw/ds001734") -> Dict[str, np.ndarray]:
    """Load NARPS (ds001734) mixed-gambles behavioral trials from downloaded events.tsv files.

    Returns equal-length arrays: subject, run, gain, loss, choice (1=accept, 0=reject), group
    (equalRange/equalIndifference). Trials without a clear accept/reject (e.g. NoResp) are dropped.
    No fMRI is used. Source: OpenNeuro ds001734 (CC0).
    """
    part = os.path.join(raw_dir, "participants.tsv")
    group: Dict[str, str] = {}
    if os.path.exists(part):
        with open(part) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                group[row["participant_id"]] = row.get("group", "")
    cols: Dict[str, list] = {k: [] for k in ("subject", "run", "gain", "loss", "choice", "group")}
    for path in sorted(glob.glob(os.path.join(raw_dir, "sub-*_task-MGT_run-*_events.tsv"))):
        base = os.path.basename(path)
        sid = base.split("_")[0]
        run = base.split("run-")[1][:2]
        with open(path) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                resp = row.get("participant_response", "")
                if "accept" in resp:
                    choice = 1
                elif "reject" in resp:
                    choice = 0
                else:
                    continue
                try:
                    gain = float(row["gain"])
                    loss = float(row["loss"])
                except (KeyError, ValueError, TypeError):
                    continue
                cols["subject"].append(sid)
                cols["run"].append(run)
                cols["gain"].append(gain)
                cols["loss"].append(loss)
                cols["choice"].append(choice)
                cols["group"].append(group.get(sid, ""))
    return {
        "subject": np.array(cols["subject"]),
        "run": np.array(cols["run"]),
        "gain": np.array(cols["gain"], dtype=float),
        "loss": np.array(cols["loss"], dtype=float),
        "choice": np.array(cols["choice"], dtype=int),
        "group": np.array(cols["group"]),
    }
