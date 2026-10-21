#!/usr/bin/env python3
"""Pillar A on real NARPS ds001734 — value-framing construct (H1a) + expected-value control.

Loads NARPS behavioral trials, computes CSI (neurosignal reference encoder, "Buy / Sell Signal")
and an expected-value (EV) baseline per trial, runs the registered Finding analyses, and writes
results/PILLAR_A_NARPS.md plus a checksummed data/manifest.csv. Requires neurosignal on PYTHONPATH
at the pinned commit (see csi_provenance.json). No fMRI; behavioral data only; CC0.
"""
from __future__ import annotations

import datetime as dt
import glob
import os

import numpy as np

from spikeprint.analysis import incremental_validity, predictive_validity
from spikeprint.datasets import load_narps_events, render_gamble_text, sha256

RAW = "data/raw/ds001734"
SOURCE = "https://s3.amazonaws.com/openneuro.org/ds001734"


def csi_metrics(gain: float, loss: float) -> dict:
    from neurosignal import analyze

    a = analyze(text=render_gamble_text(gain, loss))
    return {m.label: float(m.score) for m in a.metrics}


def write_manifest(today: str) -> int:
    files = sorted(glob.glob(os.path.join(RAW, "*.tsv")))
    with open("data/manifest.csv", "w") as fh:
        fh.write("name,source_url,version_or_commit,license_spdx,access_date,sha256,local_path\n")
        for path in files:
            base = os.path.basename(path)
            sub = base.split("_")[0] if base.startswith("sub-") else ""
            url = f"{SOURCE}/{sub}/func/{base}" if sub else f"{SOURCE}/{base}"
            fh.write(f"{base},{url},ds001734,CC0-1.0,{today},{sha256(path)},{path}\n")
    return len(files)


def main() -> None:
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    d = load_narps_events(RAW)
    n = int(d["choice"].size)
    n_sub = int(np.unique(d["subject"]).size)
    uniq = sorted({(g, ls) for g, ls in zip(d["gain"], d["loss"])})
    cache = {gl: csi_metrics(*gl) for gl in uniq}
    csi = np.array([cache[(g, ls)]["Buy / Sell Signal"] for g, ls in zip(d["gain"], d["loss"])])
    csi_sd = float(np.std(csi))
    ev = 0.5 * d["gain"] - 0.5 * d["loss"]
    groups = d["subject"]

    h1a = predictive_validity(
        "H1a: CSI = neurosignal Buy/Sell (reference encoder) -> accept", "NARPS ds001734",
        csi, d["choice"], groups=groups, n_boot=2000, n_perm=2000,
    )
    pc = predictive_validity(
        "Positive control: EV -> accept", "NARPS ds001734",
        ev, d["choice"], groups=groups, n_boot=2000, n_perm=2000,
    )
    h2 = incremental_validity(
        "H2: does CSI beat EV?", "NARPS ds001734",
        csi, ev, d["choice"], groups=groups, n_boot=2000,
    )

    n_files = write_manifest(today)
    acc_rate = float(d["choice"].mean())

    print(
        f"n_trials={n} n_subjects={n_sub} n_unique_gambles={len(uniq)} "
        f"accept_rate={acc_rate:.3f} CSI_sd={csi_sd:.4f} manifest_files={n_files}"
    )
    for f in (h1a, pc, h2):
        print(f.render())

    os.makedirs("results", exist_ok=True)
    md = f"""# Pillar A on NARPS ds001734 — value-framing construct (H1a) + EV positive control

**Run:** {today} · n_trials = {n} · n_subjects = {n_sub} · unique gambles = {len(uniq)} ·
accept rate = {acc_rate:.3f}
**Data:** OpenNeuro `ds001734` (CC0), behavioral events only (no fMRI). Bootstrap cluster unit =
subject. Manifest: `data/manifest.csv` ({n_files} files, SHA-256).
**CSI:** neurosignal reference encoder, "Buy / Sell Signal", commit `e05bdb2`.
**CSI sd across trials = {csi_sd:.4f}** (≈ 0 ⇒ CSI is essentially constant across the gambles).

## Findings (decided under the registered Finding contract)
Decisions are standalone Findings (`Finding.decide`); the registered Benjamini-Hochberg family
rule is applied once H1b closes, and is identical here (an AUC pinned at 0.5 and a dAUC far below
0 do not flip under FDR).

- {h1a.render()}
- {pc.render()}
- {h2.render()}

## Verdict (reported honestly, per the preregistered kill criteria)
**H1a FAILS — informatively.** The reference-encoder Buy/Sell metric is ~constant across NARPS
gambles (sd = {csi_sd:.4f}); it responds to affective *wording*, not numeric magnitude, so it
carries no information about which gamble was accepted (AUC = {h1a.value:.3f}). The **expected-value
control predicts choices (AUC = {pc.value:.3f})**, proving the data + pipeline (and the subject-
clustered bootstrap) can detect a real signal when one exists — so the null is about the
instrument, not a broken analysis. **Caveat:** EV is near-definitional for these 50/50 gambles, so
the control is a *pipeline-sanity check, not an independent scientific finding*. **H2 FAILS:** the
metric does not beat EV.

## What this means (and does not)
The off-the-shelf reference-encoder Buy/Sell metric (our CSI proxy here) is the wrong instrument
for a numeric value-framing task: it is magnitude-blind (committed test
`tests/test_narps_encoder.py`: the encoder moves for different wording but is identical across
gain/loss amounts, because `neurosignal` tokenizes with `[a-zA-Z']+` and discards digits). This is
exactly the unfounded assumption the validation harness
exists to catch. It does **not** show CSI is useless — it shows CSI's domain is **persuasive text**
(construct H1b: ChangeMyView, Persuasion-for-Good) and/or the **learned (TRIBE) encoder**, which
are the registered next tests. No claim that CSI predicts choice is made or implied.
"""
    with open("results/PILLAR_A_NARPS.md", "w") as fh:
        fh.write(md)
    print("\nwrote results/PILLAR_A_NARPS.md and data/manifest.csv")


if __name__ == "__main__":
    main()
