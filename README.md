# spikeprint

**A behaviorally-validated, energy-efficient instrument for measuring choice distortion from content.**

`spikeprint` tests one falsifiable question: **does a content-level Cognitive Sovereignty
Index (CSI) predict real, measured choice distortion?** It does *not* claim to detect
"manipulation" (a normative concept) or to read anyone's brain — see
[`PREREGISTRATION.md`](PREREGISTRATION.md), §1.

- **Ground truth:** observed choice shifts in public behavioral datasets (gambles, donations,
  opinion change).
- **Instrument under test:** CSI, computed from the *stimulus* by
  [`neurosignal`](../../../Sapient/neurosignal).
- **Mechanistic arm:** a spiking decision model (snnTorch) reporting accuracy **and**
  energy/decision.
- **Discipline:** preregistered hypotheses with **kill criteria**, secondary public data only,
  every major claim cited (APA), honest negative results.

> Status: **correction in progress.** The first NARPS run mislabeled a text heuristic as "CSI";
> CSI = **vmPFC − dlPFC** (fMRI-based) and that run is **superseded**
> (`results/CORRECTION_2026-06-08.md`). Real CSI is now being computed from the NARPS fMRI. The
> heuristic's magnitude-blindness and all pipeline machinery stand.

## Why this exists
Manipulation/persuasion detection today is split between ungrounded text classifiers and
neural claims that are never checked against behavior. `spikeprint` grounds a content metric in
**validated prediction of real choices**, with the validation — not the score — as the
deliverable.

## What it is *not*
A mind-reader, a manipulation detector, a clinical/diagnostic tool, or anything that infers an
individual's neural state. It estimates population-level choice-distortion likelihood from a
fixed stimulus. (Full non-claims: `PREREGISTRATION.md` §1, §9.)

## Install (dev)
```bash
# recommended (Python 3.10+):
uv venv --python 3.12 && uv pip install -e ".[dev]" && .venv/bin/pytest
# or with pip:
python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" && pytest
```
`spikeprint` depends on `neurosignal` (CSI engine), **pinned to commit `e05bdb2`** in
`csi_provenance.json`; install it editable from that repo/commit.

## Repository layout
```
spikeprint/
├── PREREGISTRATION.md     # locked hypotheses + kill criteria (the keystone)
├── REFERENCES.md          # APA citations + verification status
├── docs/
│   ├── DATA.md            # datasheet (Gebru et al., 2021): provenance/license per dataset
│   ├── SYNTHETIC.md       # LLM-respondent calibration gate (planned)
│   ├── EXCLUSIONS.md       # exclusion rules, fixed pre-analysis (planned)
│   └── ETHICS.md          # secondary-data / exemption rationale (planned)
├── spikeprint/
│   ├── validate.py        # Finding contract + CI gate (every result carries its evidence)
│   └── datasets.py        # provenance registry + checksum-guarded loaders
├── data/                  # raw third-party data (git-ignored); manifest.csv committed
├── tests/
└── .github/workflows/ci.yml
```

## The validation contract
Every reported number is a `Finding`, never a bare scalar:
```python
Finding(value, dataset, baseline, effect_size, ci95, n, passed)
```
A `Finding` that has not cleared its preregistered test is `passed=False` and is rendered as
`UNVALIDATED`. The registered confirmatory rule is `decide_family` (Benjamini-Hochberg FDR across
the H1–H4 family); `gate()` raises on any unvalidated claim *(active once analyses exist — the
scaffold ships the contract + tests only)*. See `spikeprint/validate.py`.

## Roadmap
1. ✅ Scaffold + preregistration + datasheet + references.
2. ✅ `data/manifest.csv` (SHA-256) + NARPS ds001734 loader (CC0; 433 files).
3. ✅ H1a + EV control + H2 on **NARPS** → honest null for CSI; EV control AUC 0.883 (`results/`).
4. ☐ H1b: CSI on persuasive text (ChangeMyView, Persuasion-for-Good) — CSI's actual domain.
5. ☐ Spiking decision model + energy benchmark vs. rate/Transformer controls.
6. ☐ Learned (TRIBE) encoder for CSI; synthetic-respondent calibration gate.

## License
Code: MIT (`LICENSE`). Data: each dataset retains its own license (`docs/DATA.md`).

## Citation
See `CITATION.cff`.
