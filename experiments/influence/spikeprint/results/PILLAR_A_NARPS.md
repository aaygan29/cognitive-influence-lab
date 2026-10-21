# ⚠️ SUPERSEDED — this run did NOT test CSI

> **Correction (2026-06-08).** The score below labeled "CSI" was a *text heuristic* (`neurosignal`
> Buy/Sell on a text rendering of the gamble), **not** the Cognitive Sovereignty Index. CSI =
> **vmPFC − dlPFC activation** and requires fMRI (or a validated fMRI-predicting model). This run
> therefore tests a text heuristic, not CSI; it is retained only for the record. See
> `results/CORRECTION_2026-06-08.md`. The magnitude-blindness finding stands (relabeled).

# (original) NARPS ds001734 — text-heuristic probe + EV positive control

**Run:** 2026-06-08 · n_trials = 27454 · n_subjects = 108 · unique gambles = 416 ·
accept rate = 0.554
**Data:** OpenNeuro `ds001734` (CC0), behavioral events only (no fMRI). Bootstrap cluster unit =
subject. Manifest: `data/manifest.csv` (433 files, SHA-256).
**CSI:** neurosignal reference encoder, "Buy / Sell Signal", commit `e05bdb2`.
**CSI sd across trials = 0.0000** (≈ 0 ⇒ CSI is essentially constant across the gambles).

## Findings (decided under the registered Finding contract)
Decisions are standalone Findings (`Finding.decide`); the registered Benjamini-Hochberg family
rule is applied once H1b closes, and is identical here (an AUC pinned at 0.5 and a dAUC far below
0 do not flip under FDR).

- [UNVALIDATED] H1a: CSI = neurosignal Buy/Sell (reference encoder) -> accept: 0.500 (AUC) on NARPS ds001734 95%CI=[0.500, 0.500] n=27454 p=1 baseline=0.500
- [VALIDATED] Positive control: EV -> accept: 0.883 (AUC) on NARPS ds001734 95%CI=[0.857, 0.908] n=27454 p=0.0004998 baseline=0.500
- [UNVALIDATED] H2: does CSI beat EV?: -0.383 (dAUC(CSI-baseline)) on NARPS ds001734 95%CI=[-0.408, -0.357] n=27454 p=0.0004998 baseline=0.000

## Verdict (reported honestly, per the preregistered kill criteria)
**H1a FAILS — informatively.** The reference-encoder Buy/Sell metric is ~constant across NARPS
gambles (sd = 0.0000); it responds to affective *wording*, not numeric magnitude, so it
carries no information about which gamble was accepted (AUC = 0.500). The **expected-value
control predicts choices (AUC = 0.883)**, proving the data + pipeline (and the subject-
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
