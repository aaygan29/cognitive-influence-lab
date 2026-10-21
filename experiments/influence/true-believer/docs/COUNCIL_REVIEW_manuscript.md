# Council Review - cultist manuscript (draft 1)

Mode: manuscript + artifact. Every headline number cross-checked against results/synthetic_validation.json.

## Verdict: MAJOR REVISION

The apparatus is sound and the numbers in the prose match the results file exactly. The work fails one
gate, and it is a calibration gate, not a data gate: the title and abstract claim "a demonstration and
quantification of risk," but the artifact demonstrates only that the measurement apparatus behaves
correctly on a synthetic stand-in. No real human belief data has been touched. The science is honest in
the body (Section 3.1 explicitly says "these are not findings about human belief"), but the framing at the
top outruns it. Fix the framing and this is an ACCEPT as a registered-apparatus + pre-registration paper.

## Claim Ledger
| # | Claim | Asserted tier | Actual tier | Gate status |
|---|---|---|---|---|
| C1 | The instrument detects a neural belief signal when present, null when content-only, chance when scrambled | T4 (controlled, multi-condition, CIs) | T4 verified on synthetic ground truth | PASS |
| C2 | Headline numbers (r, dAUC, AUC) as stated | T3/T4 | matches JSON to 3 decimals | PASS |
| C3 | "A demonstration and quantification of [real] risk" (title/abstract) | implies T4 on real data | T1 on real data (none run) | Gate 6 FAIL |
| C4 | Encoder-swap stability | T4 | verified (reference + 2 perturbed) but perturbed encoders are not independent substrates | PARTIAL |

## Gate Ladder
- Gate 0 Provenance ...... N/A-by-design, loudly labeled. Data is a synthetic stand-in; the manuscript
  says so repeatedly. This is legitimate for an apparatus-validation paper, but C3's framing must not let
  a reader infer a real-data result. The chance baseline (scrambled control, AUC 0.497 [0.466,0.527]) is
  established. PASS as apparatus validation; the real run has no Gate 0 yet.
- Gate 1 Variance ........ PASS. Bootstrap CIs on every estimate; repeated stratified CV (5x10) for AUC.
  Single data seed though (seed=7); see Gandalf.
- Gate 2 Spec robustness . PARTIAL. One generator configuration (n, noise, weights). Not yet swept.
- Gate 3 Specificity ..... PASS. Scrambled-B control isolates the signature: content baseline intact,
  B permuted, everything to chance. This is the right specificity test.
- Gate 4 Confounds ....... PASS for the apparatus. The content-only baseline is the confound control:
  B must beat content, and in the content-only world it correctly does not (dAUC -0.001).
- Gate 5 Necessity ....... PASS on synthetic. The positive/negative/scramble triad shows B is necessary
  for the increment (remove or scramble it, increment vanishes).
- Gate 6 Calibration ..... FAIL on the title/abstract (C3). PASS in the body.

## Council Objections
- **Elrond (stats):** CIs and repeated CV are present and correct. Single generator seed is the one gap;
  report stability across seeds. [Minor]
- **Gandalf (replication):** Every number is from seed=7. Run 10+ data seeds and report the
  detect/null/chance verdict rate, so the controls are not themselves a single draw. [Major]
- **Aragorn (provenance):** Data is synthetic and labeled as such. No leakage (the response latent is
  explicitly withheld from the baseline). The real-data Gate 0 does not exist yet; do not let the
  abstract imply it does. [Major, ties to C3]
- **Galadriel (construct):** B is built from a heuristic encoder, so "neural" is by-construction here.
  The construct claim is only cashed when a learned encoder replaces the reference one. The manuscript
  says this; keep it prominent. [Major]
- **Boromir (overclaim):** The title "a demonstration and quantification of risk" reads as though the
  risk has been demonstrated in humans. It has not. Retitle to name the apparatus and the
  pre-registration. This is the single highest-leverage fix. [Blocker for framing]
- **Gimli (single-instrument):** Strip the synthetic generator and nothing real remains yet, which is
  fine for an apparatus paper but must be stated as the scope. The perturbed encoders are not independent
  substrates; call H-encoder a within-family robustness check, not cross-encoder replication, until a
  genuinely different encoder is wired. [Major]
- **Legolas (literature):** The cited works support the sentences they are attached to (Falk MPFC,
  Genevsky/Knutson NAcc, Kaplan/Kossowska resistance, Hackenburg/Bai behavioral persuasion). The novelty
  claim (no one has joined the two literatures in a closed measurement) is defensible. [no objection]

## Severity-ranked findings
1. [Blocker, framing] Title/abstract imply a real-data risk demonstration. Retitle and rewrite the
   abstract to: a validated instrument + pre-registration; the risk demonstration is the registered next
   step, not a completed result.
2. [Major] Single data seed. Add a multi-seed control-stability report.
3. [Major] Call H-encoder a within-family robustness check until an independent encoder is wired.
4. [Minor] Content-only H-load r=0.144 fails on the magnitude threshold (min_r=0.15), not because the CI
   includes zero; state this precisely.

## What would change the verdict
Retitle to an apparatus + pre-registration paper and add a 10-seed control-stability table. Both are
edits, not new experiments, and they move this to ACCEPT for a registered-report or methods venue.

## Evidence log
- All numbers verified against results/synthetic_validation.json (positive: r 0.356 [0.312,0.400], dAUC
  0.0669 [0.0622,0.0715], AUC_B 0.628; content-only: dAUC -0.0008, r 0.144; scrambled: AUC_B 0.497
  [0.466,0.527], r 0.025). Manuscript Section 3.1 matches.
- Leakage check: src/baseline.py uses CONTENT_FEATURES (6); response_latent is encoder-only
  (src/encoders.py ENCODER_FEATURES). No baseline access to the withheld latent.
