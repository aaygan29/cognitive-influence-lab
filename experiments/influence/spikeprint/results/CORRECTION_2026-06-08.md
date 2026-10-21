# Correction — 2026-06-08: the NARPS "CSI" run did not test CSI

## What was wrong
`results/PILLAR_A_NARPS.md` (commit `9f4143c`) reported an "H1a" result on a score it labeled
"CSI." That score was the `neurosignal` reference encoder's "Buy / Sell Signal" computed on a
*text rendering* of each gamble. **That is not the Cognitive Sovereignty Index.**

CSI is defined (by the author) as **vmPFC activation − dlPFC activation** — a contrast computed
from **fMRI**, or from a model validated to predict those regions' activation. A text heuristic
neither uses fMRI nor produces vmPFC/dlPFC activations, so it cannot be CSI. The reported "H1a
null" is therefore a null about a *text heuristic*, not about CSI.

## Root cause
The preregistration originally framed CSI as "a score computed from the content/stimulus," which
baked in the error. CSI is a *neural* measure; that framing is corrected in `PREREGISTRATION.md`
(§1, §6).

## Corrected plan
Validate **real CSI** on NARPS using the dataset's **fMRI** (`ds001734` is a task-fMRI dataset):
per subject, estimate vmPFC and dlPFC activation (ROI betas), compute CSI = vmPFC − dlPFC, then
test its relationship to the choices already loaded. Output: `results/PILLAR_A_NARPS_CSI_fmri.md`.

## What is retracted, and what stands
- **Retracted:** any claim that the heuristic run validated or invalidated CSI.
- **Stands (relabeled):** the `neurosignal` reference encoder is magnitude-blind on gambles; and
  the analysis pipeline, validation contract, leakage-safe splits, and the expected-value positive
  control are all sound and reused unchanged.
