# Evidence classes, separated with controls (reviewer action #1)

The reviewer's point 2: "the body-signal, behavioural and records evidence
should be presented separately" -- the two strongest decodes are from the terms
of the offer shown, the Kiva arm is ordinary credit scoring, and the one cheap
consumer sensor was null. This splits the arms into three classes and attaches
the control that identifies what each signal actually is.

Figure: [`figures/evidence_class_split.pdf`](../figures/evidence_class_split.pdf)
(`scripts/evidence_class_figure.py`). Script:
[`scripts/evidence_class_separation.py`](../scripts/evidence_class_separation.py).
Numbers: [`results/evidence_class_separation/summary.json`](../results/evidence_class_separation/summary.json)
(Kiva re-derived here; per-modality rows read from each arm's shipped
`run_record.json`). Balanced accuracy throughout.

## (a) Offer-terms / behavioural -- real, significant, but NOT a biosignal

| arm | balanced acc | majority | perm p | modality present |
| --- | --- | --- | --- | --- |
| delay discounting | 0.882 | 0.531 | <0.001 | `behavior` only |
| NARPS behaviour | 0.814 | 0.554 | <0.001 | `behavior` only |

Control: the only modality in these arms is `behavior` -- the terms of the offer
the subject saw (delay/amount; gain/loss). There is no body-signal channel, so
these strong decodes are economics / task structure, not a read of the body.

## (b) Records / credit-scoring -- the headline was a leak

Kiva (n=3,023 loans, target = repaid vs not, majority 0.723):

| feature set | balanced acc |
| --- | --- |
| **with leak-prone `nonpayment` column** | **0.959** |
| clean tabular (loan amount, sex, description length, sector) | 0.604 |
| demographics only (amount, sex) | 0.527 |

Control: adding the outcome-adjacent `nonpayment` column reproduces the ~0.95
the reviewer flagged; with it removed the arm sits at 0.604, **below** the 0.723
majority baseline (and the shipped run's permutation p = 1.0). This is ordinary,
weak credit scoring from loan records -- not a biosignal inference.

## (c) Biosignal -- cheap sensors null; only research neuroimaging carries signal

| arm | balanced acc | majority | perm p | reads as |
| --- | --- | --- | --- | --- |
| PPG fingertip pulse -> hypertension | 0.58 | 0.753 | 1.0 | null |
| PPG fingertip pulse -> diabetes | 0.542 | 0.826 | 1.0 | null |
| PPG fingertip pulse -> sex | 0.528 | 0.525 | 1.0 | null |
| DEAP peripheral (GSR/plethysmo/temp) | 0.501 | 0.567 | -- | null (per-modality) |
| DEAP lab EEG -> liking | 0.659 | 0.567 | <0.001 | real (behavior modality 0.516 = chance, so it is the EEG) |
| NARPS research fMRI -> choice | 0.809 | 0.583 | <0.001 | real (not a cheap sensor) |
| clinical EEG (ASZED) -> diagnosis | 0.705 raw | 0.78 | -- | **refuted** (confound audit: 0.542 residualized) |

Control: for DEAP the signal is isolated to the lab-EEG modality (0.659) while
the `behavior` modality is at chance (0.516) and the cheap `peripheral` channel
is at chance (0.501). So the one place a biosignal decodes above chance is
research-grade neuroimaging (DEAP 64-ch EEG, NARPS fMRI), not a cheap consumer
sensor. Every cheap-sensor arm (PPG pulse, DEAP peripheral) is null, and the one
clinical-EEG "protected attribute" decode is confounded
([`eeg_confound_audit.md`](eeg_confound_audit.md)).

## Verdict
Presented separately, the evidence supports a **narrower** thesis than "each
cheap channel already reads a real person":
- The strong, honest decodes (0.81-0.88) are **offer-terms** -- predicting choice
  from the offer shown, which is behavioural economics, not a body read.
- The **records** arm is modest credit scoring once de-leaked (0.60 < 0.72
  majority); the 0.96 was the `nonpayment` leak.
- **Cheap consumer biosignals are null** (PPG pulse, DEAP peripheral). Only
  research-grade neuroimaging (DEAP EEG, NARPS fMRI) decodes above chance, and
  the clinical-EEG attribute claim does not survive confound control.

The repo's safety argument ("govern the join; independence sets the ceiling")
stands on the offer-terms + records joins and on research neuroimaging. It is
**not** yet supported by cheap consumer body sensors, which is the honest scope
to carry into the writeup and figures.
