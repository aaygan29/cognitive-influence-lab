# Reviewer report -- response and action list (2026-10)

External reviewer report on `behavioral_decoding` (PDF:
`~/Desktop/Delete Later/papers/Review of Biosignals and Behavioural Prediction.pdf`). The reviewer
installed the package, ran the suite, and raised four load-bearing points. This
file records the response so the critique is integrated into the work rather
than lost in a download folder.

## What the reviewer endorsed
- Engineering/evaluation: person-level splits, in-fold resampling enforced by
  tests, balanced accuracy throughout, per-run records. Suite ran cleanly apart
  from one test needing an optional imaging library.
- Honest reporting: nulls reported (fingertip pulse, the chaining experiment),
  the broken permutation test explained not hidden, the leaky Kiva column (that
  inflated accuracy to 0.95) caught and removed.
- "Govern the join, not the sensor" and the independence-sets-the-ceiling
  projection flagged as a genuinely useful governance idea.

## Actions to take (each is a concrete repo change)

1. **Separate the three evidence classes in the writeup and figures.**
   The reviewer's central point: the two strongest "results" (bids 0.882,
   gambles 0.814) predict choice from the *terms of the offer shown* (delay,
   gain/loss), and the Kiva arm (loan amount, gender, description length) is
   ordinary credit scoring -- neither is a body-signal claim. The only cheap
   consumer sensor tested (fingertip pulse) was null; clinical hospital EEG is
   the one real biosignal result.
   -> Action: restructure results into (a) offer-terms/behavioural, (b)
   records/credit-scoring, (c) biosignal (EEG), and drop any framing that says
   "each cheap channel already reads a real person." **DONE** -- see
   `docs/evidence_class_separation.md` (`scripts/evidence_class_separation.py`).
   Controlled result: offer-terms arms (delay 0.882, NARPS 0.814) are the
   `behavior` modality only; Kiva clean = 0.604 < 0.723 majority (the 0.959 was
   the `nonpayment` leak, reproduced); cheap sensors null (PPG 0.53-0.58, DEAP
   peripheral 0.501); only research neuroimaging decodes (DEAP EEG 0.659, NARPS
   fMRI 0.809), and clinical EEG is refuted. Figures still to be redrawn to this
   split.

2. **EEG diagnosis result needs a confound check.** (reviewer point 3) -- **DONE.**
   Ran the confound audit (`scripts/eeg_confound_audit.py`,
   `docs/eeg_confound_audit.md`): site/device are near-constant and not the
   confound, but patients are ~10 yr older with longer recordings. Using Snoek
   et al. (2019) controls, a confound-only baseline (age+sex+duration) reaches
   0.723 balanced accuracy vs the pipeline's 0.705, and per-fold confound
   regression collapses the EEG decode to 0.542 (chance). **Verdict: refuted** at
   this cohort. README part 3 and `docs/eeg_neuroprivacy.md` updated.

3. **Keep the governance framing but scope it to the supported evidence.**
   -> Action: in `README.md`/`docs/design.md`, state that
   "govern the join" is demonstrated on the offer-terms + records joins; the
   biosignal ceiling claim rests on the single EEG arm pending (2). **DONE** --
   README §3 now scopes the claim: the fusion/independence result is a synthetic
   positive control, the real signal is in the offer-terms + records joins and
   research neuroimaging, and the cheap-consumer-hardware risk is a *projection*
   under the independence assumption (every cheap-sensor arm was null; clinical
   EEG refuted), not a measured leak.

4. **Agent-contamination of online cohorts** (cross-reference the AGEA entry in
   `docs/literature.md`): add an agent-plausibility screen / sensitivity analysis
   to the aggregate arm. **DONE** -- see `docs/agent_contamination_sensitivity.md`
   (`scripts/agent_contamination_sensitivity.py`). Treating the agent fraction `f`
   as a nuisance parameter on the Kiva aggregate arm, balanced accuracy degrades
   ~linearly (0.605 clean -> 0.550 at f=0.5 -> 0.528 at f=0.7); the lift over
   chance halves at f~0.5. Conclusion: report aggregate numbers with a
   contamination band, not a point estimate.
