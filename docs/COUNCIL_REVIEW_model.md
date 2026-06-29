# Council Review - MODEL.md (formal belief-dynamics model)

Mode: manuscript + artifact (src/belief_model.py, tests/test_belief_model.py, 11/11 pass).

## Verdict: ACCEPT WITH MINOR REVISIONS

The model is internally consistent, correctly implemented, and honestly scoped as an assembly of existing
components. All four predictions are reproduced by the code and locked by tests. Two calibration fixes are
required before it reads correctly to a reviewer, and both are edits, not new work.

## Claim Ledger
| # | Claim | Asserted tier | Actual tier | Status |
|---|---|---|---|---|
| C1 | The cusp dynamics produce bistability, hysteresis, foot-in-the-door, and alpha^1.5 robustness scaling | T1 (illustrative, by construction) | verified in code (hysteresis area 2.40; seq +0.93 vs single -0.75; exponent 1.500; chi 0.50 -> 19.2) | PASS as analytic consequence |
| C2 | These are "falsifiable predictions" | implies T-empirical | they are predictions FOR experiment, not yet confirmed BY experiment | needs relabel |
| C3 | Each equation is attributed to its literature | T-citation | mostly correct; the attitude-cusp specifically is van der Maas et al. 2003, not cited | needs citation |
| C4 | Novelty is assembly only | T-honest | correct, stated plainly in Section 7 | PASS |

## Gate Ladder
- Gate 0 Provenance .... N/A (theory artifact, no data). Stated. The hierarchical model (Eq 13) is a
  specification, not yet fit to data, and is labeled as such.
- Gate 1 Variance ...... N/A for analytic results; the code is deterministic and tested.
- Gate 2 Spec robustness PASS. Predictions hold across the parameter settings in the tests; the
  alpha^1.5 law is exact, not knife-edge.
- Gate 3 Specificity ... PARTIAL. Hysteresis and foot-in-the-door follow from the cusp + bounded
  confidence by construction, so they confirm the implementation, not the model's empirical truth.
- Gate 4 Confounds ..... N/A (no empirical estimate yet).
- Gate 5 Necessity ..... N/A (theory).
- Gate 6 Calibration ... FAIL on C2 wording. "Falsifiable predictions" is correct in spirit (they ARE
  falsifiable), but the prose must not imply they are already empirically confirmed; the code confirms
  the math, not the world.

## Council Objections
- **Elrond:** The alpha^1.5 scaling and susceptibility divergence are exact analytic facts of the cusp,
  correctly recovered. No statistical objection; just do not present analytic identities as empirical
  wins. [Minor]
- **Gandalf:** Deterministic model, tests pin it. No seed issue. [no objection]
- **Aragorn:** No data, so no provenance risk, but Eq 13 is a promise; mark it explicitly as not-yet-fit.
  [Minor]
- **Galadriel:** The persuasive-force decomposition (Eq 3) is where the construct lives; B(m)'s role is a
  hypothesis inherited from the instrument, correctly flagged. [no objection]
- **Boromir:** "Falsifiable predictions" risks being read as "demonstrated effects." Relabel as
  predictions to be tested, with the code showing the model is correctly implemented. [Major, the fix]
- **Gimli:** Strip the neural term B(m) and the model is a standard cusp-plus-bounded-confidence opinion
  model. The novel content is exactly the B(m) coupling and the robustness functionals tied to it; say so.
  Section 7 already does. [Minor]
- **Legolas:** The attitude-cusp has specific prior art: van der Maas, Kolstein, and van der Pligt (2003),
  Sudden transitions in attitudes, Sociological Methods and Research. Crano and Prislin 2005 is a general
  review, not the cusp source. Add the specific citation so the lineage is honest. [Major, citation]

## Severity-ranked findings
1. [Major] Add van der Maas et al. 2003 as the specific attitude-cusp prior art (Section 1 and refs).
2. [Major] Relabel the four results as predictions to be tested empirically; the code verifies correct
   implementation and internal consistency, not empirical truth.
3. [Minor] Mark Eq 13 explicitly as a specification not yet fit to data.

## What would change the verdict
Nothing blocks ACCEPT as a theory-synthesis paper; the two Major edits are wording and one citation. The
empirical step (fitting Eq 13 to the public corpora and testing prediction 1 with a reversal design) is
the future-work that would turn the predictions from analytic to confirmed.

## Evidence log
- Predictions recomputed: hysteresis loop area 2.395; sequential b_final 0.929 vs single -0.746;
  beta_crit exponent 1.500; susceptibility 0.500 -> 19.201 toward the fold. All from src/belief_model.py,
  locked in tests/test_belief_model.py (11/11 pass).
- Fold-set constant corrected to 4 alpha^3 = 27 beta^2 (beta_crit = sqrt(4 alpha^3/27)).
