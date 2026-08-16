# WARDEN: implementation status, audited

This table exists because Reviewers #1 and #2 both independently flagged the same problem: the paper reads
as a description of an instrument without making clear which parts are running code, which parts are
validated findings, and which parts are still design proposals. This document is the fix: one row per
component, each with a status tag, the exact file and evidence for that tag, and, where relevant, what the
status explicitly does not mean.

**Status legend.**
- **CODE-VERIFIED**: a real, runnable implementation exists and was executed to produce the cited output.
  It runs on simulated data with parameters pinned to published effect sizes. It is not a validated
  real-world instrument.
- **PARTIALLY IMPLEMENTED**: a real implementation exists but omits a behavior the design proposal commits
  to. The gap is stated explicitly.
- **DESIGN PROPOSAL ONLY**: named or described in the paper. No code exists. Nothing has been run.
- **VALIDATED RESULT (real data)**: a statistical finding computed on real (non-simulated) data, with its
  correction method and power check stated.
- **CONCEPTUAL / ASPIRATIONAL**: a framing or claim (e.g., "H1/H2/H3 as validated instruments on real human
  data") that is not true today and is not implied to be true by any other row in this table.

---

## The WARDEN codebase (`/tmp/warden_check/`)

| Component | File | Status | Evidence | What this does NOT mean |
|---|---|---|---|---|
| Assumption ledger `Theta`, `enforce_scope`, `ScopeRefusal` | `warden/assumptions.py` | **CODE-VERIFIED** | Every WARDEN result carries a serializable `Theta`; `results/warden_results.json` includes a `C8_scope_enforcement` entry showing the refusal firing on `context="employment"` | Does not mean the refusal is enforced on all three heads (see H3 row below) |
| H1 sensitivity assay | `warden/heads.py::h1_assay` | **CODE-VERIFIED** | `results/warden_results.json`, block `H1_brain_beats_behavior`: r2_behavior_only=0.1338, r2_full=0.3200, delta=0.1863, CI [0.1289, 0.2279] | Runs on simulated data pinned to Falk 2010's effect size. Not a measurement of any real person's manipulability |
| H1 operator-class fragility check | `run_warden.py` lines 29-38 | **CODE-VERIFIED** | `results/warden_results.json`, block `H1_operator_class_fragility`: transfer R-squared = -0.2155 | Confirms non-transfer in this simulator's parameterization; not a general theorem about all possible operator-class pairs |
| H1 calibration-quality ABSTAIN gate ("if any head cannot be calibrated, it ships ABSTAIN-only," base proposal Sec. 7) | not present in `h1_assay` | **DESIGN PROPOSAL ONLY** | Reviewed `warden/heads.py::h1_assay`: no calibration-quality check, no ABSTAIN return path | The base proposal's global honesty check for H1 is not yet enforced in code |
| H1b bounded state-moderator sweep | `warden/heads.py::h1_state_sweep` | **CODE-VERIFIED** | `results/warden_results.json`, block `H1_state_moderator_sweep`: S ranges 0.353 to 0.727 over a bounded state axis | A bounded sweep over a simulated moderator, not a calibrated real drug/arousal coefficient (the code's own docstring states this) |
| H2 live-engagement detector, three-gate ABSTAIN logic | `deceptkit/fusion.py::MultivariateFusion.decide`, wired via `warden/heads.py::h2_detect` | **CODE-VERIFIED** | `results/warden_results.json`, blocks `H2_naturalistic` (coverage 0.354, simulator-PPV at 1% prior 0.023) and `H2_countermeasure_collapse` (coverage falls to 0.272 under covert countermeasure) | Field sensitivity/specificity are explicitly declared unmeasurable by the code's own output note; simulator-PPV is not a field PPV |
| H2 base-rate honest simulator-PPV reweighting | `warden/heads.py::_ppv_at_prevalence` | **CODE-VERIFIED** | Same block as above | Reweights the simulator's own TPR/FPR; does not create a field-validated PPV out of simulated inputs |
| H2 oracle-hardening / boundary-hugging meta-detector (base proposal C6: residual leakage when the adversary observes WARDEN's output) | not present in `deceptkit/fusion.py` or `warden/heads.py` | **DESIGN PROPOSAL ONLY** | `Theta.adversary_observes_output` exists as a field but is not read by any decision logic in the reviewed code | The oracle-leakage mitigation described in the base proposal (Sec. 4, H2 correction 4) is not implemented; today, setting this flag has no computational effect |
| H3 adaptive minimax ordering over singleton/pairwise defenses | `warden/heads.py::h3_floor` | **CODE-VERIFIED** | `results/warden_results.json`, block `H3_minimax_floor`: d* blocks {z_choice, z_rt}, worst-case manipulability 0.4111 vs. no-defense 0.4183 | Searches only singletons and pairs (29 of 128 possible subsets for a 7-channel set), not the full power set; not an exhaustive minimax |
| H3 scope enforcement (`enforce_scope` call) | `warden/heads.py::h3_floor` | **PARTIALLY IMPLEMENTED (gap)** | Reviewed `h3_floor`: no call to `enforce_scope(theta)` anywhere in the function body | H1 and H2 refuse per-individual output in adverse contexts; H3, as shipped, does not. This is a real gap, not a design choice, and is the top priority fix |
| H3 fragility coupling with suppressed H2 signal (base proposal C14: widen uncertainty or ABSTAIN when input is suppressed) | not present in `h3_floor` | **DESIGN PROPOSAL ONLY** | Reviewed `h3_floor`: always returns a numeric `d_star` and full ordering table; no suppressed-signal branch, no ABSTAIN return | The coupling described in Sec. 4 (H3 correction 2) of the base proposal is not yet coded. As shipped, H3 will compute and report a floor even when the underlying signal is actively suppressed |
| SPRT-style channel fusion (log-odds accumulation, three-region decision) | `deceptkit/fusion.py::BayesianFusion`, `MultivariateFusion` | **CODE-VERIFIED** | Formal match given in `REVISED_MANUSCRIPT_SECTIONS.md` Section A.2, checked against the source | This is a single-epoch, multi-channel instantiation of the Wald three-region rule. It is not a multi-epoch sequential test (see next row) |
| True multi-epoch sequential accumulation (evidence carried epoch-to-epoch within a session) | not present anywhere in the codebase | **DESIGN PROPOSAL ONLY** | No state is carried between calls to `decide()`; each row/epoch is scored independently | "Sequential-testing machinery" in the original paper's prose should not be read as implying this exists yet |
| Vendored `deceptkit` fusion layer, isolated so WARDEN runs standalone | `deceptkit/` (full package) | **CODE-VERIFIED** | `run_warden.py` imports and runs it end to end; `results/warden_results.json` is its output | This is a research prototype (deceptkit's own stated status), not a fielded polygraph-replacement product |
| End-to-end driver | `run_warden.py` | **CODE-VERIFIED** | Executes all of the above and writes `results/warden_results.json`; the file in this repository is that literal output | Every number quoted anywhere in the revised manuscript sections that is attributed to WARDEN's own output is this file, not a hand-constructed example |

---

## Claims about real human data (the part reviewers most needed disambiguated)

| Claim | Status | Evidence | What this does NOT mean |
|---|---|---|---|
| H1, H2, H3 as calibrated, validated instruments on real human physiological/behavioral data | **CONCEPTUAL / ASPIRATIONAL. Not true today.** | No component in the table above has been run on real human neural, behavioral, or physiological data in an influence-detection setting; no such paired dataset exists (base proposal Sec. 6, "no physiological ground truth," is unchanged by this revision) | This is the single most important line in this document. Nothing in this package should be read, quoted, or cited as claiming WARDEN has been validated on real people |
| Falk 2010 neural-behavior increment (~20-25% of variance), used to pin the H1 simulator | **VALIDATED RESULT (real data), pre-existing literature** | Falk et al. 2010, doi:10.1523/JNEUROSCI.0063-10.2010, cited in the base proposal | Confirms the simulator's parameter is grounded in a real, published effect size; does not mean WARDEN's H1 has itself been run on that or any other real dataset |
| Motion/hand telemetry re-identification (Nair 2023, Miller 2020, Liebers 2021) | **VALIDATED RESULT (real data), independent published literature, not WARDEN output** | See Section (d) of `REVISED_MANUSCRIPT_SECTIONS.md` for full citations and numbers | These are established facts about XR hardware and third-party re-identification research; they are not outputs of the WARDEN or deceptkit codebases and are not claims this program is making about its own instrument |
| Structural connectivity identification is cheap and near-ceiling | **VALIDATED RESULT (real data)** | Committed N=248 baseline (`wiring-not-weights/results/exp05_abide_results.json`): weighted 97.98% vs. binarized 97.17%, gap 0.81 pt; TOST equivalence at a 5% margin, 95% CI on the gap [-1.9%, +3.5%] (`RIGOR_FINDINGS.md` F2). Independently replicated fresh N=75 (`identification_rigor_results.json`): both arms hit ceiling (100%/100%), and the RMT null is rejected there (top eigenvalue 22.25 vs. random-matrix null mean 1.12, empirical p about 0.005) | The N=75 replication's OWN TOST (CI about +/-0.0002%) and its "0.0 pt" node-strength-regression drop are ceiling artifacts (both arms saturate at 1.0) and are not cited as the equivalence evidence; the substantive equivalence claim rests on the committed N=248 CI. Confirms a structural, not neural-individuation, finding; do not conflate with the individuation claims below |
| Digital-brain stimulus-evoked individuation (N=4, 25 ROIs) | **VALIDATED RESULT (real data), corrected: null, underpowered** | `digitalbrain_geometry_results.json`: 0/25 BH-FDR, 0/25 Bonferroni after correction; MDES for 80% power = 0.832 accuracy | Does NOT mean the channel is proven safe. Means the study could only have detected a near-perfect individuation effect and did not find even that; a moderate true effect would not have been detectable by this design |
| Functional-axis reconstruction probe, exp04 (N=8, 25 ROIs) | **VALIDATED RESULT (real data), corrected: null, underpowered** | `RIGOR_FINDINGS.md` F1, `MATH_FRAMEWORK.md` Sec. 7: 0/25 significant, MDES approximately 0.59 accuracy, observed 0.175 | Same caveat as above. "Reconstruction is dear" (the original framing) is retired in this revision in favor of "reconstruction is currently underdetermined at achievable N" |
| Bagley's mathematical framework for the security of cognition (arXiv:2403.07945) | **Pre-existing literature, cited, not a claim of this program** | Cited throughout the base proposal as the theoretical companion WARDEN operationalizes | WARDEN is described as an empirical companion to this framework, not a proof or extension of its mathematics |

---

## Summary judgment for the reviewers

Everything under "The WARDEN codebase" above ran, on this machine, on simulated data, and produced the
numbers quoted throughout `REVISED_MANUSCRIPT_SECTIONS.md`. Nothing under "Claims about real human data,"
except the three rows explicitly marked VALIDATED RESULT (real data) with independent third-party or this
program's own corrected re-analysis, should be read as a claim about real people. The two gaps found while
preparing this revision (H3's missing scope enforcement, H3's missing fragility-coupling ABSTAIN path) are
disclosed here rather than silently fixed and left undocumented, because the reviewers' complaint was
specifically about not being able to tell implemented from aspirational, and the right response to that
complaint is maximal legibility, not a cleaner-looking but less honest table.
