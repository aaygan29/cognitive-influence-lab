# warden

A calibrated, abstaining **cognitive-security reasoning instrument**. WARDEN reasons over a *stated set of assumptions* about a person, an influence channel, and an adversary; it does **not** measure a real mind. Its value is making a threat model's implications explicit, falsifiable, and honestly bounded.

This is a reference skeleton (v0.1). Every result here is produced on a **simulator whose parameters are fixed to published effect sizes** — it is honest by construction and is **not evidence about any real person**.

## What it does

Three heads, each calibrated and abstaining:

- **H1 — Assay** (`h1_assay`): an *assumption-relative* manipulability **sensitivity** estimate (how far a modeled influence input shifts a modeled behavioral output), reported with a bootstrap CI that may include zero. Never a per-person exploitability score.
- **H2 — Detector** (`h2_detect`): a live-engagement detector reusing a calibrated fusion layer (SPRT, isotonic calibration, StratifiedGroupKFold, abstention gates). Reports **coverage** as the headline and a simulator-referenced PPV at the caller's prevalence prior. Field validity is treated as unmeasurable.
- **H3 — Floor** (`h3_floor`): an adaptive-minimax countermeasure **ordering** (`d* = argmin_d max_u M(d,u)`), with the adversary free to reallocate after seeing the defense. A simulated ordering, not a guarantee.

**Enforced scope:** `enforce_scope` refuses per-individual H1/H2 output in any asymmetric-power context (employment, custody, immigration, security clearance, interrogation). The refusal is enforced at the API boundary, not merely stated.

## Run it

```bash
pip install -r requirements.txt
PYTHONPATH=. python run_warden.py     # writes results/warden_results.json
```

## Layout

```
warden/            the instrument: assumptions (Theta + scope), influence simulator, the three heads
deceptkit/         vendored calibrated fusion layer that H2 reuses (SPRT, isotonic calibration, abstention)
run_warden.py      end-to-end driver
results/           simulated results
WARDEN_proposal_v2.md   the design, hardened against a six-persona red team
revision_log_warden.md  what the red team changed and why
```

## spikeprint (folded in 2026-09-15)

`spikeprint/` is the behavioral-validation arm: it asks whether a content-level Cognitive
Sovereignty Index (CSI = vmPFC minus dlPFC activation) predicts real, measured choice distortion,
under preregistered hypotheses with kill criteria, on public data only. It was folded in with
full commit history via `git subtree` from `aaygan29/spikeprint`, which remains the place to
develop it (pull updates with `git subtree pull --prefix=spikeprint spikeprint main`).

What is established there, and what is not:
- **Study 2 (complete, reproduced 2026-09-15).** On NARPS ds001734 behavior (27,454 trials,
  108 subjects, subject-grouped CV), a spiking LIF decision model matches prospect theory
  (AUC 0.892 vs 0.888; paired dAUC +0.0046 [+0.0025, +0.0066], practically negligible). The
  preregistered lower-energy criterion **fails** (about 13x the rate-MLP energy proxy at this
  2-feature scale) and a gradient-boosted ceiling beats prospect theory by +0.011 AUC.
- **Expected-value positive control** AUC 0.883 [0.858, 0.907]: a pipeline sanity check, since
  EV is near-definitional for 50/50 gambles.
- **No CSI result exists yet.** The early NARPS "CSI" null tested a text heuristic and is
  retracted (`spikeprint/results/CORRECTION_2026-06-08.md`). The real-CSI fMRI pipeline is built
  and unit-tested but not run.

It installs and tests independently: `cd spikeprint && pip install -e ".[dev,stats,neuro]" && pytest`
(35 tests; the encoder test needs `neurosignal` on the path). Its `.github/workflows/ci.yml`
only runs in the standalone repo.

## Grounding

- Design companion to Bagley's mathematical framework for the security of cognition (arXiv:2403.07945); situated against the neurosecurity (Denning, Matsuoka & Kohno) and neurorights literature.
- The simulator's effect sizes are pinned to published values (e.g. Falk 2010 neural increment; a single-trial detection effect on the deceptkit scale; covert-countermeasure detection collapse).
- `deceptkit/` is vendored so the repo runs standalone; it is a calibrated concealed-information-style fusion layer reused for H2.

## Scope and intent

WARDEN is a **defensive** analyst's tool. It is explicitly not a lie detector, manipulation scanner, screening tool, admissible instrument, or targeting product, and it refuses per-individual use in asymmetric-power settings by design. It is exploratory research on simulated data, not a validated or deployable system.

## License

MIT — see [LICENSE](LICENSE).
