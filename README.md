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

## Grounding

- Design companion to Bagley's mathematical framework for the security of cognition (arXiv:2403.07945); situated against the neurosecurity (Denning, Matsuoka & Kohno) and neurorights literature.
- The simulator's effect sizes are pinned to published values (e.g. Falk 2010 neural increment; a single-trial detection effect on the deceptkit scale; covert-countermeasure detection collapse).
- `deceptkit/` is vendored so the repo runs standalone; it is a calibrated concealed-information-style fusion layer reused for H2.

## Verification side-studies

- `verification/cognitive_virus/`: Lean 4 proofs plus numerical and agent-based checks of the U/C/D mean-field model in Sole et al., *Large-Language Models as a Cognitive Virus* (arXiv:2609.03344). Includes a list of what is proven, what was checked numerically, and small discrepancies found.

## Scope and intent

WARDEN is a **defensive** analyst's tool. It is explicitly not a lie detector, manipulation scanner, screening tool, admissible instrument, or targeting product, and it refuses per-individual use in asymmetric-power settings by design. It is exploratory research on simulated data, not a validated or deployable system.

## License

MIT — see [LICENSE](LICENSE).
