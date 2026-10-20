# individuation-mcmc

**Can we infer a person's decision parameters and their learning rule from behavior, well enough that the per-person fit is (a) recoverable, (b) better than a group model out-of-sample, and (c) a handle on how that person would be steered?**

Simulation only, synthetic subjects with known ground truth. This is the DDM-reduction route: the decision circuit is reduced to a drift-diffusion process with an analytic likelihood, so the whole model is fit by NUTS. A simulation-based-inference (SBI) cross-check on the full nonlinear circuit is the planned second route.

## Why this exists

The `persuasion-control` and `bellwether` experiments use canonical decision-circuit constants, so their capture dose and stealth floor are population numbers. The individuation-gap thesis says the quantity that matters is *per-person*: a model fit to person i should predict person i, including how person i is moved. This experiment builds the inference object that makes a fit person-specific, and it does so for two coupled layers at once:

- **Layer A, individuation.** A per-person state vector `theta_i = (a, t0, w, beta)` of the drift-diffusion model. The posterior over `theta_i` predicts that person's choices and reaction times with calibrated uncertainty.
- **Layer B, learning-rule inference.** A per-person `phi_i = (alpha, w_q)`: a Rescorla-Wagner learning rate and the coupling from learned value into the drift. Inferring `phi_i` from the behavioral stream is a model of *how that person learns*.

Joint inference of `(theta_i, phi_i)` from the same choices and reaction times is the scientific core.

## Honest prior art

The MCMC itself is not the novelty, and we clear that gate up front:

- **HDDM** (Wiecki, Sofer & Frank, 2013, *Front. Neuroinform.* 7:14) already does hierarchical Bayesian MCMC fitting of drift-diffusion models per subject.
- **RL model fitting via hierarchical Bayes** (Daw, 2011, in *Decision Making, Affect, and Learning*) already infers learning rules from choice sequences.
- **RL-DDM** combinations (Pedersen, Frank & Biele, 2017, *Psychon. Bull. Rev.* 24:1234-1251) already couple a learning rule to a drift-diffusion choice model.
- **Simulation-based inference** (Cranmer, Brehmer & Louppe, 2020, *PNAS* 117:30055-30062; the `sbi` toolkit) already gives likelihood-free posteriors for nonlinear simulators like the attractor circuit.

## The novel wedge

Three things this lab couples that the prior art does not:

1. **The steering channel is in the generative model.** A posterior over `theta_i` feeds the `persuasion-control` machinery to produce a *person-specific capture dose and stealth floor*: individuation that predicts manipulability, not just choices.
2. **Joint `(theta_i, phi_i)` recovery with an identifiability story.** Which behavioral regimes let you separate "who this person is" (state) from "how they learn" (rule)? The reversal-bandit task here is chosen precisely so the learning rule has leverage; the recovery results below report, honestly, which parameters are well identified and which are not.
3. **A Lean-verified well-posedness layer.** `proofs/RWInvariant.lean` proves the learning-rule update is a convex combination that keeps the latent value in `[0,1]` and the value-difference drift term in `[-1,1]`, so the inferred learning rule is numerically well-posed by construction, not by hope.

## Method

- **Task.** A two-armed reversal bandit: one option is "good" (higher reward probability) and reverses periodically, with a weak, noisy perceptual cue. Value genuinely drives choice, which is what makes the learning rule identifiable.
- **Likelihood.** The Wiener first-passage-time density (Navarro & Fuss, 2009, *J. Math. Psychol.* 53:222-230), implemented in JAX (`ddm.wfpt_logpdf`) and validated against the Euler-Maruyama simulator (choice-rate and RT-density match; see tests).
- **Model.** Non-centered hierarchical priors over subjects; the learned value is reconstructed deterministically from each subject's observed choices and rewards given `alpha`, so `alpha` is identified through its effect on the drift. Fit with NUTS (numpyro).
- **Group model.** A pooled single-parameter-vector model, the baseline the individual model must beat out-of-sample.

## Results

### Parameter recovery (gate 1)

14 synthetic subjects, 300 trials each, NUTS 2 chains × 600 draws. Group-mean R̂ = 1.004 (converged). Recovery correlation is truth vs posterior mean across subjects; coverage is the fraction of true values inside the 90% posterior interval.

| parameter | kind | recovery r | 90% coverage | RMSE |
|---|---|---|---|---|
| `t0` (non-decision time) | state | +0.99 | 0.71 | 0.007 |
| `a` (boundary) | state | +0.98 | 0.71 | 0.053 |
| `w` (start bias) | state | +0.97 | 0.93 | 0.014 |
| `beta` (cue sensitivity) | state | +0.86 | 0.79 | 0.159 |
| `alpha` (learning rate) | learning rule | +0.67 | 0.71 | 0.166 |
| `w_q` (value→drift coupling) | learning rule | +0.59 | 0.86 | 0.198 |

The honest ordering: the state parameters that shape the reaction-time distribution recover almost perfectly; the learning-rule parameters are identified but harder, as the RL-DDM literature finds. The learning rule needs the value-driven trials the reversal bandit supplies, and more of them than the state parameters need.

### Individuation, out-of-sample (gate 2)

14 subjects, 300 trials, fit on the first 210 trials, scored on the held-out last 90. Metric is mean held-out log-likelihood per trial (higher is better).

| predictor | held-out log-lik / trial | wins vs individual |
|---|---|---|
| **individual** (own posterior-mean params) | **+0.139** | — |
| pooled (shared group params) | +0.040 | individual better in 10/14 |
| shuffled (another subject's params) | −0.201 | individual better in 14/14 |

The per-person fit beats the pooled group model out-of-sample (+0.099 per trial, better in 10 of 14 subjects) and beats the identity-shuffle control in all 14 subjects (+0.340 per trial). That identity-shuffle gap is the load-bearing one: it shows the fitted parameters carry *who the subject is*, not just generic structure a group model also has.

### The bridge to persuasion-control: per-person manipulability (gate 3)

The fitted posterior is not just a predictor of choices; fed through the steering model it yields how much covert influence it would take to flip one of that person's decisions. A covert persuasion dose is an added drift at a decision the person finds ambiguous; the capture dose is the analytic added drift that lifts their probability of the target choice to 75%, and the stealth floor is the added drift at which a monitor watching a window of their decisions detects the steering (`individuation_mcmc/steering.py`, validated against the simulator). From `results/perperson_capture.json` (14 subjects):

- **Manipulability is person-specific.** Capture dose ranges 0.55 to 1.70 across subjects, a **3.1× spread**, median 0.98. There is no single population capture dose.
- **It is predicted by the fitted model.** Capture dose correlates **−0.61** with the fitted decision boundary `a`: people with smaller boundaries (quicker, less cautious deciders) are flipped by less covert drift. So the individuation posterior predicts not just what a person chooses but how cheaply they can be moved.
- **The covert regime depends on how many decisions the monitor sees.** With the monitor accumulating a window of the person's decisions:

| monitor window | median gap (floor − capture) | subjects with a covert gap |
|---|---|---|
| 1 decision | +1.76 (detection rarely reaches 0.75) | 14/14 (wide, often unbounded) |
| 5 decisions | −0.12 | 5/14 |
| 10 decisions | −0.42 | 2/14 |

A monitor that watches a single decision leaves everyone exposed; one that accumulates ten of the person's decisions catches covert steering before it flips the choice for 12 of 14 people. The population-level `persuasion-control` gap (positive, at a fixed monitor) and this per-person result are consistent: both say the covert regime is real but shrinks as the defender gets more to look at. Here that trade-off is resolved per person.

These capture doses are defined *within the reduced DDM*. Gate 4 shows they do not transfer to the full attractor circuit's manipulability ranking, so read them as a property of the fitted model, not as validated full-circuit susceptibility.

### SBI cross-check on the full attractor circuit (gate 4)

The results above use an analytic likelihood on a reduced model. This gate checks that route against the full nonlinear attractor circuit (bellwether's reduced mean-field model) using likelihood-free inference, with approximate Bayesian computation (`individuation_mcmc/sbi_attractor.py`). From `results/sbi_crosscheck.json`:

**A. Likelihood-free recovery works.** ABC against a 1200-draw prior bank recovers the circuit's own parameters from behavior: `sigma_noise` recovery r = 1.00 (RMSE 0.0007), `J_cross` recovery r = 0.66 (RMSE 0.005). So the full nonlinear circuit is identifiable from choices and reaction times without any likelihood.

**B. The reduction fits behavior but does not transfer manipulability.** Across 14 circuit settings, the capture dose measured directly in the full attractor circuit and the capture dose predicted by the DDM-reduction route (an ML-fitted DDM on the same behavior) do **not** track each other: Spearman r = −0.23, Pearson r = −0.27. Both have real spread (attractor CV 0.30, DDM CV 0.19), so this is a genuine mismatch, not a degenerate range.

This is the honest and consequential result of the cross-check. The DDM reduction is faithful for *fitting behavior* (it fits choices and RTs, and ABC confirms the circuit is identifiable), but its capture dose, a derivative quantity, does **not** rank who is manipulable the way the full nonlinear circuit does. The practical conclusion: the per-person capture doses in gate 3 are well-defined *within the reduced model*, but they are not validated as proxies for full-circuit manipulability. Manipulability should be read from the full circuit directly, and since ABC recovers the full-circuit parameters cleanly, simulation-based inference, not the analytic reduction, is the right tool for that step. This tempers the gate-3 claim rather than confirming it.

## Has and has not been shown

**Shown.** On synthetic subjects with known ground truth: (1) NUTS recovers the per-person state parameters almost perfectly (r = 0.86 to 0.99) and the learning-rule parameters moderately (alpha r = 0.67, w_q r = 0.59), with converged chains (R̂ = 1.004); (2) the per-person posterior predicts held-out behavior better than a pooled group model (10/14 subjects) and better than an identity-shuffle control (14/14 subjects), so the fit individuates; (3) the learning-rule update is Lean-verified to keep the latent value bounded, so the inferred rule is well-posed. The analytic WFPT likelihood matches the simulator to within Monte-Carlo noise.

**Not shown.** Synthetic subjects only; the generative model and the fitted model share a functional form, so recovery here is a necessary check, not evidence the form is right for real people. The learning-rule parameters are harder to identify than the state parameters, and the amount of data needed is reported, not assumed away. The per-person capture dose is computed at an idealised ambiguous decision (baseline drift zero) and the monitor is a drift-direction detector on the person's own decision stream, which a real overseer may not have; the capture dose and the window-dependence should be read as properties of this model, not calibrated real-world numbers. The SBI cross-check (gate 4) found that the reduced-model capture dose does not rank-match the full-circuit capture dose, so the reduction is validated for behavior-fitting but not for manipulability transfer; that is a limitation to resolve by computing manipulability in the full circuit via SBI, which gate 4 shows is identifiable.

## Run it

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e . pytest
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q
JAX_PLATFORMS=cpu .venv/bin/python run_recovery.py --subjects 14 --trials 300
JAX_PLATFORMS=cpu .venv/bin/python run_individuation.py --subjects 14 --trials 300
JAX_PLATFORMS=cpu .venv/bin/python run_perperson_capture.py --subjects 14 --trials 300
JAX_PLATFORMS=cpu .venv/bin/python run_sbi_crosscheck.py --bank 1200 --cohort 14
```

The SBI cross-check needs the sibling `bellwether` package: `uv pip install --python .venv/bin/python -e ../bellwether`.

Proofs:

```bash
cd proofs && lake exe cache get && lake build
```

## Files

- `individuation_mcmc/ddm.py` : WFPT likelihood (JAX), DDM simulator, reversal-bandit subject with a learning rule.
- `individuation_mcmc/model.py` : hierarchical NUTS model, deterministic value reconstruction, individual and pooled forms.
- `run_recovery.py` : synthetic parameter recovery and coverage.
- `run_individuation.py` : individual vs pooled vs identity-shuffle, out-of-sample.
- `individuation_mcmc/steering.py` : the bridge to persuasion-control: per-person capture dose (analytic hitting probability) and stealth floor (windowed drift-direction detector).
- `run_perperson_capture.py` : fits the cohort and reports per-person capture dose, stealth floor by monitor window, and how manipulability tracks the fitted boundary.
- `individuation_mcmc/sbi_attractor.py` : likelihood-free (ABC) inference on the full attractor circuit, plus the full-circuit capture dose.
- `run_sbi_crosscheck.py` : ABC parameter recovery and the attractor-vs-DDM capture-dose cross-check.
- `proofs/RWInvariant.lean` : Lean 4 proof that the learning-rule layer is well-posed (value bounded in [0,1]).
- `tests/test_ddm.py` : WFPT-vs-Monte-Carlo validation and the value-bound invariant.

## Citations

- Navarro D.J., Fuss I.G. *Fast and accurate calculations for first-passage times in Wiener diffusion models.* J. Math. Psychol. 53:222-230, 2009.
- Wiecki T.V., Sofer I., Frank M.J. *HDDM: hierarchical Bayesian estimation of the drift-diffusion model in Python.* Front. Neuroinform. 7:14, 2013.
- Pedersen M.L., Frank M.J., Biele G. *The drift diffusion model as the choice rule in reinforcement learning.* Psychon. Bull. Rev. 24:1234-1251, 2017.
- Cranmer K., Brehmer J., Louppe G. *The frontier of simulation-based inference.* PNAS 117:30055-30062, 2020.
