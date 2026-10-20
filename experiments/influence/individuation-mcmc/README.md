# individuation-mcmc

**Can we infer a person's decision parameters and their learning rule from behavior, well enough that the per-person fit is (a) recoverable, (b) better than a group model out-of-sample, and (c) a handle on how that person would be steered?**

Mostly synthetic subjects with known ground truth, plus one real-data gate. The core is the DDM-reduction route: the decision circuit is reduced to a drift-diffusion process with an analytic likelihood fit by NUTS; a simulation-based-inference (SBI) route cross-checks it on the full nonlinear circuit, and a real reversal-learning dataset tests the one claim that is falsifiable on humans.

## Summary (plain language)

From a person's choices and reaction times alone you can fit a small model of how they decide. Three findings, in decreasing order of how well they are established:

1. **Individuation works, including on real people.** The fitted decision fingerprint is stable and specific enough to re-identify the *same* person in a later session better than anyone else's fingerprint (115 of 150 real subjects; test-retest reliability up to r = 0.71). Behavior to a stable individual decision model is the solid result.
2. **Manipulability is person-specific, but shown only in simulation.** Feeding the fit through the steering model gives how small a covert nudge flips a decision; it varies ~3x across simulated people and is predicted by the fit. On the full nonlinear circuit it must be computed by SBI, not the reduction (the reduction's version does not transfer, gate 4 vs gate 5). None of this is validated on a real person, because that would require actually covertly steering someone.
3. **Reading a person's learning rule is the weak link.** Recoverable in simulation (moderately), but not reliable across sessions on real data.

The honest one-liner: *this measures an individual's decision fingerprint from behavior (works on real humans) and estimates, in simulation only, how cheaply that individual could be covertly steered*. The system estimates susceptibility that is already there; it does not create it, and the same number a manipulator would want is the one a monitor uses to catch steering before it works.

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

### Full-circuit manipulability via SBI (gate 5, the gate-4 fix)

Gate 4 showed the reduction's capture dose does not transfer. The fix: skip the reduction for the manipulability step. For each subject, infer the full-circuit parameters from behavior with ABC, then compute the capture dose *in the attractor itself* from the inferred parameters, propagating posterior uncertainty (`run_sbi_manipulability.py`). The validation is the loop the reduction failed: does the SBI-inferred capture dose track the ground-truth capture dose (computed from each subject's true parameters)?

Across 14 subjects (1200-draw ABC bank):

| metric | value |
|---|---|
| Spearman, SBI-inferred vs ground-truth capture dose | **0.90** |
| Pearson, SBI-inferred vs ground-truth capture dose | **0.92** |
| median absolute error in capture dose | 0.0001 nA |
| 90% posterior-interval coverage of the true capture dose | 1.00 |

Computing manipulability directly in the full circuit from ABC-inferred parameters recovers the ground-truth capture dose (r = 0.90), where the DDM reduction failed (gate 4, r = −0.23). The posterior intervals contain the truth for every subject (coverage 1.00, if anything mildly conservative). So the working per-person manipulability pipeline is: behavior → ABC posterior over full-circuit parameters → capture dose computed in the attractor, with uncertainty. The reduction is kept for its analytic individuation of choices and learning rule; the full-circuit SBI route is used for manipulability.

### Real data: individuation on a reversal-learning dataset (gate 6)

Everything above is synthetic. This gate runs the one claim that is falsifiable on real human data, individuation across sessions, on the Waltmann et al. (2022) probabilistic reversal-learning dataset (CC-BY 4.0; 150 participants, each tested twice). Manipulability is deliberately not computed here: you cannot covertly steer a real participant to obtain a ground-truth capture dose, so any capture dose from these fits would be an unvalidated model readout.

The per-trial choice modelled is whether the participant repeated their previous choice (`stay`), with a win-stay/lose-shift Rescorla-Wagner value, driven by the recorded feedback, tilting the drift; reaction time is the first-passage time. Each participant is fit separately in each session (`individuation_mcmc/realdata.py`, `run_realdata.py`).

**Test-retest reliability** (Pearson, session-1 vs session-2 fitted parameters, 150 subjects):

| parameter | kind | test-retest r |
|---|---|---|
| `t0` (non-decision time) | state | 0.71 |
| `a` (boundary) | state | 0.60 |
| `v0` (baseline stay drift) | state | 0.51 |
| `w` (start bias) | state | 0.37 |
| `alpha` (learning rate) | learning rule | 0.15 |
| `w_q` (value→drift coupling) | learning rule | −0.01 |

The decision-state parameters are moderately to substantially reliable across sessions; the learning-rule parameters are not. That is consistent with the dataset's own finding that the computational RL readouts of this task have limited reliability, and it mirrors the synthetic result that learning-rule parameters are the hard ones.

**Cross-session individuation.** Predicting each subject's session-2 behavior from session-1 fits, mean log-likelihood per trial: own session-1 fit **+0.126**, a random other subject's session-1 fit **−0.248**. The own fit wins in **115 of 150 subjects** (+0.37 nats/trial). So a person's decision fingerprint fit in one session predicts the same person months later better than another person's fingerprint: individuation holds on real human behavior. (The pooled mean-of-fits baseline is degenerate here, because averaging non-decision times across people violates the per-subject fastest-RT bound and floors some trials, so the identity-shuffle is the individuation control reported.)

The reliable, individuating part is the decision-state fingerprint; the learning-rule layer, which was the harder-to-recover part even in simulation, does not individuate reliably on this dataset.

## Has and has not been shown

**Shown.** On synthetic subjects with known ground truth: (1) NUTS recovers the per-person state parameters almost perfectly (r = 0.86 to 0.99) and the learning-rule parameters moderately (alpha r = 0.67, w_q r = 0.59), with converged chains (R̂ = 1.004); (2) the per-person posterior predicts held-out behavior better than a pooled group model (10/14 subjects) and better than an identity-shuffle control (14/14 subjects), so the fit individuates; (3) the learning-rule update is Lean-verified to keep the latent value bounded, so the inferred rule is well-posed. The analytic WFPT likelihood matches the simulator to within Monte-Carlo noise.

**Shown on real data (gate 6).** On the Waltmann et al. reversal-learning dataset, the decision-state fingerprint is reliable across sessions (t0 r=0.71, a r=0.60) and individuates: a session-1 fit predicts the same person's session-2 behavior better than a random other person's fit in 115/150 subjects. This is the falsifiable individuation claim, and it passes on real human behavior for the decision-state layer.

**Not shown.** Most gates use synthetic subjects, where the generative and fitted models share a functional form, so recovery there is a necessary check, not evidence the form is right for real people. On real data the learning-rule parameters are not reliable (alpha r=0.15), so the learning-rule layer is not yet a dependable individual readout. Manipulability (capture dose) is never validated on real data: there is no ground-truth covert-steering experiment on humans here, so real-data capture doses are not computed. The real-data model is a reduced win-stay/lose-shift RL-DDM, not a claim about the true generative process. The learning-rule parameters are harder to identify than the state parameters, and the amount of data needed is reported, not assumed away. The per-person capture dose is computed at an idealised ambiguous decision (baseline drift zero) and the monitor is a drift-direction detector on the person's own decision stream, which a real overseer may not have; the capture dose and the window-dependence should be read as properties of this model, not calibrated real-world numbers. Gate 4 found the reduced-model capture dose does not transfer to the full circuit; gate 5 resolves this by computing manipulability in the full circuit via SBI (recovering the ground-truth capture dose at r = 0.90), so the gate-3 reduced-model doses should be read only as a within-DDM quantity and the gate-5 full-circuit route is the validated manipulability measure. The manipulability gates (3-5) are entirely synthetic; the only real-data result is the gate-6 individuation of the decision-state fingerprint. No neural data is used anywhere.

## Run it

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e . pytest
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q
JAX_PLATFORMS=cpu .venv/bin/python run_recovery.py --subjects 14 --trials 300
JAX_PLATFORMS=cpu .venv/bin/python run_individuation.py --subjects 14 --trials 300
JAX_PLATFORMS=cpu .venv/bin/python run_perperson_capture.py --subjects 14 --trials 300
JAX_PLATFORMS=cpu .venv/bin/python run_sbi_crosscheck.py --bank 1200 --cohort 14
JAX_PLATFORMS=cpu .venv/bin/python run_sbi_manipulability.py --bank 1200 --cohort 14
# Real data (download the CC-BY dataset yourself; it is not in this repo):
JAX_PLATFORMS=cpu .venv/bin/python run_realdata.py /path/to/RL_Data_Retest_R_wo.txt
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
- `run_sbi_manipulability.py` : per-subject manipulability computed in the full circuit from ABC-inferred parameters, validated against ground truth.
- `individuation_mcmc/realdata.py`, `run_realdata.py` : load a real reversal-learning dataset and run the cross-session individuation test (dataset not redistributed here).
- `proofs/RWInvariant.lean` : Lean 4 proof that the learning-rule layer is well-posed (value bounded in [0,1]).
- `tests/test_ddm.py` : WFPT-vs-Monte-Carlo validation and the value-bound invariant.

## Citations

- Navarro D.J., Fuss I.G. *Fast and accurate calculations for first-passage times in Wiener diffusion models.* J. Math. Psychol. 53:222-230, 2009.
- Wiecki T.V., Sofer I., Frank M.J. *HDDM: hierarchical Bayesian estimation of the drift-diffusion model in Python.* Front. Neuroinform. 7:14, 2013.
- Pedersen M.L., Frank M.J., Biele G. *The drift diffusion model as the choice rule in reinforcement learning.* Psychon. Bull. Rev. 24:1234-1251, 2017.
- Cranmer K., Brehmer J., Louppe G. *The frontier of simulation-based inference.* PNAS 117:30055-30062, 2020.
- Waltmann M., Schlagenhauf F., Deserno L. *Sufficient reliability of the behavioral and computational readouts of a probabilistic reversal learning task.* Behav. Res. Methods 54:2993-3014, 2022 (data CC-BY 4.0, https://osf.io/4ng3e/).
