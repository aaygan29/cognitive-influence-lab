# bellwether

**An early-warning instrument for value-decision capture.**

> Part of the [cognitive-influence-stack](https://github.com/anonymous/cognitive-influence-stack)
> program: this is the **Level-3 (neural value channel)** early-warning
> instrument, the neural-level analogue of that project's Level-1
> critical-slowing-down early warning. It lives here as its own repository while
> its submission is open, and folds into the stack's `level3_substrate/` later.

A bellwether is the animal whose bell tells the shepherd which way the flock is
about to turn. This project asks whether a decision circuit gives off a warning
of the same kind: a signal, readable from population activity, that a choice is
being pushed before the choice is made.

Everything here runs on a transparent synthetic model with known ground truth.
That is deliberate. The goal is to demonstrate, under conditions we fully
control, that covert steering of a value decision is detectable in principle,
to measure how large the steering must be before it becomes visible, and to show
where the detector fails. A result on a generative model whose every knob is
known is the right first step before pointing the same monitor at real
recordings, where the ground truth is not available.

---

## Why this matters: a risk vector that is getting closer

Neurotechnology is moving from reading coarse summaries of brain state toward
reading structured, moment-to-moment representations. Invasive and non-invasive
interfaces already decode remembered locations, intended movements, attended
features, and the contents of working memory from population activity. In
parallel, closed-loop systems can write to the brain: deep-brain stimulation,
transcranial magnetic and focused-ultrasound stimulation, and, in animal models,
optogenetic control that can switch specific cell populations on and off with
light. Read access and write access are both improving, and they are beginning
to meet in the same devices.

The uncomfortable implication is simple. As an automated system gains finer
access to how a person represents options and forms intentions, the distance
between *reading* a decision and *nudging* it shrinks. A system does not need to
seize a choice to change it. The science of decision circuits says the cheapest
place to intervene is near the point where a decision tips, because that is
exactly where the circuit is most sensitive to small inputs. An influence that
would be negligible at any other moment can decide the outcome if it arrives
when the circuit is balanced on its own knife edge. A sufficiently instrumented
agent, whether it writes through stimulation or simply shapes the stream of
evidence a person sees, can in principle exploit that sensitivity. The same
logic that makes modern language-model agents able to imitate human behavior
closely enough to pass as human participants in online studies is the logic that
makes covert, individually-targeted influence plausible rather than fanciful.

If that influence is possible, a defender needs a way to notice it. Not after the
fact, in the behavioral record, but during the approach to the decision, while
there is still time to flag it. That is what this instrument is a prototype of.
It does not claim any of the above is happening today. It builds the smallest
honest testbed in which the detection question can be asked and answered with
numbers.

---

## What the instrument does

Four pieces fit together.

1. **A decision substrate.** A reduced two-variable attractor circuit models a
   value-based choice between two options. With ambiguous evidence the circuit
   sits near a symmetric unstable point; noise or input tips it into one of two
   winner-take-all states. That tipping is a bifurcation, and near it the circuit
   is both maximally sensitive to small inputs and slow to recover from
   perturbations.

2. **A population readout.** The latent circuit is embedded into many noisy units
   whose rates are a random nonlinear mixture of the two internal variables. The
   monitor never sees the latent decision variable, the injected drive, or the
   realized choice. It sees population activity and the evidence the task claims
   to present, which is all a real defender would have.

3. **An adversary.** A covert steering drive is added to one option. It is
   subthreshold by construction: small relative to the evidence scale, so it does
   not look like evidence. A stealthy variant hides the same steering inside added
   diffusion so the trajectory looks as wandering as an honest hard decision.

4. **A monitor.** During the approach to the decision, the monitor looks for one
   thing: a decision that resolves **sooner and straighter than the presented
   evidence justifies**. That signature has two channels. The **timing** channel
   is how early the decision commits relative to what the evidence predicts. The
   **geometry** channel is the **directedness** of the recovered decision axis
   (net displacement over path length): a steered decision moves in one
   direction, an honest ambiguous one diffuses. Both channels are measured
   relative to the presented evidence, so neither is credited for merely tracking
   evidence strength, and the monitor raises a sequential alarm before commitment
   when the anomaly is larger than honest ambiguous decisions produce.

---

## Headline results

Numbers below are from a representative five-seed run (`experiments/run_validation.py`
and `experiments/run_leadtime.py`) and regenerate on any machine; exact digits
shift slightly with seeds.

- **Detection.** Covert steering versus an honest ambiguous decision of the same
  presented evidence is detectable before commitment. The geometry channel
  separates them at **AUC 0.97** (95% bootstrap CI about 0.95 to 0.97;
  permutation test against chance p < 0.001; 750 trials per class). The
  evidence-relative timing channel is slightly stronger at **AUC 0.99**. Against
  legitimate strong evidence both channels are near perfect, because both are
  calibrated against the evidence actually presented.
- **Robust across random embeddings.** The geometry result is not an artifact of
  one lucky projection: across ten independent random population embeddings the
  AUC is **0.97 on average (standard deviation 0.016, range 0.93 to 0.99)**.
- **False alarms are controlled.** The alarm threshold is set to a 5% false-alarm
  rate on honest ambiguous decisions, the condition a covert attack imitates. On
  held-out honest trials the realized rate is about 3%.
- **Early warning.** When the monitor fires on a steered trial, it fires **before
  the behavioral commitment about 92% of the time**, median lead about 40 ms.
- **Dose-response and a stealth floor.** Detectability rises smoothly with the
  steering dose. Below a **stealth floor of about 0.008 nA·s** the geometry
  channel cannot separate steering from an honest hard decision. A stealthy
  (diffusion-hiding) adversary roughly **doubles that floor to about 0.019 nA·s**
  for the geometry channel. The timing channel, however, is **not** evaded by the
  stealthy adversary (see below), so the floor is channel-specific.

### Two channels, and which one is really doing the work

The honest result, and the one a review forced into the open, is that the two
channels are not equal. In a symmetric attractor you cannot bias the outcome
without also speeding the escape from the saddle, so steering and acceleration are
mechanistically coupled. As a consequence an evidence-relative **timing** cue is
the dominant channel: it equals or beats the geometry channel on every contrast
(manip versus hard 0.99 against 0.97, manip versus easy 1.0 against 1.0), and it
is **not** fooled by the stealthy adversary (timing AUC about 1.0 where geometry
drops to about 0.79), because injected diffusion that hides the straightness still
leaves the decision committing too soon.

So the defensible claim is not that directedness is a superior detector. It is
that covert steering leaves a **"sooner and straighter than the evidence
warrants"** signature, that this signature is detectable before commitment via
either channel, that timing is the stronger and harder-to-evade of the two in this
model, and that the geometry channel is a correlated signal an adversary can
partly hide. The open and important question is whether a richer or asymmetric
circuit could permit biasing a choice **without** speeding it, which is exactly the
regime where the timing monitor would fail and the geometry channel would have to
carry the detection on its own.

### Substrate checks (secondary, reported for honesty)

- The chosen option's representation **expands** after commitment (variance ratio
  about 3.9), reproducing a known commitment-geometry effect.
- Critical slowing down is **partial**: the variance of the decision variable
  rises significantly (Kendall-tau trend test) on about 54% of honest ambiguous
  trials, but lag-1 autocorrelation does not reliably rise. This matches the
  known caveat that early-warning indicators are not universal and depend on the
  noise structure. The detector does **not** rely on critical slowing down; it
  rests on directedness. The CSD measurement is reported only to characterize the
  substrate.
- The two-variable substrate does not reproduce the full rotation of option
  subspaces seen in richer cortical data (the measured rotation is small). That is
  a limitation of the reduced model, noted below, not a claim.

---

## How this connects to the methods it borrows from

The components are standard families, used the way the source literature uses
them, so the instrument is not inventing machinery that would need separate
defense.

- The decision circuit is the conventional reduced mean-field model of a
  mutual-inhibition choice, the same object used to study persistent activity,
  winner-take-all dynamics, and the bifurcation underlying commitment.
- Reading a decision from the geometry of population activity, and the idea that
  the chosen option is reorganized and amplified at commitment, follows the
  population-geometry approach to value decisions.
- Variance and autocorrelation in sliding windows as warnings of an approaching
  tipping point is the standard critical-slowing-down toolkit, with a Kendall-tau
  trend test as the field-standard way to test for a monotone rise.
- Recovering the governing drift of the latent variable from noisy observations,
  up to an affine gauge, is in the spirit of latent-dynamics recovery methods for
  neural data; here it is used as an interpretive landscape tool while detection
  rests on the simpler directedness statistic.

---

## Statistical validity

The evaluation is built to survive methodological scrutiny.

- **Train/test separation.** The monitor is calibrated on honest trials only and
  evaluated on held-out trials from different seeds. The adversary's trials never
  touch any threshold, so nothing leaks from the thing being detected into the
  detector.
- **Calibration on the right null.** Directedness residuals are heteroscedastic:
  ambiguous decisions wander more than strong-evidence ones. The alarm threshold
  is therefore set on honest ambiguous trials specifically, the hardest and most
  relevant false-alarm case.
- **Effect size with uncertainty.** AUC is reported with a stratified bootstrap
  confidence interval and the count of events and non-events, not a bare number.
- **Significance against chance.** A label-permutation test gives a p-value for
  each AUC.
- **Confound handling.** Commitment speed is not treated as a nuisance to be
  explained away but as a second detection channel, compared head to head with the
  geometry channel using a paired DeLong test (the correct test for two scores on
  one shared test set) and reported honestly as the stronger of the two.
- **Specification robustness.** Results are stable across ten random population
  embeddings (AUC 0.97, SD 0.016), across the smoothing hyperparameter, and across
  seeds. A stealthy adversary is run as an adversarial specification.
- **Gauge invariance.** The detection statistic is invariant to the sign and
  scale of the recovered decision axis, so no gauge or sign-alignment choice can
  leak the target of the manipulation.
- **Measuring along the bifurcation.** Early-warning signals vanish if one
  measures orthogonal to the direction the bifurcation acts on. The monitor
  estimates the decision axis by principal components of the population, so it
  measures along the direction the choice varies.

---

## Longitudinal deployment and decoded precision

Two additions bring the instrument closer to how it would run on real recordings,
each grounded in recent human and BCI work rather than in the project's own
assumptions.

**Drift-robust deployment (`bellwether.drift`, `experiments/run_drift.py`).**
Real neural codes drift: a decoder fixed on one session decays on the next, and
the published remedy is to align the low-dimensional latent dynamics across
sessions. Reproducing that here: a decision axis fixed on a reference session
degrades as the code drifts (detection AUC falls from about 0.997 to 0.86 at
strong drift), an orthogonal-Procrustes alignment of honest-trial subspaces
carries the reference axis into each drifted session and restores it (about
0.998 throughout), and the default per-trial estimation is inherently
drift-robust (about 1.0 throughout) at the cost of re-estimating every trial.
A secondary honest finding is that the directedness statistic is unusually
drift-tolerant to begin with, because it reads trajectory shape rather than the
absolute population code.

**Decoded-precision margins (`bellwether.precision`, `experiments/run_precision.py`).**
Human intracranial recordings show uncertainty and prediction error are
explicitly decodable, with precision gating how strongly signals propagate. In
the model, an uncertainty proxy decoded from population activity alone (how
slowly the decision resolves) tracks evidence strength monotonically, reproducing
that decodability, and using it to set an adaptive margin modestly evens the
honest false-alarm rate across the uncertainty range. The honest limit, which the
experiment makes explicit, is that decoded precision cannot by itself replace the
presented-evidence reference for detection: covert steering speeds resolution and
is therefore decoded as precision, the same coupling that makes the timing
channel dominant. This confirms, rather than removes, the need for an
evidence-relative baseline.

## Limitations

Stated plainly, because the artifact is only as credible as its honesty.

- It is a synthetic model. It demonstrates a capability and a risk logic under
  known ground truth. It is not evidence that the effect exists at any particular
  magnitude in a real brain. The next step is to apply the same monitor to real
  value-decision recordings, where the decision axis must be estimated without
  ground truth.
- Steering and commitment speed are mechanically coupled in this symmetric
  circuit, so the timing channel does much of the detection work and the geometry
  channel is only partly independent of it. The claim is framed around the joint
  "sooner and straighter than warranted" signature, not around directedness as a
  stand-alone detector.
- The stealth floor is channel-specific. A diffusion-hiding adversary can push the
  geometry channel below its floor at low dose, but does not evade the timing
  channel in this model. Whether a circuit exists in which bias is possible
  without speedup, defeating the timing channel, is the key open question.
- Critical slowing down is only partially present in this substrate, and the
  reduced model does not reproduce the full subspace rotation of richer cortex.
- The reduced two-variable circuit omits much biology. Conclusions should be read
  as properties of this model class, pending replication in a higher-dimensional
  circuit and in data.

---

## Install and run

Requires Python 3.10 or newer.

```bash
cd bellwether
uv venv --python 3.12 .venv            # or: python3 -m venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q          # 14 tests

cd experiments
../.venv/bin/python run_validation.py  # detection, confounds, substrate checks
../.venv/bin/python run_leadtime.py    # lead time, dose-response, stealth floor
```

Outputs land in `results/*.json` and `figures/*.png`.

## Repository layout

```
bellwether/
  dynamics.py     reduced two-variable decision circuit, adversary, simulation
  population.py   random high-dimensional neural readout + decision-axis estimation
  estimator.py    directedness statistic; drift-field recovery (interpretive)
  ews.py          critical-slowing-down statistics + Kendall-tau trend test
  geometry.py     commitment geometry (subspace angle, chosen-subspace expansion)
  precision.py    decoded-uncertainty proxy and adaptive (ratcheting) margins
  drift.py        representational-drift simulation + Procrustes latent alignment
  monitor.py      the online early-warning monitor and its calibration
  detect.py       ROC/AUC, bootstrap CI, permutation test, paired DeLong,
                  commit-time-stratified AUC, lead-time and stealth-floor summaries
experiments/
  configs.py          shared scenario and calibration configuration
  run_validation.py   main empirical validation
  run_leadtime.py     lead time, dose-response, stealth floor
  run_robustness.py   embedding-seed robustness and the two detection channels
  run_drift.py        longitudinal detection under representational drift
  run_precision.py    decoded-precision margins vs a fixed threshold
tests/
  test_core.py        unit and integration tests
```

## License

MIT. See `LICENSE`.
