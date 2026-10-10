# bellwether

**An early-warning instrument for value-decision capture.**

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

4. **A monitor.** During the approach to the decision, the monitor estimates the
   decision axis from population activity and measures its **directedness** (net
   displacement over path length). A covertly steered decision moves in one
   direction; an honest ambiguous decision diffuses. The monitor compares the
   observed directedness to what the presented evidence predicts, and raises a
   sequential alarm when the excess is larger than honest ambiguous decisions
   ever produce.

---

## Headline results

Numbers below are from a representative five-seed run (`experiments/run_validation.py`
and `experiments/run_leadtime.py`) and regenerate on any machine; exact digits
shift slightly with seeds.

- **Detection.** Covert steering versus an honest ambiguous decision of the same
  presented evidence is separated with **AUC 0.96** (95% bootstrap CI about
  0.95 to 0.97; permutation test against chance p < 0.001; 750 trials per class;
  across-seed standard deviation 0.005). Against legitimate strong evidence the
  separation is near perfect, because the monitor calibrates directedness against
  the evidence actually presented.
- **False alarms are controlled.** The alarm threshold is set to a 5% false-alarm
  rate on honest ambiguous decisions, the condition a covert attack imitates. On
  held-out honest trials the realized false-alarm rate is about 3%.
- **Early warning.** When the monitor fires on a steered trial, it fires **before
  the behavioral commitment about 92% of the time**, with a median lead of about
  40 ms at the demonstrated dose.
- **Dose-response and a stealth floor.** Detectability rises smoothly with the
  steering dose. Below a **stealth floor of about 0.008 nA·s** the steering is, by
  this detector, statistically indistinguishable from an honest hard decision.
  A stealthy (diffusion-hiding) adversary roughly **doubles that floor to about
  0.019 nA·s** but still cannot hide at high dose. The floor is itself a result:
  it is a quantitative statement of how cheap an attack can be while remaining
  invisible.

### The confound we had to defeat, honestly

A naive adversary that steers hard also commits fast. A detector that keyed only
on commitment speed would therefore score very well here (AUC about 0.99), and in
a paired DeLong test it formally beats the directedness detector for this one
contrast. That is reported in the output, not hidden. The reason directedness is
still the right signal is twofold. First, within strata of commitment time, where
the speed cue is held fixed, directedness still separates steered from honest
decisions at **AUC about 0.93**, so it is carrying information beyond speed.
Second, the speed cue is useless against an adversary that paces its steering to
commit at a normal latency, and against the legitimate-strong-evidence contrast,
whereas directedness handles both. A real adversary adapts; a defender needs a
signal that survives adaptation.

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
- **Confound handling.** The commitment-speed confound is tested with a paired
  DeLong test (the correct test for two scores on one shared test set) and
  controlled with a commitment-time-stratified AUC.
- **Specification robustness.** A stealthy adversary is run as an adversarial
  specification, and results are stable across the smoothing hyperparameter and
  across seeds.
- **Gauge invariance.** The detection statistic is invariant to the sign and
  scale of the recovered decision axis, so no gauge or sign-alignment choice can
  leak the target of the manipulation.
- **Measuring along the bifurcation.** Early-warning signals vanish if one
  measures orthogonal to the direction the bifurcation acts on. The monitor
  estimates the decision axis by principal components of the population, so it
  measures along the direction the choice varies.

---

## Limitations

Stated plainly, because the artifact is only as credible as its honesty.

- It is a synthetic model. It demonstrates a capability and a risk logic under
  known ground truth. It is not evidence that the effect exists at any particular
  magnitude in a real brain. The next step is to apply the same monitor to real
  value-decision recordings, where the decision axis must be estimated without
  ground truth.
- The naive adversary is partly confounded with commitment speed. The defensible
  claim rests on the stratified analysis and the adaptive-adversary argument, not
  on the raw contrast.
- A stealthy adversary raises the stealth floor and, at low dose, can stay below
  it. The instrument quantifies that escape rather than eliminating it.
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
  monitor.py      the online early-warning monitor and its calibration
  detect.py       ROC/AUC, bootstrap CI, permutation test, paired DeLong,
                  commit-time-stratified AUC, lead-time and stealth-floor summaries
experiments/
  configs.py          shared scenario and calibration configuration
  run_validation.py   main empirical validation
  run_leadtime.py     lead time, dose-response, stealth floor
tests/
  test_core.py        unit and integration tests
```

## License

MIT. See `LICENSE`.
