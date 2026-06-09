# Study 2, explained from zero to deep

A complete walkthrough of the NARPS spiking-decision experiment: the idea, the statistics, and the
code line by line. Written to take a reader from no background to a graduate-level understanding of
*exactly* what this experiment did and why.

---

## 0. The whole thing in one paragraph
People were shown gambles ("50% chance to win \$20, 50% chance to lose \$10 — take it?") and said
yes or no. We have 27,454 of those real yes/no decisions from 108 people (the public NARPS dataset).
We asked: **can a brain-inspired "spiking" neural network predict those decisions as well as the
textbook economic theory of risky choice — and more cheaply?** Answer, reported honestly: it
predicts them *on par* with the theory (no meaningful accuracy difference), the theory is already
near the best any model can do on these inputs, and at this tiny scale the spiking model is *not*
more energy-efficient. No win was invented; every number carries a confidence interval.

---

## 1. The question, built from nothing

**A decision under risk** is a choice where outcomes are uncertain. NARPS used the simplest kind: a
50/50 gamble with a **gain** (how much you win on heads) and a **loss** (how much you lose on tails).
On each trial a person chooses **accept** (take the bet) or **reject** (decline).

We want a **model**: a rule that takes (gain, loss) and outputs a guess for what the person did.
A good model = its guesses match the real choices on people it has never seen.

Three reasons this is worth doing:
1. *Behavioral economics:* it lets us measure how people trade off gains vs. losses.
2. *Neuromorphic engineering:* it lets us test whether a brain-style "spiking" computer can do this
   task, and at what energy cost.
3. *Honesty as method:* the simplest economic model is already very good, so this is a stress test
   of whether fancier/brain-inspired models actually add anything — a great place to practice not
   over-claiming.

---

## 2. The five models (the ideas behind them)

We compare five predictors, from simplest to most flexible.

**(a) Expected Value (EV) — the "rational robot" null.**
EV = `0.5·gain − 0.5·loss` (average outcome of the 50/50 bet). A purely rational agent accepts when
EV > 0. This is the baseline every other model must beat to be interesting.

**(b) Prospect theory — the canonical *human* model (Kahneman & Tversky, 1979/1981).**
Humans are **loss-averse**: a \$10 loss hurts more than a \$10 gain pleases. So instead of weighting
gain and loss equally, humans use roughly `gain − λ·loss`, where **λ (lambda) is the loss-aversion
coefficient** — how many times more a loss looms than a gain. We *estimate* λ from the data; we got
**λ = 1.69**, meaning losses felt ~1.7× as strong as equivalent gains (textbook range is ~1.5–2.5).
This is the model the spiking net has to match to be credible.

**(c) Rate MLP — a standard (non-spiking) neural network.**
A small "multi-layer perceptron": inputs → a hidden layer of artificial neurons (continuous numbers)
→ an output. It learns whatever mapping from (gain, loss) to choice best fits the training data. It
is the *capacity-matched control*: same size as the spiking net, but ordinary arithmetic, so any
difference between it and the spiking net is due to the spiking, not the model size.

**(d) Spiking LIF network — the neuromorphic model (the star).**
A **spiking neural network (SNN)** uses neurons that behave like real ones: each neuron has a
**membrane potential** (a running charge). Input current adds to it; it **leaks** away over time
(hence *Leaky Integrate-and-Fire*, LIF); when it crosses a **threshold** the neuron emits a **spike**
(a 1) and resets. Information is carried by *when and how often neurons spike*, not by smooth numbers.
- *Why anyone cares:* spikes are sparse, binary events, so specialized "neuromorphic" chips can run
  SNNs at very low energy — but only when activity is sparse.
- *The training trick:* a spike is a hard 0/1 step, which has no usable gradient (you can't do
  calculus on a step). **Surrogate gradients** replace the step with a smooth function *only during
  the backward (learning) pass*, so standard deep-learning training works. We use snnTorch's
  `fast_sigmoid` surrogate.
- *Input encoding:* our inputs are 2 plain numbers (gain, loss). We feed them as **direct current**
  into the first layer every timestep ("current/direct encoding"). (The alternative, turning numbers
  into random spike trains, throws away precision on only 2 inputs — bad for accuracy.)

**(e) Gradient-Boosted Trees (GBT) — the "expressivity ceiling."**
A flexible non-neural model that can capture arbitrary nonlinear patterns. Its job is diagnostic: if
even GBT can't beat prospect theory, then the **accuracy ceiling is set by the data itself** (you
simply cannot predict these choices better from gain & loss alone), not by any model's weakness.

---

## 3. How we score a model (the metrics)

**AUC (Area Under the ROC Curve) — the main score.**
Pick one trial the person *accepted* and one they *rejected*. AUC = the probability the model gives
the accepted one a higher "accept" score. **0.5 = coin flip (useless); 1.0 = perfect.** We use AUC
instead of plain accuracy because it doesn't depend on an arbitrary cutoff and isn't fooled by class
imbalance (people accepted 55% of the time). Mathematically AUC equals the normalized Mann-Whitney
*U* statistic — a rank comparison — which is how we compute it (Section 6).

**Balanced accuracy.** Average of (correct-accepts rate, correct-rejects rate) at a 0.5 cutoff — a
sanity-check companion to AUC that treats both choices equally.

**Energy.** The neuromorphic claim is about cost, so we count operations:
- Ordinary net: **MACs** (multiply-accumulate ops) per decision.
- Spiking net: **SynOps** (synaptic operations — one per spike that travels down a synapse) plus the
  input-current multiplications. We report **energy as a *range*** across published assumptions for
  how much cheaper a spike-op is than a MAC (roughly 5–30×), because the exact ratio is
  hardware-specific. It is a transparent **proxy, not measured joules**.

---

## 4. How we keep it honest (experimental design)

**Train/test separation.** A model that is tested on the same data it learned from can cheat by
memorizing. We always test on *held-out* data.

**The leakage trap (subjects).** If the *same person* appears in both training and test data, the
model can memorize that person's quirks and look better than it is — fake performance. Fix:
**group-by-subject cross-validation.** We split the 108 people into 5 groups; train on 4 groups,
test on the 5th, rotate 5 times so every person is tested exactly once, always by a model that never
saw them. This is "**5-fold subject-grouped cross-validation**," a tractable form of
leave-subjects-out.

**Standardize on training only.** We rescale features (subtract mean, divide by SD) using statistics
computed *only on the training fold*, then apply them to the test fold — otherwise the test data
would leak into preprocessing.

**Exclude reaction time.** RT happens *during/after* the decision, so using it to predict the
decision is circular leakage. We never feed it in.

---

## 5. The statistics, properly

A single number ("AUC = 0.892") is meaningless without **how uncertain it is** and **whether a
difference is real or luck**. Two tools do this without strong assumptions.

**(i) The bootstrap → confidence intervals.**
We can't re-run the experiment 1,000 times, so we *simulate* re-running it by **resampling our own
data with replacement**: draw a new dataset the same size by picking rows at random (some appear
twice, some not at all), recompute AUC, repeat 2,000 times. The middle 95% of those 2,000 AUCs is
the **95% confidence interval (CI)** — the plausible range for the true AUC. Wide CI = uncertain;
narrow CI = precise.

**Cluster (subject) bootstrap — avoiding pseudoreplication.** Trials from the same person are
correlated (they're not 27,454 independent facts; they're 108 people × ~250 trials). If we resampled
individual trials we'd *pretend* to have more independent information than we do and get CIs that are
too narrow (overconfident). Fix: **resample whole *subjects*** (with replacement), keeping each
chosen subject's trials together. This respects that the real unit of replication is the person.

**(ii) The permutation test → p-values.**
Is an AUC of 0.89 better than chance, or a fluke? We build the **null distribution** (what we'd see
if the model had *no* real signal) by **shuffling the choice labels** so any link between model score
and choice is destroyed, then recomputing the statistic — thousands of times. The **p-value** is how
often the shuffled (null) result is as extreme as the real one. Tiny p = the real result is very
unlikely to be luck.

**(iii) Paired vs. marginal comparison — the critic's key fix.**
To ask "is model A better than model B?", the *wrong* way is to check whether their two separate
confidence intervals overlap. Two models evaluated on the **same trials** rise and fall together
(an easy person is easy for both), so comparing their independent CIs throws away that linkage and
can hide a real difference. The *right* way is a **paired** test: for each bootstrap resample,
compute **ΔAUC = AUC(A) − AUC(B) on the *same* resampled data**, and build a CI for the *difference*.
If that ΔAUC CI excludes 0, the difference is real. This is exactly why, after switching to the
paired test, we found the spiking model is *slightly significantly above* prospect theory (ΔAUC
+0.0046 [+0.0025, +0.0066]) rather than merely "matching" — a nuance the overlap method had hidden.

**(iv) Effect size vs. significance.** With 27,454 trials, *tiny* differences become "statistically
significant." +0.005 AUC is **significant but practically negligible** — detectable, but it doesn't
mean the spiking model is meaningfully better. We report both and say so. Conflating "significant"
with "important" is a classic error; we avoid it.

**(v) Preregistration & kill criteria.** *Before* modeling we wrote down what would count as success
or failure (the "kill criteria") so we couldn't move the goalposts after seeing results (a form of
p-hacking). E.g., KC1: "the energy claim is only allowed if the spiking net is iso-accurate." KC2:
"only claim lower energy if it holds across the *entire* assumed energy range." We then reported KC2
and KC3 as **FAIL** honestly.

**(vi) Multiple comparisons (FDR).** When you run many tests, some will look "significant" by chance.
Benjamini-Hochberg FDR controls that. Study 2's model-vs-model comparisons are not part of the
project's confirmatory hypothesis family, so FDR isn't applied to them — and we state that explicitly
rather than hide it.

---

## 6. The code, line by line

Three files do the work. (`spikeprint/` is the package; `scripts/` runs the study.)

### 6a. `spikeprint/decision_models.py` — the two neural models

```python
class RateMLP(nn.Module):
    def __init__(self, hidden: int = 16):
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
```
A plain network: `Linear(2, hidden)` is a learnable matrix mapping the 2 inputs to `hidden` neurons;
`ReLU` is the nonlinearity (keeps positives, zeros negatives); `Linear(hidden, 1)` collapses to one
output number (a "logit" — a pre-probability).
```python
    def forward(self, x):
        return self.net(x).squeeze(-1)        # logit per trial
    def macs_per_decision(self):
        return 2 * self.hidden + self.hidden  # input→hidden MACs + hidden→out MACs
```
`forward` is one prediction pass. `macs_per_decision` counts its multiply-accumulates: `2*hidden`
for the first layer (2 inputs × hidden) plus `hidden` for the readout — its energy unit.

```python
class SpikingMLP(nn.Module):
    def __init__(self, hidden=16, t_steps=20, beta=0.9):
        self.fc1  = nn.Linear(2, hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc2  = nn.Linear(hidden, 1)
```
`fc1` injects input current into `hidden` LIF neurons. `snn.Leaky` is the LIF layer: `beta=0.9` is
the membrane "leak" (each step it keeps 90% of its charge); `spike_grad=fast_sigmoid()` is the
**surrogate gradient** that makes the spike trainable. `fc2` reads the hidden activity out to one
logit. `t_steps=20` = we simulate 20 timesteps per decision.
```python
    def forward(self, x):
        mem1 = self.lif1.init_leaky()                 # start membrane at rest
        spk_sum = torch.zeros(x.shape[0], self.hidden)
        spike_count = torch.zeros(())
        for _ in range(self.t_steps):                 # simulate time
            spk1, mem1 = self.lif1(self.fc1(x), mem1) # inject current → maybe spike, update membrane
            spk_sum = spk_sum + spk1                   # accumulate spikes (the neural "rate")
            spike_count = spike_count + spk1.sum()     # tally spikes (for energy)
        logit = self.fc2(spk_sum / self.t_steps).squeeze(-1)  # read out the firing RATE
        return logit, spike_count
```
This is the heart of the SNN. Each timestep: `fc1(x)` is the input current; `lif1` integrates it into
the membrane `mem1` and returns a spike `spk1` (0/1 per neuron) if threshold is crossed. We **sum
spikes over time** (`spk_sum`) — that running spike-rate is the neuron's "answer" — and separately
**count total spikes** for the energy proxy. After 20 steps we divide by `t_steps` to get the average
firing rate and pass it through `fc2` to get the choice logit. The model is differentiable end-to-end
*because* of the surrogate gradient, so it trains like any net.
```python
    def input_macs_per_decision(self):
        return 2 * self.hidden * self.t_steps   # current injected EVERY timestep → ×T
```
The honest catch: because we inject current every timestep, the input layer costs `2*hidden*T` MACs —
**T times more than the rate net's input layer**. This is precisely why the SNN loses on energy at
this scale (Section 7).

```python
def train_model(model, x, y, epochs=60, lr=0.01, seed=0, spiking=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logit = model(xt)[0] if spiking else model(xt)   # SNN returns (logit, spikes)
        lossf(logit, yt).backward()                       # compute gradients (surrogate for SNN)
        opt.step()                                        # nudge weights to reduce error
```
Standard training: `BCEWithLogitsLoss` is the right loss for yes/no targets; `Adam` is the optimizer;
each epoch we compute the error, backpropagate, and update weights. The *only* SNN-specific line is
unpacking `(logit, spikes)`. We use the **same epochs/optimizer/size** for both nets so the
comparison is fair.

```python
@torch.no_grad()
def predict_proba(model, x, spiking=False):
    if spiking:
        logit, spikes = model(xt)
        return torch.sigmoid(logit).numpy(), float(spikes) / xt.shape[0]   # probs, spikes/decision
    return torch.sigmoid(model(xt)).numpy(), None
```
`sigmoid` turns a logit into a probability in [0,1]. For the SNN we also return **spikes per
decision** (total spikes ÷ number of trials) — the sparsity/energy measurement. `@torch.no_grad()`
means "don't track gradients" (we're predicting, not learning).

### 6b. `spikeprint/analysis.py` — the statistics engine

```python
def auc(scores, labels):
    r = _rankdata(s)                              # rank every score (ties get average ranks)
    u = r[y == 1].sum() - n_pos*(n_pos+1)/2.0     # Mann-Whitney U for the positive class
    return float(u / (n_pos * n_neg))             # normalize → AUC in [0,1]
```
AUC computed exactly, via ranks: rank all scores, sum the ranks of the "accept" trials, convert to
the U statistic, normalize by the number of accept×reject pairs. A constant predictor → all ties →
AUC exactly 0.5 (the honest "no information" value).

```python
def cluster_bootstrap_ci(scores, labels, groups, n_boot=2000, alpha=0.05, seed=0):
    rows = [np.nonzero(codes == k)[0] for k in range(n_groups)]   # trial indices per subject
    for _ in range(n_boot):
        chosen = rng.integers(0, n_groups, n_groups)              # resample SUBJECTS w/ replacement
        idx = np.concatenate([rows[k] for k in chosen])           # gather their trials
        boots.append(auc(s[idx], y[idx]))                          # recompute AUC
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])             # middle 95%
```
The **subject** bootstrap from Section 5(i): resample *people*, not trials, recompute AUC 2,000
times, take the 2.5th–97.5th percentiles as the 95% CI. (Degenerate resamples with only one choice
class are skipped and counted — surfaced on the Finding so they can't silently bias the CI.)

```python
def permutation_pvalue(scores, labels, observed, n_perm, seed):
    for _ in range(n_perm):
        perm = rng.permutation(y)            # shuffle the choice labels (destroy real signal)
        if abs(auc(s, perm) - 0.5) >= eff:   # is the NULL result as extreme as ours?
            count += 1
    return (count + 1) / (used + 1)          # p-value (Phipson–Smyth correction)
```
The permutation test from Section 5(ii): shuffle labels to simulate "no signal," see how often pure
chance reaches our effect, report that fraction as the p-value.

```python
def predictive_validity(name, dataset, scores, labels, groups, ...):
    a  = auc(scores, labels)
    ci = cluster_bootstrap_ci(scores, labels, groups, ...)
    p  = permutation_pvalue(scores, labels, a, ...)
    return Finding(name, value=a, metric="AUC", baseline=0.5, ci95=ci, n=..., p_value=p, null=0.5).decide()
```
This bundles a model's result into a **`Finding`** — our rule that *no number ships without its
evidence*: the value, its 95% CI, its p-value, its sample size, and the null it's tested against
(0.5 for AUC). `.decide()` stamps it VALIDATED only if the CI excludes the null.

```python
def incremental_validity(name, dataset, csi_scores, baseline_scores, labels, groups, ...):
    d = auc(csi, y) - auc(base, y)                  # observed ΔAUC (model A − model B)
    for _ in range(n_boot):
        idx = resample_subjects()
        boots.append(auc(csi[idx], y[idx]) - auc(base[idx], y[idx]))   # ΔAUC on the SAME resample
    # one-sided pass: A beats B only if the whole CI is above 0
    return replace(f, passed=bool(f.ci95[0] > 0.0))
```
This is the **paired** comparison from Section 5(iii): the difference is recomputed on the *same*
resampled subjects each time, giving a CI for ΔAUC directly. This is the function we used to compare
spiking vs. prospect-theory and GBT vs. prospect-theory.

### 6c. `scripts/run_spiking_decision_narps.py` — the experiment

```python
d = load_narps_events("data/raw/ds001734")
gain, loss, y, subj = d["gain"], d["loss"], d["choice"], d["subject"]
x  = np.column_stack([gain, loss]).astype(float)   # model inputs
ev = (0.5*gain - 0.5*loss).reshape(-1, 1)          # EV feature
```
Load the real choices; build the feature matrix (gain, loss) and the EV feature.

```python
for tr, te in GroupKFold(n_splits=5).split(x, y, groups=subj):   # 5 subject-grouped folds
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-8                   # standardize on TRAIN only
    xtr, xte = (x[tr]-mu)/sd, (x[te]-mu)/sd
    preds["EV"][te]       = LogisticRegression().fit(etr, y[tr]).predict_proba(ete)[:,1]
    preds["Prospect"][te] = LogisticRegression().fit(xtr, y[tr]).predict_proba(xte)[:,1]
    preds["GBT"][te]      = GradientBoostingClassifier().fit(x[tr], y[tr]).predict_proba(x[te])[:,1]
    preds["RateMLP"][te]  = predict_proba(train_model(RateMLP(16), xtr, y[tr]), xte)[0]
    spk = train_model(SpikingMLP(16,20), xtr, y[tr], spiking=True)
    preds["Spiking"][te], spd = predict_proba(spk, xte, spiking=True)
```
The core loop. `GroupKFold(...split(..., groups=subj))` guarantees no subject is in both train and
test. Inside each fold we **fit every model on the training people and predict the held-out people**,
storing each test prediction in its slot (`[te]`). EV uses the single EV feature; Prospect uses
(gain, loss) separately so it can learn loss aversion; GBT uses raw features (trees don't need
scaling); the two nets train on standardized features. After all 5 folds, every trial has exactly one
**out-of-fold** prediction from each model.

```python
findings = {k: predictive_validity(f"Study2 {k}", "NARPS", preds[k], y, groups=subj) for k in MODELS}
d_sp = incremental_validity("dAUC Spiking-Prospect", "NARPS", preds["Spiking"], preds["Prospect"], y, groups=subj)
d_gp = incremental_validity("dAUC GBT-Prospect",      "NARPS", preds["GBT"],     preds["Prospect"], y, groups=subj)
```
Turn each model's pooled predictions into a `Finding` (AUC + subject CI + p), and run the two
**paired** comparisons (spiking vs. prospect, GBT vs. prospect).

```python
coef = LogisticRegression().fit(x, y).coef_[0]
lam  = -coef[1] / coef[0]                 # loss aversion λ = −β_loss / β_gain
```
Loss aversion read straight off the prospect-theory logistic's two coefficients → **λ = 1.69**.
(Labeled "in-sample descriptive" because it's fit on all data, not cross-validated.)

```python
rate_macs      = 2*HIDDEN + HIDDEN          # 48
snn_input_macs = 2*HIDDEN*T_STEPS           # 640  (current injected every timestep)
snn_synops     = mean(spikes_per_dec)       # ~99 spikes/decision
energy_ratio   = [(snn_input_macs + snn_synops*r)/rate_macs for r in (1/30, 1/5)]  # E_snn / E_rate
```
The honest energy accounting: the SNN's per-decision cost is `640` input-MACs + `~99` SynOps versus
the rate net's `48` MACs, giving an energy ratio of **~13×** across the assumed spike:MAC cost range.

```python
matches = d_sp.ci95[0] <= 0 <= d_sp.ci95[1]   # does the spiking−prospect difference include 0?
kc1 = d_sp.ci95[1] >= 0                        # spiking not significantly worse than prospect
kc2 = e_hi < 1.0                               # SNN cheaper across the WHOLE range?
kc3 = d_gp.ci95[0] <= 0                         # GBT does NOT significantly beat prospect?
```
The preregistered kill criteria, evaluated from the paired CIs — not from eyeballing overlap.

---

## 7. The results, decoded

| Model | held-out AUC | reading |
|---|---|---|
| EV | 0.882 | a rational-robot baseline already predicts human choice well |
| Prospect-theory | 0.888 | adding loss aversion (λ=1.69) helps a little |
| Rate MLP | 0.893 | a small net ≈ the theory |
| **Spiking LIF** | **0.892** | the brain-style model ≈ the theory |
| GBT | 0.899 | a flexible model is the best, but only barely |

- **Spiking − Prospect ΔAUC = +0.0046 [+0.0025, +0.0066].** The CI is just above 0, so the spiking
  net is *statistically* a hair better — but +0.005 AUC is **practically nothing**. Honest reading:
  **the spiking model performs on par with the canonical economic theory.** No accuracy win claimed.
- **GBT − Prospect ΔAUC = +0.0107 [+0.0049, +0.0158].** A flexible model beats the linear theory by
  ~0.011 AUC — a *small but real* nonlinear residual. So "the ceiling is purely the data" is **not**
  quite true (KC3 fails, reported); rather, prospect theory captures *nearly* all the predictable
  structure, with a sliver left over.
- **λ = 1.69:** losses weigh ~1.7× gains — a clean, literature-consistent measurement of human loss
  aversion, recovered straight from the data (pipeline sanity check).
- **Energy: ~13× *more* for the spiking net (KC2 fails).** Because we inject input current every one
  of 20 timesteps, the input layer alone costs 20× a normal net's. At 2 inputs there's no sparsity to
  exploit. Honest conclusion: **neuromorphic energy advantages are a property of large, sparse
  workloads — not a 2-number task.** We say so plainly instead of hiding it.

---

## 8. Why this matters / what you can now say

- **A brain-inspired spiking network reproduces the canonical prospect-theory account of human risky
  choice on public data** — a clean demonstration that neuromorphic models can do real
  decision-science, stated without over-claiming.
- **The accuracy ceiling is essentially the data:** from gain & loss alone, prospect theory is
  near-optimal; flexibility buys ~0.01 AUC. That's a genuine scientific statement about how much of
  human risky choice is captured by the textbook model.
- **Honest nulls are the headline, not an embarrassment:** no fake accuracy edge, no fake energy
  win. That credibility is the asset.

## 9. The meta-lesson: the critic loop

The result is shaped as much by *what we refused to claim* as by what we found. A standing reviewer
panel changed the science twice: at **design** it reframed the goal from "more accurate" (impossible
— EV is already at ~0.88) to **iso-accuracy / lower-energy**, and swapped a pointless Transformer for
the GBT ceiling; at **results** it forced the **paired ΔAUC** test (which revealed the spiking net is
*slightly above*, not merely "matching") and made the energy claim **scoped** to this encoding/scale.
Every number that left the room carried a confidence interval and a preregistered pass/fail rule.
That discipline — not any single model — is the real instrument here.
