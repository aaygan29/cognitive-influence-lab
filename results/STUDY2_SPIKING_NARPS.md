# Study 2 — Spiking vs prospect-theory on NARPS (iso-accuracy, lower-energy)

**Run:** 2026-06-09 · n_trials = 27454 · subjects = 108 ·
5-fold subject-grouped CV · **in-sample descriptive** loss-aversion lambda = 1.69 (gain/loss
logistic; not cross-validated). Behavioral only; RT excluded; standardized on train folds. SNN:
16 LIF hidden, T=20, current/direct encoding, surrogate gradient. (Study 2 is a model
comparison — outside the registered H1–H4 FDR family.)

## Held-out performance (subject-clustered bootstrap CI)
| Model | AUC | 95% CI | balanced acc |
|---|---|---|---|
| EV | 0.882 | [0.855, 0.907] | 0.804 |
| Prospect | 0.888 | [0.864, 0.910] | 0.811 |
| GBT | 0.899 | [0.874, 0.920] | 0.817 |
| RateMLP | 0.892 | [0.868, 0.915] | 0.810 |
| Spiking | 0.892 | [0.868, 0.916] | 0.811 |

## Paired contrasts (subject-clustered ΔAUC — not marginal-CI overlap)
- **Spiking − Prospect ΔAUC = +0.0046, 95% CI [+0.0025, +0.0066]**
  → CI > 0 -> spiking slightly exceeds prospect.
- **GBT − Prospect ΔAUC = +0.0107, 95% CI [+0.0049, +0.0158]**
  → significantly exceeds prospect -> residual structure (gain, loss) models miss.

## Energy (proxy, not joules) — current/direct-coded SNN
rate MLP = 48 MACs/decision; SNN = 640 input MACs/decision +
99.4 SynOps/decision. **E_snn / E_rate ∈ [13.4, 13.7]** across AC:MAC
energy-per-op of 1/30..1/5.

## Kill criteria (preregistered)
- **KC1 iso-accuracy** (spiking not worse than prospect): **PASS**
- **KC2 lower-energy** (E_snn < E_rate across the whole range): **FAIL**
- **KC3 ceiling-is-data** (GBT does not beat prospect): **FAIL**

## Honest verdict
Spiking AUC = 0.892 vs prospect-theory 0.888;
paired ΔAUC = +0.0046 [+0.0025, +0.0066] (CI > 0 -> spiking slightly exceeds prospect). The gap is
statistically detectable at n=27454 but **practically negligible (~0.005 AUC) — the spiking model
performs on par with the canonical prospect-theory account; no meaningful accuracy advantage is
claimed.** The GBT ceiling significantly exceeds prospect -> residual structure (gain, loss) models miss (+0.0107 AUC), so linear prospect-theory already
captures nearly all the predictable structure, with only a small nonlinear residual. On energy: for
a **current/direct-coded SNN at this 2-feature scale**, input current injection costs T× MACs, so it
uses ~13× more energy than the rate MLP — **no energy advantage here**; a sparse rate-coded
SNN was not evaluated and at 2 inputs is unlikely to reverse this. Neuromorphic energy benefits are
a property of large, sparse workloads.
