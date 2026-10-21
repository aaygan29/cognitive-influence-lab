# Machine-checked results in this tree

Anonymous. What has actually been checked by Lean, when, and against what. A proof nobody
has built is a document, not a verification, so this file separates the two.

## Checked on 2026-10-20

Toolchain: **Lean 4.34.0** (`leanprover/lean4:v4.34.0`), **Mathlib `v4.34.0`**, arm64-darwin.
Each file below compiled with no errors and no `sorry`, and every theorem listed was put
through `#print axioms`.

**Every one depends on exactly `[propext, Classical.choice, Quot.sound]`** — the three
standard axioms of Lean's logic — and on nothing else. In particular **`sorryAx` appears
nowhere**, which is the mechanical check that no proof was stubbed out.

| File | Theorem | Axioms |
|---|---|---|
| [`bounded-influence/proofs/BarrierFloor.lean`](experiments/influence/bounded-influence/proofs/BarrierFloor.lean) | `barrier_floor_bounds_crossing` | standard three |
| | `dose_cap_cannot_bound` | standard three |
| | `rate_le_of_barrier_ge` | standard three |
| [`learner-tampering/proofs/DetectionFloor.lean`](experiments/influence/learner-tampering/proofs/DetectionFloor.lean) | `power_le_of_likelihood_ratio_le` | standard three |
| | `detection_floor` | standard three |
| | `no_detector_is_certain` | standard three |
| [`individuation-mcmc/proofs/IndividuationAdvantage.lean`](experiments/influence/individuation-mcmc/proofs/IndividuationAdvantage.lean) | `popCost_decomposition` | standard three |
| | `popCost_ge_spread` | standard three |
| | `advantage_zero_iff_homogeneous` | standard three |
| | `advantage_pos_of_differ` | standard three |
| [`individuation-spine/individuation-gap/proof/IndividuationGap.lean`](experiments/brain/individuation-spine/individuation-gap/proof/IndividuationGap.lean) | `individuation_gap` | standard three |
| | `aggregation_improves` | standard three |

The individuation-gap proof had **never been built** in this tree — it arrived from
`sapient` with no `lakefile.toml` and no `lean-toolchain`, and `SCOPE.md` recorded that as
an open item. It has now been built and audited. The open item is closed; the *scope*
caveat in `SCOPE.md` is not, and still applies.

## What was added on 2026-10-20, and why

Three claims in this tree were theorem-shaped but held only by simulation. Each is now a
theorem about the **inference**, taking the measured quantities as hypotheses — the same
pattern `persuasion-control/proofs/PersuasionGap.lean` already used.

**`BarrierFloor.lean` — the defence that works, and the one that doesn't.**
`bounded-influence`'s headline result is that a cumulative-dose cap is the wrong defence
and a barrier floor is the right one. The first is now a two-line proof that a cap
admitting the measured exploit (dose 0.001, crossing probability 1.00) bounds nothing, plus
a proof that crossing probability is not monotone in dose, so no threshold on dose can
separate safe from unsafe. The second derives the node's own floor formula
`u ≥ D·log(kT / −log(1−ε)) ⟹ P(cross) ≤ ε` from the Kramers rate.

**`DetectionFloor.lean` — the floor is a bound on every detector, not one detector.**
`learner-tampering` reports that attacks shorter than `⌈log₂(1/α)⌉` blocks are undetectable
"by any test of this form". The general bound behind it is the elementary half of
Neyman–Pearson: `power ≤ L·α` whenever the likelihood ratio is bounded by `L`. Putting
`L = 2^k` gives the floor. The file also proves the trap in the obvious defence: one extra
block of attack exactly cancels a halving of the false-alarm rate.

**`IndividuationAdvantage.lean` — why the chain's middle link is the dangerous one.**
The README argues that identification buys an attacker nothing and individuation buys them
everything. That is now exact: an attacker restricted to one population dose pays
`∑(dᵢ − c)² = ∑(dᵢ − mean)² + n(mean − c)²`, so the irreducible cost is precisely the
spread of the per-person optimum, zero exactly when everyone is identical. Identity does
not appear in it.

## Standing caveat: these verify inferences, not neuroscience

Every theorem here takes the empirical content as a hypothesis. `barrier_floor_bounds_crossing`
assumes Kramers escape over a barrier that never dips; `detection_floor` assumes the
likelihood-ratio bound; `IndividuationAdvantage` assumes squared-error cost and that
per-person doses vary at all. Whether those hypotheses hold of **real people** is measured
elsewhere in this tree, often with the answer *undetermined*. A machine-checked proof of a
conditional is not evidence for its antecedent.

One discrepancy the proofs expose rather than resolve: E4 reports crossing probabilities of
0.06–0.115 against a target of ε = 0.05 **with the floor enforced**. `barrier_floor_bounds_crossing`
says that cannot happen if the barrier truly never drops below `u`. So either the enforced
strategies dip transiently, or the constant-rate approximation under-predicts here. That is
now a checkable question about two named hypotheses rather than a loose end.

## Not yet built in this tree

These shipped with their source repos and have not been rebuilt here. They are documents
until someone runs them:

- `warden/verification/cognitive_virus/lean/` — 26 theorems with their own
  `axioms_output.txt` from the source repo, including a `#print axioms` audit. Carries its
  own build files; not re-run here.
- `learning-rule-fusion/proofs/` — `REPRODUCE.md` records a successful build on 2026-09-28
  (Lean v4.34.1, 8927 jobs, exit 0, no `sorry`). That is the author's record, not this one.
- `decision-phenotype/proofs/`, `behavioral_decoding/proofs/`,
  `persuasion-control/proofs/`, `manifold-topology/archive/.../CliqueBetti.lean`.

## Reproducing this

```bash
curl -fsSL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y
cd experiments/influence/bounded-influence/proofs   # or any other proofs/ directory
lake exe cache get      # mathlib oleans, about 6 GB
lake build
```

To re-run the axiom audit, `#print axioms <theorem>` on each name in the table above and
check that the output is `[propext, Classical.choice, Quot.sound]` with no `sorryAx`.
