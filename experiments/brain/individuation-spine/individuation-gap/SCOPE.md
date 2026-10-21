# What the individuation-gap proof does and does not establish

Anonymous. 2026-10-20. Read this before citing `proof/IndividuationGap.lean` or the manuscripts in
this directory. It exists because this node is the one place in the tree where a **formal result and
an empirical correction sit next to each other and could easily be read as agreeing more than they
do.**

## Where this came from

Folded in 2026-10-20 from the `sapient` repo (`paper/individuation-gap/`), by `git subtree` with
history preserved and author identities scrubbed. The tree's README had recorded an open question —
the parent description promised a "Lean individuation-gap proof" that was not in the spine, and
guessed it meant `neural-manifold-topology-attribution` or the well-posedness proofs in
`individuation-mcmc` / `learning-rule-fusion`. **Both guesses were wrong.** The proof was in
`sapient` all along. The open question is closed; the README now points here.

Author names in `paper/main.tex` and `tmlr/OPENREVIEW_FIELDS.md` are left intact, matching how
`experiments/influence/individuation-mcmc/paper/` is handled: a submission's author block is not a
stray personal string.

## What the theorem actually says

With `rhoSq a v = a / (a + v)` — best-case squared correlation for a predictor with identifiable
between-unit signal variance `a` against a target with idiosyncratic noise variance `v`:

| Theorem | Statement |
|---|---|
| `aggregation_improves` | for `N > 1`, `rhoSq a v < rhoSq a (v/N)` — aggregating over units strictly improves the population term |
| `unseen_individual_is_zero` | `rhoSq 0 v = 0` |
| `unseen_individual_zero_any_N` | `rhoSq 0 (v/N) = 0` — pooling more data cannot manufacture a signal the predictor never had |
| `individuation_gap` | for `a > 0, v > 0, N > 0`, `rhoSq 0 (v/N) < rhoSq a (v/N)` |

Complete, no `sorry`. Build infrastructure (`lakefile.toml`, `lean-toolchain` at Lean v4.34.0) was
added on fold-in, because it had none in the source repo. **It has since been built and audited**
(2026-10-20, Lean 4.34.0 / Mathlib v4.34.0): it compiles clean, and `individuation_gap` and
`aggregation_improves` depend on exactly `[propext, Classical.choice, Quot.sound]` with no
`sorryAx`. See [`VERIFICATION.md`](../../../../VERIFICATION.md).

## The scope limit, stated plainly

**The proof is elementary arithmetic, and it assumes the thing that does the work.** Its content is:
*if* a predictor has no between-unit variance (`a = 0`), *then* its best-case individual correlation
is zero. That is `0/(0+v) = 0`. The file's own docstring says as much — "the arithmetic is
elementary" — and frames the result as "a theorem about the estimand, not a limitation of a
particular model or dataset."

That framing is correct and it is also the whole limitation. The theorem is a **definitional
separation between two estimands**, not an impossibility result about individuation. Everything
substantive lives in the premise `a = 0`, which is an empirical claim about a particular class of
predictor, and which the theorem does not establish.

## How it sits with this tree's own correction

The manuscripts here are written around the claim that AI **cannot read what an individual is
thinking**. This tree has already retracted exactly that strength of claim. `README.md` claim 1 and
`rigor/RIGOR_FINDINGS.md` record that the human side is **undetermined rather than dear**: the
headline null (0/25 ROIs) does not survive its own power analysis, MDES ≈ 0.59 against an observed
0.175.

These are not in conflict once the scope is stated, and the distinction is worth being precise about
because it is the most citable thing in this node:

- **The theorem bounds a zero-shot predictor** — one that emits the same population feature for
  every unseen unit, so `a = 0` by construction. For that object the individual term is exactly zero,
  and no amount of data changes it. This is solid and unsurprising.
- **The empirical question is whether a fitted per-subject encoder has `a > 0`**, and by how much.
  That is what `digital-brain` and `wiring-not-weights/exp04` actually tested, and the answer at
  available power is **undetermined**. The theorem says nothing about this case, because a fitted
  encoder is not the `a = 0` object.

So: cite the proof for *zero-shot* individuation. Do not cite it for individuation in general, and
do not let the manuscripts' stronger abstract phrasing re-import a claim the program has already
corrected. If these papers go out, the abstract needs reconciling with the power correction first.

## Open items carried in with this fold

- The manuscripts' abstract claim is stronger than the tree allows (above). **Unresolved** — it is a
  writing decision for whoever submits.
- `README.md` in this directory notes that real-data numbers "should be regenerated from a fixed
  commit before camera-ready". They are **not currently pinned to a commit**.
- ~~The proof has not been built in this tree.~~ **Closed 2026-10-20**: built and axiom-audited. The
  *scope* limit above is unaffected and still applies — building a proof says nothing about whether
  its premise holds of real encoders.
