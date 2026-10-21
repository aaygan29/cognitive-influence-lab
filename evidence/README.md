# Evidence

External literature, kept separate from `experiments/` because it is other people's work, not this
tree's. Nothing in here was run here.

Each node answers one question the chain depends on but this tree cannot settle by simulation:
**is there a real, manipulable parameter in a real human that would make a link of the chain work?**

| Node | The proposed parameter | Link | Verdict |
|---|---|---|---|
| [`lc-ne`](lc-ne) | Noradrenergic gain — the knob `bounded-influence` uses to set barrier height, and the arousal covariate it uses as a specificity control | 2, 3 | **Much weaker than its reputation.** Continuous taVNS does not move the field's own mechanism markers (pupil g=0.002, BF01=21.9 over 771 participants); the only completed double-masked trial in MCI missed its primary endpoint; the direction of effect flips with stimulation duration; and the standard LC measure is in worse shape than the effect. |
| [`psilocybin-precision`](psilocybin-precision) | Prior precision / context binding — the quantity link 2 is blocked on | 2 | **Suggestive, not load-bearing.** A real controlled manipulation with a specific behavioural correlate (ρ=0.455, and ρ=0.008 against the control scale), but the mechanism is reverse inference, the central paper is a preprint, and the key association is between-person. |

## The rule

Every citation in this tree passes through [`corpus/references.json`](../corpus/references.json),
which records a resolvable identifier, the URL it was fetched from, and the date. **Only entries
marked `verified` may support a claim.** Entries that failed their check are kept with the reason
they failed, so a bad citation cannot quietly return; `preprint` entries resolve but are unreviewed
and are marked at every use. `python3 corpus/validate_corpus.py` enforces this.

## What both nodes have in common

Both arrived as a plausible mechanism story with a strong reputation, and both came back weaker than
advertised once the citations were actually fetched. Both also surfaced the same structural problem
rather than a merely smaller effect:

- `lc-ne`'s central measure may not measure what its name says, and seven published atlases of the
  structure overlap at Dice 0.2–0.4.
- `psilocybin-precision`'s central finding is between-person, and its own authors flag that it may
  not hold within individuals.

That second one is this tree's link-2 problem appearing in someone else's data, which is a useful
thing to be able to point at: the population-to-individual gap is not an artefact of how this
program measures things.
