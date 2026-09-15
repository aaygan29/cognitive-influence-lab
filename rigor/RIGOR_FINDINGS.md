# Stress-test re-analysis of the program's real results

2026-08-16. Re-analysis of committed result JSONs with the statistics the
original framing skipped. Run `reanalysis.py` (Rungs 1-2 + synthetic) and the digital-brain FDR
block to reproduce. No em dashes.

Purpose: apply the tests that decide whether each headline claim survives, and register the
confounds. Findings are ordered by how much they change the story.

---

## F1 (biggest). The human reconstruction result is UNDETERMINED, not "dear"
Two projects measure human stimulus-evoked individuation and reach opposite headlines. Both are
underpowered; **neither survives correction**.

| Source | N | chance | headline | after stress test |
|---|---|---|---|---|
| `digital-brain` (amplitude fingerprint) | 4 | 25% | 21/25 ROIs significant, "N=8 -> p<5e-8" | **0/25 survive BH-FDR or Bonferroni.** p-values cluster at 0.03-0.048 (just under .05) across 25 tests with only 4 trials/ROI = textbook uncorrected small-N over-read. The projected p<5e-8 is asserted, never measured. |
| `wiring-not-weights/exp04` (functional axis) | 8 | 12.5% | 0/25 ROIs significant ("reconstruction is dear") | Per-ROI needs >=4/8 correct (acc>=0.50) for p<.05; **MDES for 80% power ~ 0.59 accuracy**. Observed 0.175. The null is near-guaranteed by power, not principle. A naive pool gives z=2.14 (p=.016) but violates independence (shared 8 subjects), so it cannot rescue the claim either. |

**Verdict:** human reconstruction-individuation is currently **undetermined**. The program should
present this as a powered open question with a stated MDES, not as an established "reconstruction
is dear" result. This is the single most important correction, and it makes the program's honesty
discipline (it caught its own over-reads) the real contribution.

## F2. "Identification is cheap" holds, but via the RIGHT test, and with a live confound
- The committed `perm_p = 5e-4` is each-arm-vs-chance (the 1/2001 permutation floor). It does
  **not** test binarized vs weighted. Citing it for the gap is a category error.
- Correct paired test: full 243/248 vs binarized 241/248 = a **2-subject** net difference.
  Max-possible McNemar significance is **p >= 0.50** -> weighted is NOT significantly better.
- Correct equivalence test: **TOST vs a 5% margin, p = 0.0012 -> statistically equivalent**
  (95% CI on the gap [-1.9%, +3.5%], includes 0). So "binarized ~ weighted" is real, but it is an
  *equivalence* claim and must be made with an equivalence test, which was missing.
- **Confound (open):** the split is **within-session**, and node-strength alone identifies at
  44.6% (chance 0.4%). Within-session identification can index session-specific SNR/motion, not
  the stable individual. Needs a **between-session** replication before "structural identity" is
  safe. Report node-strength/motion as a nuisance-regressor control.

## F3. Synthetic effect sizes are being reported as empirical ones
- The within-class degeneracy is mean 1.000, **sd 0.000** -> Cohen's d ("d=33") is an artifact of
  dividing by ~0 in a noiseless simulation. Report as "perfect separation in simulation," never as
  an effect size that could transfer to data.
- `alpha*` inconsistency: seed-robust `exp06` gives **0.642 (sd .008)**; `THESIS_ARC`/`PROGRAM_ARC`
  cite ~0.8. Use 0.64 and cite exp06.
- These do not weaken the synthetic existence proof (the dissociation is real in simulation); they
  fix how it is reported.

## F4. The AI-side result is the cleanest in the program
`neuro-ai-pipeline`: imposing brain-like topology on transformers hurts, d = -1.20 over 120 runs
(20 seeds), one-tailed p = 0.9997 for the "helps" direction. This is a genuine, seed-stable,
capacity-matched negative. It is the strongest empirical claim after F2 and should anchor the
AI half of the synthesis.

---

## What this implies for the claims
- **Keep and strengthen:** identification-is-cheap (as a TOST equivalence + between-session
  replication); the synthetic functional-equivalence-class existence proof (with corrected
  numbers); the AI topology negative.
- **Downgrade to "open, powered":** human reconstruction individuation. Merge digital-brain (N=4)
  and wiring-not-weights (N=8) into a single, honestly-underpowered "bracket," not two rungs.
- **The methodological contribution is now front-and-center:** a control-first, power-and-confound-
  audited, conformal-gated discipline that catches its own false positives (BOLD5000, N=8, and now
  the N=4 FDR failure). That is publishable in itself.
