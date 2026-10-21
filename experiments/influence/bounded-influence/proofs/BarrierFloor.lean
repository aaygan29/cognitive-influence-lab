/-
  Why a cumulative-dose cap is the wrong defence and a barrier floor is the right one.

  This file verifies the *inference*, not the physics. The bistable belief model, the
  fitted Kramers constant `k = 0.190` and the measured exploit are all empirical and
  live in `PROVENANCE.md` / `experiments/e4_dose_budget.py`. Here we prove that, GIVEN
  those measurements, the two headline conclusions of E4 follow necessarily:

    * `dose_cap_cannot_bound` — a cap on cumulative dose that admits the measured
      exploit strategy provides no bound at all on crossing probability. The exploit
      parks just below the fold: dose 0.001, crossing probability 1.00.

    * `barrier_floor_bounds_crossing` — a floor on barrier height does bound it, and
      the sufficient floor is exactly the formula E4 uses:
          u ≥ D * log (k * T / (-log (1 - ε)))  ⟹  crossing probability ≤ ε.
      At D = 0.03, k = 0.190, T = 120, ε = 0.05 this gives u ≥ 0.1829, the value in
      `e4_results.json`.

  The asymmetry between the two is the point. Dose is a property of what the attacker
  spends; barrier height is a property of how exposed the believer is. Only the second
  appears in the escape rate, so only the second can bound the outcome.
-/
import Mathlib

namespace BoundedInfluence

/-! ## The model

Kramers escape over a barrier of height `u` at noise level `D`, and the probability of
at least one crossing in a horizon `T` for a constant rate (a Poisson first-passage). -/

/-- Kramers escape rate at barrier height `u`, noise `D`, prefactor `k`. -/
noncomputable def rate (k D u : ℝ) : ℝ := k * Real.exp (-u / D)

/-- Probability of at least one crossing in horizon `T` at constant rate `r`. -/
noncomputable def crossProb (r T : ℝ) : ℝ := 1 - Real.exp (-(r * T))

/-! ## 1. The dose cap is vacuous

The content here is entirely in the hypotheses, which is the honest way to state it:
E4 *measured* a strategy that spends almost nothing and crosses almost surely. Given
that measurement, no cap set above its dose can promise anything. -/

/-- **A dose cap that admits the exploit bounds nothing.**

    If some strategy `exploit` has dose within the cap `B` and crossing probability `1`,
    then for any claimed bound `p < 1` the statement "every strategy within the cap has
    crossing probability at most `p`" is false.

    E4 measures exactly this: at `a = 0.370` (just under the fold `a* = 0.3849`) the
    attacker spends dose `0.001` and crosses with probability `1.00`. Any cap at or
    above `0.001` therefore guarantees nothing. -/
theorem dose_cap_cannot_bound
    {S : Type*} (dose cross : S → ℝ) (exploit : S) (B p : ℝ)
    (hin : dose exploit ≤ B) (hcross : cross exploit = 1) (hp : p < 1) :
    ¬ (∀ s, dose s ≤ B → cross s ≤ p) := by
  intro hcap
  have := hcap exploit hin
  rw [hcross] at this
  exact absurd this (not_le.mpr hp)

/-- **Crossing probability is not monotone in dose**, so no threshold rule on dose can
    separate safe from unsafe strategies. E4 measures a ramp that spends *more* dose
    than the exploit and crosses far less often. -/
theorem cross_not_monotone_in_dose
    {S : Type*} (dose cross : S → ℝ) (exploit ramp : S)
    (hdose : dose exploit ≤ dose ramp) (hcross : cross ramp < cross exploit) :
    ¬ (∀ s t, dose s ≤ dose t → cross s ≤ cross t) := by
  intro hmono
  exact absurd (hmono exploit ramp hdose) (not_le.mpr hcross)

/-! ## 2. The barrier floor does bound crossing probability

Unlike dose, barrier height enters the escape rate directly, so a floor on it
propagates to a bound on the outcome. -/

/-- A barrier floor caps the escape rate: if the barrier never drops below `u`, the
    rate never rises above `rate k D u`. -/
theorem rate_le_of_barrier_ge
    {k D u b : ℝ} (hk : 0 ≤ k) (hD : 0 < D) (hb : u ≤ b) :
    rate k D b ≤ rate k D u := by
  unfold rate
  have hinv : (0:ℝ) ≤ D⁻¹ := inv_nonneg.mpr hD.le
  have hstep : (-b) * D⁻¹ ≤ (-u) * D⁻¹ :=
    mul_le_mul_of_nonneg_right (by linarith) hinv
  have hdiv : -b / D ≤ -u / D := by simpa [div_eq_mul_inv] using hstep
  exact mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr hdiv) hk

/-- **The barrier floor theorem.**

    If the barrier height never drops below
        `u ≥ D * log (k * T / (-log (1 - ε)))`
    then the probability of crossing within horizon `T` is at most `ε`.

    This is the formula `e4_results.json` records as `ΔU_floor`, and the direction of
    the inequality is what makes it a *defence*: raising the floor lowers the risk. -/
theorem barrier_floor_bounds_crossing
    {k D T u ε : ℝ}
    (hk : 0 < k) (hD : 0 < D) (hT : 0 < T)
    (hε0 : 0 < ε) (hε1 : ε < 1)
    (hu : D * Real.log (k * T / (-Real.log (1 - ε))) ≤ u) :
    crossProb (rate k D u) T ≤ ε := by
  set L := -Real.log (1 - ε) with hLdef
  have h1e : 0 < 1 - ε := by linarith
  have hL : 0 < L := by
    have hneg : Real.log (1 - ε) < 0 := Real.log_neg h1e (by linarith)
    simpa [hLdef] using hneg
  have hkT : 0 < k * T := mul_pos hk hT
  have hE : (0:ℝ) < Real.exp (u / D) := Real.exp_pos _
  -- The floor hypothesis gives `log (kT/L) ≤ u/D`, hence `kT/L ≤ exp (u/D)`.
  have hlog : Real.log (k * T / L) ≤ u / D := by
    rw [le_div_iff₀ hD, mul_comm]
    exact hu
  have hexp : k * T / L ≤ Real.exp (u / D) := by
    calc k * T / L = Real.exp (Real.log (k * T / L)) :=
          (Real.exp_log (div_pos hkT hL)).symm
      _ ≤ Real.exp (u / D) := Real.exp_le_exp.mpr hlog
  have h2 : k * T ≤ L * Real.exp (u / D) := by
    rw [div_le_iff₀ hL] at hexp; linarith
  -- Therefore the escape rate over the horizon is at most `L`.
  have hrate : rate k D u * T ≤ L := by
    have hne : Real.exp (-u / D) = (Real.exp (u / D))⁻¹ := by
      rw [← Real.exp_neg]; congr 1; ring
    unfold rate
    rw [hne, show k * (Real.exp (u / D))⁻¹ * T = (k * T) / Real.exp (u / D) by
      field_simp, div_le_iff₀ hE]
    linarith
  -- And `1 - exp (-rT) ≤ 1 - exp (-L) = ε`.
  have hmono : Real.exp (-L) ≤ Real.exp (-(rate k D u * T)) :=
    Real.exp_le_exp.mpr (by linarith)
  have hEL : Real.exp (-L) = 1 - ε := by
    rw [hLdef, neg_neg, Real.exp_log h1e]
  unfold crossProb
  linarith [hmono, hEL.le, hEL.ge]

/-! ## 3. What this does and does not license

`barrier_floor_bounds_crossing` is a statement about a constant-rate model with a
barrier that never drops below `u`. Two gaps between it and the simulation are worth
naming, because the simulation does not fully satisfy its hypothesis:

* E4 reports crossing probabilities of 0.06, 0.105 and 0.115 against a target of
  ε = 0.05 *with the floor enforced*. The theorem says that cannot happen when the
  barrier stays above `u` for the whole horizon. So either the enforced strategies dip
  below the floor transiently, or the constant-rate Kramers approximation
  under-predicts in this regime. The theorem turns that discrepancy from a puzzle into
  a checkable question about the two hypotheses.

* Nothing here says real beliefs behave like a particle in a potential. That is the
  assumption the node exists to test, and `theory/model.md` is explicit that the
  potential is chosen for tractability. -/

end BoundedInfluence
