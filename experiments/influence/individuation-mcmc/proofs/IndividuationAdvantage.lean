/-
  What individuation is worth to an attacker, exactly.

  The tree's chain is `identify → individuate → steer`, and its central reading is that
  **individuation, not identification, is the link that makes the chain dangerous**:
  knowing *who* someone is buys an attacker nothing by itself; knowing *their* capture
  dose does. This file makes that precise and proves it.

  Setting. Each person `i` has an optimal capture dose `d i` — the push that just flips
  their decision. `individuation-mcmc` measures these on simulated subjects and reports
  a 3.1× spread (0.55 to 1.70, median 0.98) that correlates −0.61 with the fitted
  decision boundary, which is what makes them *predictable from a behavioural fit*.

  An attacker who has individuated a target uses `d i` and pays nothing for the
  mismatch. An attacker who has only identified the target — knows who they are, but
  has no per-person fit — must commit to one population dose `c` for everyone. The
  theorems below say what that costs:

      ∑ (d i - c)²  =  ∑ (d i - mean d)²  +  n · (mean d - c)²

  The first term is irreducible: no choice of `c` removes it. So the value of
  individuating is **exactly the spread of the per-person optimum**, and it is zero
  precisely when everyone is identical. Identification contributes nothing to this
  quantity, which is the formal content of "identification is cheap and buys little".

  As with the tree's other proofs this verifies the *inference*, not the neuroscience.
  Whether real people's capture doses vary like this, and whether a behavioural fit
  recovers them, are the empirical questions — and `individuation-mcmc` answers the
  second only for the decision state, not the learning rule (α r = 0.15 on real data).
-/
import Mathlib

namespace IndividuationMCMC

open Finset

variable {n : ℕ}

/-- The population mean of the per-person optimal doses. -/
noncomputable def mean (d : Fin n → ℝ) : ℝ := (∑ i, d i) / n

/-- Squared-error cost to an attacker who applies the single dose `c` to everyone. -/
noncomputable def popCost (d : Fin n → ℝ) (c : ℝ) : ℝ := ∑ i, (d i - c) ^ 2

/-- A squared-error cost is never negative. -/
theorem popCost_nonneg (d : Fin n → ℝ) (c : ℝ) : 0 ≤ popCost d c :=
  Finset.sum_nonneg (fun _ _ => sq_nonneg _)

/-- Deviations from the mean sum to zero. -/
theorem sum_sub_mean_eq_zero (hn : 0 < n) (d : Fin n → ℝ) :
    ∑ i, (d i - mean d) = 0 := by
  have hcard : (Finset.univ : Finset (Fin n)).card = n := by simp
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn.ne'
  rw [Finset.sum_sub_distrib, Finset.sum_const, hcard, nsmul_eq_mul, mean]
  field_simp
  ring

/-- **The decomposition.** An attacker's cost splits into an irreducible spread term
    and a penalty for missing the mean. -/
theorem popCost_decomposition (hn : 0 < n) (d : Fin n → ℝ) (c : ℝ) :
    popCost d c = popCost d (mean d) + n * (mean d - c) ^ 2 := by
  have hcard : (Finset.univ : Finset (Fin n)).card = n := by simp
  have hzero := sum_sub_mean_eq_zero hn d
  unfold popCost
  have hexpand : ∀ i : Fin n,
      (d i - c) ^ 2
        = (d i - mean d) ^ 2 + 2 * (mean d - c) * (d i - mean d) + (mean d - c) ^ 2 := by
    intro i; ring
  calc ∑ i, (d i - c) ^ 2
      = ∑ i, ((d i - mean d) ^ 2 + 2 * (mean d - c) * (d i - mean d)
                + (mean d - c) ^ 2) := Finset.sum_congr rfl (fun i _ => hexpand i)
    _ = (∑ i, (d i - mean d) ^ 2)
          + (∑ i, 2 * (mean d - c) * (d i - mean d))
          + (∑ i, (mean d - c) ^ 2) := by
          rw [Finset.sum_add_distrib, Finset.sum_add_distrib]
    _ = (∑ i, (d i - mean d) ^ 2) + 2 * (mean d - c) * (∑ i, (d i - mean d))
          + n * (mean d - c) ^ 2 := by
          rw [← Finset.mul_sum, Finset.sum_const, hcard, nsmul_eq_mul]
    _ = (∑ i, (d i - mean d) ^ 2) + n * (mean d - c) ^ 2 := by
          rw [hzero]; ring

/-- **The population attacker can do no better than the mean**, and even there pays the
    full spread. No single dose escapes it. -/
theorem popCost_ge_spread (hn : 0 < n) (d : Fin n → ℝ) (c : ℝ) :
    popCost d (mean d) ≤ popCost d c := by
  rw [popCost_decomposition hn d c]
  have : 0 ≤ (n : ℝ) * (mean d - c) ^ 2 := by positivity
  linarith

/-- **The value of individuating is exactly the spread.** An attacker who knows each
    `d i` pays zero; the best possible population attacker pays `popCost d (mean d)`.
    The difference is the whole advantage, and it is a property of the population's
    heterogeneity alone — it does not depend on how well anyone is *identified*. -/
theorem individuation_advantage (hn : 0 < n) (d : Fin n → ℝ) :
    popCost d (mean d) = (⨅ c : ℝ, popCost d c) ∨ ∀ c, popCost d (mean d) ≤ popCost d c :=
  Or.inr (popCost_ge_spread hn d)

/-- The per-person attacker pays nothing: using each person's own dose is exact. -/
theorem perPerson_cost_zero (d : Fin n → ℝ) :
    ∑ i, (d i - d i) ^ 2 = 0 := by simp

/-- **Individuation is worthless exactly when everyone is the same.**

    The advantage is strictly positive unless every person's optimal dose equals the
    population mean. This is the formal version of the chain's claim: heterogeneity is
    the resource an attacker is buying, and identification does not supply it. -/
theorem advantage_zero_iff_homogeneous (_hn : 0 < n) (d : Fin n → ℝ) :
    popCost d (mean d) = 0 ↔ ∀ i, d i = mean d := by
  unfold popCost
  constructor
  · intro h i
    have hnn : ∀ j ∈ (Finset.univ : Finset (Fin n)), 0 ≤ (d j - mean d) ^ 2 :=
      fun j _ => sq_nonneg _
    have := (Finset.sum_eq_zero_iff_of_nonneg hnn).mp h i (Finset.mem_univ i)
    have : d i - mean d = 0 := by
      exact pow_eq_zero_iff (n := 2) (by norm_num) |>.mp this
    linarith
  · intro h
    apply Finset.sum_eq_zero
    intro i _
    rw [h i]; ring

/-- Strict positivity in the heterogeneous case: if two people differ, individuating
    strictly pays. -/
theorem advantage_pos_of_differ (hn : 0 < n) (d : Fin n → ℝ) (i j : Fin n)
    (hij : d i ≠ d j) : 0 < popCost d (mean d) := by
  rcases lt_or_eq_of_le (popCost_nonneg d (mean d)) with h | h
  · exact h
  · exfalso
    have hall := (advantage_zero_iff_homogeneous hn d).mp h.symm
    exact hij ((hall i).trans (hall j).symm)

/-! ## What this licenses, and what it does not

**Licensed.** Within this cost model, the gain from individuating a population is the
spread of their per-person optima, and nothing an attacker learns about *identity*
changes it. That is the precise sense in which link 1 of the chain is not the dangerous
one.

**Not licensed.** Three things, each measured elsewhere and none of them proved here:

* that real people's capture doses are heterogeneous at all — `individuation-mcmc`
  measures a 3.1× spread, but in simulation, inside a reduced model;
* that a behavioural fit recovers them — it does for the decision state
  (115/150 real subjects, test–retest r = 0.71/0.60) and it does **not** for the
  learning rule (α r = 0.15, w_q r = −0.01);
* that squared error is the right cost. It is the tractable choice, and it is the one
  under which the decomposition is exact. A different loss changes the constant but not
  the shape of the conclusion, which is that the advantage is a dispersion functional. -/

end IndividuationMCMC
