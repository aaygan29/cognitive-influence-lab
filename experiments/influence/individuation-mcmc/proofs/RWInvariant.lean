/-
  Formal verification of the well-posedness of the learning-rule layer.

  The model infers a Rescorla-Wagner learning rule from behavior. Its update is

      Q_next = Q + α (r - Q)

  with learning rate α ∈ [0,1] and reward r ∈ [0,1]. Two facts make the layer
  numerically well-posed and the value-difference drift term bounded, which the
  Python model and its tests assume:

    * `rw_step_mem` : the update is the convex combination
      `Q_next = (1-α) Q + α r`.
    * `rw_mem_invariant` : if `Q ∈ [0,1]` then `Q_next ∈ [0,1]`; hence the whole
      trajectory stays in `[0,1]` from a start in `[0,1]`.
    * `qdiff_bounded` : the value difference `Q₁ - Q₀` used as the learning
      contribution to the drift lies in `[-1,1]`.

  These verify the invariant checked empirically in tests/test_ddm.py
  (`test_rw_value_bounded_in_unit_interval`).
-/
import Mathlib

namespace IndividuationMCMC

/-- The Rescorla-Wagner update written as a convex combination. -/
theorem rw_step_mem (Q r α : ℝ) :
    Q + α * (r - Q) = (1 - α) * Q + α * r := by ring

/-- One update keeps the value in `[0,1]`. -/
theorem rw_mem_invariant (Q r α : ℝ)
    (hQ0 : 0 ≤ Q) (hQ1 : Q ≤ 1)
    (hr0 : 0 ≤ r) (hr1 : r ≤ 1)
    (ha0 : 0 ≤ α) (ha1 : α ≤ 1) :
    0 ≤ Q + α * (r - Q) ∧ Q + α * (r - Q) ≤ 1 := by
  rw [rw_step_mem]
  have h1a : 0 ≤ 1 - α := by linarith
  constructor
  · have := mul_nonneg h1a hQ0
    have := mul_nonneg ha0 hr0
    linarith
  · -- (1-α)Q + α r ≤ (1-α)·1 + α·1 = 1
    have hQ : (1 - α) * Q ≤ (1 - α) * 1 := by
      apply mul_le_mul_of_nonneg_left hQ1 h1a
    have hr : α * r ≤ α * 1 := by
      apply mul_le_mul_of_nonneg_left hr1 ha0
    nlinarith [hQ, hr]

/-- The value difference fed to the drift is bounded in `[-1,1]`. -/
theorem qdiff_bounded (Q0 Q1 : ℝ)
    (h00 : 0 ≤ Q0) (h01 : Q0 ≤ 1) (h10 : 0 ≤ Q1) (h11 : Q1 ≤ 1) :
    -1 ≤ Q1 - Q0 ∧ Q1 - Q0 ≤ 1 := by
  constructor <;> linarith

/-- The Rescorla-Wagner trajectory from a start `Q0` under reward and rate
    sequences `r, α`. -/
def rwSeq (Q0 : ℝ) (r α : ℕ → ℝ) : ℕ → ℝ
  | 0 => Q0
  | (n + 1) => rwSeq Q0 r α n + α n * (r n - rwSeq Q0 r α n)

/-- Iterating the update from a start in `[0,1]` stays in `[0,1]` for any finite
    sequence of rewards and (possibly trial-varying) rates in `[0,1]`. -/
theorem rw_trajectory_mem (n : ℕ) (Q0 : ℝ) (r α : ℕ → ℝ)
    (hQ0 : 0 ≤ Q0) (hQ1 : Q0 ≤ 1)
    (hr : ∀ k, 0 ≤ r k ∧ r k ≤ 1)
    (ha : ∀ k, 0 ≤ α k ∧ α k ≤ 1) :
    0 ≤ rwSeq Q0 r α n ∧ rwSeq Q0 r α n ≤ 1 := by
  induction n with
  | zero => exact ⟨hQ0, hQ1⟩
  | succ k ih =>
      obtain ⟨hr0, hr1⟩ := hr k
      obtain ⟨ha0, ha1⟩ := ha k
      simpa [rwSeq] using
        rw_mem_invariant (rwSeq Q0 r α k) (r k) (α k) ih.1 ih.2 hr0 hr1 ha0 ha1

end IndividuationMCMC
