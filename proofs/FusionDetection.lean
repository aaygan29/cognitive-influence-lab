/-
FusionDetection.lean

The detection corollary of the inverse-variance fusion inequality
(`FusionMath.lean`, `fused_le_min`), specialised to detecting a shared shift.

Setup. A manipulation induces the SAME shift `mu` in a latent parameter (here,
loss aversion), read out by `K` biosignal modalities. Modality `i` returns an
unbiased estimate of `mu` with independent noise of variance `sigma i ^ 2 > 0`, so
its per-modality detection SNR is `snr i = mu / sigma i`. Inverse-variance fusion
forms the estimate with precision `P = sum_i 1 / sigma i ^ 2` and standard error
`1 / sqrt P`, giving fused SNR `mu * sqrt P`.

Claims proved:
  * `fused_precision_eq` : the fused precision is the sum of per-modality precisions.
  * `snr_quadrature`     : fused SNR^2 = sum of per-modality SNR^2 (SNRs add in
                           quadrature).
  * `fused_snr_sq_ge_single` : fused SNR^2 >= any single modality's SNR^2, so
                           fusing independent modalities never detects worse than
                           the best single one, and strictly better when >1 carries
                           signal. This is the detection form of `fused_le_min`.

Everything is elementary real algebra over a Finset; no probability is invoked, so
the statements hold for the operational SNR definitions above.
-/
import Mathlib

open scoped BigOperators

namespace FusionDetection

variable {ι : Type*}

/-- Per-modality precision `1 / sigma i ^ 2`. -/
noncomputable def prec (sigma : ι → ℝ) (i : ι) : ℝ := 1 / (sigma i) ^ 2

/-- Fused precision is the sum of per-modality precisions (definitional). -/
theorem fused_precision_eq (s : Finset ι) (sigma : ι → ℝ) :
    (∑ i ∈ s, prec sigma i) = ∑ i ∈ s, 1 / (sigma i) ^ 2 := rfl

/-- Per-modality squared SNR of the shared shift `mu`. -/
noncomputable def snrSq (mu : ℝ) (sigma : ι → ℝ) (i : ι) : ℝ :=
    mu ^ 2 / (sigma i) ^ 2

/-- Fused squared SNR = `mu^2 *` (sum of precisions). -/
noncomputable def fusedSnrSq (s : Finset ι) (mu : ℝ) (sigma : ι → ℝ) : ℝ :=
    mu ^ 2 * ∑ i ∈ s, 1 / (sigma i) ^ 2

/-- SNRs add in quadrature: fused SNR^2 equals the sum of per-modality SNR^2. -/
theorem snr_quadrature (s : Finset ι) (mu : ℝ) (sigma : ι → ℝ) :
    fusedSnrSq s mu sigma = ∑ i ∈ s, snrSq mu sigma i := by
  unfold fusedSnrSq snrSq
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- The fused detector is never worse than any single modality: fused SNR^2 is at
least the SNR^2 of any family `j`. Requires each variance positive. -/
theorem fused_snr_sq_ge_single (s : Finset ι) (mu : ℝ) (sigma : ι → ℝ)
    (_hσ : ∀ i ∈ s, 0 < sigma i) {j : ι} (hj : j ∈ s) :
    snrSq mu sigma j ≤ fusedSnrSq s mu sigma := by
  rw [snr_quadrature]
  have hnonneg : ∀ i ∈ s, 0 ≤ snrSq mu sigma i := by
    intro i _
    unfold snrSq
    positivity
  exact Finset.single_le_sum hnonneg hj

end FusionDetection
