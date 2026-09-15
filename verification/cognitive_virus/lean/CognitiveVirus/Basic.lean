/-
Formal checks of the mean-field model in
Sole, Ruffini, Castaldo, Tuccio, Seoane, De Domenico, Elena, Krakauer, Levin,
"Large-Language Models as a Cognitive Virus", arXiv:2609.03344 (2026).

  dU/dt = -l U C + r C + k U^2 C
  dC/dt =  l U C - (m + r) C + s D - k U^2 C
  dD/dt =  m C - s D
with l = lambda, r = rho, k = kappa, m = mu, s = sigma.

Everything here is algebra about the vector field (identities, roots, signs,
exact linearizations). Statements about flows (e.g. "a 2x2 linearization with
det > 0 and trace < 0 is asymptotically stable", "triangular matrices have
their diagonal as eigenvalues") are standard facts and are NOT formalized.
-/
import Mathlib

namespace CognitiveVirus

section Ring
variable {R : Type*} [CommRing R]

def dU (l r k U C : R) : R := -l*U*C + r*C + k*U^2*C
def dC (l r k m s U C D : R) : R := l*U*C - (m+r)*C + s*D - k*U^2*C
def dD (m s C D : R) : R := m*C - s*D
def f (l r k U : R) : R := r - l*U + k*U^2

/-- Claim (1): U + C + D is conserved. -/
theorem conservation (l r k m s U C D : R) :
    dU l r k U C + dC l r k m s U C D + dD m s C D = 0 := by
  unfold dU dC dD; ring

/-- Claim (2): dU/dt = C f(U). -/
theorem dU_factor (l r k U C : R) : dU l r k U C = C * f l r k U := by
  unfold dU f; ring

/-- Eq. (5): with D eliminated, dC/dt = -C f(U) - m C + s (1 - U - C). -/
theorem dC_reduced (l r k m s U C : R) :
    dC l r k m s U C (1 - U - C) = -(C * f l r k U) - m*C + s*(1 - U - C) := by
  unfold dC f; ring

/-- Linearization of dC/dt at E_U = (1,0,0) in (C,D) coordinates, exact with
explicit quadratic remainder. Jacobian = [[l-r-k-m, s],[m, -s]]. -/
theorem dC_linearization_EU (l r k m s C D : R) :
    dC l r k m s (1 - C - D) C D
      = (l - r - k - m)*C + s*D + ((2*k - l)*(C+D)*C - k*(C+D)^2*C) := by
  unfold dC; ring

/-- Jacobian at E_U: determinant and trace. -/
theorem EU_det (l r k m s : R) :
    (l - r - k - m)*(-s) - s*m = s*(r + k - l) := by ring

/-- Exact linearization at an interior equilibrium (U*,C*) in (U,C) coordinates
(D = 1 - U - C). With f(U*) = 0 and -m C* + s (1 - U* - C*) = 0, the Jacobian is
lower triangular: [[C* f'(U*), 0], [-C* f'(U*) - s, -(m+s)]], f'(U) = -l + 2kU. -/
theorem interior_linearization (l r k m s U0 C0 u c : R)
    (hf : f l r k U0 = 0) (heq : -m*C0 + s*(1 - U0 - C0) = 0) :
    (C0 + c) * f l r k (U0 + u)
        = C0*(-l + 2*k*U0)*u + (c*(-l + 2*k*U0)*u + (C0 + c)*k*u^2)
    ∧ -((C0 + c) * f l r k (U0 + u)) - m*(C0 + c) + s*(1 - (U0+u) - (C0+c))
        = (-C0*(-l + 2*k*U0) - s)*u - (m+s)*c
          - (c*(-l + 2*k*U0)*u + (C0 + c)*k*u^2) := by
  have hf' : r - l*U0 + k*U0^2 = 0 := hf
  constructor
  · unfold f; linear_combination (C0 + c) * hf'
  · unfold f; linear_combination (-(C0 + c)) * hf' + heq

/-- Claim (6): <Gamma> at a coupled equilibrium is affine in U*. Gamma_u = 1. -/
theorem gamma_affine {K : Type*} [Field K] (Gc Gd m s U : K) :
    1*U + Gc*(s/(m+s)*(1-U)) + Gd*(m/(m+s)*(1-U))
      = (Gc*s + Gd*m)/(m+s) + (1 - (Gc*s + Gd*m)/(m+s))*U := by
  ring

/-- Claim (7), algebraic part: Delta Gamma formulas. -/
theorem delta_gamma (g x : R) : (g + (1-g)*1) - (g + (1-g)*x) = (1-g)*(1-x) := by ring

end Ring

section Field
variable {K : Type*} [Field K]

/-- Claim (3): C*, D* from Eq. (6) together with f(U*) = 0 make all three RHS vanish. -/
theorem coupled_equilibrium (l r k m s U : K) (hf : f l r k U = 0) :
    dU l r k U (s/(m+s)*(1-U)) = 0
    ∧ dC l r k m s U (s/(m+s)*(1-U)) (m/(m+s)*(1-U)) = 0
    ∧ dD m s (s/(m+s)*(1-U)) (m/(m+s)*(1-U)) = 0 := by
  have hf' : r - l*U + k*U^2 = 0 := hf
  refine ⟨?_, ?_, ?_⟩
  · rw [dU_factor, hf, mul_zero]
  · unfold dC; linear_combination (-(s/(m+s)*(1-U))) * hf'
  · unfold dD; ring

/-- Claim (8): D*/(C*+D*) = m/(m+s), for U* ≠ 1. -/
theorem D_share (m s U : K) (hms : m + s ≠ 0) (hU : 1 - U ≠ 0) :
    (m/(m+s)*(1-U)) / (s/(m+s)*(1-U) + m/(m+s)*(1-U)) = m/(m+s) := by
  have h : s/(m+s)*(1-U) + m/(m+s)*(1-U) = 1 - U := by
    field_simp; ring
  rw [h]; field_simp

/-- The 1D reduction is not an invariant manifold: W = C - s/(m+s) (1-U) obeys
dW/dt = -(m+s) W - (m/(m+s)) C f(U). The reduction is exact only where C f(U) = 0. -/
theorem offmanifold (l r k m s U C : K) (hms : m + s ≠ 0) :
    dC l r k m s U C (1 - U - C) + s/(m+s) * dU l r k U C
      = -(m+s)*(C - s/(m+s)*(1-U)) - m/(m+s) * (C * f l r k U) := by
  unfold dC dU f; field_simp; ring

end Field

section Real
open Real

/-- Any q with q^2 = l^2 - 4kr and 2kU = l + q gives a root of f (covers U+ and U-). -/
theorem root_of_q (l r k U q : ℝ) (hk : k ≠ 0) (hq : q^2 = l^2 - 4*k*r)
    (hU : 2*k*U = l + q) : f l r k U = 0 := by
  have h : (4*k) * f l r k U = 0 := by
    unfold f; linear_combination (2*k*U - l + q) * hU + hq
  exact (mul_eq_zero.mp h).resolve_left (by intro h'; apply hk; linarith)

/-- Claim (3): U± = (l ± sqrt(l^2 - 4kr)) / (2k) are roots whenever the discriminant is ≥ 0. -/
theorem U_plus_root (l r k : ℝ) (hk : k ≠ 0) (hd : 0 ≤ l^2 - 4*k*r) :
    f l r k ((l + √(l^2 - 4*k*r)) / (2*k)) = 0 :=
  root_of_q l r k _ _ hk (sq_sqrt hd) (by field_simp)

theorem U_minus_root (l r k : ℝ) (hk : k ≠ 0) (hd : 0 ≤ l^2 - 4*k*r) :
    f l r k ((l - √(l^2 - 4*k*r)) / (2*k)) = 0 :=
  root_of_q l r k _ (-√(l^2 - 4*k*r)) hk (by rw [neg_sq, sq_sqrt hd]) (by field_simp; ring)

/-- Real roots exist iff l ≥ 2 sqrt(kr) (for l ≥ 0, k, r > 0). -/
theorem disc_nonneg_iff (l r k : ℝ) (hl : 0 ≤ l) (hk : 0 < k) (hr : 0 < r) :
    0 ≤ l^2 - 4*k*r ↔ 2*√(k*r) ≤ l := by
  have hs : (√(k*r))^2 = k*r := sq_sqrt (by positivity)
  have h0 : 0 ≤ √(k*r) := sqrt_nonneg _
  constructor
  · intro h; nlinarith
  · intro h; nlinarith

/-- Claim (5): width identity l_TC - l_SN = (sqrt k - sqrt r)^2. -/
theorem width_identity (r k : ℝ) (hr : 0 ≤ r) (hk : 0 ≤ k) :
    (r + k) - 2*√(k*r) = (√k - √r)^2 := by
  rw [sqrt_mul hk, sub_sq, sq_sqrt hk, sq_sqrt hr]; ring

/-- CORRECTION to a literal reading of claim (5): the open interval (2 sqrt(kr), r+k)
is nonempty iff k ≠ r, not iff k > r. Physical bistability for k > r only (see below). -/
theorem interval_nonempty_iff (r k : ℝ) (hr : 0 < r) (hk : 0 < k) :
    2*√(k*r) < r + k ↔ k ≠ r := by
  rw [← sub_pos, width_identity r k hr.le hk.le]
  constructor
  · intro h heq; subst heq; simp at h
  · intro h
    have : √k ≠ √r := fun e => h ((sqrt_inj hk.le hr.le).mp e)
    exact lt_of_le_of_ne (sq_nonneg _) (Ne.symm (pow_ne_zero 2 (sub_ne_zero.mpr this)))

/-- For k < r and l < r + k (l > 0) there is no root of f in (-inf, 1]:
the formal saddle-node branch lies outside the simplex (paper, Sec. II). -/
theorem no_physical_root_of_k_lt_r (l r k U : ℝ) (hk : 0 < k) (hkr : k < r) (hl : 0 < l)
    (hlt : l < r + k) (hU1 : U ≤ 1) : f l r k U ≠ 0 := by
  unfold f
  intro h
  rcases le_or_gt U 0 with hU0 | hU0
  · nlinarith [sq_nonneg U]
  · nlinarith [mul_pos hU0 (sub_pos.mpr hlt), mul_nonneg (sub_nonneg.mpr hU1) (sub_nonneg.mpr hU1),
      mul_nonneg (sub_nonneg.mpr hU1) hU0.le]

/-- For k > r and l < r + k, U+ < 1 (so the unstable branch is physical on the whole
bistable interval) and U- > 0. -/
theorem U_plus_lt_one (l r k : ℝ) (hr : 0 < r) (hkr : r < k) (hlt : l < r + k)
    (hd : 0 ≤ l^2 - 4*k*r) : (l + √(l^2 - 4*k*r)) / (2*k) < 1 := by
  have hk : 0 < k := hr.trans hkr
  have hq0 := sqrt_nonneg (l^2 - 4*k*r)
  have hq2 := sq_sqrt hd
  have hb : 0 < 2*k - l := by linarith
  have hlt' : √(l^2 - 4*k*r) < 2*k - l := by
    by_contra h; rw [not_lt] at h
    nlinarith [mul_le_mul h h hb.le hq0]
  rw [div_lt_one (by positivity)]; linarith

theorem U_minus_pos (l r k : ℝ) (hr : 0 < r) (hk : 0 < k) (hl : 0 < l)
    (hd : 0 ≤ l^2 - 4*k*r) : 0 < (l - √(l^2 - 4*k*r)) / (2*k) := by
  have hq0 := sqrt_nonneg (l^2 - 4*k*r)
  have hq2 := sq_sqrt hd
  have : √(l^2 - 4*k*r) < l := by
    by_contra h; rw [not_lt] at h
    nlinarith [mul_le_mul h h hl.le hq0, mul_pos hk hr]
  apply div_pos <;> linarith

/-- Claim (7): at l_TC = r + k (with k ≥ r), U- = r/k. -/
theorem U_minus_at_TC (r k : ℝ) (hk : 0 < k) (hkr : r ≤ k) :
    ((r + k) - √((r + k)^2 - 4*k*r)) / (2*k) = r / k := by
  have : (r + k)^2 - 4*k*r = (k - r)^2 := by ring
  rw [this, sqrt_sq (by linarith)]; field_simp; ring

/-- Claim (7): at l_SN = 2 sqrt(kr) the double root is sqrt(r/k). -/
theorem U_at_SN (r k : ℝ) (hr : 0 ≤ r) (hk : 0 < k) :
    (2*√(k*r)) / (2*k) = √(r/k) ∧ (2*√(k*r))^2 - 4*k*r = 0 := by
  have hs : (√(k*r))^2 = k*r := sq_sqrt (by positivity)
  refine ⟨?_, by nlinarith⟩
  have h1 : r / k = (√(k*r)/k)^2 := by rw [div_pow, hs]; field_simp
  rw [h1, sqrt_sq (by positivity)]; field_simp

/-- At k = r the two thresholds coincide at 2r. -/
theorem thresholds_coincide (r : ℝ) (hr : 0 ≤ r) : 2*√(r*r) = r + r := by
  rw [sqrt_mul_self hr]; ring

/-- E_U stability conditions (Routh-Hurwitz for the 2x2 Jacobian [[l-r-k-m, s],[m,-s]]):
det > 0 iff l < r + k (s > 0), and then trace < 0 automatically. -/
theorem EU_stability_conditions (l r k m s : ℝ) (hm : 0 < m) (hs : 0 < s) :
    (0 < s*(r + k - l) ↔ l < r + k) ∧ (l < r + k → l - r - k - m - s < 0) := by
  constructor
  · constructor
    · intro h; by_contra h'; rw [not_lt] at h'; nlinarith
    · intro h; exact mul_pos hs (by linarith)
  · intro h; linarith

/-- Sign of f' at the two roots: if 2kU = l ± q then f'(U) = -l + 2kU = ± q.
With q = sqrt(disc) > 0: f'(U-) < 0 < f'(U+). Combined with `interior_linearization`,
the interior Jacobian has eigenvalues C* f'(U*) and -(m+s): U- is a stable node,
U+ a saddle, in the full 2D system (no slow-manifold assumption needed). -/
theorem fprime_signs (l r k : ℝ) (hk : 0 < k) (hd : 0 < l^2 - 4*k*r) :
    -l + 2*k*((l - √(l^2 - 4*k*r)) / (2*k)) < 0
    ∧ 0 < -l + 2*k*((l + √(l^2 - 4*k*r)) / (2*k)) := by
  have hq := sqrt_pos.mpr hd
  have e : ∀ x : ℝ, 2*k*(x/(2*k)) = x := fun x => by field_simp
  rw [e, e]; constructor <;> linarith

theorem interior_eigs_signs (Cst fp m s : ℝ) (hC : 0 < Cst) (hm : 0 < m) (hs : 0 < s) :
    (fp < 0 → Cst*fp < 0 ∧ -(m+s) < 0) ∧ (0 < fp → 0 < Cst*fp ∧ -(m+s) < 0) :=
  ⟨fun h => ⟨mul_neg_of_pos_of_neg hC h, by linarith⟩,
   fun h => ⟨mul_pos hC h, by linarith⟩⟩

end Real
end CognitiveVirus
