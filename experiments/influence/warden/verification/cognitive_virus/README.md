# Verification: "Large-Language Models as a Cognitive Virus"

This directory checks the mean-field model from Sole R, Ruffini G, Castaldo F, Tuccio M, Seoane LF, De Domenico M, Elena SF, Krakauer DC, Levin M, *Large-Language Models as a Cognitive Virus*, arXiv:2609.03344 (2026). The model has three states: uncoupled (U), coupled (C) and dependent (D).

```
dU/dt = -lam U C + rho C + kap U^2 C
dC/dt =  lam U C - (mu + rho) C + sig D - kap U^2 C
dD/dt =  mu C - sig D,          U + C + D = 1
```

The paper has three parts, and each is checked a different way. The algebra is proven in Lean 4. The dynamical claims are checked numerically. A stochastic agent-based model probes what the mean-field picture leaves out. The arXiv v1 PDF has no supplementary material, so the "1D reduction" behind the potential (paper Sec. III, "see SM") had to be reconstructed. See the Maxwell point section below.

## A. Formal proofs (`lean/`)

A Lake project on Lean `v4.33.1` with Mathlib `v4.33.1` (prebuilt cache). To build it, run `cd lean && lake exe cache get && lake build`; to print the axioms, run `lake env lean Axioms.lean`. The build succeeds with zero `sorry`. Every theorem depends only on `propext`, `Classical.choice` and `Quot.sound` (the standard Mathlib axioms). The full output is in `lean/axioms_output.txt`.

Proven (file `lean/CognitiveVirus/Basic.lean`):

| Theorem | Content |
|---|---|
| `conservation` | dU + dC + dD = 0 (any commutative ring) |
| `dU_factor`, `dC_reduced` | dU = C f(U) with f = rho - lam U + kap U^2; Eq. (5) |
| `coupled_equilibrium` | C* = sig/(mu+sig)(1-U*), D* = mu/(mu+sig)(1-U*) with f(U*)=0 zero all three right-hand sides (any field) |
| `root_of_q`, `U_plus_root`, `U_minus_root` | U± = (lam ± sqrt(lam^2-4 kap rho))/(2 kap) are roots when the discriminant is ≥ 0 |
| `disc_nonneg_iff` | real roots iff lam ≥ 2 sqrt(kap rho) |
| `width_identity` | (rho+kap) - 2 sqrt(kap rho) = (sqrt kap - sqrt rho)^2 |
| `interval_nonempty_iff` | (2 sqrt(kap rho), rho+kap) is nonempty iff kap ≠ rho (see discrepancies) |
| `no_physical_root_of_k_lt_r` | if kap < rho and lam < rho+kap, then f has no root with U ≤ 1 |
| `U_plus_lt_one`, `U_minus_pos` | if kap > rho and lam < rho+kap, then U+ < 1; U- > 0 always |
| `U_minus_at_TC`, `U_at_SN` | U-(lam_TC) = rho/kap (kap ≥ rho); double root at lam_SN is sqrt(rho/kap) |
| `thresholds_coincide` | at kap = rho, lam_SN = lam_TC = 2 rho |
| `gamma_affine`, `delta_gamma` | <Gamma>* = g + (1-g) U*; Delta Gamma = (1-g)(1-x) |
| `D_share` | D*/(C*+D*) = mu/(mu+sig) for U* ≠ 1 |
| `dC_linearization_EU`, `EU_det`, `EU_stability_conditions` | exact linearization at E_U in (C,D) coordinates, Jacobian [[lam-rho-kap-mu, sig],[mu,-sig]], det = sig(rho+kap-lam) > 0 iff lam < rho+kap, which also forces trace < 0 |
| `interior_linearization` | at any interior equilibrium the (U,C) Jacobian is lower triangular: [[C* f'(U*), 0],[-C* f'(U*) - sig, -(mu+sig)]] |
| `fprime_signs`, `interior_eigs_signs` | f'(U-) = -sqrt(disc) < 0 < f'(U+) = sqrt(disc) |
| `offmanifold` | W = C - sig/(mu+sig)(1-U) satisfies dW/dt = -(mu+sig) W - (mu/(mu+sig)) C f(U) |

Stated but not formalized, because these are standard textbook facts: a triangular matrix has its diagonal as its eigenvalues; a 2x2 linearization with det > 0 and trace < 0 is asymptotically stable (Routh-Hurwitz, Hartman-Grobman). Combined with the proven identities, they give the following. E_U is stable iff lam < rho + kap. The lower coupled branch U- is a stable node and U+ is a saddle, in the full 2D system and without any slow-manifold assumption. Note that the eigenvalue at E_U is not literally rho + kap - lam: that value is the Jacobian entry f(1) in (U,C) coordinates, and it is the determinant (divided by sig) that changes sign at lam_TC. Nothing about the potential or the Maxwell point is formalized.

## B. Synthetic checks (`synthetic/`)

Run with `python run_all.py` (needs numpy, scipy, matplotlib, tqdm). Per-check JSON goes to `results/`, the merged file is `results/results.json`, and figures go to `figures/`.

1. **Hysteresis by continuation** (`hysteresis.py`, `figures/hysteresis_grid.png`). The sweep raises lam, then lowers it, with a 1e-3 kick at every step. For kap > rho the measured jump points match theory to within one grid step (about 0.002 to 0.004). At rho=0.1, kap=0.4, the up jump is at 0.5025 (lam_TC = 0.5) and the down jump at 0.3994 (lam_SN = 0.4), a width of 0.103 against a theoretical 0.100. At rho=0.05, kap=0.5 the width is 0.242 against 0.234, and at rho=0.2, kap=0.3 it is 0.016 against 0.010. For kap < rho (three cases) the sweeps are continuous: the largest gap between the up and down curves is at most 0.003, and the largest single-step change is about 0.013. For kap = rho there are no jumps, but the up/down gap is 0.093 at rho=0.1. This is critical slowing: E_U is a degenerate (double-root) equilibrium there, and the finite integration time does not fully relax. It is not hysteresis.
2. **Random fuzz** (`fuzz.py`, 1200 log-uniform draws). The largest conservation residual is 2e-16, the equilibrium residual 1.6e-14, the D-share error 2e-16 and the Gamma formula error 1.4e-14. The analytic interior Jacobian eigenvalues match finite differences to 7e-6. E_U stability was misclassified 0 times. Predicted attractor sets disagreed with integration from 5 initial conditions 0 times; 137 draws within 3% of a bifurcation were skipped for slow convergence. U+ > 1 inside the bistable window occurred 0 times in 113 bistable draws, and bistability with kap ≤ rho occurred 0 times.
3. **Agent-based model** (`abm.py`, `figures/abm.png`). The rules per agent are U→C at rate lam·(C fraction among contacts), C→U at rho + kap·(U fraction among contacts)^2, C→D at mu and D→C at sig. The simulation uses tau-leaping with dt = 0.1 and T = 800, 3 replicates, rho=0.1, kap=0.4, mu=0.2, sig=0.1. It runs from a near-uncoupled start (2% C) and from a coupled start (U=0.2, C=0.3).
   - *Well-mixed, N = 10^4.* Invasion first happens at lam = 0.5 (mean U=0.244 against the mean-field 0.25; U=0.994 at 0.475). The coupled state survives down to lam = 0.40 and collapses at 0.375. At lam = 0.45 the coupled U is 0.295 against the mean-field 0.305. This agrees with mean field to the resolution of the lam grid (0.025).
   - *Well-mixed, N = 10^3.* Same threshold location, but at lam = 0.5 and 0.525 replicates split between the two states (mean U about 0.49). The transition is smeared by demographic noise. In any finite population E_U (C = D = 0) is absorbing, so the coupled state is only metastable.
   - *Networks, N = 2000: Erdős–Rényi with mean degree 10, Barabási–Albert with m = 5 (max degree 207).* With contact-local fractions, the coupled branch sits higher than mean field: U = 0.40 (ER) and 0.39 (BA) at lam = 0.45, against 0.305. Invasion from the uncoupled state already occurs at lam = 0.475 < lam_TC, in all ER replicates and some BA replicates. The coupled state still collapses at lam ≤ 0.40. So network structure shifts the upper threshold down and moves the coupled branch toward autonomy, which narrows the hysteresis window from above. One plausible mechanism is that local fluctuations in u make E[u^2] exceed E[u]^2, strengthening the local kap term. Another is clustering of C agents, which raises the local exposure c. We did not separate these two. With 3 replicates and a 0.025 lam grid these are coarse, qualitative results.
4. **Timescale separation and Maxwell point** (`timescale_maxwell.py`, `figures/timescale_potential.png`). At the stable coupled state the fast rate is exactly mu + sig = 0.3 (proven above), and the slow rate is |C* f'(U-)|. The ratio of fast to slow is 197 just above lam_SN, 6.5 at lam = 0.45, and falls to 1.5 by lam = 0.8. The separation is reasonable in the bistable window but not far above it. Starting on the manifold, the full 2D trajectory differs from the 1D reduction by at most 0.026 in U for lam in {0.42, 0.45}, and by up to 0.094 at lam = 0.6. Both converge to the same fixed points. The Maxwell (equal-depth) point is 0.4196 for the flow dU/dt = sig/(mu+sig)·(1-U)·f(U), which reproduces the paper's 0.420. For the time-rescaled flow dU/dtau = f(U) it is 0.4389.
5. **Table I** (`interventions.py`). Signs from finite differences confirm all 10 qualitative entries at the base parameters. The claim that mu and sig raise <Gamma> depends on the illustrative choice Gamma_c > Gamma_d; with Gamma_c < Gamma_d (a scaffolding regime) the sign reverses.

## Discrepancies, glosses and caveats

- **"Exist iff lam ≥ 2 sqrt(kap rho)" (Eq. 7) is about real roots, not physical states.** For kap < rho the roots with lam in [lam_SN, lam_TC) are both greater than 1 (proven). For lam > lam_TC, U+ > 1. The paper acknowledges the kap < rho case in words ("the formal saddle-node lies outside the physical simplex") but states the existence condition without this qualification.
- **Interval (9) is nonempty for every kap ≠ rho.** 2 sqrt(kap rho) < rho + kap is AM-GM and holds whenever kap ≠ rho. Physical bistability requires kap > rho. The paper's claims read correctly in context, but "bistable interval nonempty iff kap > rho" is false as a statement about the interval alone.
- **Stability of the branches.** The paper describes U- as stable and U+ as unstable via the figure and the potential. That labelling is correct and holds in the full 2D system (triangular Jacobian), which is a stronger statement than the paper makes.
- **The 1D reduction is not an invariant manifold.** C = sig/(mu+sig)(1-U) is exact only at equilibria (theorem `offmanifold`). The potential is an approximation whose accuracy depends on timescale separation. The separation is good near lam_SN and weak for large lam.
- **The Maxwell point is not uniquely defined.** The 2D system is not a gradient system, and the equal-depth point depends on the choice of 1D flow: 0.4196 against 0.4389 for two reductions with identical fixed points. Using <Gamma> instead of U does not change it, because the map is affine. A deterministic ODE gives the Maxwell point no dynamical meaning. In a noisy system, which state is preferred is set by the quasi-potential, which depends on the noise model. The paper's 0.420 matches the (1-U)·f reduction, which is probably the one in its SM.
- **kap = rho.** The thresholds coincide and E_U is a degenerate equilibrium, a codimension-2 point where the transcritical and saddle-node bifurcations merge. Convergence there is algebraic rather than exponential. The paper's statement is correct.
- **Initial conditions with D.** A population with C = 0 but D > 0 is not at E_U: D feeds C at rate sig. Invasion analysis must perturb both C and D, which the (C,D) Jacobian does.
- **The Gamma values are illustrative**, as the paper itself says; g = 7/30 and the numbers 0.425 and 0.617 follow from them exactly.
- **Finite populations and networks** (flagged by the paper as future work). Thresholds hold for large well-mixed populations. Small populations smear the transition and make E_U absorbing. On ER/BA contact networks the invasion threshold moves below lam_TC and the coupled branch shifts toward higher U.

These are checks of internal mathematical consistency on simulated data. They say nothing about whether the model describes real LLM adoption.
