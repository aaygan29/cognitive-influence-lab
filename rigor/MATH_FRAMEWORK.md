# A rigorous mathematical framework for "Structure identifies, function individuates"

2026-08-16. Derives the program's spine claims from information geometry, sloppy-model
theory, Lie-group quotients, structural-vs-practical identifiability, neural-manifold capacity, random
matrix theory, equivalence/conformal/power statistics, and rate-distortion. Every concept is tied to a
quantity computable on committed artifacts and generates at least one new falsifiable prediction.
Companion to `RIGOR_FINDINGS.md`, `SYNTHESIS_STRESSTESTED_2026-08.md`, `behavioral-preservation/SYNTHESIS.md`.
No em dashes.

---

## 0. The objects, made precise

The whole program studies one kind of object: a parametric stimulus-to-response map ("twin")
`f(x; theta)` fit per individual, evaluated by whether held-out responses match the right twin.

- **Linear apparatus** (`wiring-not-weights/exp01,03a`): individual `i` has `A_i in R^{p x d}`,
  `y = A_i x + eps`, `eps ~ N(0, sigma^2 I)`, SNR = 6 dB so `sigma^2 = 0.251 * signal power`.
  Stimuli excite only the first `k` of `d` coordinates; the remaining `d - k` are a zero-variance
  ("null") subspace by construction (`make_stimuli`, `make_world`). Configs: exp01 `(d,k,p)=(64,40,128)`,
  exp03a `(64,40,8)`.
- **Nonlinear apparatus** (`exp03b`): `f_i(x) = W2_i relu(W1_i x)`, `theta_i = (W1_i, W2_i)`,
  `(d,H,p)=(32,24,8)`.
- **Real connectome fingerprint** (`exp05_abide`): per-subject functional connectivity `FC_i in R^{200x200}`,
  `N=248`, split-half within session.
- **Behavioral capture** (`behavioral-preservation`): a lossy capture operator `K_f` degrades `W_i` at
  storage fidelity `f`, and `S(f)` is mean behavioral agreement of the reconstruction.

The spine has three theses to formalize:
(T-id) identification is cheap and structural; (T-recon) reconstruction-identity is the
functional-equivalence class of the weights; (T-budget) a preservation bit-budget `S(f)` measures how
much of the brain-behavior map must be kept. The eight sections below give each its exact machinery.

The organizing result is that four different formalisms compute the **same crossing number** on the
same Fisher spectrum: `alpha*` (sufficiency fraction, exp06 = 0.642 +/- 0.008), `f*` (rate at acceptable
distortion), the practical-identifiability sample threshold, and the stiff-eigenvalue cumulative
crossing. Section 9 states this unification and the single experiment that tests it.

---

## 1. Information geometry / the Fisher Information Metric

### Definition
For `p(y | x; theta)` the FIM is `g_{ab}(theta) = E[ d_a log p * d_b log p ]`. For the program's Gaussian
observation model `y = f(x;theta) + eps`, `eps ~ N(0, sigma^2 I)`, this collapses to the pullback of the
Euclidean output metric through the parameter-to-output Jacobian `J(x) = df/dtheta`:

```
g_{ab}(theta) = (1/sigma^2) E_x[ J(x)^T J(x) ]_{ab}.
```

Averaged over the probe battery `S`, the empirical FIM is `G_hat = (1/(sigma^2 |S|)) sum_{x in S} J(x)^T J(x)`.
This is the model-manifold metric of Transtrum-Machta-Sethna: the manifold is the image `{f(.;theta)}` in
output space, `G_hat` is its induced Riemannian metric, and the eigenvectors of `G_hat` are the parameter
combinations ordered by how strongly the data constrain them.

### Exact quantity on this program's artifacts
For the **linear** apparatus, `f(x;A)=Ax`, so `dF/dA_{alpha beta} = x_beta e_alpha` and

```
G_hat = (1/sigma^2) (Sigma_S kron I_p),   Sigma_S = (1/|S|) sum_x x x^T.
```

The spectrum is `{ lambda_j(Sigma_S)/sigma^2 }` each with multiplicity `p`. The apparatus engineers
`Sigma_S` to have `k=40` positive eigenvalues on `[0.02, 1.0]` (exp03a uses `eig = linspace(1.0,0.02,40)`)
and `d - k = 24` exact zeros. Therefore the FIM spectrum is known in closed form:

- **Exact-zero eigenvalues:** `(d - k) * p` = `24 * 8 = 192` (exp03a), `24 * 128 = 3072` (exp01).
- **Nonzero eigenvalues:** `k * p` = `320` (exp03a), spanning a designed 50x (1.7 decade) range.

Compute `G_hat` on the fitted twins over `S_test`, report the eigenspectrum, and read off two things:
the **kernel** (exact zeros) and the **conditioning** (`lambda_max / lambda_min` over nonzeros). Then use
Cramer-Rao: `Cov(theta_hat) >= G_hat^{-1}`, so the estimator variance in eigen-direction `a` is `>= 1/lambda_a`.

The reframe of the spine:
- **Identification** needs only the top (stiff) eigendirections. Those carry almost all between-subject
  discriminative signal and have the smallest Cramer-Rao variance, so they are estimable at tiny `N`.
  Identification is therefore a **practically identifiable coarse coordinate** (Section 4).
- **Reconstruction** needs the identity-carrying **sloppy** directions (small nonzero `lambda_a`), whose
  Cramer-Rao variance `1/lambda_a` is enormous. At achievable `N` they are practically non-identifiable.

"Structure identifies, function individuates" becomes: *the coarse structural summaries (node strength,
binarized topology) load on the stiff eigendirections; the fine individual weight-function is spread
across the sloppy tail.*

### New falsifiable predictions
- **P1a (zero-parameter kernel prediction).** Perturbations confined to the measured FIM kernel leave
  identification exactly flat. The kernel dimension must equal `(d-k)*p` (linear) or the symmetry-group
  dimension (Section 3). This reproduces `beta_null` range `0.000` and `PERM_SCALE` id-diff `0.000` with
  no free parameters. Falsifier: a nonzero id change along a numerically-confirmed kernel direction.
- **P1b (Cramer-Rao reconstruction floor).** The variance of any individuating parameter estimate scales
  as `sigma^2 / (N * lambda_a)` for its FIM eigenvalue `lambda_a`. Predict the digital-brain N=4 and
  exp04 N=8 nulls are the direct consequence of the smallest individuating `lambda_a`; the FIM computed
  on the synthetic twin should predict the observed reconstruction MDES (0.59 at N=8) to within its CI.
  Falsifier: observed MDES not tracking `1/sqrt(N lambda_a)`.

---

## 2. Sloppy-model theory (Transtrum, Machta, Sethna)

### Definition
A model is **sloppy** when its FIM eigenvalues are roughly log-uniformly spaced across many decades:
`log lambda_a ~ log lambda_1 - a * Delta`. Stiff directions (large `lambda`) are tightly constrained;
sloppy directions (small `lambda`) are barely constrained. The model manifold is a hyperribbon whose
widths `1/sqrt(lambda_a)` form a geometric hierarchy.

The functional-equivalence class splits cleanly in this language:
- **Exact symmetries** = **exactly-zero** FIM eigenvalues (flat gauge). These are `beta_null` and
  `PERM_SCALE`: moving along them changes nothing.
- **Sloppy directions** = **nonzero but tiny** eigenvalues spanning orders of magnitude. These are
  `beta_lowvar` (gentle decline). Moving along them changes identity a little.

The consequence is that the equivalence class is **sample-size relative**: the *practical* equivalence
class at `(N, sigma)` is the exact kernel PLUS every direction with `lambda_a < lambda_crit(N,sigma)`.
"Identity is the functional-equivalence class of the weights" is therefore a precise, `N`-dependent
statement, not a slogan.

### Exact quantity on this program's artifacts
Rank the individuating directions by between-subject signal energy `s_j` (this is exactly what exp03a's
`alpha` does: it retains the top-`m` functional dims ranked by deviation energy `sum(dev^2)`). Because the
linear model has white residual, `s_j` is proportional to the Fisher information in dim `j`. Define the
cumulative individuating-information curve `C(m) = sum_{j<=m} s_j / sum_j s_j`, and let identification
accuracy be the monotone (logistic-type) decoder response to retained SNR `sum_{j<=m} s_j / sigma^2`.

**The alpha-star prediction (the headline).** `alpha*` must equal the fraction `m*/k` at which the
cumulative stiff information first drives identification across the halfway-to-ceiling mark
`half = chance + 0.5(FULL - chance)`. On the engineered linear spectrum (`eig = linspace(1,0.02,40)`)
the top-26/40 dims carry 87.4% of stimulus energy, and the measured `alpha* = 0.642 +/- 0.008` (exp06)
is the accuracy-halfway crossing (`half = 0.401`, curve crosses between `alpha=0.6 -> 0.326` and
`alpha=0.65`). The 0.8 quoted in older docs is a grid artifact (exp03a's coarse grid jumps 0.6 -> 0.8);
use 0.642. The framework's testable claim: compute `G_hat`, form `C(m)`, and the predicted accuracy-halfway
`m*/k` must reproduce 0.642. Disagreement localizes the error (dev-energy ranking not equal to FIM ranking,
or accuracy not governed by cumulative Fisher information).

### New falsifiable predictions
- **P2a (real spectra are sloppier, so alpha-star drops).** The apparatus spectrum is engineered *linear*,
  not log-uniform. On a genuinely sloppy substrate (flyvis or Beiran connectome-constrained ensemble) the
  between-individual FIM spectrum will be log-uniform spanning >= 3 decades, which concentrates identity in
  few stiff directions and therefore predicts `alpha* (or f*)` **strictly smaller than 0.64**. Falsifier: a
  flat (non-sloppy) spectrum, or `alpha* >= 0.64` on a real ensemble.
- **P2b (sloppy floor equals the null-of-unknown-meaning).** The eigenvalue index where
  `lambda_a < lambda_crit(N=8, sigma_fMRI)` must coincide with the ROI count that goes non-significant in
  exp04. Predict: raising `N` until `lambda_crit` drops below the individuating eigenvalue is exactly the
  `N` at which digital-brain flips from 0/25 to >0/25 FDR-surviving. Ties Section 2 to the powered design.

---

## 3. Lie-group symmetry and the identity quotient manifold

### Definition
For the ReLU network the symmetry group is `G = S_H |x (R_{>0})^H` (permutations semidirect positive
per-unit rescaling), acting by

```
g = (P, c):  (W1, W2)  ->  ( diag(c) P W1 ,  W2 P^T diag(c)^{-1} ).
```

Invariance is exact by positive homogeneity `relu(c z) = c relu(z)` (`c>0`) and permutation symmetry of the
hidden sum, and `perm_scale` in `exp03b` implements precisely this with `c ~ U[0.5, 2.0]`. `G` is a Lie
group with `H!` connected components each of continuous dimension `H`. For a generic one-hidden-layer ReLU
net with distinct weights, `G` is the *entire* symmetry group (an identifiability theorem), so the statement
below is tight.

**Identity is the equivalence class = the orbit.** Define parameter space `Theta`, the group action above,
and the quotient `M = Theta / G`. Reconstruction-identity of network `i` is the point
`[theta_i] = G . theta_i in M`, not any representative. A capture preserves identity iff it lands in the
same orbit. The metric that matters is the quotient/orbit distance `d_M([theta],[theta'])`, never the
Euclidean `||theta - theta'||`.

### Exact quantity on this program's artifacts
- Dimension count (exp03b, `(d,H,p)=(32,24,8)`): `dim Theta = Hd + pH = 960`; `dim G = H = 24` (continuous
  scaling; permutations add 0 dimensions); generic orbit is 24-dimensional; `dim M = 936`. So the model
  manifold has codimension `>= 24`, i.e. **at least 24 exactly-zero FIM eigenvalues** (Section 1 kernel test,
  nonlinear case).
- exp03b already measures the orbit vs off-orbit contrast: `PERM_SCALE` moves `||dW||/||W|| = 1.569`
  (huge, on-orbit) with id-acc unchanged (`FULL - PERM_SCALE = 0.000`), while `FUNC_PERTURB` of *matched*
  magnitude `1.569` (off-orbit) collapses id-acc `1.000 -> 0.243`. This is the empirical statement that
  `d_M`, not `||dtheta||`, governs identity.
- **This also fixes the `d = 33` error.** The reported Cohen `d = 33.3` for PERM_SCALE-vs-FUNC_PERTURB is a
  `sd -> 0` artifact (`RIGOR_FINDINGS.md` F3). The quotient view explains *why*: PERM_SCALE has orbit
  distance `d_M = 0` exactly, so any Euclidean-in-Theta effect size is category-mistaken. Report the
  invariant orbit distance and "perfect separation," never `d = 33`.

### New falsifiable predictions
- **P3a (invariant decoder dominates the weight decoder).** Build a `G`-invariant embedding (canonicalize:
  sort hidden units by `||W1 row|| * ||W2 col||`, normalize per-unit scale; or use the function-space Gram
  matrix on a fixed probe battery). A decoder on this embedding must (i) have zero variance across a
  PERM_SCALE orbit and (ii) strictly beat a raw-weight-distance decoder, which is fooled by PERM_SCALE.
  Falsifier: raw weight-space distance predicts id-acc as well as orbit distance, in which case the
  equivalence-class framing adds nothing.
- **P3b (gauge is never worth storing).** The permutation gauge alone is `log2(H!) = 79 bits` for `H=24`,
  plus `H` continuous scale bits; a capture that gauge-fixes before quantizing achieves identical `S(f)` at
  strictly fewer bits (quantified in Section 8). Falsifier: gauge-fixed capture needs the same or more bits.

---

## 4. Structural vs practical identifiability (profile likelihood, Raue et al.)

### Definition
- **Structural identifiability:** with unlimited noise-free data, is `theta` (modulo declared symmetries)
  uniquely recoverable? A direction is structurally non-identifiable iff a continuum of `theta` gives
  identical predictions for all data. That is exactly the FIM exact-zero direction = the `G`-orbit / null
  space (Sections 1, 3).
- **Practical identifiability (Raue 2009):** with the actual finite noisy data, is `theta_a` pinned to a
  finite confidence interval? Tested by the **profile likelihood** `PL(theta_a) = min_{theta_{!=a}} chi^2(theta)`.
  Non-identifiable iff `PL` stays below the threshold `chi^2_{1-alpha}` out to `+/- infinity`, even though
  the parameter is structurally identifiable.

Mapping to the spine:
- **Identification = practically identifiable coarse coordinate.** The "which subject" label is a coarse
  projection onto the top stiff eigendirections; its profile likelihood is sharply peaked, identifiable at
  `N = 248` (ABIDE) even from binarized topology.
- **Reconstruction = structurally identifiable (given the ReLU quotient) but practically non-identifiable
  at achievable N.** The sloppy individuating combinations have flat profiles at `N = 4` or `N = 8`. This is
  precisely the honest correction (`RIGOR_FINDINGS.md` F1): digital-brain and exp04 are "no power," not "no
  effect." Claim 3 (noise-free simulation) proves the effect is structurally present; profile-likelihood
  flatness proves it is practically unresolved at that `N`.

### Exact quantity on this program's artifacts
For the human reconstruction twins, take the scalar individuating coordinate `alpha` (fraction of individual
weight-function present) or a between-subject discriminant, and compute its profile likelihood. Predict its
confidence interval spans essentially chance-to-full at `N = 8` (flat profile) and collapses to a point at
large `N`. The `N` at which the profile first rises past `chi^2_{0.95}` is the **practical-identifiability
threshold** `N*`, which is the sample-size target for the preregistered powered test. This is the MDES of
Section 7 rendered in likelihood geometry.

### New falsifiable prediction
- **P4 (the reconstruction sample threshold has a closed form).** `N* ~ sigma^2 / (lambda_min-indiv * effect^2)`,
  set by the *smallest FIM eigenvalue that still carries individuating signal*. Because that eigenvalue is
  sloppy (tiny), `N*` is large, quantifying "reconstruction is hard." Predict the concrete `N*` at which
  digital-brain reaches >=1/25 FDR-surviving and exp04's per-ROI MDES falls from 0.59 to the observed 0.175;
  the FIM-derived `N*` and the empirical `N*` must agree. Falsifier: reconstruction still practically
  non-identifiable at the predicted `N*`, which would mean either even worse sloppiness or that Claim 3's
  structural identifiability does not transfer to human data.

---

## 5. Neural-manifold capacity + participation ratio (Chung, Cohen, Sompolinsky 2018)

### Definition
- **Participation ratio** `PR = (sum lambda_i)^2 / sum lambda_i^2` of a representation's covariance
  eigenvalues is its effective dimensionality.
- **Manifold classification capacity:** for `P` object manifolds in feature dimension `D`, the critical load
  `alpha_c = P/D` at which random dichotomies stop being linearly separable is a function of the manifold
  **radius** `R_M` (extent relative to center separation) and **dimension** `D_M` (a PR-like effective dim):
  `alpha_c = alpha_M(R_M, D_M)`, with the point-manifold anchor `alpha_c = 2` (Cover).

Mapping "function individuates" rigorously: each subject `i`, across the stimulus battery and trial noise,
traces a **per-subject response manifold** `M_i` in `R^p`. Individuation = the `{M_i}` are linearly
separable. Manifold capacity is the exact measure: identification succeeds iff `N/p < alpha_c(R_M, D_M)`.
This decomposes the spine geometrically:
- **Center separation** is coarse and structural (where a subject's mean pattern sits, driven by node
  strength / topology). It drives identification.
- **Radius and dimension** are the fine geometry. They drive reconstruction / within-class individuation.

### Exact quantity on this program's artifacts
Measure `R_M`, `D_M`, and center-separation for the ABIDE fingerprint manifolds (`exp05`), and the
between-subject `PR` of stimulus-evoked geometry in digital-brain / exp04. ABIDE sits at `N/p = 248/200 = 1.24`;
identification at 0.98 requires `alpha_c > 1.24`, achievable only with small `R_M` and well-separated
centers, consistent with node strength alone reaching 0.446 and topology 0.972.

### New falsifiable predictions
- **P5a (capacity collapses inside the equivalence class).** After regressing out node strength and
  binarized topology, the residual per-subject manifolds must have `alpha_c ~ chance` at `N = 8`
  (explaining the reconstruction null), while the noise-free simulation (Claim 3, unlimited trials) has
  `alpha_c >> N/p` (perfect separation). MDES becomes the `N` at which `N/p` crosses the residual `alpha_c`.
  Falsifier: residual manifolds already separable at `N = 8` (would mean the human null is principle, not power).
- **P5b (function individuates in more dimensions than structure).** Between-subject `PR` of stimulus-evoked
  geometry `>` between-subject `PR` of connectome fingerprints. This is *why* reconstruction needs more data
  (more effective dimensions to estimate), unifying with Sections 1, 2, 4. Falsifier: comparable `PR`, which
  would deny the structure/function dimensional distinction.

---

## 6. Random matrix theory / Marchenko-Pastur null

### Definition
For `X` of shape `n x q` with iid entries variance `sigma^2`, the sample covariance `X^T X / n` has limiting
spectral density the **Marchenko-Pastur** law on `[sigma^2 (1 - sqrt(gamma))^2, sigma^2 (1 + sqrt(gamma))^2]`
with `gamma = q/n`. Eigenvalues above the upper edge `lambda_+ = sigma^2 (1 + sqrt(gamma))^2` are signal;
those in the bulk are finite-sampling noise.

### Exact quantity on this program's artifacts
The ABIDE fingerprint (`exp05`) uses `q = 200` ROIs and `T` timepoints per split-half (`MIN_T = 120`).
The **correct null** for "is there subject structure beyond finite-sampling noise" is MP, not the
label-shuffle floor (`perm_p = 5e-4`, which is the 1/2001 permutation floor and, as `RIGOR_FINDINGS.md` F2
notes, does not even compare arms). Compute `lambda_+`:

```
T = 120 -> gamma = 1.67, lambda_+ = 5.25   (gamma > 1: FC rank-deficient, huge bulk)
T = 150 -> gamma = 1.33, lambda_+ = 4.64
T = 200 -> gamma = 1.00, lambda_+ = 4.00
```

At `T = 120` the FC is rank-deficient and the MP bulk dominates, so much of the fingerprint can be
*reproducible finite-T noise structure* shared between split-halves of the same session. That is a rigorous
statement of the within-session confound flagged in Claim 1.

### New falsifiable predictions
- **P6a (the confound has an exact signature).** Project each fingerprint onto (i) the signal subspace
  (FC eigenvalues `> lambda_+`) and (ii) the MP bulk. Predict within-session identification draws
  substantially on the bulk, and a between-session split (HCP REST1/REST2) retains only the
  signal-subspace accuracy. If binarized-topology identification survives projection onto the signal
  subspace *and* survives between sessions, Claim 1 is safe; if it collapses to the bulk, "structure
  identifies" is a within-session SNR artifact. This turns the "must fix before publication" confound into
  a computable pass/fail.
- **P6b (TOST equivalence has an RMT mechanism).** The weighted-vs-binarized equivalence (Section 7) holds
  because both are dominated by the same few supra-MP eigenmodes. Predict the top signal-subspace
  eigenvectors of weighted FC and binarized FC have inner product ~ 1. Falsifier: divergent signal subspaces
  despite equal identification accuracy.

---

## 7. Equivalence testing (TOST), conformal prediction, power/MDES

### Definitions and mapping
- **TOST:** to establish equivalence within `+/- Delta`, reject both `theta <= -Delta` and `theta >= +Delta`.
  Already applied to Claim 1 (gap 0.008, 95% CI `[-1.9%, +3.5%]`, `p = 0.0012`). The framework's addition is
  a principled margin: `Delta` = the identification-irrelevant resolution = one stiff-FIM-direction's worth
  of accuracy, equivalently the design MDES. General recipe: two capture formats are identification-equivalent
  iff TOST on their id-accuracy gap falls within `Delta = MDES`.
- **Conformal prediction:** split-conformal gives distribution-free finite-sample coverage
  `P(y in C_hat(x)) >= 1 - alpha` under exchangeability (neurobridge target 0.90, achieved `>= 0.92`;
  specificity gate abstains under shift). The honest reconstruction claim is a conformal statement:
  reconstruction succeeds iff the conformal behavioral-prediction set for the reconstructed model contains
  the true behavior at coverage `1 - alpha` with set size below a usefulness threshold. Abstention is the
  correct output when the sloppy directions are unresolved (Section 4).
- **Power / MDES:** exp04 per-ROI MDES `~ 0.59` accuracy at `N = 8`, observed 0.175. MDES is the statistical
  shadow of the FIM sloppy floor: `MDES ~ 1 / sqrt(N * lambda_stiffest-individuating)`.

TOST margin, conformal set size, and MDES are three views of the same resolution scale set by the FIM
spectrum at sample size `N`.

### New falsifiable predictions
- **P7a (the cheapest dissociation in the program).** Weighted `~` binarized equivalence must **fail** for
  reconstruction. Run the identical TOST on a behavioral-agreement endpoint instead of identification:
  predict the binarized-vs-weighted gap exceeds any reasonable equivalence margin (fine weights matter for
  behavior, not for ID). One connectome ensemble, one figure: TOST-equivalent for identification,
  TOST-inequivalent for behavioral reconstruction. Falsifier: binarized `~` weighted for behavior too, which
  is also the `S(0)` test of Section 8 and would collapse the spine.
- **P7b (conformal set size is the dual of S(f)).** Coverage of the reconstructed-behavior conformal set
  degrades gracefully with fidelity `f`, and set size explodes precisely at `f < f*`. Predict `set-size(f)`
  is the operational dual of `S(f)`; the abstention rate crosses 50% at `f*`.

---

## 8. Rate-distortion theory

### Definition
For source `W ~ p(W)` and distortion `D(W, W_hat)`, the rate-distortion function
`R(D) = min_{p(W_hat|W): E[D] <= D} I(W; W_hat)` is the minimum bits to reconstruct within average
distortion `D`. It is convex, decreasing, `R(D_max) = 0`.

**S(f) is a distortion-rate curve with a behavioral distortion measure.** The program's move is to measure
distortion in **behavior space**, not weight space:

```
d_beh(W, W_hat) = E_{x in S_test} [ Delta( g(W, x), g(W_hat, x) ) ].
```

Because distortion is behavioral, weight changes *within* the functional-equivalence class have exactly
zero distortion (Section 3). So the rate-distortion source is naturally the **quotient** `W / G`. The rate is
`bits/synapse * synapses captured`, anchored to Bartol 4.7 bits/synapse. `S(f) = mean_i A_i(f)` is the
achievable (operational) distortion-rate curve of the program's specific capture family, an upper bound on
the true `R(D)`, and

```
f* = min { f : S(f) >= tau }   <->   the rate R(D = 1 - tau) at acceptable behavioral distortion.
```

### Rigorous statements and exact quantities
1. **The budget excludes the gauge.** Behavioral distortion is `G`-invariant, so the relevant source entropy
   is `H(W/G) = H(W) - H(orbit) < H(W)`: never store the permutation label or per-unit scale. Minimum
   behavioral-revival budget `= R_{W/G}(D)`. For the exp03b net, the permutation gauge alone is
   `log2(H!) = 79 bits` (`H = 24`) that a gauge-fixed capture provably saves (Section 3, P3b). On flyvis,
   the headline number is `f* * N_syn * 4.7 - log2|G|` bits.
2. **Reverse water-filling sets the shape.** For a Gaussian source with eigen-energies `s_j`,
   `R(D) = sum_j (1/2) log2(s_j / D_j)` with a water level, spending bits on the stiff (high-`s_j`)
   directions first. That is exactly exp03a's `alpha`-ranking by deviation energy. So the rate-optimal
   capture and the exp03a ranking coincide, and optimal allocation dominates uniform quantization:
   `f*(optimal) < f*(uniform)`.
3. **Wiring-only anchor.** `S(0)` at chance means the connectome carries ~0 bits about the behavioral
   quotient beyond the group, i.e. `I(connectome; individual behavior | group) ~ 0`.

### New falsifiable predictions
- **P8a (one number, four derivations).** `f*` (rate at halfway behavioral distortion) equals `alpha*`
  (Section 2 sufficiency crossing) equals the stiff-eigenvalue cumulative crossing on the same FIM spectrum,
  when both use the same halfway threshold. Predict `f* ~ alpha* = 0.642 +/- 0.008` on the linear apparatus,
  and both shift together (smaller) on a sloppier real ensemble (P2a). Falsifier: `f*` and `alpha*` diverge
  under matched thresholds, which would mean behavioral distortion and identification resolution are set by
  different parameter combinations.
- **P8b (gauge-fixing saves a countable number of bits).** Gauge-fixed capture achieves the same `S(f)` at
  `log2(H!) + H * b_scale` fewer bits than naive per-weight quantization (`b_scale` = bits per scale value).
  Quantify on flyvis/Beiran and check the `S(f)` curves overlay after the shift. Falsifier: no bit saving.
- **P8c (S(0) is the strong preservation test).** Predict `S(0) ~ chance` and `I(C; individual behavior|group) ~ 0`
  on the connectome-constrained ensemble. This is the falsifier for genesis's individuation-from-connectome
  claim: if `I(C; individual behavior) > 0` substantially, wiring is not free-and-empty and `S(0) > chance`,
  which would be the most important negative in the program.

---

## 9. The unification and the one decisive experiment

The four resolution scales below are the same crossing on one Fisher spectrum, computed four ways:

| Scale | Formalism | Definition | Value on the linear apparatus |
|---|---|---|---|
| `alpha*` | sloppy models (Sec 2) | sufficiency fraction, accuracy halfway crossing | `0.642 +/- 0.008` (exp06) |
| `f*` | rate-distortion (Sec 8) | rate at acceptable behavioral distortion | predicted `~ alpha*` |
| `m*/k` | information geometry (Sec 1) | cumulative stiff-FIM crossing | predicted `~ alpha*` |
| `N*`/MDES | practical identifiability (Sec 4, 7) | sample threshold for the individuating coordinate | set by smallest individuating `lambda` |

**Decisive test.** Compute the between-individual FIM `G_hat` on one real connectome-constrained ensemble
(flyvis or the Beiran RNN, both scanner-free). Then, from the *single* eigenspectrum, predict, ahead of
running the tasks: (i) the number of exact-zero directions (Section 3 dimension count), (ii) `alpha*` via
the cumulative-stiffness crossing (Section 2), (iii) `f*` and the gauge-adjusted bit-budget (Section 8), and
(iv) the reconstruction `N*` / MDES (Section 4). Running the identification and behavioral-reconstruction
tasks then tests all four predictions at once. If the four numbers agree and match the spectrum, the spine
is a theorem with a measured constant. If `alpha* / f*` come out below 0.64 with a log-uniform spectrum, the
program gains its first real-network confirmation that sloppiness concentrates identity. If they diverge, the
divergence localizes exactly which link (ranking, decoder, distortion measure) is wrong.

### Corrections this framework enforces (consistent with RIGOR_FINDINGS.md)
- Never report `d = 33`: PERM_SCALE has orbit distance exactly 0; use the `G`-invariant orbit distance and
  "perfect separation" (Sec 3).
- Use `alpha* = 0.642`, not 0.8; the 0.8 is a coarse-grid artifact (Sec 2).
- Replace the ABIDE `perm_p = 5e-4` between-arm claim with the MP null and TOST equivalence (Sec 6, 7).
- State reconstruction as practically non-identifiable at achievable `N` (structurally identifiable per
  Claim 3), never as "reconstruction is dear" (Sec 4).
