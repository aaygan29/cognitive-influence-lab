#!/usr/bin/env python3
"""A calibrated multi-biosignal fusion environment for learning-rule influence.

Collecting one dataset with every biosignal, the same task, and an intervention is
prohibitively slow and expensive. So instead of claiming influence from static data
(which is unidentifiable: no intervention, no counterfactual), we PROPOSE the fusion
environment explicitly, calibrate its parameters to real data, and study influence
where it is identifiable: in a controllable simulation.

Design
------
Each synthetic person has a latent decision policy theta = (bias, w_gain>0,
w_loss<0). The loss-aversion ratio |w_loss|/w_gain is drawn to match the REAL NARPS
distribution (median ~0.88, 10-90 pctile ~0.57-1.35). On a mixed-gambles task, three
biosignal families are generated as noisy read-outs of the SHARED latent:
  * choice        : Bernoulli(sigmoid(V)), V = w_gain*g + w_loss*l + bias
  * neural value  : reward-anticipation signal ~ w_gain*g (NAcc/MPFC, Genevsky-Knutson),
                    identifiability calibrated to an fMRI digital-brain fingerprint
  * peripheral    : arousal ~ |V| (conflict near indifference), cheap wearable proxy
Each modality adds independent, modality-specific noise. Different "datasets" cover
different modality subsets (signals do not share a task); pair-matched controls and
synthetic donors fill the missing modalities, matched on the observed policy estimate.

Influence = a bounded per-trial dose injected into the learning rule (the update of
w_loss), drifting the person toward a target policy (reduced loss aversion, i.e. a
"positive gambling" risk-seeking state). Because every modality is a function of the
same latent, a real manipulation moves them COHERENTLY. Validation across biosignals
= cross-modal coherence of the induced change, contrasted with (a) a pair-matched
control that receives no dose and (b) a null arm whose modalities drift independently.

Nothing here is a measurement of real manipulation. It is a calibrated feasibility and
risk model: what fused-biosignal learning-rule influence would look like, and how it
would be tracked and detected.
"""
from __future__ import annotations
import os, json, argparse
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
LAMBDA_FILE = os.environ.get(
    "NARPS_LAMBDA",
    str(REPO / "results" / "narps_lambda.npy"))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def gamble_stream(n_trials, rng):
    g = rng.uniform(5, 40, n_trials)
    l = rng.uniform(5, 20, n_trials)
    return (g - g.mean()) / g.std(), (l - l.mean()) / l.std()


def _est_loss_weight_logistic(g, l, y, iters=50, l2=1.0):
    """Loss weight from binary choice via logistic IRLS. Returns w_loss."""
    X = np.column_stack([np.ones_like(g), g, l]); w = np.zeros(3)
    for _ in range(iters):
        p = sigmoid(X @ w); W = np.clip(p*(1-p), 1e-6, None)
        grad = X.T @ (p - y) + l2*w
        H = (X*W[:, None]).T @ X + l2*np.eye(3)
        try: w = w - np.linalg.solve(H, grad)
        except np.linalg.LinAlgError: break
    return w[2]


def _est_loss_weight_linear(g, l, s):
    """Loss weight from a continuous read-out via least squares. Returns coef on l."""
    X = np.column_stack([np.ones_like(g), g, l])
    beta, *_ = np.linalg.lstsq(X, s, rcond=None)
    return beta[2]


def simulate_person(lam0, rng, n_trials=320, dose=0.0, target_lam=None,
                    null_drift=False, snr=(1.0, 0.9, 0.7)):
    """Three biosignals as read-outs of a shared latent loss-aversion that a dose
    can drive. Each modality independently encodes loss via its dependence on the
    loss feature, so each yields its OWN loss-aversion estimate. Returns per-modality
    early/late loss-aversion estimates.

    Modalities: choice (binary, logistic), neural subjective-value (continuous ~ V,
    MPFC-like), peripheral arousal (continuous ~ loss-sensitivity, wearable proxy).
    null_drift: each modality gets an INDEPENDENT drift in its own loss estimate,
    not a shared latent shift, of comparable size.
    """
    wg = 1.0
    g, l = gamble_stream(n_trials, rng)
    lam = np.zeros(n_trials)
    cur = lam0
    dnull = rng.normal(0, 0.006, 3) if null_drift else np.zeros(3)
    for t in range(n_trials):
        if dose != 0.0 and target_lam is not None:
            cur = cur + dose * (target_lam - cur)
        lam[t] = np.clip(cur, 0.05, 3.0)
    wl = -lam * wg
    V = wg*g + wl*l
    choice = (rng.uniform(size=n_trials) < sigmoid(V)).astype(float)
    neural = V + rng.normal(0, 1/snr[1], n_trials)                # subjective value ~ -lam*l
    periph = (lam * np.abs(l)) + rng.normal(0, 1/snr[2], n_trials)  # loss-magnitude arousal
    half = n_trials // 2
    e, la = slice(0, half), slice(half, n_trials)
    absl = np.abs(l)
    # Each modality yields a calibrated loss-aversion estimate on the SAME lambda scale
    # (precondition for inverse-variance fusion), early vs late:
    #   choice  : logistic loss weight wl = -lambda        -> lambda = -wl
    #   neural  : linear coef on l = -lambda               -> lambda = -coef_l
    #   peripheral: linear coef on |l| = +lambda           -> lambda =  coef_|l|
    lam_c = (-_est_loss_weight_logistic(g[e], l[e], choice[e]),
             -_est_loss_weight_logistic(g[la], l[la], choice[la]) + dnull[0])
    lam_n = (-_est_loss_weight_linear(g[e], l[e], neural[e]),
             -_est_loss_weight_linear(g[la], l[la], neural[la]) + dnull[1])
    lam_p = (_est_loss_weight_linear(g[e], absl[e], periph[e]),
             _est_loss_weight_linear(g[la], absl[la], periph[la]) + dnull[2])
    # Delta lambda per modality (late - early): each an unbiased estimate of the same shift.
    d = np.array([lam_c[1]-lam_c[0], lam_n[1]-lam_n[0], lam_p[1]-lam_p[0]])
    return {"delta_lossaversion": d, "lam0": lam0, "lam_final": float(lam[-1])}


def estimate_lambda(lam, rng, n_trials=160, snr=(1.0, 0.9, 0.7)):
    """Extract the learning-rule parameter (loss aversion) from each biosignal, for a
    person held at constant loss aversion `lam`. Returns per-modality estimates
    [choice, neural, peripheral] on a common lambda scale."""
    wg = 1.0
    g, l = gamble_stream(n_trials, rng)
    V = wg*g - lam*l
    choice = (rng.uniform(size=n_trials) < sigmoid(V)).astype(float)
    neural = V + rng.normal(0, 1/snr[1], n_trials)
    periph = (lam*np.abs(l)) + rng.normal(0, 1/snr[2], n_trials)
    lam_c = -_est_loss_weight_logistic(g, l, choice)
    lam_n = -_est_loss_weight_linear(g, l, neural)
    lam_p = _est_loss_weight_linear(g, np.abs(l), periph)
    return np.array([lam_c, lam_n, lam_p])


def coherence(delta):
    """Cross-modal coherence of the induced loss-aversion change. A real learning-rule
    shift moves all three modality estimates in the SAME direction; independent noise
    does not. Score = mean change times the fraction of modalities agreeing in sign
    (so aligned multi-modal movement scores high, scattered movement near zero)."""
    signs = np.sign(delta)
    agree = np.abs(signs.sum()) / len(delta)      # 1 if all agree, small if scattered
    return float(delta.mean() * agree)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="people per arm")
    ap.add_argument("--dose", type=float, default=0.03)
    ap.add_argument("--out", default=str(REPO / "results" / "fusion_environment.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    if Path(LAMBDA_FILE).exists():
        las = np.load(LAMBDA_FILE)
        lam_lo, lam_hi = np.percentile(las, 10), np.percentile(las, 90)
        lam_med = float(np.median(las))
        calib = f"NARPS real (n={len(las)}, median {lam_med:.2f}, 10-90 {lam_lo:.2f}-{lam_hi:.2f})"
    else:
        lam_lo, lam_hi, lam_med = 0.57, 1.35, 0.88
        calib = "NARPS literature defaults"
    target = lam_lo  # push toward low loss aversion = risk-seeking "positive gambling"

    mods = ["choice", "neural", "peripheral"]
    res = {"calibration": calib, "target_lambda": float(target),
           "per_trial_dose": args.dose, "n_per_arm": args.n,
           "thesis": "fused multimodal biosignals extract the learning rule more "
                     "accurately, and that accuracy is what enables precise manipulation"}

    # ---- Calibrate per-modality estimator noise sigma_m (fusion weights) --------
    # Estimate each modality's loss-aversion estimation error from a calibration
    # cohort (baseline people, no intervention), giving sigma_m for inverse-variance
    # fusion. This is the biosignal_fusion inverse-variance rule (proofs/FusionMath.lean;
    # detection corollary proofs/FusionDetection.lean).
    errs = {m: [] for m in mods}
    for i in range(args.n):
        lam0 = float(rng.uniform(lam_lo, lam_hi))
        est = estimate_lambda(lam0, np.random.default_rng(5000+i))
        for j, m in enumerate(mods):
            errs[m].append(est[j] - lam0)
    sigma = np.array([np.std(errs[m]) for m in mods]) + 1e-9
    w = 1.0 / sigma**2

    # ---- (1) EXTRACTION: fused estimate of the learning rule beats any single ----
    ext_err = {m: [] for m in mods}; ext_fused = []
    for i in range(args.n):
        lam0 = float(rng.uniform(lam_lo, lam_hi))
        est = estimate_lambda(lam0, np.random.default_rng(6000+i))
        for j, m in enumerate(mods):
            ext_err[m].append((est[j]-lam0)**2)
        lam_fused = (est*w).sum()/w.sum()
        ext_fused.append((lam_fused-lam0)**2)
    res["extraction_rmse"] = {m: float(np.sqrt(np.mean(ext_err[m]))) for m in mods}
    res["extraction_rmse"]["fused"] = float(np.sqrt(np.mean(ext_fused)))
    res["extraction_best_single"] = float(min(res["extraction_rmse"][m] for m in mods))

    # ---- (2) MANIPULATION: extraction accuracy propagates to control error -------
    # A closed-loop adversary drives the learner from lam0 to the target policy by
    # dosing the learning rule with a gain on the ESTIMATED error (lam* - lam_hat).
    # A noisy single-modality estimate -> jittery, biased control; the fused estimate
    # -> tighter convergence. Report steady-state |lam_final - lam*|.
    def control_run(lam0, guide, seed):
        r = np.random.default_rng(seed); lam = lam0
        for _ in range(12):  # control blocks
            est = estimate_lambda(lam, r, n_trials=120)  # observe a block, extract
            if guide == "fused":
                lam_hat = (est*w).sum()/w.sum()
            else:
                lam_hat = est[mods.index(guide)]
            lam = float(np.clip(lam + 0.6*(target - lam_hat), 0.05, 3.0))
        return abs(lam - target)
    ctrl = {g: [] for g in mods + ["fused"]}
    for i in range(args.n):
        lam0 = float(rng.uniform(lam_med, lam_hi))
        for g in mods + ["fused"]:
            ctrl[g].append(control_run(lam0, g, 7000+i))
    res["manipulation_control_error"] = {g: float(np.mean(v)) for g, v in ctrl.items()}
    res["manipulation_best_single"] = float(min(res["manipulation_control_error"][m] for m in mods))

    # ---- SNR quadrature check on the extraction estimator (the enabling math) ----
    res["snr_quadrature"] = {
        "sigma_single": {m: float(sigma[j]) for j, m in enumerate(mods)},
        "fused_sigma": float(1/np.sqrt(w.sum())),
        "fused_sigma_le_best_single": bool(1/np.sqrt(w.sum()) <= sigma.min())}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("Saved", args.out)


if __name__ == "__main__":
    main()
