#!/usr/bin/env python3
"""Learning-rule vector on a real gambling task: individuation, fusion, and risk.

Represents each learner in the NARPS mixed-gambles task (Botvinik-Nezer et al.,
2020, ds001734) by the dynamics of their decision rule (a PsyTrack-style
logistic-GLM trajectory; Roy et al. 2021), then asks three questions that connect
the digital-brain individuation thread to the AIxBio biosignal-fusion safety
thread:

  E1  Individuation. Does the learning-rule vector re-identify a person from held
      -out trials, above chance? (fingerprinting analog of digital-brain)
  E2  Forecasting robustness. Does adding the personal learning-rule vector to a
      population choice model improve prediction of that person's *later* choices,
      and does a calibrated neural individuation vector help most when behavioural
      data is scarce? (the fusion thesis)
  E3  Risk. Is the loss-aversion coefficient a control surface: does a bounded
      adversarial dose shift choices predictably? (learning-as-control)

All numbers are behaviour-only except the explicitly-labelled *calibrated* neural
vector, whose identifiability is set to the measured digital-brain fingerprinting
rate. Nothing here is a claim about brains.

Usage:
    python scripts/run_learning_rule_risk.py \
        --narps ~/Downloads/datasets/ds001734-master \
        --neural-reid 1.0     # digital-brain BOLD5000 fingerprinting = 100%
"""
from __future__ import annotations

import argparse, glob, csv, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from learning_rule import (  # noqa: E402
    extract_learning_rule, perturb_loss_weight, _fit_logistic)

from sklearn.metrics import roc_auc_score  # noqa: E402

ACCEPT = {"strongly_accept": 1, "weakly_accept": 1,
          "weakly_reject": 0, "strongly_reject": 0}


def load_narps(narps_dir: Path):
    """Return {subject_id: dict(gain,loss,rt,accept)} in trial order."""
    subs = {}
    files = sorted(glob.glob(str(narps_dir / "sub-*/func/*_events.tsv")))
    if not files:  # allow a flat extracted layout too
        files = sorted(glob.glob(str(narps_dir / "**/sub-*/func/*_events.tsv"),
                                 recursive=True))
    per = {}
    for f in files:
        sid = f.split("/sub-")[1][:3]
        per.setdefault(sid, [])
        with open(f) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                r = row.get("participant_response", "").strip()
                if r not in ACCEPT:
                    continue
                try:
                    g = float(row["gain"]); l = float(row["loss"])
                    rt = float(row.get("RT", "nan"))
                except ValueError:
                    continue
                per[sid].append((g, l, rt, ACCEPT[r]))
    for sid, rows in per.items():
        a = np.array(rows, float)
        if len(a) >= 96:  # need enough trials for a temporal split
            subs[sid] = dict(gain=a[:, 0], loss=a[:, 1], rt=a[:, 2], accept=a[:, 3])
    return subs


def e1_individuation(subs, seed=0):
    """Re-identify subjects: fit learning-rule vector on each half, match A->B."""
    rng = np.random.default_rng(seed)
    ids = sorted(subs)
    va, vb = [], []
    for sid in ids:
        d = subs[sid]; n = len(d["accept"]); h = n // 2
        ra = extract_learning_rule(sid, d["gain"][:h], d["loss"][:h], d["accept"][:h])
        rb = extract_learning_rule(sid, d["gain"][h:], d["loss"][h:], d["accept"][h:])
        va.append(ra.vector()); vb.append(rb.vector())
    A = np.vstack(va); B = np.vstack(vb)
    mu, sd = A.mean(0), A.std(0) + 1e-8
    A = (A - mu) / sd; B = (B - mu) / sd
    # nearest-neighbour match A[i] -> nearest B
    D = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    pred = D.argmin(1)
    rank1 = float((pred == np.arange(len(ids))).mean())
    return dict(n_subjects=len(ids), rank1=rank1, chance=1.0 / len(ids))


def _design(d, idx):
    g, l, rt = d["gain"][idx], d["loss"][idx], d["rt"][idx]
    def z(a, ref):
        s = ref.std(); return (a - ref.mean()) / (s if s > 1e-8 else 1.0)
    rt = np.nan_to_num(rt, nan=np.nanmean(d["rt"]))
    X = np.column_stack([np.ones(len(idx)), z(g, d["gain"]), z(l, d["loss"]),
                         z(rt, np.nan_to_num(d["rt"], nan=np.nanmean(d["rt"])))])
    return X, d["accept"][idx]


def e2_forecasting(subs, neural_reid=1.0, n_early=None, seeds=(0, 1, 2, 3, 4)):
    """Late-trial AUC: population vs +personal learning-rule vs +neural vector."""
    ids = sorted(subs)
    out = {"pop": [], "pop_personal": [], "pop_personal_neural": [],
           "scarce_pop_personal": [], "scarce_pop_personal_neural": []}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(ids)
        train_ids = perm[: int(0.7 * len(ids))]
        test_ids = perm[int(0.7 * len(ids)):]
        # population model on TRAIN subjects (all their trials)
        Xtr, ytr = [], []
        for sid in train_ids:
            d = subs[sid]; X, y = _design(d, np.arange(len(d["accept"])))
            Xtr.append(X); ytr.append(y)
        Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
        w_pop = _fit_logistic(Xtr, ytr, l2=1.0)
        # calibrate neural-vector noise so re-ID ~ neural_reid (proxy: map rate to snr)
        # higher reid -> lower noise; simple monotone map
        neural_noise = max(1e-3, (1.0 - neural_reid) * 2.0 + 0.05)
        for sid in test_ids:
            d = subs[sid]; n = len(d["accept"]); h = n // 2
            early = np.arange(h); late = np.arange(h, n)
            Xl, yl = _design(d, late)
            if len(np.unique(yl)) < 2:
                continue
            # population-only
            p_pop = 1 / (1 + np.exp(-np.clip(Xl @ w_pop, -30, 30)))
            # personal model from EARLY trials (temporal split, no leak)
            Xe, ye = _design(d, early)
            w_per = _fit_logistic(Xe, ye, l2=2.0)
            p_per = 1 / (1 + np.exp(-np.clip(Xl @ w_per, -30, 30)))
            p_fuse = 0.5 * (p_pop + p_per)
            # calibrated neural vector: reads the subject's loss-aversion latent
            # (from early trials) with calibrated noise; used to shrink the
            # personal loss weight toward the neural read-out.
            latent_la = w_per[2]
            neural_read = latent_la + rng.normal(0, neural_noise)
            w_neu = w_per.copy(); w_neu[2] = 0.5 * (w_per[2] + neural_read)
            p_neu = 1 / (1 + np.exp(-np.clip(Xl @ w_neu, -30, 30)))
            p_fuse_neu = (p_pop + p_per + p_neu) / 3.0
            out["pop"].append(roc_auc_score(yl, p_pop))
            out["pop_personal"].append(roc_auc_score(yl, p_fuse))
            out["pop_personal_neural"].append(roc_auc_score(yl, p_fuse_neu))
            # AMBIGUOUS trials only: where the population model is unsure
            # (p in [0.3,0.7]) the choice is set by the person's policy, not the
            # gamble magnitudes. This is where individuation should matter.
            amb = (p_pop >= 0.3) & (p_pop <= 0.7)
            if amb.sum() >= 8 and len(np.unique(yl[amb])) == 2:
                out.setdefault("amb_pop", []).append(roc_auc_score(yl[amb], p_pop[amb]))
                out.setdefault("amb_pop_personal", []).append(roc_auc_score(yl[amb], p_fuse[amb]))
                out.setdefault("amb_pop_personal_neural", []).append(roc_auc_score(yl[amb], p_fuse_neu[amb]))
            # SCARCE regime: only first 20 early trials for the behavioural estimate
            sc = early[:20]
            Xsc, ysc = _design(d, sc)
            if len(np.unique(ysc)) < 2:
                continue
            w_sc = _fit_logistic(Xsc, ysc, l2=4.0)
            p_sc = 1 / (1 + np.exp(-np.clip(Xl @ w_sc, -30, 30)))
            p_sc_fuse = 0.5 * (p_pop + p_sc)
            neural_read_sc = latent_la + rng.normal(0, neural_noise)
            w_scn = w_sc.copy(); w_scn[2] = 0.5 * (w_sc[2] + neural_read_sc)
            p_scn = 1 / (1 + np.exp(-np.clip(Xl @ w_scn, -30, 30)))
            p_sc_neu = (p_pop + p_sc + p_scn) / 3.0
            out["scarce_pop_personal"].append(roc_auc_score(yl, p_sc_fuse))
            out["scarce_pop_personal_neural"].append(roc_auc_score(yl, p_sc_neu))
    return {k: dict(mean=float(np.mean(v)), sd=float(np.std(v)), n=len(v))
            for k, v in out.items() if v}


def e3_risk(subs, doses=(-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)):
    """Dose-response: bounded perturbation of the loss weight vs accept-rate shift."""
    ids = sorted(subs)
    curve = {}; curve_amb = {}
    for dose in doses:
        flips, flips_amb = [], []
        for sid in ids:
            d = subs[sid]
            r = extract_learning_rule(sid, d["gain"], d["loss"], d["accept"])
            p0, _ = perturb_loss_weight(r.weights_mean, d["gain"], d["loss"], 0.0)
            p1, _ = perturb_loss_weight(r.weights_mean, d["gain"], d["loss"], dose)
            dec0 = (p0 >= 0.5); dec1 = (p1 >= 0.5)
            flips.append(float((dec0 != dec1).mean()))          # fraction flipped
            amb = (p0 >= 0.3) & (p0 <= 0.7)
            if amb.sum() >= 4:
                flips_amb.append(float((dec0[amb] != dec1[amb]).mean()))
        curve[str(dose)] = dict(frac_decisions_flipped=float(np.mean(flips)),
                                sd=float(np.std(flips)))
        if flips_amb:
            curve_amb[str(dose)] = dict(frac_decisions_flipped=float(np.mean(flips_amb)),
                                        sd=float(np.std(flips_amb)))
    def _flip(c):
        for dose in sorted(doses, key=abs):
            if str(dose) in c and c[str(dose)]["frac_decisions_flipped"] >= 0.10:
                return dose
        return None
    return dict(dose_response=curve, dose_response_ambiguous=curve_amb,
                dose_for_10pct_flip=_flip(curve),
                dose_for_10pct_flip_ambiguous=_flip(curve_amb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--narps", default=str(Path.home() / "Downloads/datasets/ds001734-master"))
    ap.add_argument("--neural-reid", type=float, default=1.0,
                    help="digital-brain fingerprinting rate the neural vector is calibrated to")
    ap.add_argument("--out", default=str(REPO / "results" / "learning_rule_risk.json"))
    args = ap.parse_args()

    subs = load_narps(Path(args.narps).expanduser())
    print(f"Loaded {len(subs)} subjects with >=96 trials")
    if len(subs) < 10:
        print("Not enough subjects; check --narps path"); return 1

    res = {"dataset": "NARPS ds001734 (mixed gambles)",
           "n_subjects": len(subs),
           "neural_reid_calibration": args.neural_reid}
    print("E1 individuation..."); res["E1_individuation"] = e1_individuation(subs)
    print("  ", res["E1_individuation"])
    print("E2 forecasting..."); res["E2_forecasting"] = e2_forecasting(subs, args.neural_reid)
    for k, v in res["E2_forecasting"].items():
        print(f"   {k}: AUC {v['mean']:.3f} ± {v['sd']:.3f} (n={v['n']})")
    print("E3 risk..."); res["E3_risk"] = e3_risk(subs)
    print("   dose_for_10pct_flip (all):", res["E3_risk"]["dose_for_10pct_flip"])
    print("   dose_for_10pct_flip (ambiguous):", res["E3_risk"]["dose_for_10pct_flip_ambiguous"])
    print("   ambiguous flip curve:", {k: round(v["frac_decisions_flipped"], 3)
          for k, v in res["E3_risk"]["dose_response_ambiguous"].items()})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("Saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
