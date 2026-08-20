"""Real fMRI arm: neural grounding of the affective channel on NARPS ds001734.

Downloads fmriprep-preprocessed MNI152 BOLD (one run per subject, streamed then deleted to bound
disk), fits a first-level GLM with gain and loss parametric regressors plus motion confounds, and
extracts nucleus-accumbens (NAcc) and anterior-insula betas. This tests the affective grounding the
estimator assumes:

- does NAcc track anticipated gain and anterior insula track anticipated loss (AIM / Knutson)?
- does a NEURAL loss-aversion index (how much NAcc drops per unit loss vs rises per unit gain)
  correlate across subjects with the BEHAVIORAL loss aversion recovered from choices (Tom, Fox,
  Trepel & Poldrack 2007)?

This is a real neural result on a subset of subjects, single run, honestly caveated. It grounds the
affective channel; it is not the Genevsky aggregate-market forecast (NARPS has no separate market
outcome), which needs a shared-stimulus design.

Coordinates (MNI152): NAcc +/-(10,12,-8); anterior insula +/-(36,20,-4); 6mm spheres.
Provenance: ds001734 v1.0.5 fmriprep derivatives, task-MGT, TR=1.0s.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import honesty as H  # noqa: E402
import real_narps as RN  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = "https://s3.amazonaws.com/openneuro.org/ds001734"
TMP = os.path.join(ROOT, "data", "narps", "fmri_tmp")
ROI_JSON = os.path.join(ROOT, "results", "narps_fmri_roi.json")
TR = 1.0
NACC = [(-10, 12, -8), (10, 12, -8)]
AINS = [(-36, 20, -4), (36, 20, -4)]


def _curl(url: str, out: str) -> bool:
    r = subprocess.run(["curl", "-sf", url, "-o", out])
    return r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 1000


def _subject_events(sub: str) -> pd.DataFrame:
    """Concatenate the subject's run-01 events into a parametric design (gamble/gain/loss)."""
    path = os.path.join(RN.EVENTS, f"{sub}_task-MGT_run-01_events.tsv")
    df = pd.read_csv(path, sep="\t")
    df = df[df["participant_response"].isin(RN.ACCEPT)].copy()
    g = df["gain"].to_numpy(float); l = df["loss"].to_numpy(float); on = df["onset"].to_numpy(float)
    dur = df["duration"].to_numpy(float)
    def block(name, amp):
        return pd.DataFrame({"onset": on, "duration": dur, "trial_type": name, "modulation": amp})
    return pd.concat([block("gamble", np.ones_like(g)),
                      block("gain", g - g.mean()),
                      block("loss", l - l.mean())], ignore_index=True)


def extract_subject(sub: str) -> Dict[str, float]:
    from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
    from nilearn.maskers import NiftiSpheresMasker
    import nibabel as nib

    os.makedirs(TMP, exist_ok=True)
    bold = os.path.join(TMP, f"{sub}_bold.nii.gz")
    conf = os.path.join(TMP, f"{sub}_conf.tsv")
    burl = f"{BASE}/derivatives/fmriprep/{sub}/func/{sub}_task-MGT_run-01_bold_space-MNI152NLin2009cAsym_preproc.nii.gz"
    curl_ok = _curl(burl, bold) and _curl(
        f"{BASE}/derivatives/fmriprep/{sub}/func/{sub}_task-MGT_run-01_bold_confounds.tsv", conf)
    if not curl_ok:
        return {}
    try:
        img = nib.load(bold)
        n_scans = img.shape[3]
        frame_times = np.arange(n_scans) * TR
        cf = pd.read_csv(conf, sep="\t")
        motion = cf[["X", "Y", "Z", "RotX", "RotY", "RotZ"]].fillna(0.0).to_numpy()
        events = _subject_events(sub)
        dm = make_first_level_design_matrix(frame_times, events=events, hrf_model="spm",
                                            drift_model="cosine", high_pass=0.01,
                                            add_regs=motion,
                                            add_reg_names=[f"m{i}" for i in range(6)])
        model = FirstLevelModel(t_r=TR, minimize_memory=True)
        model.fit(img, design_matrices=dm)
        out = {}
        for cname in ("gain", "loss"):
            emap = model.compute_contrast(cname, output_type="effect_size")
            for roi, seeds in (("nacc", NACC), ("ains", AINS)):
                masker = NiftiSpheresMasker(seeds=seeds, radius=6.0, allow_overlap=True)
                vals = masker.fit_transform(emap)  # (1, n_seeds)
                out[f"{roi}_{cname}"] = float(np.mean(vals))
        return out
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}
    finally:
        for f in (bold, conf):
            if os.path.exists(f):
                os.remove(f)


def run(subject_list: List[str]) -> Dict[str, object]:
    roi = {}
    if os.path.exists(ROI_JSON):
        with open(ROI_JSON) as fh:
            roi = json.load(fh)
    for i, sub in enumerate(subject_list):
        if sub in roi and "nacc_gain" in roi[sub]:
            continue
        print(f"[{i+1}/{len(subject_list)}] {sub} ...", flush=True)
        r = extract_subject(sub)
        roi[sub] = r
        with open(ROI_JSON, "w") as fh:
            json.dump(roi, fh, indent=2)
    return roi


def analyze(roi: Dict[str, dict]) -> Dict[str, object]:
    from scipy import stats
    good = {s: r for s, r in roi.items() if r and "nacc_gain" in r}
    nacc_gain = np.array([r["nacc_gain"] for r in good.values()])
    nacc_loss = np.array([r["nacc_loss"] for r in good.values()])
    ains_gain = np.array([r["ains_gain"] for r in good.values()])
    ains_loss = np.array([r["ains_loss"] for r in good.values()])

    def ttest(x):
        t, p = stats.ttest_1samp(x, 0.0)
        return {"mean": float(np.mean(x)), "t": float(t), "p": float(p), "n": int(len(x))}

    # neural loss aversion per subject: NAcc drop per unit loss vs rise per unit gain
    neural_la = -nacc_loss / nacc_gain
    fin = np.isfinite(neural_la) & (np.abs(nacc_gain) > 1e-6)
    # behavioral lambda for the same subjects
    real, _, per_subject = RN.run_real()
    beh = np.array([per_subject.get(s, {}).get("loss_aversion", np.nan) for s in good.keys()])
    both = fin & np.isfinite(beh)
    corr = stats.pearsonr(neural_la[both], beh[both]) if both.sum() >= 6 else (None, None)

    return {
        "experiment": "REAL_narps_fmri_affective_grounding",
        "n_subjects": len(good),
        "NAcc_tracks_gain": ttest(nacc_gain),
        "NAcc_response_to_loss": ttest(nacc_loss),
        "aIns_tracks_gain": ttest(ains_gain),
        "aIns_response_to_loss": ttest(ains_loss),
        "neural_vs_behavioral_loss_aversion": {
            "pearson_r": (float(corr[0]) if corr[0] is not None else None),
            "p": (float(corr[1]) if corr[1] is not None else None),
            "n": int(both.sum()),
        },
        "provenance": {"dataset": "ds001734 fmriprep", "run": "run-01", "TR": TR,
                       "rois_mni": {"nacc": NACC, "ains": AINS}, "sphere_mm": 6.0},
        "caveats": "single run per subject, ROI-sphere GLM on a subset; grounding test, not the market forecast",
    }


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    all_subs = sorted({os.path.basename(p).split("_")[0]
                       for p in glob.glob(os.path.join(RN.EVENTS, "*_run-01_events.tsv"))})
    subs = all_subs[:n]
    roi = run(subs)
    res = analyze(roi)
    with open(os.path.join(ROOT, "results", "real_narps_fmri.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print("\nREAL fMRI affective grounding (NARPS, n=%d subjects, run-01)" % res["n_subjects"])
    for k in ("NAcc_tracks_gain", "NAcc_response_to_loss", "aIns_tracks_gain", "aIns_response_to_loss"):
        e = res[k]
        print(f"  {k:22s} mean={e['mean']:+.3f}  t={e['t']:+.2f}  p={e['p']:.3f}  (n={e['n']})")
    nb = res["neural_vs_behavioral_loss_aversion"]
    print(f"  neural vs behavioral loss aversion: r={nb['pearson_r']} p={nb['p']} (n={nb['n']})")
