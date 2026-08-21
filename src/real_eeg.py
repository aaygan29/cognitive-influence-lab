"""EEG link (L7): the Reward Positivity on a real open EEG gambling dataset (OpenNeuro ds003458).

DEAP is licence-gated (EULA + academic-position verification), so we use an openly-downloadable EEG
reward dataset instead: the three-armed bandit gambling task (ds003458, 23 subjects, EEGLAB format).
We compute the Reward Positivity (RewP), the frontocentral ERP difference between win and loss
feedback around 250-350 ms. A win-positive RewP is the canonical EEG signature of reward processing,
so it grounds the affective/reward channel in a second modality (EEG), independent of the fMRI arm.

Streams each subject's data (download -> epoch -> extract -> delete) to bound disk. Frontocentral
cluster = FCz, Cz, FC1, FC2. Provenance: ds003458, task ThreeArmedBandit, feedback win=S5 / loss=S6.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from typing import Dict, List

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import honesty as H  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://s3.amazonaws.com/openneuro.org/ds003458"
TMP = os.path.join(ROOT, "data", "eeg_tmp")
ROI_JSON = os.path.join(ROOT, "results", "eeg_rewp_subjects.json")
CLUSTER = ["FCz", "Cz", "FC1", "FC2"]
WIN_WINDOW = (0.25, 0.35)   # RewP window (s)


def _curl(url, out):
    return subprocess.run(["curl", "-sf", url, "-o", out]).returncode == 0 and \
        os.path.exists(out) and os.path.getsize(out) > 500


def _win_loss_onsets(events_tsv: str):
    win, loss = [], []
    with open(events_tsv) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        oi, ti = hdr.index("onset"), hdr.index("trial_type")
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) <= ti:
                continue
            t = p[ti]
            if "Feedback Win" in t:
                win.append(float(p[oi]))
            elif "Feedback Loss" in t:
                loss.append(float(p[oi]))
    return win, loss


def extract_subject(sub: str) -> Dict[str, float]:
    import mne
    mne.set_log_level("ERROR")
    os.makedirs(TMP, exist_ok=True)
    stem = f"{sub}_task-ThreeArmedBandit"
    setf = os.path.join(TMP, f"{stem}_eeg.set")
    fdtf = os.path.join(TMP, f"{stem}_eeg.fdt")
    evf = os.path.join(TMP, f"{stem}_events.tsv")
    ok = (_curl(f"{BASE}/{sub}/eeg/{stem}_eeg.set", setf)
          and _curl(f"{BASE}/{sub}/eeg/{stem}_eeg.fdt", fdtf)
          and _curl(f"{BASE}/{sub}/eeg/{stem}_events.tsv", evf))
    if not ok:
        return {}
    try:
        raw = mne.io.read_raw_eeglab(setf, preload=True)
        raw.filter(0.1, 30.0, verbose="ERROR")
        sf = raw.info["sfreq"]
        chans = [c for c in CLUSTER if c in raw.ch_names] or (["Cz"] if "Cz" in raw.ch_names else [])
        if not chans:
            return {"error": "no frontocentral channel"}
        win_on, loss_on = _win_loss_onsets(evf)
        def erp(onsets):
            ev = np.array([[int(o * sf), 0, 1] for o in onsets if 0 < o * sf < raw.n_times - int(0.6 * sf)])
            if len(ev) < 10:
                return None
            ep = mne.Epochs(raw, ev, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0.0),
                            picks=chans, preload=True, verbose="ERROR")
            data = ep.get_data().mean(axis=1)  # (n_epochs, n_times), avg over cluster
            times = ep.times
            m = (times >= WIN_WINDOW[0]) & (times <= WIN_WINDOW[1])
            return float(np.mean(data[:, m]) * 1e6)  # microvolts
        w, l = erp(win_on), erp(loss_on)
        if w is None or l is None:
            return {"error": "too few epochs"}
        return {"win_uV": w, "loss_uV": l, "rewp_uV": w - l, "n_win": len(win_on), "n_loss": len(loss_on)}
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}
    finally:
        for f in (setf, fdtf, evf):
            if os.path.exists(f):
                os.remove(f)


def run(subject_list: List[str]) -> Dict[str, object]:
    roi = json.load(open(ROI_JSON)) if os.path.exists(ROI_JSON) else {}
    for i, sub in enumerate(subject_list):
        if sub in roi and "rewp_uV" in roi.get(sub, {}):
            continue
        print(f"[{i+1}/{len(subject_list)}] {sub} ...", flush=True)
        roi[sub] = extract_subject(sub)
        json.dump(roi, open(ROI_JSON, "w"), indent=2)
    return roi


def analyze(roi: Dict[str, dict]) -> Dict[str, object]:
    good = {s: r for s, r in roi.items() if r and "rewp_uV" in r}
    rewp = np.array([r["rewp_uV"] for r in good.values()])
    win = np.array([r["win_uV"] for r in good.values()])
    loss = np.array([r["loss_uV"] for r in good.values()])
    t, p = stats.ttest_rel(win, loss) if len(good) > 2 else (float("nan"), float("nan"))
    ci = H.bootstrap_ci(rewp, np.mean, seed=0)
    d = float(np.mean(rewp) / (np.std(rewp, ddof=1) + 1e-9)) if len(good) > 1 else float("nan")
    return {
        "experiment": "REAL_eeg_reward_positivity",
        "dataset": "OpenNeuro ds003458 (three-armed bandit, EEG)",
        "n_subjects": len(good),
        "cluster": CLUSTER, "rewp_window_s": WIN_WINDOW,
        "mean_rewp_uV": float(np.mean(rewp)), "rewp_ci95": [ci["lo"], ci["hi"]],
        "win_vs_loss_paired_t": float(t), "p": float(p), "cohens_d": d,
        "reward_positivity_present": bool(np.mean(rewp) > 0 and p < 0.05),
        "note": "win-positive frontocentral RewP = EEG signature of reward processing; grounds L7",
    }


if __name__ == "__main__":
    import glob as _g
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 23
    # discover subjects from participants (sub-001..sub-0NN); ds003458 uses sub-001 style
    subs = [f"sub-{i:03d}" for i in range(1, n + 1)]
    roi = run(subs)
    res = analyze(roi)
    json.dump(res, open(os.path.join(ROOT, "results", "real_eeg.json"), "w"), indent=2)
    print(f"\nREAL EEG Reward Positivity (ds003458, n={res['n_subjects']} subjects)")
    print(f"  mean RewP (win-loss) = {res['mean_rewp_uV']:+.2f} uV "
          f"CI[{res['rewp_ci95'][0]:+.2f},{res['rewp_ci95'][1]:+.2f}]")
    print(f"  win vs loss paired t={res['win_vs_loss_paired_t']:+.2f} p={res['p']:.4f} d={res['cohens_d']:+.2f}")
    print(f"  reward positivity present: {res['reward_positivity_present']}")
