"""Detection lead time, dose-response, and the stealth floor.

Two questions this answers:

  1. When the monitor does fire on a manipulated trial, how far ahead of the
     behavioral commitment does it fire? (Lead time.)
  2. How does detectability depend on the adversary's dose, and below what dose
     does covert steering become statistically invisible to this detector?
     (Dose-response and the stealth floor.)

Outputs results/leadtime.json and figures/leadtime_*.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import configs as C
from bellwether.detect import roc_auc, stealth_floor, summarize_lead_times
from bellwether.dynamics import Adversary, Stimulus, simulate
from bellwether.monitor import Monitor, records_from_trajectory

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

DOSES_AMP = (0.002, 0.004, 0.006, 0.008, 0.012, 0.016, 0.020, 0.024)
LEADTIME_AMPLITUDE = 0.020
SEED = 2024


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)

    readout = C.make_readout()
    cfg = C.make_monitor_config()
    mon = Monitor(readout=readout, config=cfg).calibrate(C.calibration_records(readout))

    # --- Lead time at a fixed, clearly-detectable dose. ---
    manip = C.test_records(readout, "manip", seed=SEED, amplitude=LEADTIME_AMPLITUDE)
    alarms = [mon.alarm(r) for r in tqdm(manip, desc="lead-time alarms")]
    lead = summarize_lead_times(alarms)

    # Map amplitude to realized dose for the reporting axis.
    dt_axis = np.arange(int(round(C.DURATION / C.DT))) * C.DT
    amp_to_dose = {a: Adversary(amplitude=a, onset=C.ONSET, offset=C.OFFSET).dose(dt_axis)
                   for a in DOSES_AMP}

    # --- Dose-response: AUC(manip vs honest-hard) as dose grows. ---
    hard = C.test_records(readout, "hard", seed=SEED + 100)
    s_hard = np.array([mon.score(r) for r in hard])
    doses, aucs, tprs = [], [], []
    for amp in tqdm(DOSES_AMP, desc="dose-response"):
        recs = C.test_records(readout, "manip", seed=SEED + 200 + int(amp * 1000),
                              amplitude=amp)
        s = np.array([mon.score(r) for r in recs])
        doses.append(amp_to_dose[amp])
        aucs.append(roc_auc(s, s_hard))
        tprs.append(float(np.mean(s > mon.alarm_threshold)))

    doses = np.array(doses)
    aucs = np.array(aucs)
    floor = stealth_floor(doses, aucs, threshold=0.75)

    # Stealthy adversary: same steering dose, but hidden inside added diffusion so
    # the trajectory looks as wandering as an honest ambiguous decision. This is
    # the adversary's escape route and it should raise the stealth floor.
    STEALTH_NOISE = 0.03
    aucs_stealth = []
    for amp in tqdm(DOSES_AMP, desc="stealthy dose-response"):
        # Same steering amplitude, but with per-step jitter mixed into the drive.
        tr = simulate(Stimulus(coherence=C.MANIP_COHERENCE),
                      Adversary(amplitude=amp, noise=STEALTH_NOISE,
                                onset=C.ONSET, offset=C.OFFSET),
                      n_trials=C.N_TEST, duration=C.DURATION, dt=C.DT,
                      seed=SEED + 400 + int(amp * 1000))
        s = np.array([mon.score(r) for r in
                      records_from_trajectory(tr, readout, C.MANIP_COHERENCE)])
        aucs_stealth.append(roc_auc(s, s_hard))
    aucs_stealth = np.array(aucs_stealth)
    floor_stealth = stealth_floor(doses, aucs_stealth, threshold=0.75)

    summary = {
        "leadtime_amplitude_nA": LEADTIME_AMPLITUDE,
        "leadtime_dose_nAs": amp_to_dose[LEADTIME_AMPLITUDE],
        "detection_rate": lead.detection_rate,
        "median_lead_s": lead.median_lead_s,
        "lead_iqr_s": list(lead.iqr_lead_s),
        "pre_commit_alarm_rate": lead.pre_commit_rate,
        "dose_response": {
            "amplitude_nA": list(DOSES_AMP),
            "dose_nAs": doses.tolist(),
            "auc_vs_hard": aucs.tolist(),
            "tpr": tprs,
            "auc_vs_hard_stealthy": aucs_stealth.tolist(),
            "stealth_noise_nA": STEALTH_NOISE,
        },
        "stealth_floor_dose_nAs_at_auc0.75": floor,
        "stealth_floor_dose_nAs_stealthy_adversary": floor_stealth,
    }
    (RESULTS / "leadtime.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ("detection_rate", "median_lead_s", "pre_commit_alarm_rate",
                       "stealth_floor_dose_nAs_at_auc0.75")}, indent=2))

    _figures(doses, aucs, aucs_stealth, tprs, floor, alarms)


def _figures(doses, aucs, aucs_stealth, tprs, floor, alarms) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))

    ax[0].plot(doses, aucs, "o-", color="tab:red", label="AUC (naive adversary)")
    ax[0].plot(doses, aucs_stealth, "^-", color="tab:purple",
               label="AUC (stealthy adversary)")
    ax[0].plot(doses, tprs, "s--", color="tab:orange", label="detection rate")
    ax[0].axhline(0.75, color="k", ls=":", lw=1)
    ax[0].axhline(0.5, color="gray", ls=":", lw=1)
    if np.isfinite(floor):
        ax[0].axvline(floor, color="tab:red", ls="--", lw=1,
                      label=f"stealth floor ~ {floor:.4f} nA·s")
    ax[0].set_xlabel("adversarial dose (nA·s)")
    ax[0].set_ylabel("detectability")
    ax[0].set_title("dose-response and stealth floor")
    ax[0].legend(fontsize=8)

    leads = np.array([a["lead_time"] for a in alarms
                      if a["alarmed"] and np.isfinite(a["lead_time"])])
    if leads.size:
        ax[1].hist(leads, bins=20, color="tab:green", alpha=0.8)
        ax[1].axvline(0, color="k", lw=1)
        ax[1].axvline(np.median(leads), color="tab:red", ls="--", lw=1,
                      label=f"median {np.median(leads):.2f}s")
        ax[1].legend(fontsize=8)
    ax[1].set_title("detection lead time\n(commit time - alarm time)")
    ax[1].set_xlabel("lead time (s); positive = warned before commitment")

    fig.tight_layout()
    fig.savefig(FIGS / "leadtime_overview.png", dpi=140)
    print("wrote", FIGS / "leadtime_overview.png")


if __name__ == "__main__":
    main()
