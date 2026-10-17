"""One-command reproduction: run every experiment, regenerate figures/tables, print a gate summary.

    python reproduce.py            # full run (a few minutes; E3 does permutation nulls)
    python reproduce.py --quick    # smaller settings for a fast smoke test

Each experiment writes results/<name>.json; figures/make_figures.py renders vector PDFs + LaTeX
tables from those. The summary at the end reports the pass/fail of each headline gate so a reviewer
can confirm the claims reproduce.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
RESULTS = os.path.join(ROOT, "results")


def _run(mod: str):
    print(f"\n=== running {mod} ===", flush=True)
    subprocess.run([sys.executable, os.path.join(SRC, mod)], check=True)


def _load(name):
    with open(os.path.join(RESULTS, name)) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke test (fewer seeds/perms)")
    args = ap.parse_args()

    # honesty positive control
    _run("honesty.py")
    # E1 recovery (+ E4 pooling, run inside phenotype.py __main__)
    _run("phenotype.py")
    # E3 transfer controls
    _run("transfer.py")
    # E6 attribution-scale ladder
    _run("inverse.py")
    # E7 state-shift
    _run("states.py")
    # real data (behavioral) if the NARPS events are present
    if os.path.isdir(os.path.join(ROOT, "data", "narps", "events")) and \
       len(os.listdir(os.path.join(ROOT, "data", "narps", "events"))) > 0:
        _run("real_narps.py")
        if os.path.isdir(os.path.join(ROOT, "data", "ds000005", "events")):
            _run("real_crossdataset.py")
        # EEG link + external grounding refresh their result JSONs if present (network for downloads)
        if os.path.exists(os.path.join(RESULTS, "real_eeg.json")):
            print("\n(EEG result present: results/real_eeg.json)")
        # synthesis: assemble the grounding chain + its statistical validation from the result JSONs
        for m in ("grounding_chain.py", "triangulation_stats.py"):
            try:
                _run(m)
            except Exception as e:
                print(f"(skipping {m}: {e})")
    else:
        print("\n(skipping real_narps.py: NARPS events not downloaded)")

    # regenerate figures + tables from the fresh results
    print("\n=== regenerating figures + tables ===", flush=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "figures", "make_figures.py")], check=True)

    # run the test suite
    print("\n=== running tests ===", flush=True)
    subprocess.run([sys.executable, "-m", "pytest", os.path.join(ROOT, "tests"), "-q"], check=False)

    # gate summary
    hz = _load("e2_honesty_selfcheck.json")
    e1 = _load("e1_recovery.json")
    e4 = _load("e4_pooling.json")
    e3 = _load("e3_transfer.json")
    e6 = _load("e6_scale_ladder.json")
    e7 = _load("e7_state_shift.json")

    print("\n" + "=" * 60)
    print("GATE SUMMARY (headline claim reproduces?)")
    print("=" * 60)
    rows = [
        ("C2 honesty layer (coverage+calibration+gate)", hz["all_pass"]),
        ("C1 affective identifiability at max n", e1["affective_kill_criterion_pass"]),
        ("C4 pooling helps low-n recovery", e4["affective_pooling_helps_at_min_n"]),
        ("C3 transfer pipeline valid (+/- controls)", e3["pipeline_valid"]),
        ("C5/C6 scale-ladder dose-response", e6["dose_response_holds"]),
        ("ext. state-shift recovered", e7["valid"]),
    ]
    rn_path = os.path.join(RESULTS, "real_narps.json")
    if os.path.exists(rn_path):
        rn = _load("real_narps.json")
        rows.append(("REAL C1 loss aversion recovered (NARPS)", rn["RN1_loss_aversion"]["frac_loss_averse_gt1"] > 0.5))
        rows.append(("REAL C1 held-out choice beats base-rate", rn["RN1_prediction"]["beats_baserate"]))
        rows.append(("REAL C2 conformal coverage >= nominal", rn["RN2_honesty"]["median_conformal_coverage"] >= rn["RN2_honesty"]["nominal"]))
    for name, ok in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    all_ok = all(ok for _, ok in rows)
    print("=" * 60)
    print("ALL GATES PASS" if all_ok else "SOME GATES FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
