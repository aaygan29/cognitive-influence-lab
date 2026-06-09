#!/usr/bin/env python3
"""Real CSI = vmPFC − dlPFC on NARPS ds001734 — REQUIRES preprocessed fMRI; never fabricates.

NARPS on OpenNeuro ships only **raw native-space** BOLD (~688 MB/run, ~300 GB total, no fMRIPrep
derivatives). Computing vmPFC/dlPFC activation needs template-space (MNI) data, i.e. fMRIPrep
preprocessing. This runner therefore expects fMRIPrep output under ``$NARPS_FMRIPREP`` and a
preregistered ``$CSI_CONTRAST``. If either is missing it STOPS with a message rather than producing
anything. See docs/CSI_FMRI_PLAN.md.
"""
from __future__ import annotations

import glob
import os
import sys

PREP_DIR = os.environ.get("NARPS_FMRIPREP", "data/raw/ds001734_fmriprep")
CONTRAST = os.environ.get("CSI_CONTRAST", "")  # e.g. "gain" — preregistered; none assumed


def main() -> int:
    if not CONTRAST:
        print(
            "STOP: set CSI_CONTRAST — the preregistered contrast that defines 'activation' for CSI "
            "(e.g. the gain parametric effect). This is a neuroscience choice; none is assumed."
        )
        return 2
    preproc = glob.glob(os.path.join(PREP_DIR, "sub-*", "func", "*preproc_bold.nii.gz"))
    if not preproc:
        print(
            f"STOP: no fMRIPrep-preprocessed BOLD found under {PREP_DIR!r}.\n"
            "NARPS ds001734 ships only raw native-space BOLD (~688 MB/run, ~300 GB, no derivatives),"
            " so CSI cannot be computed without preprocessing. Run fMRIPrep first (e.g. on Modal/"
            "Colab), point NARPS_FMRIPREP at its output, then re-run. This runner does not "
            "preprocess or fabricate data. See docs/CSI_FMRI_PLAN.md."
        )
        return 1

    # Reached only when real preprocessed data are present (per-subject loop implemented then,
    # against the confirmed contrast/ROIs — see docs/CSI_FMRI_PLAN.md).
    print(
        f"Found {len(preproc)} preprocessed runs under {PREP_DIR}. "
        f"Contrast = {CONTRAST!r}. Implement per-subject CSI loop per docs/CSI_FMRI_PLAN.md."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
