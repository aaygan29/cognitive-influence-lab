# Real CSI (vmPFC − dlPFC) on NARPS ds001734 — plan & execution requirement

CSI is a **neural** contrast (PREREGISTRATION §1): **CSI = vmPFC activation − dlPFC activation**.
This documents how it is computed on NARPS and why the numbers must be produced on real compute.

## Why it can't be run in the current sandbox
NARPS on OpenNeuro provides only **raw, native-space BOLD**: ~688 MB per run × 4 runs × 108
subjects ≈ **300 GB**, with **no fMRIPrep/MNI derivatives**. ROI activation requires template-space
data, i.e. **fMRIPrep preprocessing** (motion correction + registration to MNI), which needs
fMRIPrep/FSL/ANTs and hours of compute per subject — not available here. We will not fabricate or
fake-register data, so the GLM step runs where preprocessing exists (e.g. the user's Modal/Colab).

## Pipeline (built; `spikeprint/fmri.py` + `scripts/run_csi_fmri_narps.py`)
Per subject, on fMRIPrep-preprocessed (MNI) BOLD:
1. **First-level GLM** (`first_level_effect_map`): gamble-evaluation regressor + the parametric
   modulators **gain** and **loss** (demeaned) + motion/aCompCor confounds → effect-size map for
   the chosen **contrast**.
2. **ROIs** (`harvard_oxford_masks`): vmPFC ≈ {Frontal Medial Cortex, Frontal Orbital Cortex};
   dlPFC ≈ {Middle Frontal Gyrus} (Harvard-Oxford cortical atlas).
3. **CSI** (`subject_csi`): mean(vmPFC effect) − mean(dlPFC effect).
4. **Validation** (`correlation_validity`, tested on mock data): across subjects, Spearman
   correlation of CSI with acceptance rate (from the behavior we already loaded), with a
   bootstrap CI and permutation p, decided under the `Finding` contract (null = 0, no association).

## Decisions needed from you (I will NOT assume them — that error already happened once)
1. **The contrast that defines "activation" for CSI.** Options: the **gain parametric** effect
   (canonical value signal), task-vs-baseline, decision-period, or a gain−loss contrast. This
   determines what vmPFC/dlPFC "activation" means.
2. **ROI definitions** — accept the Harvard-Oxford defaults above, or specify your own (e.g. a
   Neurosynth vmPFC/dlPFC meta-analytic mask, or NARPS's a-priori ROIs).
3. **Unit of analysis** — subject-level (CSI vs acceptance rate, n≈108) and/or trial/condition-level.
4. **Execution route** — (a) you run fMRIPrep on NARPS on your compute and point
   `NARPS_FMRIPREP` at it; (b) you provide existing preprocessed derivatives; or (c) authorize a
   subset download knowing preprocessing still must happen on your side.

Once (1)–(4) are fixed, the per-subject loop is a short addition to the runner, and the result
goes to `results/PILLAR_A_NARPS_CSI_fmri.md` under the same kill-criteria discipline.
