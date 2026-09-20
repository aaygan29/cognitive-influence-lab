# Integration note, 2026-09-20

Three external inputs for the content-driven choice-distortion instrument.

## 1. CANlab pattern masks (canlab/Neuroimaging_Pattern_Masks) [data-scarcity fix]
Validated multivariate fMRI signatures (appetitive/reward, negative affect, pain) as .nii, applied with CANlab tools, GPL-3.0 (NPS and some patterns need a research agreement). Action item: use the appetitive/reward and negative-affect signatures as the neural axis in the NARPS real-CSI pipeline alongside or instead of a from-scratch decoder. This gives an externally-validated y and cuts the data requirement, directly addressing the "not enough neuroimaging data for human behavioral-neuro conclusions" bottleneck. Log provenance and licensing before use.

## 2. Neural subspace reorganization in value choice (Li et al., iScience 2026, Miller lab)
Chosen-vs-unchosen subspaces rotate orthogonal and the chosen subspace expands and aligns to action after a decision. Action item: a content-induced choice distortion should show up as altered subspace reorganization (e.g. premature or exaggerated chosen-subspace expansion). Adds a population-geometry readout of commitment to sit alongside the spiking DDM / actor-critic distortion measure.

## 3. Superior colliculus action-pressure term (bioRxiv 730072 v2)
SC adds an additive, side-specific action-promotion term to a logistic/softmax choice model. Action item: cite as the action-side complement to NAcc reward-anticipation value (Knutson/Genevsky). Distortion can act on the value term (evidence) or the action-pressure term (bias); keeping them separate lets the instrument attribute a distortion to "wanted it more" vs "pushed to act."

## Cross-program link
The cognitive-security framing here is the individual-scale readout of the same threat the RAND agency-erosion model (RRA4817-1) formalizes at the collective scale, and the bounded-influence repo builds the control-theoretic defense.
