To: mackenzie.mathis@epfl.ch
Subject: Detecting covert steering of a value decision before commitment (in-silico, a methods question)

Dear Professor Mathis,

Your DYSCO work on recovering latent dynamics from noisy neural observations, with identification up to an affine gauge, is close to a problem I have been testing in silico, and I would value your read on whether the methodology holds up.

I built a small, fully synthetic testbed to ask one question: if an adversary covertly steers a value decision by exploiting the circuit's sensitivity near its choice bifurcation, can a monitor detect it from population activity before the behavioral commitment? A reduced attractor circuit produces the decision, a random high-dimensional readout stands in for a recording, and the monitor estimates the decision axis and watches its directedness relative to the presented evidence.

It works in this model (AUC about 0.96 separating steered from honest ambiguous decisions, with most alarms firing pre-commitment), and it has a clear stealth floor below which steering is invisible. I have tried to be honest about the confounds: a commitment-speed cue actually beats the detector on one contrast, which I handle with a commitment-time-stratified analysis and an adaptive adversary.

A two-page brief is attached. My worry is methodological validity, exactly the kind of identifiability and gauge-invariance question your group thinks about carefully. Would you be willing to tell me whether the approach is sound, or where it would break? I am happy to share the code.

With thanks for your time,
Anonymous
ORCID 0000-0000-0000-0000
[your preferred email / affiliation]
