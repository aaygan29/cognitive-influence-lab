To: adeel.razi@monash.edu
Subject: An early-warning monitor for value-decision capture: a control-theoretic in-silico prototype

Dear Professor Razi,

I have been working on a question that sits close to your interests in brain dynamics and the neuroscience-AI interface, and I would be grateful for your view on whether it is sound.

As automated systems gain finer access to neural representations, the gap between reading a decision and nudging it narrows, and a decision circuit is cheapest to steer exactly where it tips. I built a synthetic testbed to ask the defensive version: can covert steering of a value decision be detected from population activity before the commitment, how large must it be to be visible, and where does detection fail? A reduced attractor circuit produces the choice near its bifurcation, and an online monitor measures how directed the recovered decision trajectory is relative to the evidence presented.

In this model it detects steering with AUC about 0.96 and fires before commitment most of the time, with a quantifiable dose floor below which an attack stays invisible (and a stealthy adversary that raises that floor). I have kept the limitations explicit, including a commitment-speed confound that I control for directly.

A two-page brief is attached. Two things I would value your read on: whether the dynamical framing is defensible, and whether a perturbed real dataset such as PsiConnect could be a sensible next testbed for the same monitor. Code available on request.

With thanks,
Anonymous
ORCID 0000-0000-0000-0000
[your preferred email / affiliation]
