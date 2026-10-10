"""Build a self-contained 2-page PDF brief for bellwether.

Run: ../.venv/bin/python build_brief.py
Writes: bellwether_brief.pdf
"""

from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figures"
OUT = Path(__file__).resolve().parent / "bellwether_brief.pdf"

styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.3, leading=12.6,
                      alignment=TA_JUSTIFY, spaceAfter=6)
h = ParagraphStyle("h", parent=styles["Heading2"], fontSize=11, leading=13,
                   spaceBefore=8, spaceAfter=3)
title = ParagraphStyle("title", parent=styles["Title"], fontSize=16, leading=19)
sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=9, textColor="#555555",
                     spaceAfter=10)
cap = ParagraphStyle("cap", parent=styles["Normal"], fontSize=7.8, leading=9.5,
                     textColor="#555555", spaceAfter=8)


def P(t):
    return Paragraph(t, body)


def fig(path, width=6.6 * inch):
    from PIL import Image as PILImage
    w0, h0 = PILImage.open(path).size
    return Image(str(path), width=width, height=width * h0 / w0)


story = []
story.append(Paragraph("bellwether: an early-warning instrument for value-decision capture", title))
story.append(Paragraph("A mechanistic in-silico testbed. Anonymous, ORCID 0000-0000-0000-0000. "
                       "Private code repository available on request.", sub))

story.append(Paragraph("Question", h))
story.append(P(
    "As automated systems gain finer read and write access to neural representations, through decoding, "
    "closed-loop stimulation, and, in animal models, optogenetic control, the distance between reading a "
    "decision and nudging it shrinks. Decision circuits are most sensitive to small inputs exactly where a "
    "choice tips, so the cheapest place to steer a decision is near its bifurcation. This project asks a "
    "narrow, testable version of the defensive question: can covert steering of a value decision be detected "
    "from population activity before the behavioral commitment, how large must the steering be before it "
    "becomes visible, and where does the detector fail? The work is deliberately synthetic, so that every "
    "claim is a falsifiable property of a model with known ground truth, as a first step before applying the "
    "same monitor to real recordings."))

story.append(Paragraph("Method", h))
story.append(P(
    "<b>Substrate.</b> A reduced two-variable attractor circuit models a value choice between two options. "
    "With ambiguous evidence it sits near a symmetric unstable point; noise or input tips it into one of two "
    "winner-take-all states. Near that bifurcation the circuit is both maximally input-sensitive and slow to "
    "recover. <b>Readout.</b> The latent circuit is embedded into many noisy units whose rates are a random "
    "nonlinear mixture of the two internal variables; the monitor sees only population activity and the "
    "evidence the task claims to present. <b>Adversary.</b> A subthreshold steering drive is added to one "
    "option; a stealthy variant hides the same steering inside added diffusion. <b>Monitor.</b> During the "
    "approach to the decision the monitor looks for a decision that resolves sooner and straighter than the "
    "presented evidence justifies. The signature has two channels, both measured relative to the presented "
    "evidence: a timing channel (how early the decision commits) and a geometry channel (the directedness of "
    "the recovered decision axis, estimated by principal components so it is read along the direction the "
    "choice varies). A sequential alarm fires before commitment when the anomaly exceeds what honest "
    "ambiguous decisions produce."))

story.append(fig(FIGS / "validation_overview.png"))
story.append(Paragraph(
    "Figure 1. Left: decision-variable trajectories, honest-ambiguous (blue) versus steered (red). "
    "Middle: critical slowing down on the approach to the bifurcation. Right: monitor score distributions "
    "with the calibrated alarm threshold.", cap))

story.append(PageBreak())

story.append(Paragraph("Results", h))
story.append(P(
    "<b>Detection.</b> Covert steering is detectable before commitment relative to an honest ambiguous "
    "decision of the same presented evidence. The geometry channel separates them at AUC 0.97 (95% bootstrap "
    "CI 0.95 to 0.97; label-permutation p &lt; 0.001; 750 trials per class), and the evidence-relative timing "
    "channel at AUC 0.99. Against legitimate strong evidence both channels are near perfect, because both are "
    "calibrated against the evidence presented. <b>Robust across embeddings.</b> The geometry result is not a "
    "lucky projection: across ten random population embeddings the AUC is 0.97 (SD 0.016, range 0.93 to "
    "0.99). <b>False alarms.</b> The threshold is set to a 5% false-alarm rate on honest ambiguous decisions "
    "(the condition an attack imitates and the hardest case); the realized rate on held-out trials is about "
    "3%. <b>Early warning.</b> The monitor fires before commitment about 92% of the time, median lead about "
    "40 ms. <b>Dose-response and stealth floor.</b> Detectability rises smoothly with dose; below about 0.008 "
    "nA.s the geometry channel cannot distinguish steering from an honest hard decision. A diffusion-hiding "
    "adversary roughly doubles that geometry floor (about 0.019 nA.s), but does not evade the timing channel."))

story.append(Paragraph("Which channel does the work, honestly", h))
story.append(P(
    "The two channels are not equal, a point a review forced into the open. In a symmetric attractor one "
    "cannot bias the outcome without also speeding the escape from the saddle, so steering and acceleration "
    "are mechanistically coupled. The evidence-relative timing cue is therefore the dominant channel: it "
    "equals or beats the geometry channel on every contrast (manip versus hard 0.99 against 0.97, manip "
    "versus easy 1.0 against 1.0), and it is not fooled by the stealthy adversary (timing about 1.0 where "
    "geometry drops to about 0.79). The defensible claim is thus not that directedness is a superior "
    "detector, but that covert steering leaves a sooner-and-straighter-than-warranted signature detectable "
    "via either channel, with timing the stronger and harder to evade. The key open question is whether a "
    "richer or asymmetric circuit could permit biasing a choice without speeding it, the regime where the "
    "timing channel would fail and the geometry channel would have to carry detection alone."))

story.append(fig(FIGS / "leadtime_overview.png", width=5.6 * inch))
story.append(Paragraph(
    "Figure 2. Left: dose-response and stealth floor for naive and stealthy adversaries. "
    "Right: pre-commitment detection lead times.", cap))

story.append(Paragraph("Statistical validity and limitations", h))
story.append(P(
    "Evaluation uses train/test separation (calibration on honest trials only), AUC with stratified "
    "bootstrap CIs and event counts, label-permutation tests against chance, a paired DeLong test and a "
    "commitment-time-stratified AUC for the speed confound, a stealthy adversary as a specification-"
    "robustness control, a Kendall-tau trend test for the early-warning signals, and a detection statistic "
    "invariant to the sign and scale of the recovered decision axis. Honest limits: the model is synthetic "
    "and demonstrates a capability and a risk logic, not a magnitude in a real brain; critical slowing down "
    "is only partial in this substrate (variance rises on about 54% of trials, autocorrelation does not "
    "reliably rise); the reduced circuit does not reproduce the full rotation of option subspaces seen in "
    "richer cortex (it does reproduce the roughly fourfold expansion of the chosen representation). The "
    "intended next step is to apply the same monitor to real value-decision recordings, where the decision "
    "axis must be estimated without ground truth."))

doc = SimpleDocTemplate(str(OUT), pagesize=LETTER,
                        leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                        topMargin=0.7 * inch, bottomMargin=0.6 * inch)
doc.build(story)
print("wrote", OUT)
