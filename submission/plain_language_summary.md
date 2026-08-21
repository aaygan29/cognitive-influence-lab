# Plain-language summary (for circulation / lay abstract)

Most systems that try to predict a person's decisions give a single confident answer and never say
"I'm not sure." We built a decision model that is honest by design: it predicts a person's choices
from a short list of interpretable quantities (how much they weigh gains, losses, risk, delay, and
threat), it attaches a calibrated uncertainty to every prediction, and — crucially — it *refuses to
answer* when the available data are too thin to support a real prediction.

We tested it on real human choices from public brain-imaging studies of gambling. The model recovers
people's loss aversion (and reproduces a classic value), predicts their held-out choices accurately,
can tell one person from another by their decision "fingerprint," and transfers from one lab's data to
another's. We connect the model to real brain signals across two recording methods (fMRI and EEG), and
where the brain-to-behavior link is too weak to establish, the model abstains rather than overclaims —
and we show mathematically that this direct link is not needed for the grounding to hold.

The broader point: for high-stakes behavioral prediction, knowing *when not to predict* is as
important as the prediction itself.
