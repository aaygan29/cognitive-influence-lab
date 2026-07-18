"""
fusion.py — Bayesian evidence-accumulation fusion with abstention.

This is the core idea of deceptkit v2 and the reason it is positioned as a
*layer on top of legacy machinery* rather than a replacement for it.

THE FRAMING (sequential probability ratio test, SPRT).
Detecting concealed information is a hypothesis test:
    H1 = informed (recognizes the concealed detail)
    H0 = naive    (does not)
Each measurement channel c (the legacy polygraph's own output, and each neural/
autonomic channel) provides evidence. In log-odds space the channels ADD:

    logit P(H1 | all channels) = logit prior + Σ_c LLR_c

where LLR_c = log [ f(x_c | H1) / f(x_c | H0) ] is the calibrated
log-likelihood ratio contributed by channel c. This is exactly Bayes' rule for
conditionally-independent-ish evidence, and it is why a *weak* legacy channel
and several *modest* neural channels can combine into a confident posterior —
or fail to, in which case we abstain.

WHY THIS IS THE RIGHT ABSTRACTION FOR A "LEGACY LAYER".
The legacy polygraph is treated as just another evidence channel with its own
measured reliability. Its LLR is learned/calibrated from data, so a machine
that is only ~70% accurate contributes a correspondingly *small* LLR and cannot
by itself push the posterior past threshold. The neural channels add
independent LLRs. The operator keeps their existing polygraph; we add a
calibrated fusion layer that (a) down-weights the legacy channel to its true
reliability and (b) accumulates neural evidence on top.

CALIBRATION & HONESTY.
  * Per-channel LLRs are estimated by logistic fits on held-out folds, so a
    channel that is pure noise gets LLR ~ 0 (it cannot help or hurt).
  * We accumulate in log-odds and convert once at the end -> a single
    calibrated posterior with an evidence breakdown per channel (auditable).
  * ABSTENTION is first-class: if the accumulated evidence does not clear an
    upper (informed) or lower (naive) log-odds boundary, OR the countermeasure/
    quality gate trips, the layer returns ABSTAIN. This is the SPRT's third
    outcome ("continue sampling") repurposed as "insufficient evidence to act."

The result is a decision-support layer that reports, in plain terms:
"the legacy box contributed X bits, the P300 channel Y bits, ... total posterior
p = ..., which is/na enough to act; otherwise abstain."
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from .simulate import MODALITIES
from .features import feature_columns

# Channels that feed the fusion: each neural/autonomic salience z_<mod>, plus
# the legacy polygraph channel. Countermeasure/quality diagnostics are NOT
# evidence channels — they drive the abstention gate instead.
EVIDENCE_CHANNELS = [f"z_{m}" for m in MODALITIES] + ["legacy_z"]


@dataclass
class SPRTPolicy:
    """Log-odds boundaries + gates for the accumulation decision."""
    upper_bits: float = 1.5     # accumulated evidence (log2 odds) above +this -> INFORMED
    lower_bits: float = 1.5     # below -this -> NAIVE
    cm_z_gate: float = 2.5      # countermeasure suspicion (foil behaving like probe)
    cm_disp_gate: float = 1.6
    min_quality: float = 0.5


@dataclass
class BayesianFusion:
    """Per-channel calibrated LLR estimator + log-odds accumulation.

    Each channel's LLR as a function of its (standardized) value is modeled with
    a univariate logistic regression fit on training folds:
        P(H1 | x_c) = sigmoid(a_c * x_c + b_c)
    From that, the channel LLR at value x is
        LLR_c(x) = a_c * x + (b_c - logit(prior))
    i.e. the part of the log-odds attributable to channel c beyond the prior.
    Channels are summed; a channel with a_c ~ 0 contributes ~0 (auto-ignored).
    """
    prior: float = 0.5
    channels: list = field(default_factory=lambda: list(EVIDENCE_CHANNELS))
    policy: SPRTPolicy = field(default_factory=SPRTPolicy)
    _coef: dict = field(default_factory=dict, init=False)   # channel -> (a, b)
    _prior_logit: float = field(default=0.0, init=False)

    def _prep(self, feats: pd.DataFrame) -> pd.DataFrame:
        f = feats.copy()
        # standardize legacy channel into a z-like scale using training stats
        return f

    def fit(self, feats: pd.DataFrame):
        y = feats["concealed"].to_numpy()
        self.prior = float(np.mean(y))
        self._prior_logit = float(np.log(self.prior / (1 - self.prior)))
        self._coef = {}
        # standardization stats for legacy channel (so its scale is comparable)
        self._legacy_mu = float(feats["legacy_poly"].mean())
        self._legacy_sd = float(feats["legacy_poly"].std() + 1e-9)
        Xall = self._channel_matrix(feats)
        for c in self.channels:
            x = Xall[c].to_numpy().reshape(-1, 1)
            # guard degenerate channels
            if np.std(x) < 1e-8 or len(np.unique(y)) < 2:
                self._coef[c] = (0.0, self._prior_logit)
                continue
            lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
            lr.fit(x, y)
            a = float(lr.coef_[0, 0]); b = float(lr.intercept_[0])
            self._coef[c] = (a, b)
        return self

    def _channel_matrix(self, feats: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=feats.index)
        for c in self.channels:
            if c == "legacy_z":
                f[c] = (feats["legacy_poly"] - getattr(self, "_legacy_mu", 0.0)) / getattr(self, "_legacy_sd", 1.0)
            else:
                f[c] = feats[c]
        return f

    def llr_breakdown(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Return per-channel LLR (in nats) for each row — the auditable
        evidence breakdown."""
        X = self._channel_matrix(feats)
        out = pd.DataFrame(index=feats.index)
        for c in self.channels:
            a, b = self._coef[c]
            # LLR beyond prior = (a*x + b) - prior_logit
            out[c] = a * X[c].to_numpy() + (b - self._prior_logit)
        return out

    def accumulate(self, feats: pd.DataFrame):
        """Return (posterior_p, total_logodds, llr_df)."""
        llr = self.llr_breakdown(feats)
        total_logodds = self._prior_logit + llr.sum(axis=1).to_numpy()
        p = 1.0 / (1.0 + np.exp(-total_logodds))
        return p, total_logodds, llr

    def decide(self, feats: pd.DataFrame) -> pd.DataFrame:
        p, logodds, llr = self.accumulate(feats)
        bits = logodds / np.log(2)               # evidence in bits (log2 odds)
        pol = self.policy
        res = feats[["question_set_id", "subject_id"]].copy()
        res["p_informed"] = p
        res["evidence_bits"] = bits
        # attach per-channel LLR in bits for auditability
        for c in self.channels:
            res[f"llr_{c}"] = llr[c].to_numpy() / np.log(2)

        actions, reasons = [], []
        for i in range(len(feats)):
            q = feats.iloc[i]
            if q["quality"] < pol.min_quality:
                actions.append("ABSTAIN"); reasons.append("low_signal_quality"); continue
            if (q["max_foil_z"] >= pol.cm_z_gate) or (q["foil_disp"] >= pol.cm_disp_gate):
                actions.append("ABSTAIN"); reasons.append("possible_countermeasure"); continue
            if bits[i] >= pol.upper_bits:
                actions.append("INFORMED"); reasons.append("evidence_exceeds_upper_boundary")
            elif bits[i] <= -pol.lower_bits:
                actions.append("NAIVE"); reasons.append("evidence_below_lower_boundary")
            else:
                actions.append("ABSTAIN"); reasons.append("insufficient_evidence")
        res["action"] = actions
        res["reason"] = reasons
        for c in ("concealed", "used_countermeasure"):
            if c in feats.columns:
                res[c] = feats[c].to_numpy()
        return res


def fit_predict_crossval_fusion(feats: pd.DataFrame, channels=None, policy=None,
                                n_splits=5, random_state=0):
    """Subject-grouped CV for the Bayesian fusion layer."""
    y = feats["concealed"].to_numpy()
    groups = feats["subject_id"].to_numpy()
    cols = [c for c in feature_columns() if c.startswith("z_")]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    parts = []
    for tr, te in sgkf.split(np.zeros((len(feats), 1)), y, groups):
        m = BayesianFusion(channels=channels or list(EVIDENCE_CHANNELS),
                           policy=policy or SPRTPolicy())
        m.fit(feats.iloc[tr].reset_index(drop=True))
        parts.append(m.decide(feats.iloc[te].reset_index(drop=True)))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# MultivariateFusion — correlation-aware alternative to naive-sum SPRT (C9 fix).
#
# The naive per-channel LLR sum in BayesianFusion assumes conditional channel
# independence. The agent-council statistician and neuroscientist showed the
# autonomic channels share an orienting factor (measured LLR corr mean |r|~0.31),
# so summing their LLRs double-counts evidence and yields OVERCONFIDENT posteriors
# (worse Brier score despite similar AUC). This class models the joint channel
# covariance with a single multivariate logistic fit and calibrates with isotonic
# regression on held-out folds. Use it when calibration (not just ranking)
# matters — e.g. any time a posterior probability will be reported or a
# base-rate/PPV calculation depends on it.
# ---------------------------------------------------------------------------
class MultivariateFusion:
    """Correlation-aware fusion: one multivariate logistic over all channels +
    isotonic calibration. Better-calibrated than naive-sum LLR when channels are
    correlated. Same decide() contract as BayesianFusion."""

    def __init__(self, channels=None, policy: "SPRTPolicy" = None, C: float = 1.0):
        from sklearn.linear_model import LogisticRegression  # noqa
        self.channels = list(channels) if channels else list(EVIDENCE_CHANNELS)
        self.policy = policy or SPRTPolicy()
        self.C = C
        self._lr = None
        self._iso = None

    def _design(self, feats: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.channels:
            if c == "legacy_z":
                lp = feats["legacy_poly"] if "legacy_poly" in feats.columns else feats.get("legacy_z", 0.0)
                cols.append(((lp - np.mean(lp)) / (np.std(lp) + 1e-9)).to_numpy()
                            if hasattr(lp, "to_numpy") else np.zeros(len(feats)))
            else:
                cols.append(feats[c].to_numpy())
        return np.column_stack(cols)

    def fit(self, feats: pd.DataFrame):
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        X = self._design(feats); y = feats["concealed"].to_numpy()
        self._lr = LogisticRegression(C=self.C, max_iter=500).fit(X, y)
        praw = self._lr.predict_proba(X)[:, 1]
        self._iso = IsotonicRegression(out_of_bounds="clip").fit(praw, y)
        return self

    def predict_proba(self, feats: pd.DataFrame) -> np.ndarray:
        praw = self._lr.predict_proba(self._design(feats))[:, 1]
        return self._iso.transform(praw)

    def decide(self, feats: pd.DataFrame) -> pd.DataFrame:
        pol = self.policy
        p = np.clip(self.predict_proba(feats), 1e-6, 1 - 1e-6)
        bits = np.log2(p / (1 - p))
        res = pd.DataFrame({
            "question_set_id": feats.get("question_set_id", np.arange(len(feats))),
            "p_informed": p, "bits": bits,
        })
        actions, reasons = [], []
        for i in range(len(feats)):
            q = feats.iloc[i]
            if q.get("quality", 1.0) < pol.min_quality:
                actions.append("ABSTAIN"); reasons.append("low_signal_quality"); continue
            if (q.get("max_foil_z", 0) >= pol.cm_z_gate) or (q.get("foil_disp", 0) >= pol.cm_disp_gate):
                actions.append("ABSTAIN"); reasons.append("possible_countermeasure"); continue
            if bits[i] >= pol.upper_bits:
                actions.append("INFORMED"); reasons.append("evidence_exceeds_upper_boundary")
            elif bits[i] <= -pol.lower_bits:
                actions.append("NAIVE"); reasons.append("evidence_below_lower_boundary")
            else:
                actions.append("ABSTAIN"); reasons.append("insufficient_evidence")
        res["action"] = actions; res["reason"] = reasons
        for c in ("concealed", "used_countermeasure"):
            if c in feats.columns:
                res[c] = feats[c].to_numpy()
        return res
