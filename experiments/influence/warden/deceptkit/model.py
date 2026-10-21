"""
model.py — the estimator and the decision layer.

Design commitments (these are what make the confidence score *useful*, not
decorative):

  * Output is a CALIBRATED probability of concealed information, p in [0,1].
    Calibration is enforced with isotonic regression on held-out folds, so
    "p = 0.8" actually means ~80% of such cases are informed. An uncalibrated
    classifier score is not a usable confidence.

  * Two DISTINCT uncertainties are reported, because they mean different things:
      - p_hat        : the point probability (aleatoric / how informed the case is)
      - p_lo, p_hi   : an epistemic interval from a bootstrap ensemble
                       (how much the model itself is unsure given finite data /
                       this case being unusual). A wide interval -> abstain.

  * A COUNTERMEASURE / QUALITY gate can veto a confident-looking answer:
    if foils are behaving like probes (max_foil_z / foil_disp high) or signal
    quality is low, the case is routed to ABSTAIN regardless of p_hat, because
    the probe-vs-foil contrast the whole method rests on is compromised.

  * The decision layer maps (p_hat, interval width, gate) -> one of
    {INFORMED, NAIVE, ABSTAIN} using explicit, tunable thresholds, and reports
    which rule fired. No hidden binary verdict.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold

from .features import feature_columns


@dataclass
class DecisionPolicy:
    """Thresholds for turning a calibrated probability into an action."""
    p_high: float = 0.80          # >= this (and interval clears 0.5) -> INFORMED
    p_low: float = 0.20           # <= this (and interval clears 0.5) -> NAIVE
    max_interval_width: float = 0.45   # wider epistemic interval -> ABSTAIN
    cm_z_gate: float = 2.5        # a foil this salient looks like a countermeasure
    cm_disp_gate: float = 1.6     # high foil dispersion -> countermeasure suspicion
    min_quality: float = 0.5      # below this, signal too poor to decide


@dataclass
class DeceptionModel:
    """Calibrated bootstrap-ensemble estimator for concealed-information detection."""
    n_boot: int = 40
    n_splits: int = 5
    random_state: int = 0
    policy: DecisionPolicy = field(default_factory=DecisionPolicy)
    _members: list = field(default_factory=list, init=False)
    _cols: list = field(default_factory=list, init=False)

    def _base(self):
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=self.random_state,
        )

    def fit(self, feats: pd.DataFrame):
        """Fit the ensemble. Each member is trained on a subject-grouped
        bootstrap and calibrated by isotonic regression on an out-of-bag slice,
        so calibration never sees its own training rows."""
        self._cols = feature_columns()
        X = feats[self._cols].to_numpy()
        y = feats["concealed"].to_numpy()
        groups = feats["subject_id"].to_numpy()
        uniq_subj = np.unique(groups)
        rng = np.random.default_rng(self.random_state)
        self._members = []
        for b in range(self.n_boot):
            # bootstrap at the SUBJECT level (never split a subject across
            # train/calib) to respect the grouping.
            samp = rng.choice(uniq_subj, size=len(uniq_subj), replace=True)
            in_bag = np.isin(groups, samp)
            oob = ~in_bag
            if oob.sum() < 20 or len(np.unique(y[in_bag])) < 2 or len(np.unique(y[oob])) < 2:
                continue
            clf = clone(self._base())
            clf.fit(X[in_bag], y[in_bag])
            raw_oob = clf.predict_proba(X[oob])[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_oob, y[oob])
            self._members.append((clf, iso))
        if not self._members:
            raise RuntimeError("No ensemble members fit; check data size/prevalence.")
        return self

    def predict_proba_dist(self, feats: pd.DataFrame):
        """Return (p_hat, p_lo, p_hi): calibrated mean probability and a 90%
        epistemic interval across ensemble members."""
        X = feats[self._cols].to_numpy()
        preds = np.column_stack([iso.transform(clf.predict_proba(X)[:, 1])
                                 for clf, iso in self._members])
        p_hat = preds.mean(axis=1)
        p_lo = np.quantile(preds, 0.05, axis=1)
        p_hi = np.quantile(preds, 0.95, axis=1)
        return p_hat, p_lo, p_hi

    def decide(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Full inference: probability, interval, gate, action, and reason."""
        p_hat, p_lo, p_hi = self.predict_proba_dist(feats)
        width = p_hi - p_lo
        pol = self.policy
        res = feats[["question_set_id", "subject_id"]].copy()
        res["p_informed"] = p_hat
        res["p_lo"] = p_lo
        res["p_hi"] = p_hi
        res["interval_width"] = width

        actions, reasons = [], []
        for i in range(len(feats)):
            q = feats.iloc[i]
            gate_cm = (q["max_foil_z"] >= pol.cm_z_gate) or (q["foil_disp"] >= pol.cm_disp_gate)
            gate_q = q["quality"] < pol.min_quality
            if gate_q:
                actions.append("ABSTAIN"); reasons.append("low_signal_quality"); continue
            if gate_cm:
                actions.append("ABSTAIN"); reasons.append("possible_countermeasure"); continue
            if width[i] > pol.max_interval_width:
                actions.append("ABSTAIN"); reasons.append("high_model_uncertainty"); continue
            if p_hat[i] >= pol.p_high and p_lo[i] > 0.5:
                actions.append("INFORMED"); reasons.append("confident_recognition")
            elif p_hat[i] <= pol.p_low and p_hi[i] < 0.5:
                actions.append("NAIVE"); reasons.append("confident_no_recognition")
            else:
                actions.append("ABSTAIN"); reasons.append("inconclusive_probability")
        res["action"] = actions
        res["reason"] = reasons
        # carry ground truth if present (for evaluation)
        for c in ("concealed", "used_countermeasure"):
            if c in feats.columns:
                res[c] = feats[c].to_numpy()
        return res


def fit_predict_crossval(feats: pd.DataFrame, n_splits=5, random_state=0, policy=None):
    """Honest evaluation: subject-grouped cross-validation so no subject appears
    in both train and test. Returns per-set decisions with out-of-fold estimates."""
    cols = feature_columns()
    y = feats["concealed"].to_numpy()
    groups = feats["subject_id"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    parts = []
    for tr, te in sgkf.split(feats[cols].to_numpy(), y, groups):
        m = DeceptionModel(random_state=random_state,
                           policy=policy or DecisionPolicy())
        m.fit(feats.iloc[tr].reset_index(drop=True))
        parts.append(m.decide(feats.iloc[te].reset_index(drop=True)))
    return pd.concat(parts, ignore_index=True)
