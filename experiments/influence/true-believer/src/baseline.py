"""The content-only baseline: the model B(s) must beat to demonstrate incremental risk signal.

This is a strong but transparent content model. It uses the same interpretable persuasion features
that the encoder sees, fit directly to the outcome with regularized logistic / linear regression.
If the neural signature B(s) adds nothing over this, the risk is fully content-level and already
visible to content analysis. That is the null we genuinely test against, not a strawman.

No em dashes.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV

from .encoders import CONTENT_FEATURES


def features_matrix(features_rows) -> np.ndarray:
    return np.array(
        [[float(f.get(k, 0.0)) for k in CONTENT_FEATURES] for f in features_rows],
        dtype=float,
    )


def content_only_logit(X_train, y_train, X_test):
    """Probability predictions from a content-only logistic model (binary outcome)."""
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def content_plus_signal_logit(X_train, b_train, y_train, X_test, b_test):
    """Content features augmented with the neural signature B(s)."""
    Xa = np.column_stack([X_train, b_train])
    Xb = np.column_stack([X_test, b_test])
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xa, y_train)
    return clf.predict_proba(Xb)[:, 1]


def content_only_linear(X_train, y_train, X_test):
    reg = RidgeCV(alphas=np.logspace(-3, 3, 13))
    reg.fit(X_train, y_train)
    return reg.predict(X_test)
