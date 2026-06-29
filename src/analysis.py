"""The analysis: H-incremental, H-encoder, H-bound, and conformal coverage.

All tests carry pre-committed kill criteria. Every reported number is a Finding (value, CI, n, passed).

No em dashes.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .baseline import features_matrix, content_only_logit, content_plus_signal_logit
from .signature import compute_signatures
from .encoders import get_encoder
from .fusion import Finding, estimate_latent_factor, h_load


def _cv_auc_pair(X, b, y, n_splits=5, n_repeats=10, seed=0):
    """Repeated stratified CV. Returns per-fold (auc_content, auc_content_plus_B) arrays."""
    rng = np.random.default_rng(seed)
    auc_c, auc_cb = [], []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(rng.integers(1e9)))
        for tr, te in skf.split(X, y):
            if len(np.unique(y[te])) < 2:
                continue
            p_c = content_only_logit(X[tr], y[tr], X[te])
            p_cb = content_plus_signal_logit(X[tr], b[tr], y[tr], X[te], b[te])
            auc_c.append(roc_auc_score(y[te], p_c))
            auc_cb.append(roc_auc_score(y[te], p_cb))
    return np.asarray(auc_c), np.asarray(auc_cb)


def h_incremental(rows, behavior, encoder_name="reference", seed=0, signature=None) -> dict:
    """Does B(s) add predictive value about measured behavior over the content-only model?

    Kill criterion: pass requires the paired delta-AUC (content+B minus content) bootstrap 95% CI to
    exclude 0 and be positive. Pass `signature` to override the computed B(s) (used for the scrambled
    negative control).
    """
    X = features_matrix(rows)
    y = np.asarray(behavior, dtype=int)
    if signature is not None:
        b = np.asarray(signature, dtype=float)
    else:
        enc = get_encoder(encoder_name)
        b = compute_signatures(rows, enc)

    auc_c, auc_cb = _cv_auc_pair(X, b, y, seed=seed)
    delta = auc_cb - auc_c
    rng = np.random.default_rng(seed + 1)
    boots = np.array([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(3000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    d0 = float(delta.mean())
    passed = bool(lo > 0)
    finding = Finding(
        name=f"H_incremental_deltaAUC[{encoder_name}]",
        value=d0,
        ci95=(lo, hi),
        n=len(y),
        passed=passed,
        note=f"content AUC={auc_c.mean():.3f}, content+B AUC={auc_cb.mean():.3f}",
    )
    return {
        "finding": finding.as_dict(),
        "auc_content": float(auc_c.mean()),
        "auc_content_plus_B": float(auc_cb.mean()),
    }


def h_encoder(rows, behavior, indicators, encoder_names, seed=0) -> dict:
    """Do H-load and H-incremental survive swapping the brain encoder?

    Kill criterion: pass requires the same qualitative verdict (both significant, same sign) across
    all encoders. A verdict that flips on swap is an encoder artifact and is dropped.
    """
    latent = estimate_latent_factor(indicators)
    per = {}
    load_pass, incr_pass, signs = [], [], []
    for name in encoder_names:
        enc = get_encoder(name)
        b = compute_signatures(rows, enc)
        load = h_load(b, latent, seed=seed)
        incr = h_incremental(rows, behavior, encoder_name=name, seed=seed)
        per[name] = {"h_load": load.as_dict(), "h_incremental": incr["finding"]}
        load_pass.append(load.passed)
        incr_pass.append(incr["finding"]["passed"])
        signs.append(np.sign(incr["finding"]["value"]))
    stable = bool(all(load_pass) == True and all(incr_pass) == True and len(set(signs)) == 1) or \
             bool(all(not p for p in load_pass) and all(not p for p in incr_pass))
    return {
        "per_encoder": per,
        "encoder_stable": stable,
        "note": "stable = same qualitative verdict across all encoders",
    }


def h_bound(rows, behavior, encoder_name="reference", seed=0, signature=None) -> dict:
    """Quantify the magnitude: AUC of B(s) alone for measured behavior, with bootstrap CI.

    This is the risk magnitude (how much the neural signature alone tracks the outcome), reported as a
    bounded effect with a CI. No kill criterion; this is the descriptive quantification.
    """
    if signature is not None:
        b = np.asarray(signature, dtype=float)
    else:
        enc = get_encoder(encoder_name)
        b = compute_signatures(rows, enc)
    y = np.asarray(behavior, dtype=int)
    auc0 = float(roc_auc_score(y, b))
    rng = np.random.default_rng(seed + 2)
    n = len(y)
    boots = []
    for _ in range(3000):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], b[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "finding": Finding(
            name=f"H_bound_AUC_B_alone[{encoder_name}]",
            value=auc0, ci95=(lo, hi), n=n, passed=bool(lo > 0.5),
            note="magnitude of belief-imparting signal carried by B(s) alone",
        ).as_dict()
    }


def conformal_coverage(rows, behavior, encoder_name="reference", alpha=0.10, seed=0) -> dict:
    """Split-conformal coverage for the content+B predictor of binary behavior.

    Reports empirical coverage of the conformal prediction set at target 1 - alpha. The honesty layer:
    coverage at or above nominal means the instrument abstains rather than confabulating.
    """
    rng = np.random.default_rng(seed + 3)
    X = features_matrix(rows)
    enc = get_encoder(encoder_name)
    b = compute_signatures(rows, enc)
    y = np.asarray(behavior, dtype=int)
    n = len(y)
    idx = rng.permutation(n)
    n_tr, n_cal = int(0.5 * n), int(0.25 * n)
    tr, cal, te = idx[:n_tr], idx[n_tr:n_tr + n_cal], idx[n_tr + n_cal:]

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(np.column_stack([X[tr], b[tr]]), y[tr])

    def probs(ix):
        return clf.predict_proba(np.column_stack([X[ix], b[ix]]))

    # nonconformity = 1 - p(true class) on calibration set
    p_cal = probs(cal)
    nonconf = 1.0 - p_cal[np.arange(len(cal)), y[cal]]
    qhat = np.quantile(nonconf, min(1.0, np.ceil((len(cal) + 1) * (1 - alpha)) / len(cal)))

    p_te = probs(te)
    # prediction set: labels whose (1 - p) <= qhat
    in_set = (1.0 - p_te) <= qhat
    covered = in_set[np.arange(len(te)), y[te]]
    coverage = float(covered.mean())
    avg_set = float(in_set.sum(axis=1).mean())
    return {
        "target_coverage": 1 - alpha,
        "empirical_coverage": coverage,
        "avg_set_size": avg_set,
        "passed": bool(coverage >= (1 - alpha) - 0.03),
        "note": "coverage at or above nominal = honest abstention holds",
    }
