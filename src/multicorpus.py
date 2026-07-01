"""Cross-corpus generalization test: does the engagement channel predict persuasion in a SECOND,
different domain?

Corpus 1: ChangeMyView (Tan et al. 2016). Item = challenger argument. Outcome = delta awarded (view
changed). Loaded via realdata.assemble.

Corpus 2: Persuasion-for-Good (Wang et al. 2019). Item = one dialogue. Text = the persuader's utterances
(role 0). Outcome = the persuadee actually donated money (donation_ee > 0). This is a real behavioral
outcome (money), in a completely different domain (charity solicitation dialogue), so agreement across the
two is genuine cross-domain generalization, not a within-corpus artifact.

Same deterministic features and the same E, V, R channels are computed for both. Features are standardized
WITHIN each corpus, then a logistic model success ~ content_baseline + E + V + R is fit per corpus. We
report the standardized coefficient of each channel with a bootstrap CI in each corpus, a cross-corpus
agreement verdict, and a fixed-effect meta-combination. With only two corpora we do not pretend to
estimate a random-effects variance; the honest test is sign-and-significance agreement plus the pooled
estimate.

No em dashes.
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

from .realdata import extract_features, assemble, zscore

ROOT = os.path.dirname(os.path.dirname(__file__))
CMV = os.path.join(ROOT, "data", "raw", "winning-args-corpus")
P4G = os.path.join(ROOT, "data", "raw", "persuasionforgood-corpus")
RESULTS = os.path.join(ROOT, "results", "multicorpus.json")

CHANNELS = ["E", "V", "R"]


def load_p4g(corpus_dir=P4G):
    """Per-dialogue: concatenate persuader (role 0) text, outcome = persuadee donated (>0)."""
    convos = json.load(open(os.path.join(corpus_dir, "conversations.json")))
    # gather persuader utterances per conversation
    text_by_convo = {}
    with open(os.path.join(corpus_dir, "utterances.jsonl")) as f:
        for line in f:
            u = json.loads(line)
            if u["meta"].get("role") != 0:
                continue
            cid = u.get("conversation_id") or u.get("root")
            text_by_convo.setdefault(cid, []).append(u.get("text", "") or "")
    rows = []
    for cid, meta in convos.items():
        txt = " ".join(text_by_convo.get(cid, []))
        if len(txt.split()) < 5:
            continue
        don = meta.get("donation_ee")
        if don is None:
            continue
        feats = extract_features(txt)
        rows.append({"content": feats["content"], "E": feats["E"], "V": feats["V"],
                     "R": feats["R"], "y": int(don > 0)})
    ck = list(rows[0]["content"].keys())
    return {
        "Xc": np.array([[r["content"][k] for k in ck] for r in rows], float),
        "content_keys": ck,
        "E": np.array([r["E"] for r in rows]), "V": np.array([r["V"] for r in rows]),
        "R": np.array([r["R"] for r in rows]), "y": np.array([r["y"] for r in rows]),
    }


def _design(data):
    Xc = data["Xc"]
    Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-9)
    E, V, R = zscore(data["E"]), zscore(data["V"]), zscore(data["R"])
    X = np.column_stack([Xc, E, V, R])
    names = list(data["content_keys"]) + ["E", "V", "R"]
    return X, names, data["y"]


def per_corpus_coefs(data, n_boot=1000, seed=0):
    """Standardized logistic coefficients for E, V, R (controlling content) with bootstrap CIs."""
    X, names, y = _design(data)
    idxE, idxV, idxR = names.index("E"), names.index("V"), names.index("R")
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
    point = {c: float(clf.coef_[0][names.index(c)]) for c in CHANNELS}
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = {c: [] for c in CHANNELS}
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        if len(np.unique(y[ix])) < 2:
            continue
        cf = LogisticRegression(max_iter=2000, C=1.0).fit(X[ix], y[ix]).coef_[0]
        for c in CHANNELS:
            boots[c].append(cf[names.index(c)])
    out = {}
    for c in CHANNELS:
        lo, hi = np.percentile(boots[c], [2.5, 97.5])
        out[c] = {"coef": point[c], "ci95": [float(lo), float(hi)],
                  "significant": bool(lo > 0 or hi < 0), "sign": int(np.sign(point[c]))}
    return out


def main():
    cmv = assemble(CMV)
    p4g = load_p4g()
    cmv_c = per_corpus_coefs({"Xc": cmv["Xc"], "content_keys": cmv["content_keys"],
                              "E": cmv["E"], "V": cmv["V"], "R": cmv["R"], "y": cmv["y"]})
    p4g_c = per_corpus_coefs(p4g)

    # cross-corpus verdict per channel: significant and same sign in both
    verdict = {}
    for c in CHANNELS:
        a, b = cmv_c[c], p4g_c[c]
        both_sig_same = a["significant"] and b["significant"] and a["sign"] == b["sign"]
        # fixed-effect (inverse-variance) meta on the coefficient
        va = ((a["ci95"][1] - a["ci95"][0]) / 3.92) ** 2
        vb = ((b["ci95"][1] - b["ci95"][0]) / 3.92) ** 2
        w = 1 / va + 1 / vb
        meta = (a["coef"] / va + b["coef"] / vb) / w
        meta_se = (1 / w) ** 0.5
        verdict[c] = {"generalizes": bool(both_sig_same),
                      "meta_coef": float(meta), "meta_ci95": [float(meta - 1.96 * meta_se), float(meta + 1.96 * meta_se)]}

    out = {
        "corpus1": {"name": "ChangeMyView (Tan 2016)", "n": int(len(cmv["y"])), "coefs": cmv_c},
        "corpus2": {"name": "Persuasion-for-Good (Wang 2019)", "n": int(len(p4g["y"])), "coefs": p4g_c},
        "cross_corpus_verdict": verdict,
        "note": "Two corpora only; sign+significance agreement and a fixed-effect meta, not a random-effects tau.",
    }
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    json.dump(out, open(RESULTS, "w"), indent=2)

    print(f"CMV n={out['corpus1']['n']}   P4G n={out['corpus2']['n']}")
    for c in CHANNELS:
        a, b, v = cmv_c[c], p4g_c[c], verdict[c]
        print(f"  {c}: CMV {a['coef']:+.3f}{a['ci95']} sig={a['significant']} | "
              f"P4G {b['coef']:+.3f}{b['ci95']} sig={b['significant']} | "
              f"generalizes={v['generalizes']} meta={v['meta_coef']:+.3f}{v['meta_ci95']}")
    print("wrote", RESULTS)
    return out


if __name__ == "__main__":
    main()
