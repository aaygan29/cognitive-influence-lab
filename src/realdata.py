"""Real-data loader and deterministic feature extractor for the ChangeMyView / Winning Arguments corpus.

Data: Tan et al. 2016, obtained via ConvoKit (winning-args-corpus). Each labeled utterance is a challenger
argument with a success label (1 = the original poster awarded a delta, meaning the argument changed their
view; 0 = no delta) and pair_ids linking successful and unsuccessful challengers to the SAME original post.
The paired design is the confound control: within a pair the topic and the original poster are held fixed,
so a within-pair comparison removes topic and difficulty confounds (this is Tan et al.'s own design).

Two feature sets are computed, deterministically, from text alone. No external model, no gated data.

1. content_baseline: generic style and content features (length, lexical diversity, questions, links,
   numbers, average word length). This is the strong content-only comparison, in the spirit of the
   surface features used in the original study.
2. signature channels E, V, R: persuasion-neuroscience-grounded lexical channels. E (engagement) from
   evidence and analytic-framing markers; V (value/affect) from emotional and social-proof markers; R
   (resistance) from identity-threat and aggression markers. The neural belief-formation signature is
   B = z(E) + z(V) - z(R), the text-proxy analogue of the fMRI signature. This is neuro-THEORY grounded,
   not fMRI-validated; a learned brain encoder replacing this text proxy is the stated next step.

No em dashes.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict

import numpy as np

# ---- lexicons (small, transparent, editable) ----
EVIDENCE = ["because", "evidence", "study", "studies", "data", "research", "source", "sources",
            "example", "for instance", "in fact", "statistics", "proof", "cite", "according to",
            "shows that", "demonstrates", "reason", "therefore", "thus", "hence"]
AFFECT = ["afraid", "angry", "hate", "love", "fear", "hope", "terrible", "wonderful", "amazing",
          "awful", "disgusting", "happy", "sad", "suffer", "pain", "joy", "outrage", "shocking",
          "beautiful", "horrible", "wrong", "evil", "cruel"]
SOCIAL_PROOF = ["most people", "everyone", "nobody", "we all", "consensus", "widely", "commonly",
                "the majority", "many people", "experts", "scientists agree"]
IDENTITY_THREAT = ["you people", "you clearly", "you obviously", "you fail", "your kind", "idiot",
                   "stupid", "ignorant", "ridiculous", "nonsense", "clearly wrong", "you don't",
                   "you can't", "obviously"]
ANALYTIC_FRAME = ["consider", "suppose", "imagine", "if we", "let us", "let's", "what if",
                  "on the other hand", "however", "although", "granted", "admittedly", "it depends"]
CONCESSION = ["i agree", "you're right", "fair point", "good point", "i concede", "that's true",
              "you make a good", "i see your point"]


def _count(text_low, phrases):
    return sum(text_low.count(p) for p in phrases)


def extract_features(text: str) -> dict:
    """Deterministic text features. Returns content_baseline dict and E, V, R channel raw scores."""
    t = text or ""
    low = t.lower()
    words = re.findall(r"[a-zA-Z']+", low)
    n = max(len(words), 1)
    uniq = len(set(words))
    links = len(re.findall(r"http[s]?://", low))
    numbers = len(re.findall(r"\b\d+\b", t))
    questions = t.count("?")
    exclam = t.count("!")
    avg_wlen = np.mean([len(w) for w in words]) if words else 0.0

    # per-1000-word normalized channel intensities
    scale = 1000.0 / n
    evidence = (_count(low, EVIDENCE) + links + 0.5 * numbers) * scale
    frame = _count(low, ANALYTIC_FRAME) * scale
    affect = (_count(low, AFFECT) + 0.5 * exclam) * scale
    social = _count(low, SOCIAL_PROOF) * scale
    identity = _count(low, IDENTITY_THREAT) * scale
    concession = _count(low, CONCESSION) * scale

    # neuroscience-grounded channels
    E = evidence + frame + concession          # engagement / analytic uptake (MPFC analogue)
    V = affect + social                          # value / affect / social reward (NAcc analogue)
    R = identity + 0.5 * exclam * scale          # resistance / identity threat (DMN+dmPFC analogue)

    content_baseline = {
        "log_len": np.log1p(n),
        "type_token": uniq / n,
        "avg_word_len": avg_wlen,
        "questions_per_k": questions * scale,
        "links_per_k": links * scale,
        "numbers_per_k": numbers * scale,
    }
    # interpretable, alterable content factors (the levers an author could actually change)
    factors = {
        "evidence": evidence,
        "framing": frame,
        "affect": affect,
        "social_proof": social,
        "identity_threat": identity,
        "concession": concession,
    }
    return {"content": content_baseline, "E": E, "V": V, "R": R, "factors": factors}


FACTOR_NAMES = ["evidence", "framing", "affect", "social_proof", "identity_threat", "concession"]


def load_labeled(corpus_dir: str):
    """Yield dicts with id, success, pair_ids, text, and extracted features for labeled challengers."""
    path = os.path.join(corpus_dir, "utterances.jsonl")
    with open(path) as f:
        for line in f:
            u = json.loads(line)
            m = u.get("meta", {})
            s = m.get("success")
            if s not in (0, 1):
                continue
            text = u.get("text", "") or ""
            if len(text.split()) < 5:
                continue
            feats = extract_features(text)
            yield {
                "id": u["id"],
                "success": int(s),
                "pair_ids": m.get("pair_ids") or [],
                "content": feats["content"],
                "E": feats["E"], "V": feats["V"], "R": feats["R"],
                "factors": feats["factors"],
            }


def assemble(corpus_dir: str):
    """Return arrays: content matrix Xc, channel arrays E,V,R, labels y, and a pair-group id per row.

    The pair-group id is the first pair_id, used as the CV grouping and within-pair matching unit so that
    successful and unsuccessful challengers to the same original post never split across train and test.
    """
    rows = list(load_labeled(corpus_dir))
    cont_keys = list(rows[0]["content"].keys())
    Xc = np.array([[r["content"][k] for k in cont_keys] for r in rows], dtype=float)
    E = np.array([r["E"] for r in rows]); V = np.array([r["V"] for r in rows]); R = np.array([r["R"] for r in rows])
    y = np.array([r["success"] for r in rows], dtype=int)
    groups = np.array([r["pair_ids"][0] if r["pair_ids"] else r["id"] for r in rows])
    return {
        "Xc": Xc, "content_keys": cont_keys, "E": E, "V": V, "R": R, "y": y,
        "groups": groups, "rows": rows,
    }


def zscore(a: np.ndarray) -> np.ndarray:
    sd = a.std()
    return (a - a.mean()) / sd if sd > 0 else a - a.mean()


def signature(E, V, R) -> np.ndarray:
    """B = z(E) + z(V) - z(R), the text-proxy neural belief-formation signature."""
    return zscore(E) + zscore(V) - zscore(R)
