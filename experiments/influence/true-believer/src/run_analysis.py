"""Run the full analysis on the synthetic stand-in: positive and negative controls.

Positive control (neural signal present): the pipeline should DETECT it. H-load and H-incremental pass,
B(s) carries magnitude, the verdict is stable across encoders, conformal coverage holds.

Negative control (neural signal absent): the pipeline should report NULL. H-load and H-incremental fail.

This is the discipline that makes the instrument trustworthy before it touches real public data. The
real run swaps generate(...) for the public-corpus loader (docs/DATA.md) and the reference encoder for a
learned encoder (TRIBE v2), with everything else unchanged.

No em dashes.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .make_synthetic import generate
from .fusion import estimate_latent_factor, h_load
from .signature import compute_signatures
from .encoders import get_encoder
from .analysis import h_incremental, h_encoder, h_bound, conformal_coverage

ENCODERS = ["reference", "perturbed_11", "perturbed_29"]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def run_condition(label: str, neural_weight: float, scramble: bool = False) -> dict:
    data = generate(n=1400, neural_weight=neural_weight, seed=7)
    rows = data["rows"]
    behavior = data["behavior"]
    indicators = {
        "behavior": data["behavior"],
        "self_report": data["self_report"],
        "certainty": data["certainty"],
        "mist": data["mist"],
    }
    latent = estimate_latent_factor(indicators)

    b_ref = compute_signatures(rows, get_encoder("reference"))
    sig = None
    if scramble:
        # pure negative control: permute B relative to outcomes; content baseline stays intact.
        rng = np.random.default_rng(123)
        b_ref = b_ref[rng.permutation(len(b_ref))]
        sig = b_ref

    load = h_load(b_ref, latent)
    incr = h_incremental(rows, behavior, encoder_name="reference", signature=sig)
    bound = h_bound(rows, behavior, encoder_name="reference", signature=sig)
    enc = None if scramble else h_encoder(rows, behavior, indicators, ENCODERS)
    conf = conformal_coverage(rows, behavior, encoder_name="reference")

    return {
        "label": label,
        "neural_weight": neural_weight,
        "scramble": scramble,
        "n": data["meta"]["n"],
        "H_load": load.as_dict(),
        "H_incremental": incr,
        "H_bound": bound,
        "H_encoder": enc,
        "conformal": conf,
    }


def seed_stability(n_seeds: int = 12) -> dict:
    """Repeat the three control verdicts across data seeds; report how often each lands as expected.

    Expected: positive -> H-load and H-incremental PASS; content-only -> H-incremental FAIL;
    scrambled -> H-load FAIL and H-incremental FAIL. This shows the controls are not a single draw.
    """
    pos_load, pos_incr, content_incr_null, scram_load_null, scram_incr_null = 0, 0, 0, 0, 0
    for s in range(n_seeds):
        # positive
        d = generate(n=1200, neural_weight=0.9, seed=s)
        ind = {"behavior": d["behavior"], "self_report": d["self_report"],
               "certainty": d["certainty"], "mist": d["mist"]}
        lat = estimate_latent_factor(ind)
        b = compute_signatures(d["rows"], get_encoder("reference"))
        pos_load += int(h_load(b, lat).passed)
        pos_incr += int(h_incremental(d["rows"], d["behavior"])["finding"]["passed"])
        # content-only
        d0 = generate(n=1200, neural_weight=0.0, seed=s)
        content_incr_null += int(not h_incremental(d0["rows"], d0["behavior"])["finding"]["passed"])
        # scrambled
        b2 = compute_signatures(d["rows"], get_encoder("reference"))
        rng = np.random.default_rng(1000 + s)
        b2 = b2[rng.permutation(len(b2))]
        scram_load_null += int(not h_load(b2, lat).passed)
        scram_incr_null += int(not h_incremental(d["rows"], d["behavior"], signature=b2)["finding"]["passed"])
    return {
        "n_seeds": n_seeds,
        "positive_H_load_pass_rate": pos_load / n_seeds,
        "positive_H_incremental_pass_rate": pos_incr / n_seeds,
        "content_only_H_incremental_null_rate": content_incr_null / n_seeds,
        "scrambled_H_load_null_rate": scram_load_null / n_seeds,
        "scrambled_H_incremental_null_rate": scram_incr_null / n_seeds,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {
        "program": "cultist",
        "harness": "synthetic stand-in (positive + content-only + scrambled controls); NOT a real-data result",
        "positive_control": run_condition("positive_signal_present", neural_weight=0.9),
        "content_only_world": run_condition("content_only_no_neural_signal", neural_weight=0.0),
        "scrambled_control": run_condition("scrambled_B_negative_control", neural_weight=0.9, scramble=True),
        "seed_stability": seed_stability(12),
    }
    path = os.path.join(RESULTS_DIR, "synthetic_validation.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    # console summary
    for cond in ("positive_control", "content_only_world", "scrambled_control"):
        c = out[cond]
        print(f"\n=== {cond} (neural_weight={c['neural_weight']}, scramble={c['scramble']}) ===")
        print(f"  H-load        r={c['H_load']['value']:.3f} CI={c['H_load']['ci95']} passed={c['H_load']['passed']}")
        fi = c["H_incremental"]["finding"]
        print(f"  H-incremental dAUC={fi['value']:.4f} CI={fi['ci95']} passed={fi['passed']}"
              f" (content={c['H_incremental']['auc_content']:.3f} -> +B={c['H_incremental']['auc_content_plus_B']:.3f})")
        bf = c["H_bound"]["finding"]
        print(f"  H-bound       AUC_B={bf['value']:.3f} CI={bf['ci95']}")
        enc_stable = c["H_encoder"]["encoder_stable"] if c["H_encoder"] else "n/a"
        print(f"  H-encoder     stable={enc_stable}")
        print(f"  conformal     cov={c['conformal']['empirical_coverage']:.3f} (target {c['conformal']['target_coverage']}) passed={c['conformal']['passed']}")
    print(f"\nwrote {path}")
    return out


if __name__ == "__main__":
    main()
