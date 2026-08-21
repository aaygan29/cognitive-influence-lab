"""Fast operating-characteristic tests for the estimator, pooling, and transfer pipeline.

Kept permutation-free so the suite stays quick; the full permutation nulls run in the experiment
scripts. Run: python tests/test_pipeline.py  (or python -m pytest tests/ -q)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import phenotype as P  # noqa: E402
import transfer as T   # noqa: E402


def test_recovery_improves_with_more_trials():
    pop = P.Population()
    rs = {}
    for nt in (200, 1500):
        out = P.recovery_once(seed=0, n_agents=40, n_trials=nt, pop=pop)
        from scipy import stats
        rs[nt] = stats.pearsonr(out["truth"]["threat"], out["recovered"]["threat"])[0]
    assert rs[1500] > rs[200] + 0.1  # more data -> better recovery


def test_pooling_helps_low_n():
    pop = P.Population()
    rng = np.random.default_rng(3)
    na, nt = 40, 150
    theta = pop.sample_theta(rng, na)
    truth = P.true_ratios(theta)
    Xs, ys = [], []
    for i in range(na):
        ti = {k: float(theta[k][i]) for k in theta}
        X = P.sample_contexts(rng, nt); Xs.append(X)
        ys.append(P.simulate_agent_choices(rng, X, ti))
    unp = [P.fit_agent(Xs[i], ys[i]) for i in range(na)]
    poo = P.fit_agents_pooled(Xs, ys)
    params = ["loss_aversion", "threat", "risk", "discount"]
    ru = P._recovery_r_for_betas(unp, truth, params)
    rp = P._recovery_r_for_betas(poo, truth, params)
    # pooling should help on average across the phenotype at low n
    assert np.mean([rp[k] for k in params]) > np.mean([ru[k] for k in params])


def test_transfer_positive_control_brain_beats_behavior():
    diffs = []
    for s in range(4):
        d = T.generate_transfer_data(n_subjects=30, n_stim=70, dissociation=True, seed=s)
        A = T._arms(d); y = d["market"]
        rb = T._oos_scores(y, T._ridge_oos(A["brain_only"], y, seed=s))["r2"]
        rbe = T._oos_scores(y, T._ridge_oos(A["behavior_only"], y, seed=s))["r2"]
        diffs.append(rb - rbe)
    assert np.mean(diffs) > 0.02  # brain forecasts held-out market better than behavior


def test_transfer_negative_control_is_null():
    r2s = []
    for s in range(4):
        d = T.generate_transfer_data(n_subjects=30, n_stim=70, dissociation=False, seed=s)
        y = d["market"]
        r2s.append(T._oos_scores(y, T._ridge_oos(d["brain_feat"], y, seed=s))["r2"])
    assert np.mean(r2s) < 0.1  # no predictive signal when the market is noise


def test_scale_ladder_dose_response():
    import inverse as I
    res = I.run_scale_ladder(n_entities=30, n_seeds=3)
    idc = res["identifiability_curve"]; abc = res["abstention_curve"]
    # identifiability higher at aggregate than individual; abstention lower at aggregate
    assert idc[0] > idc[-1]
    assert abc[0] < abc[-1]
    assert abc[-1] > 0.5  # individual scope mostly abstains


def test_state_shift_recovers_direction():
    import states as S
    res = S.run_e7(n_entities=25, n_trials=1200, n_seeds=2)
    assert res["positive_control"]["detects_shift"] is True
    assert res["negative_control"]["detects_shift"] is False


def test_individuation_machinery():
    import numpy as np
    import real_narps_individuation as I
    # distinct phenotype vectors -> perfect self-identification; structure runs
    rng = np.random.default_rng(0)
    db = rng.normal(0, 1, (20, 6)); tg = db + rng.normal(0, 0.01, (20, 6))
    acc, matches = I._identify(db, tg)
    assert acc > 0.9  # near-perfect when target is each subject's own vector + tiny noise
    st = I.structure_analysis(db, tg)
    assert 0 < st["participation_ratio_effective_dim"] <= 6


def test_triangulation_math():
    import math
    import triangulation_stats as TS
    # transitivity bound brackets the independent-residual product and stays in [-1,1]
    b = TS.transitivity_bound(0.5, 0.6)
    assert b["implied_low"] <= 0.5 * 0.6 <= b["implied_high"]
    assert b["implied_low"] >= -1.0 and b["implied_high"] <= 1.0
    # perfect chain (r=1,1) pins the direct correlation to exactly 1
    b2 = TS.transitivity_bound(1.0, 1.0)
    assert abs(b2["implied_low"] - 1.0) < 1e-9 and abs(b2["implied_high"] - 1.0) < 1e-9
    # Sobel: larger paths -> larger |z|; Fisher combine of tiny ps is tiny
    z_big = abs(TS.sobel(0.5, 0.1, 0.5, 0.1)["z"]) 
    z_small = abs(TS.sobel(0.1, 0.1, 0.1, 0.1)["z"])
    assert z_big > z_small
    assert TS.fisher_combine([1e-4, 1e-4])["combined_p"] < 1e-3


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"\n{len(fns)} tests passed")
