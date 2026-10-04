import numpy as np

from inactivity import evaluate as ev


def test_perfect_ranking_and_lift():
    y = np.array([0] * 90 + [1] * 10)
    s = np.r_[np.zeros(90), np.ones(10)]
    m = ev.discrimination(y, s, 0.10)
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0
    assert abs(m["lift_top"] - 10.0) < 1e-9          # top 10% are all positives, prevalence 10%


def test_calibrated_probabilities_have_slope_near_one():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.random(20000) < p).astype(int)
    slope, intercept = ev.calibration_slope_intercept(y, p)
    assert abs(slope - 1) < 0.08 and abs(intercept) < 0.08
    assert ev.ece(y, p) < 0.02


def test_overconfident_probabilities_have_slope_below_one():
    rng = np.random.default_rng(1)
    true_p = rng.uniform(0.2, 0.8, 20000)
    y = (rng.random(20000) < true_p).astype(int)
    over = 1 / (1 + np.exp(-3 * np.log(true_p / (1 - true_p))))     # stretched logits
    assert ev.calibration_slope_intercept(y, over)[0] < 0.6


def test_cluster_bootstrap_interval_covers_point_estimate():
    rng = np.random.default_rng(2)
    n_c = 300
    cust = np.repeat(np.arange(n_c), 3)
    latent = rng.normal(size=n_c).repeat(3)
    s = latent + rng.normal(scale=0.5, size=n_c * 3)
    y = (latent + rng.normal(size=n_c * 3) > 0.3).astype(int)
    boot = ev.cluster_bootstrap(y, cust, {"a": s, "b": -s}, reps=200, seed=0, pairs=(("a", "b"),))
    point = ev.discrimination(y, s)["roc_auc"]
    lo, hi = ev.ci(boot[("a", "roc_auc")])
    assert lo < point < hi
    lo_d, hi_d = ev.ci(boot[("a-b", "roc_auc")])
    assert lo_d > 0                                     # a ranks perfectly better than its negation
