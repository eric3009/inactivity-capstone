import numpy as np
import pandas as pd
import pytest

from inactivity.anomaly import fit_forest, score
from inactivity.hypothesis import feature_tests, holm, uk_comparison
from inactivity.segments import CLUSTER_FEATURES
from inactivity.selection import rank_filter, rank_l1, rank_wrapper


def test_holm_matches_a_hand_calculation_and_is_monotone():
    adj = holm(np.array([0.01, 0.04, 0.03, 0.005]))
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])          # sorted p: .005,.01,.03,.04 -> x4,x3,x2,x1 with running max
    assert (adj >= np.array([0.01, 0.04, 0.03, 0.005])).all() and adj.max() <= 1


def _labelled(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({"signal": rng.normal(0, 1, n) + 1.2 * y, "noise": rng.normal(0, 1, n), "constant": 1.0,
                       "is_uk": rng.integers(0, 2, n).astype(float)})
    df["y_inactive"] = y
    return df


def test_mann_whitney_finds_the_planted_signal_not_the_noise():
    t = feature_tests(_labelled(), ["signal", "noise", "constant"])
    row = {r.feature: r for r in t.itertuples()}
    assert row["signal"].significant and row["signal"].prob_lapser_higher > 0.7 and row["signal"].size in ("medium", "large")
    assert not row["noise"].significant and row["noise"].size == "negligible"
    assert row["constant"].p == 1.0                               # a constant input cannot differ
    assert t.iloc[0]["feature"] == "signal"                       # sorted by effect size


def test_uk_comparison_detects_a_planted_difference_and_a_null():
    rng = np.random.default_rng(1)
    n = 4000
    uk = rng.integers(0, 2, n)
    y = (rng.random(n) < np.where(uk == 1, 0.55, 0.35)).astype(int)
    d = uk_comparison(pd.DataFrame({"is_uk": uk.astype(float), "y_inactive": y}))
    assert d["difference"] > 0.1 and d["p_value"] < 1e-6 and d["ci95"][0] > 0
    null = uk_comparison(pd.DataFrame({"is_uk": uk.astype(float), "y_inactive": rng.integers(0, 2, n)}))
    assert null["p_value"] > 0.01 and abs(null["difference"]) < 0.06


def test_selection_rankers_put_the_informative_inputs_first():
    rng = np.random.default_rng(2)
    n = 1500
    y = rng.integers(0, 2, n)
    X = np.column_stack([rng.normal(0, 1, n) + 1.5 * y, rng.normal(0, 1, n) + 0.8 * y] + [rng.normal(0, 1, n) for _ in range(4)])
    feats = [f"f{i}" for i in range(6)]
    for order in (rank_filter(X, y, feats, 0), rank_wrapper(X, y, feats), rank_l1(X, y, feats)):
        assert order[0] == "f0" and set(order[:2]) == {"f0", "f1"}


def test_isolation_forest_flags_planted_outliers_and_few_normal_rows():
    rng = np.random.default_rng(3)
    train = pd.DataFrame(rng.normal(0, 1, (1500, len(CLUSTER_FEATURES))), columns=CLUSTER_FEATURES)
    model = fit_forest(train, contamination=0.02, seed=1)
    normal = pd.DataFrame(rng.normal(0, 1, (500, len(CLUSTER_FEATURES))), columns=CLUSTER_FEATURES)
    odd = pd.DataFrame(rng.normal(8, 1, (50, len(CLUSTER_FEATURES))), columns=CLUSTER_FEATURES)
    _, flag_normal = score(model, normal)
    s_odd, flag_odd = score(model, odd)
    assert flag_odd.mean() > 0.95 and flag_normal.mean() < 0.06
    assert s_odd.min() > score(model, normal)[0].mean()
