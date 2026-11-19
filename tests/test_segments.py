import numpy as np
import pandas as pd

from inactivity.features import FEATURES
from inactivity.segments import CLUSTER_FEATURES, assign, fit_segments, name_segments


def _frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    kind = rng.integers(0, 3, n)                        # three planted behaviours
    base = np.array([[10, 8, 20, 9, 10, 15, 25, 30], [90, 1, 2, 5, 6, 80, 2, 4], [40, 3, 6, 7, 8, 40, 8, 10]], float)
    X = base[kind] + rng.normal(0, 1.0, (n, 8)) * base[kind] * 0.05
    return pd.DataFrame(X, columns=CLUSTER_FEATURES), kind


def test_kmeans_recovers_planted_segments_and_picks_k():
    df, kind = _frame()
    m = fit_segments(df, k_range=range(2, 6), seed=1)
    assert m["k"] == 3 and set(m["silhouette"]) == {2, 3, 4, 5}
    labels = assign(m, df)
    # every planted group maps to a single cluster
    for g in range(3):
        assert len(set(labels[kind == g])) == 1


def test_assignment_is_deterministic_and_handles_missing_values():
    df, _ = _frame()
    m = fit_segments(df, k_range=range(2, 5), seed=1)
    holes = df.copy()
    holes.loc[:5, "median_interval_days"] = np.nan
    a1, a2 = assign(m, holes), assign(m, holes)
    assert (a1 == a2).all() and len(a1) == len(df)


def test_segment_names_are_unique_and_readable():
    prof = pd.DataFrame({"spend_180": [900, 40, 150, 60], "orders_180": [9, 1, 3, 2], "recency_days": [10, 170, 60, 100],
                         "purchase_days": [20, 1, 6, 4]}, index=[0, 1, 2, 3])
    names = name_segments(prof)
    assert len(set(names.values())) == 4 and names[0] == "Core regulars"
