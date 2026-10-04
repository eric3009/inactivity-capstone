"""Metrics, calibration diagnostics and customer-clustered bootstrap intervals."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-6


def top_share_lift(y, s, share=0.10) -> float:
    y, s = np.asarray(y), np.asarray(s)
    k = max(1, math.ceil(share * len(y)))
    top = np.argsort(-s, kind="stable")[:k]
    prev = y.mean()
    return float(y[top].mean() / prev) if prev > 0 else float("nan")


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(m.coef_[0][0]), float(m.intercept_[0])


def ece(y, p, bins=10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return float(sum(len(c) / len(y) * abs(y[c].mean() - p[c].mean()) for c in chunks if len(c)))


def reliability_table(y, p, bins=10) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    chunks = np.array_split(np.argsort(p), bins)
    return pd.DataFrame({"bin": range(len(chunks)),
                         "mean_predicted": [p[c].mean() for c in chunks],
                         "observed_rate": [y[c].mean() for c in chunks],
                         "n": [len(c) for c in chunks]})


def discrimination(y, s, share=0.10) -> dict:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "lift_top": float("nan")}
    return {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
            "lift_top": top_share_lift(y, s, share)}


def full_metrics(y, s, calibrated: bool, share=0.10) -> dict:
    out = discrimination(y, s, share)
    out["prevalence"] = float(np.mean(y))
    if calibrated:
        slope, inter = calibration_slope_intercept(y, s)
        out.update(brier=brier(y, s), cal_slope=slope, cal_intercept=inter, ece=ece(y, s))
    return out


def cluster_bootstrap(y, customers, scores: dict, reps=500, seed=42, share=0.10,
                      prob_models: tuple = (), pairs: tuple = ()) -> dict:
    """Resample whole customers (all their snapshots) with replacement.

    Returns {(model, metric): array}. `pairs` is a tuple of (a, b) model names for which the
    paired difference a - b is also returned under the key (f"{a}-{b}", metric).
    """
    y = np.asarray(y)
    codes, _ = pd.factorize(pd.Series(customers))
    order = np.argsort(codes, kind="stable")
    y = y[order]
    sc = {k: np.asarray(v)[order] for k, v in scores.items()}
    counts = np.bincount(codes[order])
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    rng = np.random.default_rng(seed)
    res: dict = {}
    n_c = len(counts)
    for _ in range(reps):
        pick = rng.integers(0, n_c, n_c)
        lens = counts[pick]
        idx = np.arange(lens.sum()) + np.repeat(starts[pick] - np.r_[0, np.cumsum(lens)[:-1]], lens)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        vals = {}
        for name, s in sc.items():
            sb = s[idx]
            m = {"roc_auc": roc_auc_score(yb, sb), "pr_auc": average_precision_score(yb, sb),
                 "lift_top": top_share_lift(yb, sb, share)}
            if name in prob_models:
                m["brier"] = brier(yb, sb)
            vals[name] = m
            for k, v in m.items():
                res.setdefault((name, k), []).append(v)
        for a, b in pairs:
            for k in ("roc_auc", "pr_auc", "lift_top"):
                res.setdefault((f"{a}-{b}", k), []).append(vals[a][k] - vals[b][k])
    return {k: np.asarray(v) for k, v in res.items()}


def ci(arr, level=0.95) -> tuple[float, float]:
    lo, hi = np.percentile(arr, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)
