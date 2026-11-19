"""Unsupervised customer segments (Session 7): standardise, PCA for plotting, K-Means with k chosen by silhouette.

The clustering is fitted on the training period only and then applied to the test period, so it uses no test information.
Each segment is profiled in business units and checked against the served model: how often its customers go quiet and how
well the model ranks them. This is a descriptive tool; it does not feed the model.

    python -m inactivity segments
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import resolve
from .features import FEATURES, build_dataset
from .splits import assign_blocks, make_snapshots

# Behavioural inputs only: how recently, how often, how much and how regularly customers buy.
CLUSTER_FEATURES = ["recency_days", "inv_count_90", "inv_count_365", "rev_180_log", "rev_365_log", "median_interval_days",
                    "n_purchase_days", "n_products_180"]


def _prep(df: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    return df[CLUSTER_FEATURES].fillna(medians)


def fit_segments(train: pd.DataFrame, k_range=range(3, 8), seed: int = 42, sample: int = 4000):
    """Scale, choose k by silhouette on a training sample, then fit K-Means and PCA on the full training window."""
    medians = train[CLUSTER_FEATURES].median()
    X = _prep(train, medians)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), min(sample, len(Z)), replace=False)
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z[idx])
        scores[k] = float(silhouette_score(Z[idx], labels))
    best_k = max(scores, key=scores.get)
    km = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(Z)
    pca = PCA(n_components=2, random_state=seed).fit(Z)
    return {"medians": medians, "scaler": scaler, "km": km, "pca": pca, "silhouette": scores, "k": best_k}


def assign(model: dict, df: pd.DataFrame) -> np.ndarray:
    return model["km"].predict(model["scaler"].transform(_prep(df, model["medians"])))


def name_segments(profile: pd.DataFrame) -> dict[int, str]:
    """Plain-language labels from each segment's profile (relative to the other segments)."""
    names = {}
    spend_rank = profile["spend_180"].rank(ascending=False)
    freq_rank = profile["orders_180"].rank(ascending=False)
    for c, row in profile.iterrows():
        if spend_rank[c] == 1 and freq_rank[c] <= 2:
            names[c] = "Core regulars"
        elif row["recency_days"] == profile["recency_days"].max():
            names[c] = "Fading"
        elif row["purchase_days"] <= profile["purchase_days"].min() + 0.5:
            names[c] = "One-off buyers"
        else:
            names[c] = "Occasional buyers"
    seen = {}
    for c, n in list(names.items()):               # keep labels unique if two segments get the same name
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[c] = f"{n} ({seen[n]})"
    return names


def run_segments(cfg: dict, tx: pd.DataFrame, art_dir: Path | None = None, reports_dir: Path | None = None) -> dict:
    art = art_dir or resolve(cfg["paths"]["artifacts_dir"])
    rep = reports_dir or resolve(cfg["paths"]["reports_dir"])
    rep.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(art / "model_bundle.joblib")
    H, sn, sp = bundle["horizon_days"], cfg["snapshots"], cfg["splits"]
    snaps = make_snapshots(sn["start"], tx["date"].max(), H, sn["freq"])
    blocks, _ = assign_blocks(snaps, sp["val_start"], sp["cal_start"], sp["test_start"], H)
    data, _ = build_dataset(tx, snaps, H, bundle["activity_window_days"], sn["segment_k"])
    train = data[data["snapshot_date"].isin(blocks["train"])]
    test = data[data["snapshot_date"].isin(blocks["test"])].copy()
    model = fit_segments(train, seed=cfg["project"]["seed"])
    test["segment"] = assign(model, test)
    test["score"] = bundle["calibrator"].transform(bundle["model"].predict_proba(test[bundle["features"]].astype(float))[:, 1])
    test["spend_180"] = np.expm1(test["rev_180_log"])

    rows = []
    for c, g in test.groupby("segment"):
        auc = float(roc_auc_score(g["y_inactive"], g["score"])) if g["y_inactive"].nunique() == 2 and len(g) >= 100 else None
        rows.append({"segment": int(c), "customers": int(len(g)), "share": float(len(g) / len(test)),
                     "lapse_rate": float(g["y_inactive"].mean()), "mean_predicted": float(g["score"].mean()),
                     "recency_days": float(g["recency_days"].median()), "orders_180": float(g["inv_count_180"].median()),
                     "spend_180": float(g["spend_180"].median()), "purchase_days": float(g["n_purchase_days"].median()),
                     "interval_days": float(g["median_interval_days"].median()) if g["median_interval_days"].notna().any() else None,
                     "model_roc_auc": auc})
    profile = pd.DataFrame(rows).set_index("segment")
    names = name_segments(profile)
    profile["name"] = pd.Series(names)
    profile = profile.sort_values("lapse_rate", ascending=False)

    # 2-D PCA coordinates of a test sample, for the picture
    sample = test.sample(min(3000, len(test)), random_state=cfg["project"]["seed"])
    xy = model["pca"].transform(model["scaler"].transform(_prep(sample, model["medians"])))
    result = {
        "k": model["k"], "silhouette_by_k": model["silhouette"], "features": CLUSTER_FEATURES,
        "pca_explained_variance": [float(v) for v in model["pca"].explained_variance_ratio_],
        "fitted_on": [str(blocks["train"][0].date()), str(blocks["train"][-1].date())],
        "applied_to": [str(blocks["test"][0].date()), str(blocks["test"][-1].date())],
        "segments": [{"name": r["name"], **{k: (None if pd.isna(v) else v) for k, v in r.items() if k != "name"}, "id": int(i)} for i, r in profile.iterrows()],
        "pca_sample": {"x": xy[:, 0].round(3).tolist(), "y": xy[:, 1].round(3).tolist(), "segment": sample["segment"].astype(int).tolist()},
    }
    (rep / "segments.json").write_text(json.dumps(result, indent=1), encoding="utf8")
    profile.reset_index().to_csv(rep / "segments.csv", index=False)
    return result


def summary(res: dict) -> str:
    lines = [f"K-Means segments: k={res['k']} (silhouette " + ", ".join(f"k={k}: {v:.3f}" for k, v in res["silhouette_by_k"].items()) + ")",
             f"Fitted on {res['fitted_on'][0]} to {res['fitted_on'][1]}; profiled on {res['applied_to'][0]} to {res['applied_to'][1]}"]
    for s in res["segments"]:
        auc = "n/a" if s["model_roc_auc"] is None else f"{s['model_roc_auc']:.2f}"
        lines.append(f"  {s['name']:<18} {s['share']:>4.0%} of customers | lapse {s['lapse_rate']:.0%} | predicted {s['mean_predicted']:.0%} | "
                     f"median spend £{s['spend_180']:,.0f}, {s['orders_180']:.0f} orders, {s['recency_days']:.0f} days since order | model ROC-AUC {auc}")
    return "\n".join(lines)
