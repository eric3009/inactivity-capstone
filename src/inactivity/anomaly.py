"""Anomaly detection: which customers behave unlike the customers the model was built on, and does the model cope with them?

An Isolation Forest is fitted on the training period (standardised behavioural inputs, no labels) and applied to the test period.
Flagged customers are profiled (they are mostly unusually large accounts) and compared with the rest on lapse rate and on how well
the served model predicts them. The share flagged is also a row-level signal of drift.

    python -m inactivity anomalies
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .blocks import load_blocks
from .config import resolve
from .segments import CLUSTER_FEATURES


def fit_forest(train: pd.DataFrame, contamination: float = 0.02, seed: int = 42):
    imp = SimpleImputer(strategy="median").fit(train[CLUSTER_FEATURES])
    sc = StandardScaler().fit(imp.transform(train[CLUSTER_FEATURES]))
    forest = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed, n_jobs=2).fit(sc.transform(imp.transform(train[CLUSTER_FEATURES])))
    return imp, sc, forest


def score(model, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    imp, sc, forest = model
    Z = sc.transform(imp.transform(df[CLUSTER_FEATURES]))
    return -forest.score_samples(Z), forest.predict(Z) == -1            # higher = more unusual


def run_anomalies(cfg: dict, tx: pd.DataFrame, art_dir: Path | None = None, reports_dir: Path | None = None, contamination: float = 0.02) -> dict:
    rep = reports_dir or resolve(cfg["paths"]["reports_dir"])
    rep.mkdir(parents=True, exist_ok=True)
    bundle, parts, blocks = load_blocks(cfg, tx, art_dir)
    tr, te = parts["train"], parts["test"].copy()
    model = fit_forest(tr, contamination, cfg["project"]["seed"])
    _, tr_flag = score(model, tr)
    te["anomaly_score"], te["flagged"] = score(model, te)
    te["p"] = bundle["calibrator"].transform(bundle["model"].predict_proba(te[bundle["features"]].astype(float))[:, 1])
    te["spend_180"] = np.expm1(te["rev_180_log"])

    def block(g: pd.DataFrame) -> dict:
        auc = float(roc_auc_score(g["y_inactive"], g["p"])) if g["y_inactive"].nunique() == 2 and len(g) >= 50 else None
        return {"rows": int(len(g)), "lapse_rate": float(g["y_inactive"].mean()), "mean_predicted": float(g["p"].mean()), "model_roc_auc": auc,
                "median_spend_180": float(g["spend_180"].median()), "median_orders_180": float(g["inv_count_180"].median()),
                "median_recency_days": float(g["recency_days"].median())}

    flagged, rest = te[te["flagged"]], te[~te["flagged"]]
    top = flagged.sort_values("anomaly_score", ascending=False).head(15)
    result = {"contamination": contamination, "train_flagged_share": float(tr_flag.mean()), "test_flagged_share": float(te["flagged"].mean()),
              "features": CLUSTER_FEATURES, "flagged": block(flagged), "rest": block(rest),
              "top": [{"customer_id": str(r.customer_id), "snapshot": str(r.snapshot_date.date()), "anomaly_score": float(r.anomaly_score),
                       "spend_180": float(r.spend_180), "orders_180": int(r.inv_count_180), "recency_days": int(r.recency_days),
                       "predicted": float(r.p), "went_quiet": int(r.y_inactive)} for r in top.itertuples()]}
    (rep / "anomalies.json").write_text(json.dumps(result, indent=1), encoding="utf8")
    top.drop(columns=["p"]).assign(predicted=top["p"]).to_csv(rep / "anomalies_top.csv", index=False)
    return result


def summary(res: dict) -> str:
    f, r = res["flagged"], res["rest"]
    auc = lambda b: "n/a" if b["model_roc_auc"] is None else f"{b['model_roc_auc']:.2f}"
    return "\n".join([
        f"Isolation Forest (contamination {res['contamination']:.0%}, fitted on the training period): {res['train_flagged_share']:.1%} of training rows "
        f"flagged, {res['test_flagged_share']:.1%} of test rows flagged",
        f"  flagged : {f['rows']:,} rows | lapse {f['lapse_rate']:.0%} | predicted {f['mean_predicted']:.0%} | model ROC-AUC {auc(f)} | median spend £{f['median_spend_180']:,.0f}, "
        f"{f['median_orders_180']:.0f} orders",
        f"  the rest: {r['rows']:,} rows | lapse {r['lapse_rate']:.0%} | predicted {r['mean_predicted']:.0%} | model ROC-AUC {auc(r)} | median spend £{r['median_spend_180']:,.0f}, "
        f"{r['median_orders_180']:.0f} orders"])
