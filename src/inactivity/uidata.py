"""Data files behind the web UI: the retention worklist and the 'how far can I trust it' page.

Everything is computed from the trained bundle, the cached clean transactions and the reports the pipeline already wrote,
so it can be regenerated without retraining (`python -m inactivity demo`).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import evaluate as ev
from .config import resolve
from .demo import write_demo
from .edadata import build_eda
from .explain import reason_codes
from .features import FEATURES, build_dataset, compute_features, prepare_index
from .models import bgnbd_scores, rule_score
from .splits import assign_blocks, make_snapshots


def clean_json(obj):
    """Recursively replace NaN/inf by None so the files are valid JSON for browsers and FastAPI."""
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _gains(y: np.ndarray, score: np.ndarray, grid: np.ndarray) -> list[float]:
    """Share of all inactive customers reached when contacting the top `grid` share by score."""
    order = np.argsort(-score, kind="stable")
    cum = np.cumsum(y[order]) / max(1, y.sum())
    n = len(y)
    return [float(cum[max(0, math.ceil(g * n) - 1)]) if g > 0 else 0.0 for g in grid]


def _band(p: float, th: dict) -> str:
    return "High" if p >= th["high"] else "Medium" if p >= th["medium"] else "Low"


def build_ui_data(cfg: dict, tx: pd.DataFrame, art_dir: Path | None = None, reports_dir: Path | None = None) -> dict:
    art = art_dir or resolve(cfg["paths"]["artifacts_dir"])
    rep = reports_dir or resolve(cfg["paths"]["reports_dir"])
    bundle = joblib.load(art / "model_bundle.joblib")
    H, aw = bundle["horizon_days"], bundle["activity_window_days"]
    sn, sp = cfg["snapshots"], cfg["splits"]
    snaps = make_snapshots(sn["start"], tx["date"].max(), H, sn["freq"])
    blocks, dropped = assign_blocks(snaps, sp["val_start"], sp["cal_start"], sp["test_start"], H)

    data, idx = build_dataset(tx, blocks["test"], H, aw, sn["segment_k"])
    X = data[FEATURES]
    raw = bundle["model"].predict_proba(X)[:, 1]
    cal = bundle["calibrator"].transform(raw)
    y = data["y_inactive"].to_numpy()
    th = bundle["thresholds"]

    # ---- gains (catch-rate) curves over the whole test block
    grid = np.linspace(0, 1, 101)
    gains = {"x": grid.tolist(), "lgbm_cal": _gains(y, cal, grid),
             "rule_recency": _gains(y, rule_score(X), grid)}
    fit_frame = compute_features(idx, max(blocks["cal"]), aw)
    bg = bgnbd_scores(fit_frame, {"t": data}, H)
    if bg is not None:
        gains["bgnbd"] = _gains(y, bg["t"], grid)

    # ---- reliability
    def rel(p):
        t = ev.reliability_table(y, p, 10)
        return [{"predicted": float(a), "observed": float(b), "n": int(n)}
                for a, b, n in zip(t["mean_predicted"], t["observed_rate"], t["n"])]

    metrics = json.loads((rep / "metrics.json").read_text(encoding="utf8"))
    quality = json.loads((rep / "data_quality.json").read_text(encoding="utf8"))
    drift = pd.read_csv(rep / "drift_psi.csv").to_dict("records")
    # Permutation importance on the held-out test period (gain-based importance favours many-valued numeric inputs)
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(bundle["model"], X, y, scoring="roc_auc", n_repeats=5, random_state=cfg["project"]["seed"], n_jobs=1)
    perm = pd.DataFrame({"feature": FEATURES, "auc_drop": pi.importances_mean, "auc_drop_std": pi.importances_std})         .sort_values("auc_drop", ascending=False).reset_index(drop=True)
    perm.to_csv(rep / "permutation_importance.csv", index=False)
    importance = perm.head(8).to_dict("records")

    # Does it work equally well for different kinds of customers? (test period, served calibrated score)
    seg_defs = {
        "Country": pd.Series(np.where(data["is_uk"] == 1, "UK", "Outside UK"), index=data.index),
        "Spend tier (365 days)": pd.qcut(data["rev_365_log"], 3, labels=["Lower third", "Middle third", "Upper third"]).astype(str),
        "Order history": pd.cut(data["n_purchase_days"], [0, 1, 4, np.inf], labels=["One purchase day", "2-4 purchase days", "5+ purchase days"]).astype(str),
    }
    segments = []
    for dim, labels_ in seg_defs.items():
        for g in sorted(labels_.unique()):
            mk = (labels_ == g).to_numpy()
            if mk.sum() < 100 or len(np.unique(y[mk])) < 2:
                continue
            segments.append({"dimension": dim, "group": g, "n": int(mk.sum()), "inactivity_rate": float(y[mk].mean()),
                             **ev.discrimination(y[mk], cal[mk], 0.1), "mean_predicted": float(cal[mk].mean())})
    trust = clean_json({
        "model_version": bundle["model_version"], "source": bundle["source"], "horizon_days": H,
        "activity_window_days": aw, "thresholds": th, "metrics": metrics["metrics"], "ci": metrics["ci"],
        "per_snapshot": metrics["per_snapshot"], "sizes": metrics["sizes"], "splits": metrics["splits"],
        "calibration": metrics["calibration"], "sensitivity": metrics["sensitivity"],
        "gains": gains, "reliability": {"lgbm_raw": rel(raw), "lgbm_cal": rel(cal)},
        "drift": drift, "importance": importance, "segments": segments, "quality": quality,
        "val_metrics": metrics.get("val_metrics"), "val_comparison": metrics.get("val_comparison"), "customer_holdout": metrics.get("customer_holdout"),
        "test_rows": int(len(y)), "test_prevalence": float(y.mean()),
    })
    (art / "trust.json").write_text(json.dumps(trust), encoding="utf8")

    # ---- worklist: every eligible customer at the final test snapshot
    last = max(blocks["test"])
    m = (data["snapshot_date"] == last).to_numpy()
    wl = data[m].reset_index(drop=True)
    wl_p = cal[m]
    codes = reason_codes(bundle["model"], wl[FEATURES], 3)
    inv = idx.invoices
    past = inv[(inv["date"] <= last) & (inv["date"] > last - pd.Timedelta(days=365))]
    fut = inv[(inv["date"] > last) & (inv["date"] <= last + pd.Timedelta(days=H))]

    def by_cust(df):
        out = {}
        for c, d, v in zip(df["customer_id"], df["date"], df["revenue"]):
            out.setdefault(c, []).append([str(d.date()), round(float(v), 2)])
        return out

    past_orders, fut_orders = by_cust(past), by_cust(fut)
    rows = []
    for i, r in wl.iterrows():
        cid = str(r["customer_id"])
        rows.append({
            "customer_id": cid,
            "p": round(float(wl_p[i]), 4),
            "band": _band(float(wl_p[i]), th),
            "spend_180": round(float(np.expm1(r["rev_180_log"])), 2),
            "orders_180": int(r["inv_count_180"]),
            "recency_days": int(r["recency_days"]),
            "reasons": [c["text"].replace(" (raises inactivity risk)", "") for c in codes[i]],
            "past_orders": past_orders.get(cid, []),
            "future_orders": fut_orders.get(cid, []),
            "inactive": int(r["y_inactive"]),
            "features": {f: (None if pd.isna(r[f]) else round(float(r[f]), 4)) for f in FEATURES},
        })
    rows.sort(key=lambda d: -d["p"])
    worklist = {"snapshot_date": str(last.date()), "horizon_days": H, "currency": "GBP",
                "thresholds": th, "customers": rows}
    (art / "worklist.json").write_text(json.dumps(clean_json(worklist)), encoding="utf8")
    # ---- exploratory data analysis over every snapshot (the modelling periods plus the dropped months)
    data_all, _ = build_dataset(tx, snaps, H, aw, sn["segment_k"])
    blocks_map = {pd.Timestamp(d): b for b, ds in blocks.items() for d in ds}
    blocks_map.update({pd.Timestamp(d): "dropped" for d in dropped})
    eda = build_eda(tx, data_all, blocks_map, quality)
    (art / "eda.json").write_text(json.dumps(clean_json(eda)), encoding="utf8")
    demo = write_demo(cfg, tx, art / "model_bundle.joblib", art / "demo_customers.json")
    return {"trust": art / "trust.json", "worklist": art / "worklist.json", "eda": art / "eda.json", "demo": demo, "n_worklist": len(rows)}
