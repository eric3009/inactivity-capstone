"""Batch prediction: score every customer at a chosen date with the trained bundle (no retraining).

    python -m inactivity predict --date 2011-08-31 --out predictions.csv
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import resolve
from .explain import reason_codes
from .features import compute_features, prepare_index

SUFFIX = " (raises inactivity risk)"


def _band(p: float, thresholds: dict) -> str:
    return "High" if p >= thresholds["high"] else "Medium" if p >= thresholds["medium"] else "Low"


def predict_frame(bundle: dict, tx: pd.DataFrame, snapshot, include_ineligible: bool = False) -> pd.DataFrame:
    """One row per customer, ranked by predicted risk. Customers with no purchase in the activity window are not scored."""
    snap = pd.Timestamp(snapshot).normalize()
    feats = compute_features(prepare_index(tx), snap, bundle["activity_window_days"])
    cols = ["customer_id", "snapshot_date", "eligible", "probability_inactive", "risk_band",
            "reason_1", "reason_2", "reason_3", "horizon_days", "model_version"]
    if feats.empty:
        return pd.DataFrame(columns=cols)
    elig = feats[feats["eligible"]]
    rows = []
    if len(elig):
        X = elig[bundle["features"]].astype(float)
        p = bundle["calibrator"].transform(bundle["model"].predict_proba(X)[:, 1])
        reasons = reason_codes(bundle["model"], X, 3)
        for cid, prob, rs in zip(elig.index, p, reasons):
            texts = [r["text"].replace(SUFFIX, "") for r in rs] + [""] * 3
            rows.append({"customer_id": str(cid), "snapshot_date": str(snap.date()), "eligible": True,
                         "probability_inactive": round(float(prob), 4), "risk_band": _band(float(prob), bundle["thresholds"]),
                         "reason_1": texts[0], "reason_2": texts[1], "reason_3": texts[2],
                         "horizon_days": bundle["horizon_days"], "model_version": bundle["model_version"]})
    if include_ineligible:
        for cid in feats.index[~feats["eligible"]]:
            rows.append({"customer_id": str(cid), "snapshot_date": str(snap.date()), "eligible": False,
                         "probability_inactive": np.nan, "risk_band": "Not eligible", "reason_1": "", "reason_2": "", "reason_3": "",
                         "horizon_days": bundle["horizon_days"], "model_version": bundle["model_version"]})
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("probability_inactive", ascending=False, na_position="last").reset_index(drop=True)


def score_snapshot(cfg: dict, snapshot=None, out: str | Path | None = None, include_ineligible: bool = False,
                   tx: pd.DataFrame | None = None, bundle_path: str | Path | None = None) -> tuple[pd.DataFrame, Path]:
    """Load the trained bundle and the cleaned transactions, score a date, and write a CSV. Returns (frame, path)."""
    if tx is None:
        from .data import prepare
        tx = prepare(cfg)
    bundle = joblib.load(bundle_path or resolve(cfg["paths"]["artifacts_dir"]) / "model_bundle.joblib")
    snap = pd.Timestamp(snapshot) if snapshot else tx["date"].max()
    frame = predict_frame(bundle, tx, snap, include_ineligible)
    path = Path(out) if out else resolve(cfg["paths"]["reports_dir"]) / f"predictions_{snap.date()}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame, path
