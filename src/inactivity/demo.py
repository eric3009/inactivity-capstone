"""Build the demo-customer file used by the web UI (real customers from the out-of-time test period).

The Online Retail II customer IDs are already public, anonymous numbers. For each sampled customer the file stores the
snapshot features and the outcome that actually happened, so the UI can show a prediction next to the truth.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import resolve
from .features import FEATURES, build_dataset
from .splits import assign_blocks, make_snapshots


def _band(p: float, thresholds: dict) -> str:
    return "High" if p >= thresholds["high"] else "Medium" if p >= thresholds["medium"] else "Low"


def write_demo(cfg: dict, tx: pd.DataFrame, bundle_path: Path | None = None, out_path: Path | None = None,
               per_band: int = 8) -> Path:
    art = resolve(cfg["paths"]["artifacts_dir"])
    bundle_path = bundle_path or art / "model_bundle.joblib"
    out_path = out_path or art / "demo_customers.json"
    bundle = joblib.load(bundle_path)
    sn, sp = cfg["snapshots"], cfg["splits"]
    H = bundle["horizon_days"]
    snaps = make_snapshots(sn["start"], tx["date"].max(), H, sn["freq"])
    blocks, _ = assign_blocks(snaps, sp["val_start"], sp["cal_start"], sp["test_start"], H)
    snapshot = max(blocks["test"])
    data, _ = build_dataset(tx, [snapshot], H, bundle["activity_window_days"], sn["segment_k"])

    raw = bundle["model"].predict_proba(data[FEATURES])[:, 1]
    data["p"] = bundle["calibrator"].transform(raw)
    data["band"] = [_band(p, bundle["thresholds"]) for p in data["p"]]
    rng = np.random.default_rng(cfg["project"]["seed"])
    picks = []
    for band in ("High", "Medium", "Low"):
        pool = data[data["band"] == band]
        take = min(per_band, len(pool))
        picks.append(pool.iloc[rng.choice(len(pool), take, replace=False)])
    chosen = pd.concat(picks)

    items = []
    for _, r in chosen.iterrows():
        items.append({
            "customer_id": str(r["customer_id"]),
            "snapshot_date": str(pd.Timestamp(r["snapshot_date"]).date()),
            "features": {f: (None if pd.isna(r[f]) else float(r[f])) for f in FEATURES},
            "actual_inactive": int(r["y_inactive"]),
            "horizon_days": H,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(items, indent=1), encoding="utf8")
    return out_path
