"""Shared helper for the analysis commands: customer-snapshot rows split into the pipeline's four periods."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .config import resolve
from .features import build_dataset
from .splits import BLOCKS, assign_blocks, make_snapshots


def load_blocks(cfg: dict, tx: pd.DataFrame, art_dir: Path | None = None):
    """Return (bundle, parts, blocks): the trained bundle, a frame per period, and the snapshot dates per period."""
    art = art_dir or resolve(cfg["paths"]["artifacts_dir"])
    bundle = joblib.load(art / "model_bundle.joblib")
    H, sn, sp = bundle["horizon_days"], cfg["snapshots"], cfg["splits"]
    snaps = make_snapshots(sn["start"], tx["date"].max(), H, sn["freq"])
    blocks, _ = assign_blocks(snaps, sp["val_start"], sp["cal_start"], sp["test_start"], H)
    data, _ = build_dataset(tx, snaps, H, bundle["activity_window_days"], sn["segment_k"])
    parts = {b: data[data["snapshot_date"].isin(blocks[b])].reset_index(drop=True) for b in BLOCKS}
    return bundle, parts, blocks
