import joblib
import numpy as np
import pandas as pd
import pytest

from inactivity.config import load_config
from inactivity.predict import predict_frame, score_snapshot

SNAP = "2011-03-31"


@pytest.fixture(scope="module")
def bundle(bundle_path):
    return joblib.load(bundle_path)


def test_predict_frame_is_ranked_and_complete(bundle, tx):
    df = predict_frame(bundle, tx, SNAP)
    assert len(df) > 0 and df["eligible"].all()
    assert df["probability_inactive"].between(0, 1).all()
    assert df["probability_inactive"].is_monotonic_decreasing                      # ranked, highest risk first
    assert set(df["risk_band"]) <= {"High", "Medium", "Low"}
    assert df[["reason_1", "reason_2", "reason_3"]].notna().all().all()               # reasons are plain text, never null
    assert df["reason_1"].str.len().gt(0).mean() > 0.9                                # nearly every scored customer has a top reason
    assert df["customer_id"].is_unique and (df["horizon_days"] == bundle["horizon_days"]).all()


def test_ineligible_customers_are_listed_only_on_request(bundle, tx):
    only = predict_frame(bundle, tx, SNAP)
    both = predict_frame(bundle, tx, SNAP, include_ineligible=True)
    extra = both[~both["eligible"]]
    assert len(both) > len(only) and (extra["risk_band"] == "Not eligible").all() and extra["probability_inactive"].isna().all()
    assert both["eligible"].sum() == len(only)


def test_batch_predictions_match_the_api_path(bundle, bundle_path, tx):
    from fastapi.testclient import TestClient
    from inactivity.features import FEATURES, compute_features, prepare_index
    from inactivity.service.app import create_app
    df = predict_frame(bundle, tx, SNAP)
    cid = df["customer_id"].iloc[len(df) // 2]
    feats = compute_features(prepare_index(tx), SNAP).loc[cid, FEATURES]
    body = {"snapshot_date": SNAP, "features": {k: (None if pd.isna(v) else float(v)) for k, v in feats.items()}}
    with TestClient(create_app(bundle_path)) as c:
        api = c.post("/score", json=body).json()
    row = df[df["customer_id"] == cid].iloc[0]
    assert api["probability_inactive"] == pytest.approx(row["probability_inactive"], abs=1e-4)
    assert api["risk_band"] == row["risk_band"]


def test_score_snapshot_writes_a_csv(bundle_path, tx, tmp_path):
    out = tmp_path / "pred.csv"
    frame, path = score_snapshot(load_config(), SNAP, out, tx=tx, bundle_path=bundle_path)
    back = pd.read_csv(path, dtype={"customer_id": str})
    assert path == out and len(back) == len(frame) > 0
    assert list(back.columns[:5]) == ["customer_id", "snapshot_date", "eligible", "probability_inactive", "risk_band"]
