"""FastAPI scoring service.

GET  /                    web UI
GET  /eda                 exploratory aggregates for the Explore page
GET  /trust               evaluation evidence for the UI trust page
GET  /worklist            ranked customers at the final test snapshot
GET  /demo/customers      real test-period examples for the UI
POST /score               snapshot features in -> calibrated probability, risk band, reason codes
POST /score/transactions  raw transaction lines in -> same, features computed by the training code
GET  /health, GET /model-info
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..explain import reason_codes
from ..features import compute_features_from_lines
from ..versions import mismatches
from .schemas import FeatureScoreRequest, ScoreResponse, TransactionScoreRequest

DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "artifacts" / "model_bundle.joblib"
STATIC = Path(__file__).resolve().parent / "static"


def _band(p: float, thresholds: dict) -> str:
    if p >= thresholds["high"]:
        return "High"
    if p >= thresholds["medium"]:
        return "Medium"
    return "Low"


def _score_frame(bundle: dict, X: pd.DataFrame, customer_id, snapshot_date, eligible: bool) -> ScoreResponse:
    base = dict(customer_id=customer_id, snapshot_date=snapshot_date, horizon_days=bundle["horizon_days"],
                model_version=bundle["model_version"])
    if not eligible:
        return ScoreResponse(eligible=False, probability_inactive=None, risk_band="Not eligible",
                             note=f"No purchase in the last {bundle['activity_window_days']} days: the customer is "
                                  "already dormant under the activity definition, so no prediction is made.", **base)
    raw = float(bundle["model"].predict_proba(X)[:, 1][0])
    p = float(bundle["calibrator"].transform(np.array([raw]))[0])
    codes = reason_codes(bundle["model"], X, 3)[0]
    return ScoreResponse(eligible=True, probability_inactive=round(p, 4), risk_band=_band(p, bundle["thresholds"]),
                         reason_codes=codes, **base)


def create_app(model_path: str | os.PathLike | None = None) -> FastAPI:
    path = Path(model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not path.exists():
            raise RuntimeError(f"Model bundle not found at {path}. Run `python -m inactivity train` first.")
        app.state.bundle = joblib.load(path)
        app.state.version_mismatch = mismatches(app.state.bundle.get("library_versions"))
        if app.state.version_mismatch:
            warnings.warn("Model was trained in a different environment: " + "; ".join(app.state.version_mismatch))
        yield

    app = FastAPI(title="Customer inactivity scoring API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ui():
        return (STATIC / "index.html").read_text(encoding="utf8")

    def _json_file(name: str):
        f = path.parent / name
        if not f.exists():
            raise HTTPException(404, f"{name} not found; run `python -m inactivity demo`.")
        return json.loads(f.read_text(encoding="utf8"))

    @app.get("/worklist")
    def worklist():
        """Every eligible customer at the final test snapshot, ranked, with spend, reasons and order history."""
        return _json_file("worklist.json")

    @app.get("/eda")
    def eda():
        """Exploratory aggregates (activity over time, distributions, feature explorer, correlations)."""
        return _json_file("eda.json")

    @app.get("/trust")
    def trust():
        """Evaluation evidence behind the 'how far can I trust it' page."""
        return _json_file("trust.json")

    @app.get("/demo/customers")
    def demo_customers():
        """Real test-period customers (features + what actually happened) for the web UI."""
        f = path.parent / "demo_customers.json"
        return json.loads(f.read_text(encoding="utf8")) if f.exists() else []

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": app.state.bundle["model_version"]}

    @app.get("/model-info")
    def model_info():
        b = app.state.bundle
        return {"model_version": b["model_version"], "trained_at": b["trained_at"], "source": b["source"],
                "horizon_days": b["horizon_days"], "activity_window_days": b["activity_window_days"],
                "features": b["features"], "risk_thresholds": b["thresholds"], "test_metrics": b["test_metrics"],
                "library_versions": b.get("library_versions"), "version_mismatch": app.state.version_mismatch}

    @app.post("/score", response_model=ScoreResponse)
    def score(req: FeatureScoreRequest):
        b = app.state.bundle
        unknown = set(req.features) - set(b["features"])
        if unknown:
            raise HTTPException(422, f"Unknown feature(s): {sorted(unknown)}")
        row = {f: req.features.get(f) for f in b["features"]}
        X = pd.DataFrame([row], columns=b["features"]).astype(float)
        recency = row.get("recency_days")
        eligible = recency is not None and recency < b["activity_window_days"]
        return _score_frame(b, X, req.customer_id, req.snapshot_date, bool(eligible))

    @app.post("/score/transactions", response_model=ScoreResponse)
    def score_transactions(req: TransactionScoreRequest):
        b = app.state.bundle
        lines = pd.DataFrame([t.model_dump() for t in req.transactions])
        lines["customer_id"] = req.customer_id or "api"
        feats = compute_features_from_lines(lines, req.snapshot_date, b["activity_window_days"])
        if feats.empty:
            raise HTTPException(422, "No purchases on or before the snapshot date.")
        row = feats.iloc[[0]]
        X = row[b["features"]].astype(float)
        return _score_frame(b, X, req.customer_id, req.snapshot_date, bool(row["eligible"].iloc[0]))

    app.mount("/ui", StaticFiles(directory=STATIC), name="ui")
    return app


app = create_app()
