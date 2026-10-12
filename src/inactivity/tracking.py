"""Experiment tracking with MLflow.

Logs a finished run (parameters, metrics, the hyper-parameter search curve, reports and the model bundle) from the
files the pipeline already wrote, so it can run at the end of training or on its own afterwards:

    python -m inactivity track

Where it goes is set by the standard MLflow environment variables. With nothing set it writes to ./mlruns. For a
hosted server such as DagsHub:

    MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
    MLFLOW_TRACKING_USERNAME=<dagshub user name>
    MLFLOW_TRACKING_PASSWORD=<dagshub access token>      (keep this out of the repository)
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from .config import ROOT, resolve


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or ("file:" + str(ROOT / "mlruns"))


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v)


def log_run(cfg: dict, reports_dir=None, artifacts_dir=None) -> str | None:
    """Log the run described by reports/metrics.json. Returns the MLflow run id, or None if tracking is off."""
    if not cfg["tracking"]["mlflow"]:
        return None
    reports = Path(reports_dir) if reports_dir else resolve(cfg["paths"]["reports_dir"])
    arts = Path(artifacts_dir) if artifacts_dir else resolve(cfg["paths"]["artifacts_dir"])
    m = json.loads((reports / "metrics.json").read_text(encoding="utf8"))

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")   # plain local store when no server is configured
    import mlflow
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(cfg["tracking"]["experiment"])

    with mlflow.start_run(run_name=f"model {m['version']}") as run:
        mlflow.log_params({
            "horizon_days": m["horizon"], "model_version": m["version"], "seed": cfg["project"]["seed"],
            "lgbm_trials": cfg["modelling"]["lgbm_trials"], "bootstrap_reps": cfg["modelling"]["bootstrap_reps"],
            "activity_window_days": cfg["snapshots"]["activity_window_days"],
            "train_snapshots": len(m["splits"]["blocks"]["train"]), "test_snapshots": len(m["splits"]["blocks"]["test"]),
            "dropped_by_embargo": len(m["splits"]["dropped_by_embargo"]),
        })
        metrics = {}
        for model, vals in m["metrics"].items():                       # test period, every model
            for k in ("roc_auc", "pr_auc", "lift_top", "brier", "cal_slope", "ece"):
                if _num(vals.get(k)):
                    metrics[f"test_{k}__{model}"] = float(vals[k])
        for model, vals in (m.get("val_comparison") or {}).items():    # validation period, every model
            for k in ("roc_auc", "pr_auc", "lift_top"):
                if _num(vals.get(k)):
                    metrics[f"val_{k}__{model}"] = float(vals[k])
        for k, v in m["metrics"]["lgbm_cal"].items():                   # headline numbers for the served model
            if _num(v):
                metrics[f"served_{k}"] = float(v)
        if m.get("customer_holdout"):
            ch = m["customer_holdout"]
            for k in ("roc_auc", "pr_auc"):
                if _num(ch.get(k)):
                    metrics[f"customer_holdout_{k}"] = float(ch[k])
        mlflow.log_metrics(metrics)

        search = reports / "lgbm_search.csv"                            # the tuning curve, one point per trial
        if search.exists():
            import csv
            with search.open(encoding="utf8") as f:
                for i, row in enumerate(csv.DictReader(f)):
                    mlflow.log_metric("trial_val_logloss", float(row["val_logloss"]), step=i)
        mlflow.set_tags({"stage": "evaluation", "data_source": "uci-online-retail-ii", "served_model": "lightgbm_calibrated"})

        mlflow.log_artifacts(str(reports), artifact_path="reports")
        bundle = arts / "model_bundle.joblib"
        if bundle.exists():
            mlflow.log_artifact(str(bundle), artifact_path="model")
        return run.info.run_id
