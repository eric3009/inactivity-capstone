import json
from pathlib import Path

import pytest

from inactivity.config import ROOT, load_config

pytest.importorskip("mlflow")


def test_log_run_writes_metrics_trials_and_artifacts(tmp_path, monkeypatch):
    if not (ROOT / "reports" / "metrics.json").exists():
        pytest.skip("run the pipeline first")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path.as_posix()}")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow
    from inactivity.tracking import log_run, tracking_uri

    assert tracking_uri().startswith("file:")
    run_id = log_run(load_config())
    assert run_id
    run = mlflow.MlflowClient(tracking_uri()).get_run(run_id)
    served = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf8"))["metrics"]["lgbm_cal"]["roc_auc"]
    assert run.data.metrics["served_roc_auc"] == pytest.approx(served)
    assert "val_roc_auc__logreg_cal" in run.data.metrics and "test_roc_auc__rule_recency" in run.data.metrics
    assert len(mlflow.MlflowClient(tracking_uri()).get_metric_history(run_id, "trial_val_logloss")) >= 1


def test_tracking_off_logs_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path.as_posix()}")
    from inactivity.tracking import log_run
    assert log_run(load_config(overrides={"tracking": {"mlflow": False}})) is None
    assert not any(Path(tmp_path).iterdir())
