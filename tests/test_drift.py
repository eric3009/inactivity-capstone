import json

import numpy as np
import pandas as pd
import pytest

from inactivity.config import load_config
from inactivity.drift import (ALERT, OK, WARN, bin_props, evaluate_drift, make_edges, psi_from_props, run_check,
                              thresholds)

FEATS = ["recency_days", "tenure_days", "n_purchase_days", "inv_count_90"]


def _frame(n=2000, seed=0, shift=None):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "recency_days": rng.gamma(2, 20, n), "tenure_days": rng.gamma(3, 90, n),
        "n_purchase_days": rng.integers(1, 30, n).astype(float), "inv_count_90": rng.poisson(2, n).astype(float)})
    for k, v in (shift or {}).items():
        df[k] = df[k] + v
    df["score"] = rng.beta(2, 3, n)
    df["y_inactive"] = (rng.random(n) < df["score"]).astype(float)
    return df


def _ref(df, th):
    feats = {f: {"edges": (e := make_edges(df[f], th["bins"])), "props": bin_props(df[f], e)} for f in FEATS}
    e = make_edges(df["score"], th["bins"])
    from inactivity.drift import _slice_auc
    return {"features": feats, "score": {"edges": e, "props": bin_props(df["score"], e)},
            "slice": {"name": "one_purchase_day", "baseline_auc": _slice_auc(df, 20) or 0.7, "baseline_share": 0.1}}


@pytest.fixture
def th():
    t = thresholds({})
    t["slice"] = {**t["slice"], "min_rows": 20}
    return t


def test_psi_is_zero_for_identical_and_large_for_shifted():
    x = np.random.default_rng(1).normal(size=5000)
    edges = make_edges(x, 10)
    ref = bin_props(x, edges)
    assert psi_from_props(ref, ref) == 0
    assert psi_from_props(ref, bin_props(x + 2.0, edges)) > 0.25
    assert abs(sum(ref) - 1) < 1e-9                                   # bins plus the missing-value bin cover every row


def test_missing_values_get_their_own_bin():
    x = np.array([1.0, 2.0, 3.0, np.nan, np.nan, 5.0])
    p = bin_props(x, [2.5])
    assert p[-1] == pytest.approx(2 / 6) and abs(sum(p) - 1) < 1e-9


def test_unshifted_window_passes_the_gate(th):
    ref_df = _frame(seed=0)
    r = evaluate_drift(_ref(ref_df, th), _frame(seed=1), th)
    assert r["exit_code"] == OK and r["status"] == "OK"


def test_shifted_inputs_trigger_an_alert(th):
    ref_df = _frame(seed=0)
    cur = _frame(seed=1, shift={"recency_days": 80, "tenure_days": 400, "n_purchase_days": 15})
    r = evaluate_drift(_ref(ref_df, th), cur, th)
    assert r["exit_code"] == ALERT and r["inputs_major"] >= 3
    assert any("major shift" in x for x in r["reasons"])


def test_a_single_moderate_shift_is_only_a_warning(th):
    ref_df = _frame(seed=0)
    ref = _ref(ref_df, th)
    for shift in np.linspace(1, 60, 60):
        cur = _frame(seed=1, shift={"recency_days": shift})
        p = psi_from_props(ref["features"]["recency_days"]["props"], bin_props(cur["recency_days"], ref["features"]["recency_days"]["edges"]))
        if th["feature_psi_warn"] <= p < th["feature_psi_alert"]:
            r = evaluate_drift(ref, cur, th)
            assert r["exit_code"] == WARN
            return
    pytest.fail("could not construct a moderate shift")


def test_score_distribution_shift_is_an_alert(th):
    ref_df = _frame(seed=0)
    cur = _frame(seed=1)
    cur["score"] = np.clip(cur["score"] + 0.45, 0, 1)
    assert evaluate_drift(_ref(ref_df, th), cur, th)["exit_code"] == ALERT


def test_slice_metric_drop_is_an_alert(th):
    ref_df = _frame(seed=0)
    ref = _ref(ref_df, th)
    ref["slice"]["baseline_auc"] = 0.80
    cur = _frame(seed=1)
    cur["n_purchase_days"] = 1.0                                       # everyone is in the slice; scores are uninformative (AUC ~ 0.5)
    cur["y_inactive"] = np.random.default_rng(5).integers(0, 2, len(cur)).astype(float)
    r = evaluate_drift(ref, cur, th)
    assert r["slice"]["drop"] > th["slice"]["auc_drop_alert"] and r["exit_code"] == ALERT


def test_tiny_window_is_flagged_not_assessed(th):
    r = evaluate_drift(_ref(_frame(seed=0), th), _frame(n=50, seed=1), th)
    assert r["exit_code"] == WARN and "not assessed" in r["reasons"][0]


def test_run_check_end_to_end_on_the_trained_bundle(bundle_path, tx, tmp_path):
    art = bundle_path.parent
    assert (art / "drift_reference.json").exists() and (art / "drift_reference.parquet").exists()
    ref = json.loads((art / "drift_reference.json").read_text())
    assert set(ref["features"]) and ref["reference_window"]["features"] and ref["slice"]["name"] == "one_purchase_day"
    cfg = load_config(overrides={"paths": {"artifacts_dir": str(art), "reports_dir": str(tmp_path)},
                                  "monitoring": {"min_rows": 50, "bins": 4}})   # tiny synthetic windows: fewer bins keep sampling noise below the thresholds
    control = run_check(cfg, tx, control=True, evidently=False, art_dir=art, reports_dir=tmp_path)
    assert control["exit_code"] == OK, control["reasons"]                                   # the gate is quiet on unshifted data
    window = run_check(cfg, tx, window="test", evidently=False, art_dir=art, reports_dir=tmp_path)
    assert window["exit_code"] in (OK, WARN, ALERT) and window["features"] and (tmp_path / "drift_check.json").exists()


def test_check_without_a_reference_fails_loudly(tmp_path, tx):
    cfg = load_config(overrides={"paths": {"artifacts_dir": str(tmp_path), "reports_dir": str(tmp_path)}})
    with pytest.raises(FileNotFoundError, match="drift --init"):
        run_check(cfg, tx, evidently=False, art_dir=tmp_path, reports_dir=tmp_path)
