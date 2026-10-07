import pandas as pd
import pytest
from fastapi.testclient import TestClient

from inactivity.features import FEATURES, compute_features, prepare_index
from inactivity.service.app import create_app

SNAP = "2011-03-31"


@pytest.fixture(scope="module")
def client(bundle_path):
    with TestClient(create_app(bundle_path)) as c:
        yield c


def _active_customer(tx):
    feats = compute_features(prepare_index(tx), SNAP)
    feats = feats[feats["eligible"]]
    cid = feats.index[len(feats) // 2]
    return cid, feats.loc[cid, FEATURES]


def test_health_and_model_info(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    info = client.get("/model-info").json()
    assert info["features"] == FEATURES and info["horizon_days"] == 90
    assert 0 < info["risk_thresholds"]["medium"] < info["risk_thresholds"]["high"] <= 1


def test_score_from_features(client, tx):
    cid, feats = _active_customer(tx)
    body = {"customer_id": cid, "snapshot_date": SNAP, "features": {k: (None if pd.isna(v) else float(v)) for k, v in feats.items()}}
    r = client.post("/score", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["eligible"] and 0.0 <= out["probability_inactive"] <= 1.0
    assert out["risk_band"] in {"Low", "Medium", "High"}
    assert len(out["reason_codes"]) <= 3
    assert all(rc["contribution"] > 0 for rc in out["reason_codes"])


def test_dormant_customer_is_not_scored(client):
    r = client.post("/score", json={"snapshot_date": SNAP, "features": {"recency_days": 400.0}})
    out = r.json()
    assert r.status_code == 200
    assert out["eligible"] is False and out["probability_inactive"] is None and out["risk_band"] == "Not eligible"


def test_unknown_feature_is_rejected(client):
    r = client.post("/score", json={"snapshot_date": SNAP, "features": {"not_a_feature": 1.0}})
    assert r.status_code == 422


def test_score_from_transactions_matches_feature_path(client, tx):
    cid, feats = _active_customer(tx)
    lines = tx[(tx["customer_id"] == cid) & (tx["date"] <= SNAP)]
    payload = [{"invoice": r.invoice, "stock_code": r.stock_code, "invoice_date": r.invoice_date.isoformat(),
                "quantity": float(r.quantity), "price": float(r.price), "country": r.country}
               for r in lines.itertuples()]
    a = client.post("/score/transactions", json={"customer_id": cid, "snapshot_date": SNAP, "transactions": payload}).json()
    b = client.post("/score", json={"customer_id": cid, "snapshot_date": SNAP,
                                    "features": {k: (None if pd.isna(v) else float(v)) for k, v in feats.items()}}).json()
    assert a["probability_inactive"] == pytest.approx(b["probability_inactive"], abs=1e-9)
    assert a["risk_band"] == b["risk_band"]


def test_transactions_after_snapshot_only_is_rejected(client):
    payload = [{"invoice": "1", "invoice_date": "2012-01-01T10:00:00", "quantity": 1, "price": 1.0}]
    r = client.post("/score/transactions", json={"snapshot_date": SNAP, "transactions": payload})
    assert r.status_code == 422


def test_ui_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Customer inactivity risk" in r.text and "/ui/js/app.js" in r.text
    for path in ("/ui/style.css", "/ui/js/app.js", "/ui/js/charts.js", "/ui/js/common.js"):
        assert client.get(path).status_code == 200, path
    assert "image/svg" in client.get("/ui/favicon.svg").headers["content-type"]
    for page in ("overview", "explore", "model", "worklist", "customer", "try", "about"):
        assert client.get(f"/ui/js/pages/{page}.js").status_code == 200, page


def test_eda_endpoint_has_what_the_explore_page_draws(client):
    e = client.get("/eda").json()
    for key in ("quality", "monthly", "countries", "distributions", "lorenz", "snapshots", "features", "correlation"):
        assert key in e, key
    feats = set(client.get("/model-info").json()["features"])
    assert set(e["features"]) == feats
    for f in e["features"].values():
        n = len(f["bins"])
        assert all(len(s["n0"]) == n and len(s["n1"]) == n for s in f["sets"].values())
    assert len(e["correlation"]["matrix"]) == len(feats)
    assert e["lorenz"]["y"][-1] == 1.0 or abs(e["lorenz"]["y"][-1] - 1) < 1e-9


def test_demo_customers_are_scorable_and_match_feature_list(client):
    demo = client.get("/demo/customers").json()
    assert len(demo) > 0
    feats = set(client.get("/model-info").json()["features"])
    for c in demo[:5]:
        assert set(c["features"]) == feats and c["actual_inactive"] in (0, 1)
        r = client.post("/score", json={"customer_id": c["customer_id"], "snapshot_date": c["snapshot_date"],
                                        "features": c["features"]})
        assert r.status_code == 200 and r.json()["eligible"]


def test_trust_endpoint_has_the_evidence_the_ui_needs(client):
    t = client.get("/trust").json()
    for key in ("metrics", "ci", "gains", "reliability", "splits", "sizes", "per_snapshot", "drift", "importance", "quality"):
        assert key in t
    g = t["gains"]
    assert g["x"][0] == 0 and g["x"][-1] == 1 and g["lgbm_cal"][-1] == pytest.approx(1.0)
    assert all(a <= b + 1e-12 for a, b in zip(g["lgbm_cal"], g["lgbm_cal"][1:]))      # a gains curve never falls
    assert len(t["reliability"]["lgbm_cal"]) == 10
    assert {"tree_depth4", "random_forest", "mlp_challenger"} <= set(t["metrics"])            # the course model ladder
    assert all(r["auc_drop"] is not None for r in t["importance"])                           # permutation importance
    assert t["segments"] and all(0 <= r["roc_auc"] <= 1 for r in t["segments"])


def test_worklist_is_ranked_and_has_what_the_ui_shows(client):
    w = client.get("/worklist").json()
    cs = w["customers"]
    assert len(cs) > 0 and w["snapshot_date"]
    probs = [c["p"] for c in cs]
    assert probs == sorted(probs, reverse=True)
    c = cs[0]
    for key in ("customer_id", "p", "band", "spend_180", "orders_180", "recency_days", "reasons", "past_orders", "future_orders", "inactive"):
        assert key in c
    assert all(c["band"] in {"High", "Medium", "Low"} for c in cs)
    assert all(o[0] <= w["snapshot_date"] for c in cs[:50] for o in c["past_orders"])       # history stops at the scoring date
    assert all(o[0] > w["snapshot_date"] for c in cs[:50] for o in c["future_orders"])      # outcome orders come after it


def test_model_info_reports_training_environment(client):
    info = client.get("/model-info").json()
    assert {"python", "scikit-learn", "lightgbm", "pandas", "numpy"} <= set(info["library_versions"])
    assert info["version_mismatch"] == []                            # trained and served in the same environment


def test_version_mismatch_helper_flags_differences():
    from inactivity.versions import current_versions, mismatches
    cur = current_versions()
    assert mismatches(cur) == []
    old = {**cur, "scikit-learn": "0.0.1"}
    assert any("scikit-learn" in d for d in mismatches(old)) and mismatches(None)


def test_worklist_rows_carry_features_for_the_what_if_page(client):
    w = client.get("/worklist").json()
    feats = set(client.get("/model-info").json()["features"])
    for c in w["customers"][:20]:
        assert set(c["features"]) == feats
        r = client.post("/score", json={"customer_id": c["customer_id"], "snapshot_date": w["snapshot_date"], "features": c["features"]}).json()
        assert abs(r["probability_inactive"] - c["p"]) < 2e-3
