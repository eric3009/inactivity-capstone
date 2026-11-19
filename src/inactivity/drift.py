"""Drift alert: stored reference window and bin edges, a PSI check, an Evidently report, a domain slice metric and an
automated gate that returns an exit code.

    python -m inactivity drift --init              # (re)build the stored reference from the trained model
    python -m inactivity drift                     # check the out-of-time test window; exit 0 ok, 1 warning, 2 alert
    python -m inactivity drift --window latest     # check the most recent snapshots (labels may not have matured)
    python -m inactivity drift --control           # unshifted control: the gate must stay quiet

Exit codes are 0 (ok), 1 (warning) and 2 (alert), so a scheduler or CI job can fail on a non-zero code.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import resolve
from .features import FEATURES, compute_features, make_labels, prepare_index
from .splits import assign_blocks, make_snapshots

OK, WARN, ALERT = 0, 1, 2
STATUS = {OK: "OK", WARN: "WARNING", ALERT: "ALERT"}
DEFAULTS = {
    "bins": 10, "feature_psi_warn": 0.10, "feature_psi_alert": 0.25, "features_in_alert_to_fail": 3,
    "score_psi_warn": 0.10, "score_psi_alert": 0.25, "min_rows": 300,
    "slice": {"name": "one_purchase_day", "auc_drop_warn": 0.03, "auc_drop_alert": 0.06, "min_rows": 150},
}


def thresholds(cfg: dict) -> dict:
    t = {**DEFAULTS, **(cfg.get("monitoring") or {})}
    t["slice"] = {**DEFAULTS["slice"], **(t.get("slice") or {})}
    return t


# ------------------------------------------------------------------ binning and PSI with stored edges
def make_edges(values, bins: int) -> list[float]:
    v = pd.Series(values).dropna().to_numpy(float)
    if len(v) == 0:
        return []
    return [float(x) for x in np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1]))]


def bin_props(values, edges: list[float]) -> list[float]:
    """Share of rows in each bin defined by the stored interior edges, plus a final bin for missing values."""
    v = pd.Series(values).to_numpy(float)
    miss = np.isnan(v)
    idx = np.searchsorted(np.asarray(edges, float), v[~miss], side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    counts = np.append(counts, miss.sum())
    return (counts / max(1, len(v))).tolist()


def psi_from_props(ref, cur) -> float:
    r, c = np.clip(np.asarray(ref, float), 1e-4, None), np.clip(np.asarray(cur, float), 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


# ------------------------------------------------------------------ windows and scoring
def _paths(cfg: dict) -> tuple[Path, Path, Path]:
    art, rep = resolve(cfg["paths"]["artifacts_dir"]), resolve(cfg["paths"]["reports_dir"])
    return art / "drift_reference.json", art / "drift_reference.parquet", rep


def _blocks(cfg: dict, tx: pd.DataFrame, horizon: int) -> dict:
    sn, sp = cfg["snapshots"], cfg["splits"]
    snaps = make_snapshots(sn["start"], tx["date"].max(), horizon, sn["freq"])
    blocks, _ = assign_blocks(snaps, sp["val_start"], sp["cal_start"], sp["test_start"], horizon)
    return blocks


def window_frame(bundle: dict, idx, snapshots, with_labels: bool, data_max) -> pd.DataFrame:
    """Eligible customer-snapshot rows with features, calibrated score and (if matured) the label."""
    frames = []
    H = bundle["horizon_days"]
    for s in snapshots:
        f = compute_features(idx, s, bundle["activity_window_days"])
        f = f[f["eligible"]].drop(columns="eligible")
        if f.empty:
            continue
        f = f.assign(snapshot_date=pd.Timestamp(s))
        if with_labels and pd.Timestamp(s) + pd.Timedelta(days=H) <= pd.Timestamp(data_max):
            f["y_inactive"] = make_labels(idx, f.index, s, H).to_numpy()
        frames.append(f.reset_index(drop=True))
    if not frames:
        return pd.DataFrame(columns=FEATURES + ["snapshot_date"])
    df = pd.concat(frames, ignore_index=True)
    X = df[bundle["features"]].astype(float)
    df["score"] = bundle["calibrator"].transform(bundle["model"].predict_proba(X)[:, 1])
    return df


def _slice_mask(df: pd.DataFrame) -> pd.Series:
    return df["n_purchase_days"] <= 1          # customers with a single purchase day: the model's weakest group


def _slice_auc(df: pd.DataFrame, min_rows: int):
    if "y_inactive" not in df:
        return None
    d = df.dropna(subset=["y_inactive"])
    d = d[_slice_mask(d)]
    if len(d) < min_rows or d["y_inactive"].nunique() < 2:
        return None
    return float(roc_auc_score(d["y_inactive"], d["score"]))


# ------------------------------------------------------------------ stored reference
def build_reference(cfg: dict, tx: pd.DataFrame, art_dir: Path | None = None) -> Path:
    """Store the reference window, bin edges and baselines the alert compares against.

    Feature bins come from the training window (what the model learned from). The score distribution and the slice
    baseline come from the validation window, which is out of sample for the model and has matured labels.
    """
    ref_json, ref_parquet, _ = _paths(cfg)
    if art_dir is not None:
        ref_json, ref_parquet = art_dir / "drift_reference.json", art_dir / "drift_reference.parquet"
    art = ref_json.parent
    bundle = joblib.load(art / "model_bundle.joblib")
    th = thresholds(cfg)
    blocks = _blocks(cfg, tx, bundle["horizon_days"])
    idx = prepare_index(tx)
    data_max = tx["date"].max()
    train = window_frame(bundle, idx, blocks["train"], with_labels=False, data_max=data_max)
    val = window_frame(bundle, idx, blocks["val"], with_labels=True, data_max=data_max)
    feats = {f: {"edges": (e := make_edges(train[f], th["bins"])), "props": bin_props(train[f], e)} for f in bundle["features"]}
    s_edges = make_edges(val["score"], th["bins"])
    base_auc = _slice_auc(val, th["slice"]["min_rows"])
    ref = {
        "created_at": datetime.now(timezone.utc).isoformat(), "model_version": bundle["model_version"],
        "reference_window": {"features": [str(blocks["train"][0].date()), str(blocks["train"][-1].date())],
                             "scores_and_slice": [str(blocks["val"][0].date()), str(blocks["val"][-1].date())]},
        "rows": {"features": int(len(train)), "scores_and_slice": int(len(val))},
        "features": feats, "score": {"edges": s_edges, "props": bin_props(val["score"], s_edges)},
        "slice": {"name": th["slice"]["name"], "definition": "customers with a single purchase day",
                  "baseline_auc": base_auc, "baseline_share": float(_slice_mask(val).mean())},
        "thresholds": th,
    }
    ref_json.write_text(json.dumps(ref, indent=1), encoding="utf8")
    train[bundle["features"]].to_parquet(ref_parquet, index=False)
    return ref_json


# ------------------------------------------------------------------ the check
def evaluate_drift(ref: dict, cur: pd.DataFrame, th: dict) -> dict:
    """Pure function: stored reference + current window -> per-check statuses, reasons and an exit code."""
    rows = []
    for f, spec in ref["features"].items():
        p = psi_from_props(spec["props"], bin_props(cur[f], spec["edges"]))
        rows.append({"feature": f, "psi": p, "status": "alert" if p >= th["feature_psi_alert"] else "warn" if p >= th["feature_psi_warn"] else "ok"})
    rows.sort(key=lambda r: -r["psi"])
    n_alert = sum(r["status"] == "alert" for r in rows)
    n_warn = sum(r["status"] == "warn" for r in rows)
    reasons, code = [], OK

    if len(cur) < th["min_rows"]:
        return {"status": "WARNING", "exit_code": WARN, "reasons": [f"only {len(cur)} rows in the window (need {th['min_rows']}); drift not assessed"],
                "features": rows, "score_psi": None, "slice": None, "rows": int(len(cur))}

    if n_alert >= th["features_in_alert_to_fail"]:
        code = ALERT
        reasons.append(f"{n_alert} inputs in major shift (PSI >= {th['feature_psi_alert']}): " + ", ".join(r["feature"] for r in rows if r["status"] == "alert"))
    elif n_alert or n_warn:
        code = max(code, WARN)
        reasons.append(f"{n_alert} inputs in major shift and {n_warn} in moderate shift (alert needs {th['features_in_alert_to_fail']} major)")

    score_psi = None
    if "score" in cur and ref.get("score", {}).get("edges"):
        score_psi = psi_from_props(ref["score"]["props"], bin_props(cur["score"], ref["score"]["edges"]))
        if score_psi >= th["score_psi_alert"]:
            code = ALERT
            reasons.append(f"score distribution shifted (PSI {score_psi:.2f} >= {th['score_psi_alert']})")
        elif score_psi >= th["score_psi_warn"]:
            code = max(code, WARN)
            reasons.append(f"score distribution moderately shifted (PSI {score_psi:.2f})")

    sl = None
    if ref.get("slice", {}).get("baseline_auc") is not None:
        auc = _slice_auc(cur, th["slice"]["min_rows"])
        share = float(_slice_mask(cur).mean())
        sl = {"name": ref["slice"]["name"], "baseline_auc": ref["slice"]["baseline_auc"], "current_auc": auc,
              "baseline_share": ref["slice"]["baseline_share"], "current_share": share}
        if auc is None:
            sl["note"] = "labels not matured or too few rows: slice AUC not assessed"
        else:
            drop = ref["slice"]["baseline_auc"] - auc
            sl["drop"] = drop
            if drop >= th["slice"]["auc_drop_alert"]:
                code = ALERT
                reasons.append(f"{ref['slice']['name']} AUC fell {drop:.3f} (baseline {ref['slice']['baseline_auc']:.3f} to {auc:.3f})")
            elif drop >= th["slice"]["auc_drop_warn"]:
                code = max(code, WARN)
                reasons.append(f"{ref['slice']['name']} AUC fell {drop:.3f}")
    if not reasons:
        reasons.append("all inputs, scores and the slice metric are within thresholds")
    return {"status": STATUS[code], "exit_code": code, "reasons": reasons, "features": rows, "score_psi": score_psi,
            "slice": sl, "rows": int(len(cur)), "inputs_major": n_alert, "inputs_moderate": n_warn}


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, out: Path) -> str | None:
    """Evidently data-drift report as HTML. Returns the path, or None if Evidently is unavailable."""
    cols = [c for c in reference.columns if c in current.columns]
    ref, cur = reference[cols].astype(float), current[cols].astype(float)
    try:
        try:                                              # Evidently >= 0.7
            from evidently import Report
            from evidently.presets import DataDriftPreset
            snap = Report([DataDriftPreset()]).run(cur, ref)
            snap.save_html(str(out))
        except ImportError:                               # older releases
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report
            rep = Report(metrics=[DataDriftPreset()])
            rep.run(reference_data=ref, current_data=cur)
            rep.save_html(str(out))
        return "reports/" + Path(out).name            # a portable path, not an absolute one
    except Exception as e:
        print(f"[warn] Evidently report skipped: {type(e).__name__}: {str(e)[:120]}")
        return None


def run_check(cfg: dict, tx: pd.DataFrame, window: str = "test", start=None, end=None, control: bool = False,
              evidently: bool = True, art_dir: Path | None = None, reports_dir: Path | None = None) -> dict:
    ref_json, ref_parquet, rep_dir = _paths(cfg)
    if art_dir is not None:
        ref_json, ref_parquet = art_dir / "drift_reference.json", art_dir / "drift_reference.parquet"
    rep_dir = reports_dir or rep_dir
    if not ref_json.exists():
        raise FileNotFoundError(f"{ref_json} not found: run `python -m inactivity drift --init` first")
    ref = json.loads(ref_json.read_text(encoding="utf8"))
    th = thresholds(cfg)                      # read from the config at check time, so thresholds can be tuned without rebuilding
    reference = pd.read_parquet(ref_parquet)
    bundle = joblib.load(ref_json.parent / "model_bundle.joblib")

    if control:
        rng = np.random.default_rng(cfg["project"]["seed"])
        order = rng.permutation(len(reference))
        half = len(reference) // 2
        ref_half, cur_half = reference.iloc[order[:half]], reference.iloc[order[half:]]
        sub = {"features": {f: {"edges": (e := make_edges(ref_half[f], th["bins"])), "props": bin_props(ref_half[f], e)} for f in bundle["features"]}}
        result = evaluate_drift(sub, cur_half, th)
        label, ref_for_report, cur_for_report = "control: random halves of the reference window", ref_half, cur_half
    else:
        blocks = _blocks(cfg, tx, bundle["horizon_days"])
        idx = prepare_index(tx)
        snaps = make_snapshots(cfg["snapshots"]["start"], tx["date"].max(), 0, cfg["snapshots"]["freq"])
        if start or end:
            lo, hi = pd.Timestamp(start or snaps[0]), pd.Timestamp(end or snaps[-1])
            chosen = [s for s in snaps if lo <= s <= hi]
        elif window == "latest":
            chosen = [s for s in snaps if s <= tx["date"].max()][-3:]
        else:
            chosen = blocks["test"]
        cur = window_frame(bundle, idx, chosen, with_labels=True, data_max=tx["date"].max())
        result = evaluate_drift(ref, cur, th)
        label = f"{chosen[0].date()} to {chosen[-1].date()}" if chosen else "empty window"
        ref_for_report, cur_for_report = reference, cur

    result.update({"window": label, "reference_window": ref["reference_window"], "model_version": ref["model_version"],
                   "checked_at": datetime.now(timezone.utc).isoformat(), "thresholds": {k: v for k, v in th.items()}})
    rep_dir.mkdir(parents=True, exist_ok=True)
    if evidently and len(cur_for_report):
        result["evidently_report"] = evidently_report(ref_for_report, cur_for_report, rep_dir / ("drift_report_control.html" if control else "drift_report.html"))
    (rep_dir / ("drift_check_control.json" if control else "drift_check.json")).write_text(json.dumps(result, indent=1, default=float), encoding="utf8")
    return result


def summary(result: dict, top: int = 6) -> str:
    lines = [f"Drift check [{result['window']}]: {result['status']} (exit code {result['exit_code']}); {result['rows']:,} rows",
             f"Reference window: features {result['reference_window']['features'][0]} to {result['reference_window']['features'][1]}, "
             f"scores and slice {result['reference_window']['scores_and_slice'][0]} to {result['reference_window']['scores_and_slice'][1]}"]
    lines += ["  - " + r for r in result["reasons"]]
    lines.append("Largest input shifts (PSI): " + ", ".join(f"{r['feature']} {r['psi']:.2f}" for r in result["features"][:top]))
    if result.get("score_psi") is not None:
        lines.append(f"Score PSI: {result['score_psi']:.3f}")
    sl = result.get("slice")
    if sl:
        cur = "n/a" if sl["current_auc"] is None else f"{sl['current_auc']:.3f}"
        lines.append(f"Slice '{sl['name']}': AUC {cur} (baseline {sl['baseline_auc']:.3f}); share of customers {sl['current_share']:.0%} (baseline {sl['baseline_share']:.0%})")
    if result.get("evidently_report"):
        lines.append(f"Evidently report: {result['evidently_report']}")
    return "\n".join(lines)
