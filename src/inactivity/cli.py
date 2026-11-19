"""Command line: `python -m inactivity <prepare|train|all|predict|serve|demo|track|drift|segments|stats|select|anomalies> [--synthetic]`."""
from __future__ import annotations

import argparse
import json
import warnings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="inactivity", description=__doc__)
    ap.add_argument("command", choices=["prepare", "train", "all", "predict", "serve", "demo", "track", "drift", "segments", "stats", "select", "anomalies"])
    ap.add_argument("--config", default=None, help="path to a YAML config (default configs/config.yaml)")
    ap.add_argument("--synthetic", action="store_true",
                    help="use generated data instead of downloading Online Retail II (offline smoke run)")
    ap.add_argument("--no-sensitivity", action="store_true", help="skip the horizon/label sensitivity runs")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--date", default=None, help="predict: scoring date, YYYY-MM-DD (default: last date in the data)")
    ap.add_argument("--out", default=None, help="predict: output CSV (default reports/predictions_<date>.csv)")
    ap.add_argument("--include-ineligible", action="store_true", help="predict: also list customers who are not scored")
    ap.add_argument("--init", action="store_true", help="drift: (re)build the stored reference window and bin edges")
    ap.add_argument("--window", default="test", choices=["test", "latest"], help="drift: window to check (default: the out-of-time test period)")
    ap.add_argument("--from", dest="date_from", default=None, help="drift: first snapshot date, YYYY-MM-DD")
    ap.add_argument("--to", dest="date_to", default=None, help="drift: last snapshot date, YYYY-MM-DD")
    ap.add_argument("--control", action="store_true", help="drift: unshifted control (random halves of the reference); the gate must stay quiet")
    ap.add_argument("--no-evidently", action="store_true", help="drift: skip the Evidently HTML report")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    from .config import load_config
    overrides = {"tracking": {"mlflow": False}} if a.no_mlflow else None
    cfg = load_config(a.config, overrides)

    if a.command == "serve":
        import uvicorn
        uvicorn.run("inactivity.service.app:app", host=a.host, port=a.port)
        return 0

    if a.command == "track":
        from .tracking import log_run, tracking_uri
        rid = log_run(cfg)
        print(f"Logged run {rid} to {tracking_uri()}" if rid else "Tracking is switched off in the config.")
        return 0

    from .data import prepare
    from .pipeline import run
    if a.command == "stats":
        from .hypothesis import run_stats, summary as stats_summary
        print(stats_summary(run_stats(cfg, prepare(cfg, synthetic=a.synthetic))))
        return 0
    if a.command == "select":
        from .selection import run_selection, summary as select_summary
        print(select_summary(run_selection(cfg, prepare(cfg, synthetic=a.synthetic))))
        return 0
    if a.command == "anomalies":
        from .anomaly import run_anomalies, summary as anomaly_summary
        print(anomaly_summary(run_anomalies(cfg, prepare(cfg, synthetic=a.synthetic))))
        return 0
    if a.command == "segments":
        from .segments import run_segments, summary as seg_summary
        print(seg_summary(run_segments(cfg, prepare(cfg, synthetic=a.synthetic))))
        return 0
    if a.command == "drift":
        from .drift import build_reference, run_check, summary
        tx = prepare(cfg, synthetic=a.synthetic)
        if a.init:
            print(f"Wrote {build_reference(cfg, tx)}")
            return 0
        result = run_check(cfg, tx, window=a.window, start=a.date_from, end=a.date_to, control=a.control, evidently=not a.no_evidently)
        print(summary(result))
        return int(result["exit_code"])
    if a.command == "predict":
        from .predict import score_snapshot
        frame, path = score_snapshot(cfg, a.date, a.out, a.include_ineligible)
        print(f"Scored {int(frame['eligible'].sum()):,} customers at {frame['snapshot_date'].iloc[0] if len(frame) else a.date}; wrote {path}")
        return 0
    if a.command == "demo":
        from .uidata import build_ui_data
        out = build_ui_data(cfg, prepare(cfg, synthetic=a.synthetic))
        print(f"Wrote {out['trust']}, {out['worklist']} ({out['n_worklist']} customers), {out['eda']} and {out['demo']}")
        return 0
    if a.command in ("prepare", "all"):
        tx = prepare(cfg, synthetic=a.synthetic, force=True)
        print(f"Prepared {len(tx):,} clean lines for {tx['customer_id'].nunique():,} customers "
              f"({tx['date'].min().date()} to {tx['date'].max().date()})")
    if a.command in ("train", "all"):
        out = run(cfg, synthetic=a.synthetic, sensitivity=not a.no_sensitivity)
        m = out["main"]["metrics"]["lgbm_cal"]
        print(json.dumps({"model_version": out["bundle"]["model_version"],
                          "test_lgbm_cal": {k: round(v, 4) for k, v in m.items()}}, indent=2))
        print("Artifacts: artifacts/model_bundle.joblib | Reports: reports/results.md")
    return 0
