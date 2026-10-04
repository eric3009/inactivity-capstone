"""Command line: `python -m inactivity <prepare|train|all|predict|serve|demo> [--synthetic]`."""
from __future__ import annotations

import argparse
import json
import warnings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="inactivity", description=__doc__)
    ap.add_argument("command", choices=["prepare", "train", "all", "predict", "serve", "demo"])
    ap.add_argument("--config", default=None, help="path to a YAML config (default configs/config.yaml)")
    ap.add_argument("--synthetic", action="store_true",
                    help="use generated data instead of downloading Online Retail II (offline smoke run)")
    ap.add_argument("--no-sensitivity", action="store_true", help="skip the horizon/label sensitivity runs")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--date", default=None, help="predict: scoring date, YYYY-MM-DD (default: last date in the data)")
    ap.add_argument("--out", default=None, help="predict: output CSV (default reports/predictions_<date>.csv)")
    ap.add_argument("--include-ineligible", action="store_true", help="predict: also list customers who are not scored")
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

    from .data import prepare
    from .pipeline import run
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
