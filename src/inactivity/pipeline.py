"""End-to-end pipeline: ingest -> snapshots -> train/tune -> calibrate -> evaluate -> artifacts."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from . import evaluate as ev
from .config import ROOT, resolve
from .data import prepare
from .explain import reason_codes
from .features import FEATURES, build_dataset, compute_features
from sklearn.metrics import log_loss

from .models import (Calibrator, bgnbd_scores, choose_calibrator, fit_lgbm, fit_logreg, fit_mlp, fit_rf,
                     fit_tree, rule_score)
from .monitoring import drift_report
from .splits import BLOCKS, assign_blocks, make_snapshots, summarise, validate_embargo
from .versions import current_versions


def _part(data: pd.DataFrame, blocks: dict, name: str) -> pd.DataFrame:
    return data[data["snapshot_date"].isin(blocks[name])].reset_index(drop=True)


def run_experiment(tx: pd.DataFrame, cfg: dict, horizon: int, label: str = "y_inactive",
                   trials: int | None = None, reps: int | None = None, ladder: bool = False,
                   holdout: bool = False) -> dict:
    sn, sp, md = cfg["snapshots"], cfg["splits"], cfg["modelling"]
    seed = cfg["project"]["seed"]
    trials = md["lgbm_trials"] if trials is None else trials
    reps = md["bootstrap_reps"] if reps is None else reps

    snaps = make_snapshots(sn["start"], tx["date"].max(), horizon, sn["freq"])
    data, idx = build_dataset(tx, snaps, horizon, sn["activity_window_days"], sn["segment_k"])
    blocks, dropped = assign_blocks(snaps, sp["val_start"], sp["cal_start"], sp["test_start"], horizon)
    validate_embargo(blocks, horizon)
    parts = {b: _part(data, blocks, b) for b in BLOCKS}
    for b, df in parts.items():
        if df.empty or df[label].nunique() < 2:
            raise ValueError(f"Block '{b}' is empty or has a single class; adjust snapshots/splits.")

    Xtr, ytr = parts["train"][FEATURES], parts["train"][label].to_numpy()
    Xva, yva = parts["val"][FEATURES], parts["val"][label].to_numpy()
    Xca, yca = parts["cal"][FEATURES], parts["cal"][label].to_numpy()
    Xte, yte = parts["test"][FEATURES], parts["test"][label].to_numpy()

    logreg, lr_table = fit_logreg(Xtr, ytr, Xva, yva, md["logreg_C"], seed)
    lgbm, lgbm_table = fit_lgbm(Xtr, ytr, Xva, yva, trials, md["early_stopping_rounds"], seed)
    ladder_models = {}
    if ladder:     # the rest of the course model ladder: tree -> forest -> neural-network challenger
        ladder_models = {"tree_depth4": fit_tree(Xtr, ytr, Xva, yva, seed),
                         "random_forest": fit_rf(Xtr, ytr, Xva, yva, seed),
                         "mlp_challenger": fit_mlp(Xtr, ytr, Xva, yva, seed)}

    raw = {"logreg": logreg.predict_proba(Xte)[:, 1], "lgbm": lgbm.predict_proba(Xte)[:, 1]}
    cal_raw = {"logreg": logreg.predict_proba(Xca)[:, 1], "lgbm": lgbm.predict_proba(Xca)[:, 1]}
    calibrators, cal_info = {}, {}
    for k in raw:
        calibrators[k], cal_info[k] = choose_calibrator(cal_raw[k], yca, parts["cal"]["customer_id"], seed)
    scores = {
        "prevalence": np.full(len(yte), ytr.mean()),
        "rule_recency": rule_score(Xte),
        "logreg_raw": raw["logreg"], "logreg_cal": calibrators["logreg"].transform(raw["logreg"]),
        "lgbm_raw": raw["lgbm"], "lgbm_cal": calibrators["lgbm"].transform(raw["lgbm"]),
    }
    fit_snapshot = max(blocks["cal"])          # history ends before the test period; no labels are used
    bg = bgnbd_scores(compute_features(idx, fit_snapshot, sn["activity_window_days"]), {"test": parts["test"]}, horizon)
    if bg is not None:
        scores["bgnbd"] = bg["test"]
    for name, mdl in ladder_models.items():
        scores[name] = mdl.predict_proba(Xte)[:, 1]
    prob_models = ("prevalence", "logreg_raw", "logreg_cal", "lgbm_raw", "lgbm_cal")

    share = md["top_share"]
    metrics = {}
    for name, s in scores.items():
        m = ev.full_metrics(yte, s, calibrated=name in prob_models, share=share)
        if name == "prevalence":
            m.update(roc_auc=0.5, pr_auc=float(np.mean(yte)), lift_top=1.0, cal_slope=float('nan'), cal_intercept=float('nan'))
        metrics[name] = m

    cis = {}
    if reps:
        boot_scores = {k: v for k, v in scores.items() if k != "prevalence"}
        pairs = [("lgbm_cal", "logreg_cal"), ("lgbm_cal", "rule_recency")]
        if "bgnbd" in boot_scores:
            pairs.append(("lgbm_cal", "bgnbd"))
        for name in ladder_models:
            pairs.append(("lgbm_cal", name))
        boot = ev.cluster_bootstrap(yte, parts["test"]["customer_id"], boot_scores, reps, seed, share,
                                    prob_models=prob_models, pairs=tuple(pairs))
        cis = {f"{m}|{k}": ev.ci(v) for (m, k), v in boot.items()}

    per_snap = []
    for s in blocks["test"]:
        mask = (parts["test"]["snapshot_date"] == s).to_numpy()
        if len(np.unique(yte[mask])) == 2:
            per_snap.append({"snapshot": str(pd.Timestamp(s).date()), "n": int(mask.sum()),
                             **ev.full_metrics(yte[mask], scores["lgbm_cal"][mask], True, share)})

    pv = lgbm.predict_proba(Xva)[:, 1]
    val_metrics = {**ev.discrimination(yva, pv, share), "logloss": float(log_loss(yva, pv, labels=[0, 1])),
                   "test_logloss_raw": float(log_loss(yte, raw["lgbm"], labels=[0, 1]))}
    # Every model scored on the validation period too, so the choice of served model rests on data that is not the test period.
    val_scores = {"rule_recency": rule_score(Xva), "logreg": logreg.predict_proba(Xva)[:, 1], "lgbm": pv}
    for name, mdl in ladder_models.items():
        val_scores[name] = mdl.predict_proba(Xva)[:, 1]
    bg_val = bgnbd_scores(compute_features(idx, max(blocks["train"]), sn["activity_window_days"]), {"val": parts["val"]}, horizon)
    if bg_val is not None:
        val_scores["bgnbd"] = bg_val["val"]
    val_comparison = {("lgbm_cal" if k == "lgbm" else "logreg_cal" if k == "logreg" else k): ev.discrimination(yva, v, share)
                      for k, v in val_scores.items()}
    customer_holdout = None
    if holdout:
        # Group leakage check: retrain without any customer who appears in the test period, then test on them.
        seen = set(parts["test"]["customer_id"])
        tr_h = parts["train"][~parts["train"]["customer_id"].isin(seen)]
        va_h = parts["val"][~parts["val"]["customer_id"].isin(seen)]
        if len(tr_h) > 200 and tr_h[label].nunique() == 2 and va_h[label].nunique() == 2:
            m_h, _ = fit_lgbm(tr_h[FEATURES], tr_h[label].to_numpy(), va_h[FEATURES], va_h[label].to_numpy(),
                              max(20, trials // 3), md["early_stopping_rounds"], seed)
            ph = m_h.predict_proba(Xte)[:, 1]
            customer_holdout = {"train_rows": int(len(tr_h)), "val_rows": int(len(va_h)),
                                **ev.discrimination(yte, ph, share), "brier_raw": ev.brier(yte, ph),
                                "main_raw": ev.discrimination(yte, raw["lgbm"], share)}

    return {
        "horizon": horizon, "label": label, "val_metrics": val_metrics, "val_comparison": val_comparison, "customer_holdout": customer_holdout,
        "splits": summarise(blocks, dropped),
        "sizes": {b: {"rows": int(len(df)), "customers": int(df["customer_id"].nunique()),
                      "prevalence": float(df[label].mean())} for b, df in parts.items()},
        "metrics": metrics, "ci": cis, "per_snapshot": per_snap,
        "calibration": cal_info, "logreg_search": lr_table, "lgbm_search": lgbm_table,
        "_objects": {"lgbm": lgbm, "logreg": logreg, "ladder": ladder_models, "calibrators": calibrators, "parts": parts,
                     "scores": scores, "y_test": yte, "blocks": blocks},
    }


def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def results_markdown(main: dict, sens: dict, quality: dict) -> str:
    m, ci = main["metrics"], main["ci"]
    lines = ["# Results (out-of-time test period)\n",
             f"Horizon: {main['horizon']} days | label: `{main['label']}`\n",
             "## Split sizes\n", "| Block | Snapshots | Rows | Customers | Inactivity rate |", "|---|---|---|---|---|"]
    for b in BLOCKS:
        sz = main["sizes"][b]
        lines.append(f"| {b} | {len(main['splits']['blocks'][b])} | {sz['rows']} | {sz['customers']} | {sz['prevalence']:.3f} |")
    lines.append(f"\nSnapshots dropped by the label-maturity embargo: {', '.join(main['splits']['dropped_by_embargo']) or 'none'}\n")
    lines += ["## Model comparison\n",
              "| Model | ROC-AUC [95% CI] | PR-AUC [95% CI] | Top-decile lift | Brier | Cal. slope | Cal. intercept | ECE |",
              "|---|---|---|---|---|---|---|---|"]
    def cell(model, key):
        v = m[model].get(key)
        if f"{model}|{key}" in ci:
            lo, hi = ci[f"{model}|{key}"]
            return f"{_fmt(v)} [{lo:.3f}, {hi:.3f}]"
        return _fmt(v)
    for name in m:
        r = m[name]
        lines.append(f"| {name} | {cell(name, 'roc_auc')} | {cell(name, 'pr_auc')} | {cell(name, 'lift_top')} | "
                     f"{_fmt(r.get('brier'))} | {_fmt(r.get('cal_slope'))} | {_fmt(r.get('cal_intercept'))} | {_fmt(r.get('ece'))} |")
    vc = main.get("val_comparison")
    if vc:
        lines += ["\n## Model selection on the validation period (before the test period was used)\n",
                  "| Model | Validation ROC-AUC | Validation PR-AUC | Validation top-decile lift |", "|---|---|---|---|"]
        for name, r_ in sorted(vc.items(), key=lambda kv: -kv[1]["roc_auc"]):
            lines.append(f"| {name} | {_fmt(r_['roc_auc'])} | {_fmt(r_['pr_auc'])} | {_fmt(r_['lift_top'])} |")
    vm = main.get("val_metrics")
    if vm:
        lines += ["\n## Honesty checks\n",
                  "- **How the test period was used:** its dates were fixed in advance and no model, hyper-parameter, calibrator or "
                  "threshold was chosen on it. Its results were, however, visible during development, and some models (Optuna "
                  "tuning, the neural-network challenger) were added after earlier runs, so treat it as a clean test of the chosen "
                  "design rather than a single-use hold-out.",
                  f"- **Optimism gap (selection set vs test period):** LightGBM validation ROC-AUC {_fmt(vm['roc_auc'])} vs test "
                  f"{_fmt(m['lgbm_raw']['roc_auc'])}; validation PR-AUC {_fmt(vm['pr_auc'])} vs test {_fmt(m['lgbm_raw']['pr_auc'])}."]
        ch = main.get("customer_holdout")
        if ch:
            lines.append(f"- **Customer-held-out retrain (group-leakage check):** with every test-period customer removed from "
                         f"training and validation ({ch['train_rows']:,} training rows left), test ROC-AUC {_fmt(ch['roc_auc'])} "
                         f"and PR-AUC {_fmt(ch['pr_auc'])}, against {_fmt(ch['main_raw']['roc_auc'])} and {_fmt(ch['main_raw']['pr_auc'])} for the served model.")
    pair_keys = sorted({k.split("|")[0] for k in ci if "-" in k.split("|")[0]})
    if pair_keys:
        lines += ["\n## Paired differences (95% customer-clustered bootstrap CI)\n", "| Comparison | ROC-AUC diff | PR-AUC diff | Lift diff |", "|---|---|---|---|"]
        for p in pair_keys:
            cells = []
            for k in ("roc_auc", "pr_auc", "lift_top"):
                lo, hi = ci[f"{p}|{k}"]
                cells.append(f"[{lo:+.3f}, {hi:+.3f}]")
            lines.append(f"| {p} | " + " | ".join(cells) + " |")
    lines += ["\n## Sensitivity (LightGBM, calibrated; 20 search trials, no bootstrap)\n",
              "| Variant | Test rows | Inactivity rate | ROC-AUC | PR-AUC | Brier |", "|---|---|---|---|---|---|"]
    for name, s in sens.items():
        r = s["metrics"]["lgbm_cal"]
        lines.append(f"| {name} | {s['sizes']['test']['rows']} | {s['sizes']['test']['prevalence']:.3f} | "
                     f"{_fmt(r['roc_auc'])} | {_fmt(r['pr_auc'])} | {_fmt(r.get('brier'))} |")
    lines += ["\nSensitivity rows use different label definitions, so their metrics are not directly "
              "comparable with each other as a ranking of models. The segment-aware label is defined partly from "
              "recency and the customer's median order interval, which are also model inputs, so its high AUC is "
              "partly circular and must not be read as better predictive skill.\n",
              "## Data quality\n", "```json", json.dumps(quality, indent=2), "```"]
    return "\n".join(lines)


def _plots(main: dict, reports_dir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    obj = main["_objects"]
    y = obj["y_test"]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    for name, c in (("lgbm_raw", "#bbb"), ("lgbm_cal", "#1f77b4"), ("logreg_cal", "#ff7f0e")):
        t = ev.reliability_table(y, obj["scores"][name])
        ax.plot(t["mean_predicted"], t["observed_rate"], marker="o", color=c, label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed inactivity rate")
    ax.set_title("Reliability (out-of-time test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "reliability.png", dpi=150)
    plt.close(fig)
    try:
        import shap
        X = obj["parts"]["test"][FEATURES]
        sv = shap.TreeExplainer(obj["lgbm"]).shap_values(X)
        sv = sv[1] if isinstance(sv, list) else sv
        plt.figure()
        shap.summary_plot(sv, X, show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(reports_dir / "shap_summary.png", dpi=150)
        plt.close()
    except Exception as e:  # shap is optional for the pipeline
        print(f"[warn] SHAP summary skipped: {e}")


def _track(cfg: dict, reports_dir, art_dir) -> None:
    try:
        from .tracking import log_run
        log_run(cfg, reports_dir, art_dir)
    except Exception as e:
        print(f"[warn] MLflow tracking skipped: {e}")


def run(cfg: dict, synthetic: bool = False, sensitivity: bool = True) -> dict:
    reports = resolve(cfg["paths"]["reports_dir"])
    art = resolve(cfg["paths"]["artifacts_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    art.mkdir(parents=True, exist_ok=True)
    tx = prepare(cfg, synthetic=synthetic)
    quality_path = reports / "data_quality.json"
    quality = json.load(open(quality_path, encoding="utf8")) if quality_path.exists() else {}

    H = cfg["snapshots"]["horizon_days"]
    main = run_experiment(tx, cfg, H, ladder=True, holdout=True)
    sens = {}
    if sensitivity:
        for h in sorted(set(cfg["sensitivity"]["horizons"]) | {H}):
            if h == H:
                sens[f"horizon {h}d (fixed label)"] = main
                continue
            try:
                sens[f"horizon {h}d (fixed label)"] = run_experiment(tx, cfg, h, trials=20, reps=0)
            except ValueError as e:
                print(f"[warn] horizon {h} skipped: {e}")
        sens[f"horizon {H}d (segment-aware label)"] = run_experiment(tx, cfg, H, "y_inactive_seg", trials=20, reps=0)

    obj = main["_objects"]
    parts = obj["parts"]
    lgbm, calib = obj["lgbm"], obj["calibrators"]["lgbm"]
    cal_p = calib.transform(lgbm.predict_proba(parts["cal"][FEATURES])[:, 1])
    bands = cfg["modelling"]["bands"]
    thresholds = {"high": float(np.quantile(cal_p, 1 - bands["high_top_share"])),
                  "medium": float(np.quantile(cal_p, 1 - bands["medium_top_share"]))}
    version = datetime.now(timezone.utc).strftime("%Y%m%d") + "-" + hashlib.sha1(
        (json.dumps(cfg, sort_keys=True, default=str) + str(len(tx)) + str(tx["date"].max())).encode()).hexdigest()[:8]
    bundle = {
        "model": lgbm, "calibrator": calib, "features": FEATURES, "thresholds": thresholds,
        "horizon_days": H, "activity_window_days": cfg["snapshots"]["activity_window_days"],
        "model_version": version, "trained_at": datetime.now(timezone.utc).isoformat(), "library_versions": current_versions(),
        "test_metrics": {k: main["metrics"][k] for k in ("lgbm_cal", "logreg_cal", "rule_recency")},
        "source": "synthetic" if synthetic else "uci-online-retail-ii",
    }
    joblib.dump(bundle, art / "model_bundle.joblib")

    # reports
    summary = {"horizon": H, "version": version, "splits": main["splits"], "sizes": main["sizes"],
               "metrics": main["metrics"], "ci": main["ci"], "per_snapshot": main["per_snapshot"],
               "calibration": main["calibration"], "thresholds": thresholds,
               "val_metrics": main["val_metrics"], "val_comparison": main["val_comparison"], "customer_holdout": main["customer_holdout"],
               "sensitivity": {k: {"sizes": v["sizes"]["test"], "metrics": v["metrics"]["lgbm_cal"]} for k, v in sens.items()}}
    with open(reports / "metrics.json", "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2, default=float)
    pd.DataFrame(main["metrics"]).T.to_csv(reports / "model_comparison.csv")
    main["lgbm_search"].to_csv(reports / "lgbm_search.csv", index=False)
    pd.DataFrame({"feature": FEATURES, "gain": lgbm.booster_.feature_importance("gain")}).sort_values(
        "gain", ascending=False).to_csv(reports / "feature_importance.csv", index=False)
    drift_report(parts["train"], parts["test"], FEATURES).to_csv(reports / "drift_psi.csv", index=False)
    test = parts["test"]
    top = test.assign(p=obj["scores"]["lgbm_cal"]).sort_values("p", ascending=False).head(15)
    rc = reason_codes(lgbm, top[FEATURES], 3)
    pd.DataFrame({"customer_id": top["customer_id"].values, "snapshot": top["snapshot_date"].dt.date.values,
                  "p_inactive": top["p"].round(3).values, "actual_inactive": top[main["label"]].values,
                  "reasons": [" | ".join(r["text"] for r in x) for x in rc]}).to_csv(reports / "sample_scores.csv", index=False)
    (reports / "results.md").write_text(results_markdown(main, sens, quality), encoding="utf8")
    _plots(main, reports)
    _track(cfg, reports, art)
    from .uidata import build_ui_data
    build_ui_data(cfg, tx, art, reports)
    return {"main": main, "sensitivity": sens, "bundle": bundle}
