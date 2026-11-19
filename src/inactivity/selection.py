"""Feature selection compared three ways (filter, wrapper, embedded): does using fewer inputs change the result?

Every method chooses its inputs on the training period only. A fixed LightGBM (the served model's hyperparameters) is then
refitted on those inputs and scored on the validation period; the test period is reported for completeness but is not used to
choose anything.

  filter    mutual information between each input and the label
  wrapper   recursive feature elimination with logistic regression
  embedded  L1-penalised logistic regression (largest coefficients) and LightGBM split gain

    python -m inactivity select
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .blocks import load_blocks
from .config import resolve

KS = (5, 8, 12)


def _prep(train: pd.DataFrame, features: list[str]):
    imp = SimpleImputer(strategy="median").fit(train[features])
    sc = StandardScaler().fit(imp.transform(train[features]))
    return imp, sc


def rank_filter(Xs: np.ndarray, y: np.ndarray, features: list[str], seed: int) -> list[str]:
    mi = mutual_info_classif(Xs, y, random_state=seed)
    return [features[i] for i in np.argsort(-mi)]


def rank_wrapper(Xs: np.ndarray, y: np.ndarray, features: list[str]) -> list[str]:
    rfe = RFE(LogisticRegression(max_iter=2000), n_features_to_select=1, step=1).fit(Xs, y)
    return [features[i] for i in np.argsort(rfe.ranking_)]


def rank_l1(Xs: np.ndarray, y: np.ndarray, features: list[str]) -> list[str]:
    m = LogisticRegression(penalty="l1", solver="liblinear", C=0.05, max_iter=2000).fit(Xs, y)
    return [features[i] for i in np.argsort(-np.abs(m.coef_[0]))]


def rank_lgbm(model_params: dict, train: pd.DataFrame, y: np.ndarray, features: list[str]) -> list[str]:
    m = lgb.LGBMClassifier(**model_params).fit(train[features], y)
    return [features[i] for i in np.argsort(-m.booster_.feature_importance("gain"))]


def _fit_score(params: dict, cols: list[str], tr, va, te) -> tuple[float, float]:
    m = lgb.LGBMClassifier(**params).fit(tr[cols], tr["y_inactive"])
    return (float(roc_auc_score(va["y_inactive"], m.predict_proba(va[cols])[:, 1])),
            float(roc_auc_score(te["y_inactive"], m.predict_proba(te[cols])[:, 1])))


def run_selection(cfg: dict, tx: pd.DataFrame, art_dir: Path | None = None, reports_dir: Path | None = None, ks=KS) -> dict:
    rep = reports_dir or resolve(cfg["paths"]["reports_dir"])
    rep.mkdir(parents=True, exist_ok=True)
    bundle, parts, _ = load_blocks(cfg, tx, art_dir)
    features = list(bundle["features"])
    tr, va, te = parts["train"], parts["val"], parts["test"]
    seed = cfg["project"]["seed"]
    served = bundle["model"]
    params = {k: v for k, v in served.get_params().items() if k in ("learning_rate", "num_leaves", "min_child_samples", "subsample",
                                                                      "colsample_bytree", "reg_lambda", "subsample_freq")}
    params.update(n_estimators=int(getattr(served, "best_iteration_", None) or 200), random_state=seed, verbose=-1, n_jobs=2)
    imp, sc = _prep(tr, features)
    Xs = sc.transform(imp.transform(tr[features]))
    y = tr["y_inactive"].to_numpy()
    rankings = {
        "Filter: mutual information": rank_filter(Xs, y, features, seed),
        "Wrapper: RFE with logistic regression": rank_wrapper(Xs, y, features),
        "Embedded: L1 logistic regression": rank_l1(Xs, y, features),
        "Embedded: LightGBM split gain": rank_lgbm(params, tr, y, features),
    }
    rows = []
    full_val, full_test = _fit_score(params, features, tr, va, te)
    rows.append({"method": "All inputs (served set)", "k": len(features), "val_auc": full_val, "test_auc": full_test, "features": ", ".join(features)})
    for name, order in rankings.items():
        for k in ks:
            cols = order[:k]
            v, t = _fit_score(params, cols, tr, va, te)
            rows.append({"method": name, "k": k, "val_auc": v, "test_auc": t, "features": ", ".join(cols)})
    table = pd.DataFrame(rows)
    table["val_vs_all"] = table["val_auc"] - full_val
    # how often each input is in the top-k lists
    votes = {f: int(sum(f in order[:max(ks)] for order in rankings.values())) for f in features}
    result = {"fixed_model": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in params.items()},
              "rows": table.to_dict("records"), "votes_top_k": dict(sorted(votes.items(), key=lambda kv: -kv[1])), "k_values": list(ks),
              "rankings": rankings}
    (rep / "feature_selection.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf8")
    table.to_csv(rep / "feature_selection.csv", index=False)
    return result


def summary(res: dict) -> str:
    lines = ["Feature selection (fixed LightGBM; inputs chosen on the training period; validation AUC is the selection criterion):"]
    for r in res["rows"]:
        lines.append(f"  {r['method']:<40} k={r['k']:<2} validation {r['val_auc']:.3f} ({r['val_vs_all']:+.3f})  test {r['test_auc']:.3f}")
    top = [f for f, v in res["votes_top_k"].items() if v == 4]
    lines.append(f"Inputs in the top {max(res['k_values'])} of all four methods: " + (", ".join(top) if top else "none"))
    return "\n".join(lines)
