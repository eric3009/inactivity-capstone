"""Input drift between the training period and a later period (Population Stability Index)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected, actual, bins: int = 10) -> float:
    e = pd.Series(expected).dropna().to_numpy(float)
    a = pd.Series(actual).dropna().to_numpy(float)
    if len(e) == 0 or len(a) == 0:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def drift_report(train: pd.DataFrame, later: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = [{"feature": f, "psi": psi(train[f], later[f])} for f in features]
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    out["status"] = pd.cut(out["psi"], [-np.inf, 0.1, 0.25, np.inf], labels=["stable", "moderate shift", "major shift"])
    return out
