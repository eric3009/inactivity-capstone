"""Reason codes from TreeSHAP contributions (LightGBM's built-in `pred_contrib`).

The text is direction-neutral on purpose: the feature's own SHAP sign decides whether it is
reported, so a reason code can never contradict the model. Only features that raise the
predicted inactivity risk are returned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = {
    "recency_days": ("Days since last purchase", "{:.0f}"),
    "tenure_days": ("Days since first purchase", "{:.0f}"),
    "n_purchase_days": ("Number of purchase days", "{:.0f}"),
    "inv_count_30": ("Orders in last 30 days", "{:.0f}"),
    "inv_count_90": ("Orders in last 90 days", "{:.0f}"),
    "inv_count_180": ("Orders in last 180 days", "{:.0f}"),
    "inv_count_365": ("Orders in last 365 days", "{:.0f}"),
    "inv_count_90_prev": ("Orders in the previous 90-day period", "{:.0f}"),
    "freq_trend": ("Change in order count vs previous 90 days", "{:+.0f}"),
    "rev_90_log": ("Spend in last 90 days", "£{:,.0f}"),
    "rev_180_log": ("Spend in last 180 days", "£{:,.0f}"),
    "rev_365_log": ("Spend in last 365 days", "£{:,.0f}"),
    "rev_trend": ("Spend vs previous 90 days (ratio)", "x{:.2f}"),
    "avg_invoice_value_180_log": ("Average order value", "£{:,.0f}"),
    "avg_items_per_invoice_180": ("Average items per order", "{:.0f}"),
    "n_products_180": ("Distinct products in last 180 days", "{:.0f}"),
    "median_interval_days": ("Customer's usual days between orders", "{:.0f}"),
    "interval_cv": ("Irregularity of order timing", "{:.2f}"),
    "recency_to_interval": ("Time since last order relative to usual interval", "{:.1f}x"),
    "return_rate_180": ("Share of value returned (180 days)", "{:.0%}"),
    "is_uk": ("UK-based customer", "{:.0f}"),
}


# Features stored on a log scale are shown in business units.
_TO_DISPLAY = {"rev_90_log": np.expm1, "rev_180_log": np.expm1, "rev_365_log": np.expm1,
               "avg_invoice_value_180_log": np.expm1, "rev_trend": np.exp}


def contributions(model, X: pd.DataFrame) -> np.ndarray:
    """Per-feature TreeSHAP values in log-odds space (bias column removed)."""
    contrib = model.booster_.predict(X, pred_contrib=True)
    return np.asarray(contrib)[:, :-1]


def reason_codes(model, X: pd.DataFrame, top_k: int = 3) -> list[list[dict]]:
    """Top-k risk-raising drivers for every row of X."""
    contrib = contributions(model, X)
    out = []
    for i in range(len(X)):
        order = np.argsort(-contrib[i])
        items = []
        for j in order[:top_k]:
            if contrib[i, j] <= 0:
                break
            name = X.columns[j]
            label, fmt = LABELS.get(name, (name, "{:.2f}"))
            val = X.iloc[i, j]
            if not pd.isna(val) and name in _TO_DISPLAY:
                val = float(_TO_DISPLAY[name](val))
            shown = "n/a" if pd.isna(val) else fmt.format(val)
            items.append({"code": name, "text": f"{label}: {shown} (raises inactivity risk)",
                          "contribution": float(contrib[i, j])})
        out.append(items)
    return out
