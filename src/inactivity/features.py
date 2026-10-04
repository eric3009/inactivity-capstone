"""Snapshot features and labels.

A snapshot is a scoring date `s`. Features use only transactions dated on or before `s`;
labels look at (s, s + horizon]. The same `compute_features` code path is used for batch
training and for the API, so there is no training/serving skew.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = [
    "recency_days", "tenure_days", "n_purchase_days",
    "inv_count_30", "inv_count_90", "inv_count_180", "inv_count_365", "inv_count_90_prev",
    "freq_trend", "rev_90_log", "rev_180_log", "rev_365_log", "rev_trend",
    "avg_invoice_value_180_log", "avg_items_per_invoice_180", "n_products_180",
    "median_interval_days", "interval_cv", "recency_to_interval", "return_rate_180", "is_uk",
]


@dataclass
class TxIndex:
    """Pre-aggregated transaction tables so each snapshot is a cheap filter + groupby."""
    invoices: pd.DataFrame   # customer_id, invoice, date, revenue, items, country
    products: pd.DataFrame   # customer_id, date, stock_code (distinct)
    returns: pd.DataFrame    # customer_id, date, value (positive = value returned)


def prepare_index(tx: pd.DataFrame) -> TxIndex:
    buys = tx[~tx["is_return"]]
    invoices = (buys.groupby(["customer_id", "invoice", "date"], as_index=False)
                .agg(revenue=("revenue", "sum"), items=("quantity", "sum"), country=("country", "last")))
    products = buys[["customer_id", "date", "stock_code"]].drop_duplicates()
    rets = tx[tx["is_return"]]
    returns = (rets.groupby(["customer_id", "date"], as_index=False)
               .agg(value=("revenue", lambda s: float(-s.sum()))))
    return TxIndex(invoices.sort_values("date"), products, returns)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURES + ["eligible"], index=pd.Index([], name="customer_id"))


def compute_features(idx: TxIndex, snapshot, activity_window: int = 180) -> pd.DataFrame:
    """Features for every customer with at least one purchase on or before `snapshot`."""
    s = pd.Timestamp(snapshot).normalize()
    inv = idx.invoices[idx.invoices["date"] <= s].copy()
    if inv.empty:
        return _empty()
    inv["age"] = (s - inv["date"]).dt.days
    out = inv.groupby("customer_id").agg(first_date=("date", "min"), last_date=("date", "max"))
    out["recency_days"] = (s - out["last_date"]).dt.days
    out["tenure_days"] = (s - out["first_date"]).dt.days

    pdays = inv[["customer_id", "date"]].drop_duplicates().sort_values(["customer_id", "date"])
    out["n_purchase_days"] = pdays.groupby("customer_id").size()
    gaps = pdays.groupby("customer_id")["date"].diff().dt.days
    gstats = gaps.groupby(pdays["customer_id"]).agg(["median", "mean", "std", "count"])
    out["median_interval_days"] = gstats["median"]
    out["interval_cv"] = np.where(gstats["count"] >= 3, gstats["std"] / gstats["mean"], np.nan)
    out["recency_to_interval"] = out["recency_days"] / out["median_interval_days"].replace(0, np.nan)

    for w in (30, 90, 180, 365):
        out[f"inv_count_{w}"] = inv[inv["age"] < w].groupby("customer_id").size()
    w90 = inv[inv["age"] < 90].groupby("customer_id")["revenue"].sum()
    prev = inv[(inv["age"] >= 90) & (inv["age"] < 180)]
    out["inv_count_90_prev"] = prev.groupby("customer_id").size()
    rev90_prev = prev.groupby("customer_id")["revenue"].sum()
    w180 = inv[inv["age"] < 180].groupby("customer_id").agg(rev=("revenue", "sum"), items=("items", "sum"))
    rev365 = inv[inv["age"] < 365].groupby("customer_id")["revenue"].sum()
    out["rev_90_log"] = np.log1p(w90.clip(lower=0))
    out["rev_180_log"] = np.log1p(w180["rev"].clip(lower=0))
    out["rev_365_log"] = np.log1p(rev365.clip(lower=0))
    out["rev_trend"] = np.log1p(w90.clip(lower=0).reindex(out.index).fillna(0)) - \
        np.log1p(rev90_prev.clip(lower=0).reindex(out.index).fillna(0))
    cols = ["inv_count_30", "inv_count_90", "inv_count_180", "inv_count_365", "inv_count_90_prev",
            "rev_90_log", "rev_180_log", "rev_365_log"]
    out[cols] = out[cols].fillna(0)
    out["freq_trend"] = out["inv_count_90"] - out["inv_count_90_prev"]
    n180 = out["inv_count_180"].replace(0, np.nan)
    out["avg_invoice_value_180_log"] = np.log1p((np.expm1(out["rev_180_log"]) / n180).fillna(0))
    out["avg_items_per_invoice_180"] = (w180["items"].reindex(out.index) / n180).fillna(0)

    pr = idx.products[idx.products["date"] <= s]
    pr = pr[(s - pr["date"]).dt.days < 180]
    out["n_products_180"] = pr.groupby("customer_id")["stock_code"].nunique().reindex(out.index).fillna(0)

    rt = idx.returns[idx.returns["date"] <= s]
    rt = rt[(s - rt["date"]).dt.days < 180]
    ret_val = rt.groupby("customer_id")["value"].sum().reindex(out.index).fillna(0).clip(lower=0)
    gross = np.expm1(out["rev_180_log"])
    out["return_rate_180"] = (ret_val / (gross + ret_val)).fillna(0)

    last_country = inv.sort_values("date").groupby("customer_id")["country"].last()
    out["is_uk"] = (last_country.reindex(out.index) == "United Kingdom").astype(float)

    out["eligible"] = out["recency_days"] < activity_window
    return out[FEATURES + ["eligible"]]


def compute_features_from_lines(lines: pd.DataFrame, snapshot, activity_window: int = 180) -> pd.DataFrame:
    """Single-customer path used by the API: raw lines in, one-row feature frame out."""
    df = lines.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["date"] = df["invoice_date"].dt.normalize()
    df["invoice"] = df["invoice"].astype(str)
    df["is_return"] = df["invoice"].str.upper().str.startswith("C")
    df["revenue"] = df["quantity"] * df["price"]
    if "country" not in df:
        df["country"] = "United Kingdom"
    return compute_features(prepare_index(df), snapshot, activity_window)


def make_labels(idx: TxIndex, customers: pd.Index, snapshot, horizon: int) -> pd.Series:
    """1 = no qualifying purchase in (snapshot, snapshot + horizon]."""
    s = pd.Timestamp(snapshot).normalize()
    fut = idx.invoices[(idx.invoices["date"] > s) & (idx.invoices["date"] <= s + pd.Timedelta(days=horizon))]
    bought = set(fut["customer_id"].unique())
    return pd.Series([0 if c in bought else 1 for c in customers], index=customers, name="y_inactive")


def build_dataset(tx: pd.DataFrame, snapshots, horizon: int, activity_window: int = 180,
                  segment_k: float = 3.0) -> tuple[pd.DataFrame, TxIndex]:
    """Stack eligible customer-snapshot rows with both label definitions."""
    idx = prepare_index(tx)
    frames = []
    for s in snapshots:
        f = compute_features(idx, s, activity_window)
        n_all = len(f)
        f = f[f["eligible"]].drop(columns="eligible")
        if f.empty:
            continue
        y = make_labels(idx, f.index, s, horizon)
        # Segment-aware label: inactive only if the elapsed gap exceeds k x the customer's own cycle.
        fallback = float(np.nanmedian(f["median_interval_days"])) if f["median_interval_days"].notna().any() else 30.0
        own = f["median_interval_days"].fillna(fallback).clip(lower=1.0)
        y_seg = ((y == 1) & ((f["recency_days"] + horizon) > segment_k * own)).astype(int)
        f = f.assign(snapshot_date=pd.Timestamp(s), y_inactive=y.values, y_inactive_seg=y_seg.values,
                     n_customers_at_snapshot=n_all)
        frames.append(f.reset_index())
    data = pd.concat(frames, ignore_index=True)
    data["customer_id"] = data["customer_id"].astype(str)
    return data, idx
