"""Exploratory-data-analysis aggregates for the web UI.

Everything is computed here once, so the browser only draws: activity over time, customer value concentration,
distributions, the lapse rate by month, a per-feature explorer and a correlation matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .explain import LABELS
from .features import FEATURES

SETS = ("all", "train", "val", "cal", "test")
BLOCK_NAMES = ("train", "val", "cal", "test")
# Stored on a log scale in the model; shown in business units in the explorer.
DISPLAY = {"rev_90_log": np.expm1, "rev_180_log": np.expm1, "rev_365_log": np.expm1,
           "avg_invoice_value_180_log": np.expm1, "rev_trend": np.exp}
MONEY = {"rev_90_log", "rev_180_log", "rev_365_log", "avg_invoice_value_180_log"}


def _num(v: float, money: bool = False) -> str:
    if money:
        return f"£{v:,.0f}"
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v)):,}"
    a = abs(v)
    if a >= 100:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.1f}".rstrip("0").rstrip(".")
    return f"{v:.2f}"


def monthly(tx: pd.DataFrame) -> list[dict]:
    b = tx[~tx["is_return"]].copy()
    r = tx[tx["is_return"]].copy()
    b["m"] = b["date"].dt.to_period("M")
    r["m"] = r["date"].dt.to_period("M")
    g = b.groupby("m").agg(orders=("invoice", "nunique"), revenue=("revenue", "sum"), customers=("customer_id", "nunique"))
    first = b.groupby("customer_id")["m"].min().value_counts()
    ret = -r.groupby("m")["revenue"].sum()
    return [{"month": str(m), "orders": int(row.orders), "revenue": round(float(row.revenue)), "customers": int(row.customers),
             "new_customers": int(first.get(m, 0)), "returns": round(float(ret.get(m, 0)))} for m, row in g.iterrows()]


def countries(tx: pd.DataFrame, top: int = 10) -> list[dict]:
    b = tx[~tx["is_return"]]
    g = b.groupby("country").agg(revenue=("revenue", "sum"), customers=("customer_id", "nunique"), orders=("invoice", "nunique"))
    g = g.sort_values("revenue", ascending=False)
    head, rest = g.head(top), g.iloc[top:]
    out = [{"country": c, "revenue": round(float(r.revenue)), "customers": int(r.customers), "orders": int(r.orders)} for c, r in head.iterrows()]
    if len(rest):
        out.append({"country": f"{len(rest)} other countries", "revenue": round(float(rest.revenue.sum())),
                    "customers": int(b[b["country"].isin(rest.index)]["customer_id"].nunique()), "orders": int(rest.orders.sum())})
    return out


def _hist(values, edges, labels, title, unit):
    counts = np.histogram(np.asarray(values, dtype=float), bins=edges)[0]
    return {"title": title, "unit": unit, "labels": labels, "counts": [int(c) for c in counts], "total": int(counts.sum())}


def distributions(tx: pd.DataFrame) -> dict:
    b = tx[~tx["is_return"]]
    n_inv = b.groupby("customer_id")["invoice"].nunique()
    rev = b.groupby("customer_id")["revenue"].sum().clip(lower=0)
    inv_val = b.groupby("invoice")["revenue"].sum().clip(lower=0)
    pdays = b[["customer_id", "date"]].drop_duplicates().sort_values(["customer_id", "date"])
    gaps = pdays.groupby("customer_id")["date"].diff().dt.days
    med = gaps.groupby(pdays["customer_id"]).median().dropna()
    big = 1e12
    return {
        "orders_per_customer": _hist(n_inv, [1, 2, 3, 4, 5, 6, 9, 13, 21, 51, big],
                                     ["1", "2", "3", "4", "5", "6-8", "9-12", "13-20", "21-50", "51+"], "Orders per customer", "customers"),
        "revenue_per_customer": _hist(rev, [0, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, big],
                                      ["<£50", "£50-100", "£100-200", "£200-400", "£400-800", "£800-1.6k", "£1.6-3.2k", "£3.2-6.4k", "£6.4-12.8k", "£12.8k+"],
                                      "Lifetime spend per customer", "customers"),
        "days_between_orders": _hist(med, [0, 7, 14, 21, 30, 45, 60, 90, 120, 180, 365, big],
                                     ["<7", "7-13", "14-20", "21-29", "30-44", "45-59", "60-89", "90-119", "120-179", "180-364", "365+"],
                                     "Customer's median days between orders", "customers with 2+ purchase days"),
        "order_value": _hist(inv_val, [0, 25, 50, 100, 200, 400, 800, 1600, 3200, big],
                             ["<£25", "£25-50", "£50-100", "£100-200", "£200-400", "£400-800", "£800-1.6k", "£1.6-3.2k", "£3.2k+"],
                             "Value of an order", "orders"),
    }


def lorenz(tx: pd.DataFrame) -> dict:
    b = tx[~tx["is_return"]]
    rev = np.sort(b.groupby("customer_id")["revenue"].sum().clip(lower=0).to_numpy())
    cum = np.cumsum(rev) / rev.sum()
    n = len(rev)
    xs = np.linspace(0, 1, 101)
    ys = [0.0] + [float(cum[max(0, int(np.ceil(x * n)) - 1)]) for x in xs[1:]]

    def top(share):
        k = max(1, int(np.ceil(share * n)))
        return float(rev[-k:].sum() / rev.sum())

    return {"x": xs.tolist(), "y": ys, "customers": int(n), "top1": top(0.01), "top10": top(0.10), "top20": top(0.20),
            "revenue": round(float(rev.sum()))}


def snapshots(data_all: pd.DataFrame, blocks_map: dict) -> list[dict]:
    g = data_all.groupby("snapshot_date").agg(eligible=("y_inactive", "size"), lapse=("y_inactive", "mean"), lapse_seg=("y_inactive_seg", "mean"))
    return [{"date": str(pd.Timestamp(d).date()), "eligible": int(r.eligible), "lapse": float(r.lapse), "lapse_seg": float(r.lapse_seg),
             "block": blocks_map.get(pd.Timestamp(d), "dropped")} for d, r in g.iterrows()]


def feature_explorer(data_all: pd.DataFrame, blocks_map: dict) -> dict:
    d = data_all.copy()
    d["block"] = d["snapshot_date"].map(lambda s: blocks_map.get(pd.Timestamp(s), "dropped"))
    d = d[d["block"].isin(BLOCK_NAMES)].reset_index(drop=True)
    y = d["y_inactive"].to_numpy()
    out = {}
    for f in FEATURES:
        x = d[f]
        label = LABELS.get(f, (f, ""))[0]
        money = f in MONEY
        tf = DISPLAY.get(f, lambda v: v)
        vals = x.dropna()
        vmin, vmax = float(vals.min()), float(vals.max())
        small_int = bool(np.all(np.mod(vals, 1) == 0)) and (vmax - vmin) <= 40
        if small_int:            # counts and flags: one bin per value, with the long tails folded into the end bins
            lo_c, hi_c = max(vmin, -6.0), min(vmax, 11.0)
            filled = x.fillna(lo_c - 1).to_numpy()
            code = np.where(x.isna().to_numpy(), -1, np.clip(filled, lo_c, hi_c) - lo_c).astype(int)
            nb = int(hi_c - lo_c) + 1
            labels = [_num(float(tf(lo_c + i)), money) for i in range(nb)]
            if vmin < lo_c:
                labels[0] = f"<={_num(lo_c)}"
            if vmax > hi_c:
                labels[-1] = f"{_num(hi_c)}+"
        else:
            edges = np.unique(np.quantile(vals, np.linspace(0, 1, 13)))
            cut = pd.cut(x, bins=edges, labels=False, include_lowest=True)
            code = cut.fillna(-1).astype(int).to_numpy()
            nb = len(edges) - 1
            labels = []
            for i in range(nb):
                lo, hi = float(tf(edges[i])), float(tf(edges[i + 1]))
                labels.append(f"{_num(lo, money)} to {_num(hi, money)}")
        if (code == -1).any():
            code = np.where(code == -1, nb, code)
            labels = labels + ["missing"]
            nb += 1
        sets = {}
        for s in SETS:
            m = np.ones(len(d), dtype=bool) if s == "all" else (d["block"] == s).to_numpy()
            n0 = np.bincount(code[m & (y == 0)], minlength=nb)
            n1 = np.bincount(code[m & (y == 1)], minlength=nb)
            sets[s] = {"n0": [int(v) for v in n0], "n1": [int(v) for v in n1]}
        out[f] = {"label": label, "money": money, "bins": labels, "sets": sets,
                  "median_active": float(tf(np.nanmedian(x[y == 0]))) if (y == 0).any() else None,
                  "median_lapsed": float(tf(np.nanmedian(x[y == 1]))) if (y == 1).any() else None}
    return out


def correlation(data_all: pd.DataFrame, blocks_map: dict) -> dict:
    d = data_all[data_all["snapshot_date"].map(lambda s: blocks_map.get(pd.Timestamp(s), "dropped")).isin(BLOCK_NAMES)]
    c = d[FEATURES].corr(method="spearman").fillna(0.0)
    return {"features": FEATURES, "labels": [LABELS.get(f, (f, ""))[0] for f in FEATURES],
            "matrix": [[round(float(v), 2) for v in row] for row in c.to_numpy()]}


def build_eda(tx: pd.DataFrame, data_all: pd.DataFrame, blocks_map: dict, quality: dict) -> dict:
    return {
        "quality": quality, "monthly": monthly(tx), "countries": countries(tx), "distributions": distributions(tx),
        "lorenz": lorenz(tx), "snapshots": snapshots(data_all, blocks_map),
        "features": feature_explorer(data_all, blocks_map), "correlation": correlation(data_all, blocks_map),
        "rows_modelling": int(data_all["snapshot_date"].map(lambda s: blocks_map.get(pd.Timestamp(s), "dropped")).isin(BLOCK_NAMES).sum()),
    }
