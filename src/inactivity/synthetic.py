"""Synthetic transactions in the raw Online Retail II layout.

Used only for tests and an offline smoke run (`prepare --synthetic`); it reproduces the
messiness the cleaner must handle (missing IDs, cancellations, non-product codes, duplicates)
and a heterogeneous purchase process with customer churn, so models have signal to find.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_raw(n_customers: int = 700, seed: int = 0,
                 start: str = "2009-12-01", end: str = "2011-12-09") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    span = (t1 - t0).days
    rows = []
    inv_no = 489000
    for cid in range(12346, 12346 + n_customers):
        first = int(rng.integers(0, int(span * 0.7)))
        rate = float(rng.gamma(2.0, 0.012))                 # purchases per day (heterogeneous)
        life = float(rng.exponential(250)) if rng.random() < 0.55 else 10_000
        day = first
        while day < span:
            day += int(max(1, rng.exponential(1 / max(rate, 1e-3))))
            if day >= span or day - first > life:
                break
            inv_no += 1
            n_lines = int(rng.integers(1, 6))
            stamp = t0 + pd.Timedelta(days=day, hours=int(rng.integers(8, 17)), minutes=int(rng.integers(0, 59)))
            for _ in range(n_lines):
                rows.append((str(inv_no), str(int(rng.integers(10000, 10400))), "ITEM",
                             int(rng.integers(1, 40)), stamp, round(float(rng.uniform(0.5, 25)), 2),
                             float(cid), "United Kingdom" if rng.random() < 0.9 else "France"))
            if rng.random() < 0.04:                          # a cancellation of the same invoice
                rows.append(("C" + str(inv_no), rows[-1][1], "ITEM", -int(rng.integers(1, 5)),
                             stamp + pd.Timedelta(days=2), rows[-1][5], float(cid), "United Kingdom"))
    df = pd.DataFrame(rows, columns=["invoice", "stock_code", "description", "quantity", "invoice_date",
                                     "price", "customer_id", "country"])
    # inject raw-data problems
    miss = rng.random(len(df)) < 0.2
    df.loc[miss, "customer_id"] = np.nan
    junk = df.sample(frac=0.01, random_state=seed).index
    df.loc[junk, "stock_code"] = rng.choice(["POST", "M", "D", "BANK CHARGES", "ADJUST"], size=len(junk))
    df = pd.concat([df, df.sample(frac=0.01, random_state=seed + 1)], ignore_index=True)   # duplicates
    return df
