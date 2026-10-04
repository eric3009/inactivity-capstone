import numpy as np
import pandas as pd
import pytest

from inactivity.features import (FEATURES, compute_features, compute_features_from_lines, make_labels,
                                 prepare_index)


def _tx(rows):
    df = pd.DataFrame(rows, columns=["customer_id", "invoice", "date", "quantity", "price"])
    df["date"] = pd.to_datetime(df["date"])
    df["invoice_date"] = df["date"] + pd.Timedelta(hours=10)
    df["stock_code"] = "10001"
    df["revenue"] = df["quantity"] * df["price"]
    df["is_return"] = df["invoice"].str.startswith("C")
    df["country"] = "United Kingdom"
    return df


SNAP = pd.Timestamp("2011-03-31")


def test_features_ignore_everything_after_the_snapshot(tx):
    idx_full = prepare_index(tx)
    cutoff = tx[tx["date"] <= SNAP]
    idx_cut = prepare_index(cutoff)
    a = compute_features(idx_full, SNAP)
    b = compute_features(idx_cut, SNAP)
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())


def test_future_rows_do_not_change_features(tx):
    future = tx[tx["date"] > SNAP].copy()
    future["quantity"] = future["quantity"] * 1000          # corrupt the future
    mutated = pd.concat([tx[tx["date"] <= SNAP], future])
    a = compute_features(prepare_index(tx), SNAP)
    b = compute_features(prepare_index(mutated), SNAP)
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())


def test_purchase_on_snapshot_day_is_history_not_label():
    tx = _tx([("A", "1", "2011-03-31", 1, 10.0), ("B", "2", "2011-03-01", 1, 10.0),
              ("B", "3", "2011-04-15", 1, 10.0)])
    idx = prepare_index(tx)
    feats = compute_features(idx, SNAP)
    assert feats.loc["A", "recency_days"] == 0
    y = make_labels(idx, feats.index, SNAP, 90)
    assert y["A"] == 1          # A has no purchase after the snapshot -> inactive
    assert y["B"] == 0          # B buys 15 days after


def test_label_window_is_inclusive_of_horizon_end():
    tx = _tx([("A", "1", "2011-03-01", 1, 5.0), ("A", "2", "2011-06-29", 1, 5.0),   # day 90 -> active
              ("B", "3", "2011-03-01", 1, 5.0), ("B", "4", "2011-06-30", 1, 5.0)])   # day 91 -> inactive
    idx = prepare_index(tx)
    feats = compute_features(idx, SNAP)
    y = make_labels(idx, feats.index, SNAP, 90)
    assert (y["A"], y["B"]) == (0, 1)


def test_return_lines_do_not_count_as_purchases():
    tx = _tx([("A", "1", "2011-01-10", 2, 10.0), ("A", "C2", "2011-03-30", -1, 10.0)])
    feats = compute_features(prepare_index(tx), SNAP)
    assert feats.loc["A", "recency_days"] == (SNAP - pd.Timestamp("2011-01-10")).days
    assert feats.loc["A", "return_rate_180"] > 0


def test_dormant_customers_are_ineligible():
    tx = _tx([("A", "1", "2010-06-01", 1, 10.0), ("B", "2", "2011-03-20", 1, 10.0)])
    feats = compute_features(prepare_index(tx), SNAP, activity_window=180)
    assert not feats.loc["A", "eligible"] and feats.loc["B", "eligible"]


def test_api_path_matches_batch_path(tx):
    """Training/serving parity: features from raw lines == features from the batch index."""
    cust = tx[tx["date"] <= SNAP].groupby("customer_id").size().sort_values().index[-5]
    batch = compute_features(prepare_index(tx), SNAP).loc[[cust]]
    lines = tx[(tx["customer_id"] == cust) & (tx["date"] <= SNAP)][
        ["invoice", "stock_code", "invoice_date", "quantity", "price", "country"]].assign(customer_id=cust)
    api = compute_features_from_lines(lines, SNAP)
    pd.testing.assert_frame_equal(batch[FEATURES + ["eligible"]].astype(float),
                                  api[FEATURES + ["eligible"]].astype(float), check_names=False)


def test_feature_list_matches_output(tx):
    feats = compute_features(prepare_index(tx), SNAP)
    assert list(feats.columns) == FEATURES + ["eligible"]
    assert not np.isinf(feats[FEATURES].to_numpy(float)).any()
