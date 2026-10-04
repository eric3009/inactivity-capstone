import pytest
import numpy as np
import pandas as pd

from inactivity.data import clean


def _row(invoice, code, qty, price, cid, when="2010-03-01 10:00"):
    return dict(invoice=invoice, stock_code=code, description="x", quantity=qty,
                invoice_date=pd.Timestamp(when), price=price, customer_id=cid, country="United Kingdom")


def test_each_cleaning_rule_is_applied():
    raw = pd.DataFrame([
        _row("1001", "85123A", 2, 3.0, 12345.0),                     # good purchase
        _row("1001", "85123A", 2, 3.0, 12345.0),                     # exact duplicate
        _row("1002", "85123A", 1, 3.0, np.nan),                      # missing customer id
        _row("1003", "POST", 1, 18.0, 12345.0),                      # non-product code
        _row("C1004", "85123A", -1, 3.0, 12345.0),                   # return (cancellation)
        _row("1005", "85123A", -5, 3.0, 12345.0),                    # negative qty without C prefix
        _row("1006", "85123A", 1, 0.0, 12345.0),                     # zero price
    ])
    df, rep = clean(raw)
    assert rep["duplicate_rows_dropped"] == 1
    assert rep["missing_customer_id_rows"] == 1
    assert rep["non_product_rows_dropped"] == 1
    assert rep["cancellation_return_lines_kept"] == 1
    assert rep["other_anomalous_lines_dropped"] == 2
    assert len(df) == 2
    assert df["is_return"].tolist().count(True) == 1
    assert df["customer_id"].tolist() == ["12345", "12345"]


def test_clean_synthetic_has_no_problem_rows(raw):
    df, rep = clean(raw)
    assert df["customer_id"].notna().all()
    assert df["stock_code"].str.match(r"^\d{5}").all()
    assert not df.duplicated(["invoice", "stock_code", "quantity", "invoice_date", "price", "customer_id"]).any()
    assert rep["rows_clean"] == len(df) < rep["rows_raw"]
    assert (df.loc[~df["is_return"], "quantity"] > 0).all()


def test_schema_check_names_the_missing_column(raw):
    from inactivity.data import validate_schema
    validate_schema(raw)                                             # the expected layout passes
    with pytest.raises(ValueError, match="customer_id"):
        validate_schema(raw.drop(columns=["customer_id"]))
    with pytest.raises(ValueError, match="no rows"):
        validate_schema(raw.iloc[0:0])


def test_file_hash_is_stable_and_content_sensitive(tmp_path):
    from inactivity.data import sha256_file
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"online retail")
    b.write_bytes(b"online retail!")
    assert sha256_file(a) == sha256_file(a) and sha256_file(a) != sha256_file(b)
    assert len(sha256_file(a)) == 64
