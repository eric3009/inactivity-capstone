"""Ingestion and cleaning of the UCI Online Retail II transactions.

The raw file is deliberately messy: missing customer IDs, cancellation invoices, returns,
non-product stock codes, duplicates and extreme quantities. Every cleaning step is counted
and written to reports/data_quality.json so the effect of each rule is auditable.
"""
from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve

RAW_COLUMNS = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


REQUIRED_COLUMNS = ["invoice", "stock_code", "quantity", "invoice_date", "price", "customer_id", "country"]


def validate_schema(raw: pd.DataFrame) -> None:
    """Fail early, with a clear message, if the input does not look like Online Retail II."""
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Input is missing expected column(s) {missing}. Found: {list(raw.columns)}. "
                         f"Expected the Online Retail II layout {list(RAW_COLUMNS)}; the source file may have changed.")
    if len(raw) == 0:
        raise ValueError("Input has no rows.")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, raw_dir: Path, xlsx_name: str) -> Path:
    """Download the UCI zip once and extract the workbook. Returns the xlsx path."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / xlsx_name
    if target.exists():
        return target
    req = urllib.request.Request(url, headers={"User-Agent": "inactivity-capstone/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]
        if not names:
            raise RuntimeError(f"No .xlsx found in downloaded archive: {zf.namelist()}")
        with zf.open(names[0]) as src, open(target, "wb") as dst:
            dst.write(src.read())
    return target


def load_raw(xlsx_path: Path) -> pd.DataFrame:
    """Read every worksheet (the file splits the two years across sheets) and stack them."""
    sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
    df = pd.concat(sheets.values(), ignore_index=True)
    df = df.rename(columns=RAW_COLUMNS)
    validate_schema(df)
    return df


def clean(raw: pd.DataFrame, product_code_regex: str = r"^\d{5}") -> tuple[pd.DataFrame, dict]:
    """Apply the documented cleaning rules. Returns (clean_lines, quality_report)."""
    rep: dict = {"rows_raw": int(len(raw))}
    df = raw.copy()
    df["invoice"] = df["invoice"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str).str.strip()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    before = len(df)
    df = df.drop_duplicates(subset=["invoice", "stock_code", "quantity", "invoice_date", "price", "customer_id"])
    rep["duplicate_rows_dropped"] = int(before - len(df))

    miss = df["customer_id"].isna()
    rep["missing_customer_id_rows"] = int(miss.sum())
    rep["missing_customer_id_share"] = float(miss.mean())
    df = df[~miss].copy()
    df["customer_id"] = df["customer_id"].astype(float).astype(int).astype(str)

    is_product = df["stock_code"].str.match(product_code_regex, case=False, na=False)
    rep["non_product_rows_dropped"] = int((~is_product).sum())
    df = df[is_product].copy()

    df["is_cancel"] = df["invoice"].str.upper().str.startswith("C")
    df["revenue"] = df["quantity"] * df["price"]
    purchase = (~df["is_cancel"]) & (df["quantity"] > 0) & (df["price"] > 0)
    ret = df["is_cancel"] & (df["quantity"] < 0) & (df["price"] > 0)
    rep["cancellation_return_lines_kept"] = int(ret.sum())
    rep["other_anomalous_lines_dropped"] = int((~(purchase | ret)).sum())
    df = df[purchase | ret].copy()
    df["is_return"] = df["is_cancel"]

    q99 = float(df.loc[~df["is_return"], "quantity"].quantile(0.999))
    rep["quantity_p99_9"] = q99
    rep["extreme_quantity_lines"] = int((df.loc[~df["is_return"], "quantity"] > q99).sum())

    df["date"] = df["invoice_date"].dt.normalize()
    df = df[["customer_id", "invoice", "invoice_date", "date", "stock_code", "quantity", "price",
             "revenue", "is_return", "country"]].reset_index(drop=True)
    rep["rows_clean"] = int(len(df))
    rep["customers"] = int(df["customer_id"].nunique())
    rep["date_min"] = str(df["date"].min().date())
    rep["date_max"] = str(df["date"].max().date())
    return df, rep


def prepare(cfg: dict, synthetic: bool = False, force: bool = False) -> pd.DataFrame:
    """Download (or synthesise), clean and cache transactions; returns the clean frame."""
    interim = resolve(cfg["data"]["interim_dir"])
    interim.mkdir(parents=True, exist_ok=True)
    clean_path = interim / cfg["data"]["clean_file"]
    reports = resolve(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    if clean_path.exists() and not force and not synthetic:
        return pd.read_parquet(clean_path)

    provenance = {}
    if synthetic:
        from .synthetic import generate_raw
        raw = generate_raw(seed=cfg["project"]["seed"])
    else:
        xlsx = download(cfg["data"]["url"], resolve(cfg["data"]["raw_dir"]), cfg["data"]["xlsx_name"])
        digest = sha256_file(xlsx)
        expected = cfg["data"].get("expected_sha256")
        provenance = {"source_url": cfg["data"]["url"], "source_bytes": xlsx.stat().st_size, "source_sha256": digest,
                      "hash_matches_expected": (digest == expected) if expected else None}
        if expected and digest != expected:
            warnings.warn(f"The input workbook's SHA-256 ({digest[:12]}...) differs from the one the reported results were built "
                          f"on ({expected[:12]}...). Results may not match the published ones.", stacklevel=2)
            print("[warn] input file differs from the version used for the reported results (SHA-256 mismatch)")
        raw = load_raw(xlsx)
    validate_schema(raw)
    df, rep = clean(raw, cfg["cleaning"]["product_code_regex"])
    rep.update(provenance)
    rep["source"] = "synthetic" if synthetic else "uci-online-retail-ii"
    with open(reports / "data_quality.json", "w", encoding="utf8") as f:
        json.dump(rep, f, indent=2)
    if not synthetic:
        df.to_parquet(clean_path, index=False)
    return df
