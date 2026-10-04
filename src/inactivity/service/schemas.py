from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


class ReasonCode(BaseModel):
    code: str
    text: str
    contribution: float = Field(description="TreeSHAP contribution in log-odds (positive raises risk)")


class ScoreResponse(BaseModel):
    customer_id: Optional[str] = None
    snapshot_date: date
    eligible: bool
    probability_inactive: Optional[float] = Field(
        None, description="Calibrated P(no qualifying purchase within the horizon); null when not eligible")
    risk_band: str = Field(description="High, Medium, Low or Not eligible")
    horizon_days: int
    reason_codes: list[ReasonCode] = []
    model_version: str
    note: Optional[str] = None


class FeatureScoreRequest(BaseModel):
    customer_id: Optional[str] = None
    snapshot_date: date
    features: dict[str, Optional[float]] = Field(
        description="Snapshot features keyed by name (see GET /model-info). Missing values may be null.")


class TransactionLine(BaseModel):
    invoice: str
    stock_code: str = "00000"
    invoice_date: datetime
    quantity: float
    price: float
    country: str = "United Kingdom"


class TransactionScoreRequest(BaseModel):
    customer_id: Optional[str] = None
    snapshot_date: date
    transactions: list[TransactionLine] = Field(min_length=1, description="The customer's lines up to the snapshot date")
