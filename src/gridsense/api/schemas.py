"""Pydantic schemas for all GridSense API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


# ---------------------------------------------------------------------------
# Solar forecast
# ---------------------------------------------------------------------------


class ForecastRequest(BaseModel):
    station_code: str = Field(
        default="A801",
        description="INMET station code (e.g. 'A801' for Florianópolis).",
        min_length=1,
        max_length=16,
    )
    horizon_hours: int = Field(
        default=24,
        ge=1,
        le=48,
        description="Forecast horizon in hours (1–48).",
    )


class HourlyForecastItem(BaseModel):
    timestamp: datetime
    predicted_kw: float
    lower_bound: float
    upper_bound: float


class ForecastResponse(BaseModel):
    predictions: List[HourlyForecastItem]
    model_version: str
    station_code: str
    generated_at: datetime


# ---------------------------------------------------------------------------
# Power quality events
# ---------------------------------------------------------------------------


class PQEventItem(BaseModel):
    timestamp: datetime
    label: str
    confidence: float
    duration_ms: Optional[int] = None


class PQEventsResponse(BaseModel):
    events: List[PQEventItem]
    total: int
    start: datetime
    end: datetime


# ---------------------------------------------------------------------------
# Battery SoC
# ---------------------------------------------------------------------------


class SoCResponse(BaseModel):
    soc: float = Field(..., ge=0.0, le=1.0, description="State of Charge 0–1.")
    soc_percent: float = Field(..., ge=0.0, le=100.0)
    updated_at: datetime
