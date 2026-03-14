"""
SQLAlchemy ORM models for all GridSense time-series tables.

All tables are TimescaleDB hypertables partitioned on the ``time`` column.
The :func:`create_all_tables` helper creates the schema and enables the
TimescaleDB extension in one call — safe to run repeatedly (idempotent).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    TIMESTAMP,
    Column,
    Double,
    Index,
    Integer,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


class SolarReading(Base):
    """Raw readings from the solar inverter (Modbus)."""

    __tablename__ = "solar_readings"

    # TimescaleDB requires the partition column to be part of the PK
    time: datetime = Column(TIMESTAMP(timezone=True), primary_key=True, nullable=False)
    station_id: str = Column(String(64), primary_key=True, nullable=False)
    power_w: float | None = Column(Double)
    voltage_v: float | None = Column(Double)
    current_a: float | None = Column(Double)

    __table_args__ = (
        Index("ix_solar_readings_time", "time"),
    )


class WeatherReading(Base):
    """Meteorological data from the INMET API."""

    __tablename__ = "weather_readings"

    time: datetime = Column(TIMESTAMP(timezone=True), primary_key=True, nullable=False)
    station_code: str = Column(String(16), primary_key=True, nullable=False)
    irradiance_wm2: float | None = Column(Double)
    temp_c: float | None = Column(Double)
    humidity_pct: float | None = Column(Double)

    __table_args__ = (
        Index("ix_weather_readings_time", "time"),
    )


class SolarForecast(Base):
    """Model-generated solar generation forecasts."""

    __tablename__ = "solar_forecasts"

    forecast_time: datetime = Column(
        TIMESTAMP(timezone=True), primary_key=True, nullable=False
    )
    generated_at: datetime = Column(
        TIMESTAMP(timezone=True), primary_key=True, nullable=False
    )
    predicted_kw: float | None = Column(Double)
    lower_bound: float | None = Column(Double)
    upper_bound: float | None = Column(Double)
    model_version: str | None = Column(String(32))

    __table_args__ = (
        Index("ix_solar_forecasts_forecast_time", "forecast_time"),
        Index("ix_solar_forecasts_generated_at", "generated_at"),
    )


class PQEvent(Base):
    """Power quality disturbance events detected by the classifier."""

    __tablename__ = "pq_events"

    time: datetime = Column(TIMESTAMP(timezone=True), primary_key=True, nullable=False)
    label: str = Column(String(64), nullable=False)
    confidence: float | None = Column(Double)
    duration_ms: int | None = Column(Integer)
    raw_data: dict | None = Column(JSONB)

    __table_args__ = (
        Index("ix_pq_events_time", "time"),
        Index("ix_pq_events_label", "label"),
    )


class BatterySoC(Base):
    """Battery State-of-Charge time series."""

    __tablename__ = "battery_soc"

    time: datetime = Column(TIMESTAMP(timezone=True), primary_key=True, nullable=False)
    soc: float | None = Column(Double)
    current_a: float | None = Column(Double)

    __table_args__ = (
        Index("ix_battery_soc_time", "time"),
    )


# ---------------------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------------------


def create_all_tables(engine: Engine) -> None:
    """Create all tables and enable TimescaleDB hypertables.

    Idempotent — safe to call multiple times.  Requires TimescaleDB to be
    installed in the target PostgreSQL instance.

    Parameters
    ----------
    engine:
        Connected SQLAlchemy engine pointing at the target database.
    """
    # 1. Enable TimescaleDB extension (no-op if already enabled)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
        conn.commit()

    # 2. Create tables via SQLAlchemy metadata
    Base.metadata.create_all(engine)

    # 3. Convert each table to a hypertable (no-op if already done)
    hypertables = {
        "solar_readings": "time",
        "weather_readings": "time",
        "solar_forecasts": "forecast_time",
        "pq_events": "time",
        "battery_soc": "time",
    }

    with engine.connect() as conn:
        for table, time_col in hypertables.items():
            conn.execute(
                text(
                    f"SELECT create_hypertable('{table}', '{time_col}', "
                    f"if_not_exists => TRUE, migrate_data => TRUE)"
                )
            )
        conn.commit()


def drop_all_tables(engine: Engine) -> None:
    """Drop all GridSense tables.  **Destructive — test environments only.**"""
    Base.metadata.drop_all(engine)
