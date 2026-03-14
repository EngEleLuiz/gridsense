"""
Prefect flow: data ingestion.

Runs every 15 minutes (configurable via schedule).  Each execution:
  1. Fetches weather data from INMET for today
  2. Reads the solar inverter via simulated Modbus
  3. Persists both to TimescaleDB

Run manually::

    python -m pipelines.ingest_flow

Deploy as a Prefect deployment::

    prefect deploy pipelines/ingest_flow.py:ingest_flow \
        --name gridsense-ingest \
        --interval 900
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from prefect import flow, task
from prefect.logging import get_run_logger

from gridsense.db.connection import create_db_engine, get_session
from gridsense.db.models import SolarReading, WeatherReading, create_all_tables
from gridsense.ingest.modbus import SimulatedModbusReader
from gridsense.ingest.weather import INMETClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@task(
    name="fetch-weather",
    retries=2,
    retry_delay_seconds=15,
    description="Fetch today's weather from INMET API, falling back to synthetic data if unavailable.",
)
def fetch_weather(station_code: str = "A801") -> pd.DataFrame:
    task_logger = get_run_logger()
    task_logger.info("Fetching INMET weather for station %s", station_code)

    try:
        client = INMETClient(station_code=station_code)
        today = datetime.now(timezone.utc).date()
        df = client.fetch(date=str(today))

        if df.empty:
            raise ValueError("INMET returned empty dataset")

        task_logger.info("Got %d weather records from INMET.", len(df))
        return df

    except Exception as exc:
        task_logger.warning(
            "INMET API unavailable (%s) — using synthetic fallback data.", exc
        )
        # Generate a single day of synthetic data as a local fallback
        from gridsense.forecast.trainer import generate_training_data
        fallback = generate_training_data(n_days=1).tail(24).reset_index(drop=True)
        task_logger.info("Synthetic fallback: %d rows generated.", len(fallback))
        return fallback


@task(
    name="fetch-inverter",
    retries=3,
    retry_delay_seconds=30,
    description="Read current power output from the solar inverter (Modbus).",
)
def fetch_inverter(station_id: str = "floripa-01") -> dict:
    task_logger = get_run_logger()

    # Use real ModbusReader if MODBUS_HOST is set, otherwise use simulator
    host = os.environ.get("MODBUS_HOST")
    if host:
        from gridsense.ingest.modbus import ModbusReader
        port = int(os.environ.get("MODBUS_PORT", "502"))
        unit_id = int(os.environ.get("MODBUS_UNIT_ID", "1"))
        reader = ModbusReader(host=host, port=port, unit_id=unit_id, station_id=station_id)
    else:
        task_logger.info("MODBUS_HOST not set — using SimulatedModbusReader")
        reader = SimulatedModbusReader(station_id=station_id)

    reading = reader.read()
    task_logger.info(
        "Inverter reading: power=%.1f W, voltage=%.1f V, current=%.2f A",
        reading.power_w,
        reading.voltage_v,
        reading.current_a,
    )
    return reading.to_dict()


@task(
    name="save-weather-to-db",
    description="Persist weather DataFrame rows to TimescaleDB.",
)
def save_weather(df: pd.DataFrame, station_code: str = "A801") -> int:
    task_logger = get_run_logger()

    if df.empty:
        task_logger.warning("Weather DataFrame is empty — nothing to save.")
        return 0

    engine = create_db_engine()
    saved = 0

    with get_session(engine) as session:
        for _, row in df.iterrows():
            record = WeatherReading(
                time=row["timestamp"],
                station_code=station_code,
                irradiance_wm2=row.get("irradiance_wm2"),
                temp_c=row.get("temp_c"),
                humidity_pct=row.get("humidity_pct"),
            )
            session.merge(record)  # upsert on (time, station_code)
            saved += 1

    task_logger.info("Saved %d weather records.", saved)
    return saved


@task(
    name="save-inverter-to-db",
    description="Persist a single inverter reading to TimescaleDB.",
)
def save_inverter(reading_dict: dict) -> None:
    task_logger = get_run_logger()
    engine = create_db_engine()

    with get_session(engine) as session:
        record = SolarReading(
            time=datetime.fromisoformat(reading_dict["timestamp"]),
            station_id=reading_dict["station_id"],
            power_w=reading_dict["power_w"],
            voltage_v=reading_dict["voltage_v"],
            current_a=reading_dict["current_a"],
        )
        session.add(record)

    task_logger.info(
        "Saved inverter reading for station '%s'.", reading_dict["station_id"]
    )


@task(name="ensure-schema", description="Create DB tables if they don't exist yet.")
def ensure_schema() -> None:
    engine = create_db_engine()
    create_all_tables(engine)


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------


@flow(
    name="ingest-flow",
    description="Collect weather + inverter data and persist to TimescaleDB.",
    log_prints=True,
)
def ingest_flow(
    station_code: str = "A801",
    station_id: str = "floripa-01",
) -> dict:
    """Main ingestion flow.

    Parameters
    ----------
    station_code:
        INMET station code (default: Florianópolis A801).
    station_id:
        Human-readable inverter identifier stored in the DB.

    Returns
    -------
    dict
        Summary with row counts for monitoring.
    """
    # Ensure schema exists (idempotent)
    ensure_schema()

    # Run ingestion tasks (can run concurrently in a Prefect work pool)
    weather_df = fetch_weather(station_code)
    inverter_dict = fetch_inverter(station_id)

    # Persist
    n_weather = save_weather(weather_df, station_code)
    save_inverter(inverter_dict)

    return {
        "weather_records_saved": n_weather,
        "inverter_power_w": inverter_dict.get("power_w"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = ingest_flow()
    print(result)
