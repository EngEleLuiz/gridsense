"""
Prefect flow: solar generation forecasting.

Runs every 15 minutes (triggered after ingest_flow completes).
  1. Loads the most recent 24 h of weather data from TimescaleDB
  2. Runs :class:`SolarForecaster` to produce 24-hour predictions
  3. Saves forecasts to the ``solar_forecasts`` hypertable
  4. Optionally runs drift detection and logs a warning if drift is detected

Run manually::

    python -m pipelines.forecast_flow
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from prefect import flow, task
from prefect.logging import get_run_logger

from gridsense.db.connection import create_db_engine, get_session
from gridsense.db.models import SolarForecast, WeatherReading, create_all_tables
from gridsense.forecast.solar import DEFAULT_MODEL_PATH, SolarForecaster
from gridsense.forecast.trainer import generate_training_data, train_and_save

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@task(
    name="load-recent-weather",
    description="Fetch last 24 h of weather from DB, falling back to synthetic data.",
    retries=1,
    retry_delay_seconds=10,
)
def load_recent_weather(station_code: str = "A801", hours: int = 24) -> pd.DataFrame:
    task_logger = get_run_logger()

    try:
        engine = create_db_engine()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        from sqlalchemy import select

        with engine.connect() as conn:
            stmt = (
                select(WeatherReading)
                .where(WeatherReading.station_code == station_code)
                .where(WeatherReading.time >= cutoff)
                .order_by(WeatherReading.time)
            )
            rows = conn.execute(stmt).fetchall()

        if rows:
            df = pd.DataFrame(
                [
                    {
                        "timestamp": row.time,
                        "irradiance_wm2": row.irradiance_wm2 or 0.0,
                        "temp_c": row.temp_c or 20.0,
                        "humidity_pct": row.humidity_pct or 60.0,
                    }
                    for row in rows
                ]
            )
            task_logger.info("Loaded %d weather rows from DB.", len(df))
            return df

    except Exception as exc:
        task_logger.warning("DB query failed (%s) — using synthetic fallback.", exc)

    task_logger.warning("No DB data found — generating synthetic fallback.")
    fallback = generate_training_data(n_days=1).tail(24).reset_index(drop=True)
    return fallback


@task(
    name="load-or-train-model",
    description="Load the saved forecasting model, training from scratch if missing.",
)
def load_or_train_model(model_path: Path = DEFAULT_MODEL_PATH) -> SolarForecaster:
    task_logger = get_run_logger()

    if model_path.exists():
        task_logger.info("Loading model from %s", model_path)
        return SolarForecaster.load(model_path)

    task_logger.warning("Model not found at %s — training from synthetic data.", model_path)
    return train_and_save(model_path=model_path)


@task(
    name="generate-forecast",
    description="Run the solar forecaster and return a 24-row DataFrame.",
)
def generate_forecast(
    forecaster: SolarForecaster,
    conditions: pd.DataFrame,
) -> pd.DataFrame:
    task_logger = get_run_logger()
    forecast_df = forecaster.predict_next_24h(conditions)
    task_logger.info("Generated %d hourly forecasts.", len(forecast_df))
    return forecast_df


@task(
    name="save-forecasts",
    description="Persist forecast rows to TimescaleDB.",
)
def save_forecasts(df: pd.DataFrame, model_version: str = "0.1.0") -> int:
    task_logger = get_run_logger()
    engine = create_db_engine()
    generated_at = datetime.now(timezone.utc)
    saved = 0

    with get_session(engine) as session:
        for _, row in df.iterrows():
            record = SolarForecast(
                forecast_time=row["timestamp"],
                generated_at=generated_at,
                predicted_kw=row["predicted_kw"],
                lower_bound=row["lower_bound"],
                upper_bound=row["upper_bound"],
                model_version=model_version,
            )
            session.merge(record)
            saved += 1

    task_logger.info("Saved %d forecast rows.", saved)
    return saved


@task(
    name="check-drift",
    description="Run data drift detection and log a warning if drift is detected.",
)
def check_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> dict:
    task_logger = get_run_logger()

    try:
        from gridsense.forecast.monitor import DriftMonitor

        monitor = DriftMonitor(reference_data=reference_df)
        result = monitor.check(current_data=current_df)

        if result["drift_detected"]:
            task_logger.warning(
                "⚠️  Data drift detected in features: %s  (report: %s)",
                result["drifted_features"],
                result["report_path"],
            )
        else:
            task_logger.info("✓ No data drift detected.")

        return result

    except Exception as exc:
        task_logger.warning("Drift check failed: %s", exc)
        return {"drift_detected": False, "drifted_features": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------


@flow(
    name="forecast-flow",
    description="Generate 24-hour solar generation forecasts and save to DB.",
    log_prints=True,
)
def forecast_flow(
    station_code: str = "A801",
    model_path: Path = DEFAULT_MODEL_PATH,
    run_drift_check: bool = True,
) -> dict:
    """Main forecast flow.

    Parameters
    ----------
    station_code:
        INMET station to source weather data from.
    model_path:
        Path to the trained model artifact.
    run_drift_check:
        Whether to run Evidently drift detection (default True).

    Returns
    -------
    dict
        Run summary including forecast count and drift status.
    """
    ensure_schema_task()

    # Load data and model
    weather_df = load_recent_weather(station_code)
    forecaster = load_or_train_model(model_path)

    # Generate and persist forecasts
    forecast_df = generate_forecast(forecaster, weather_df)
    n_saved = save_forecasts(forecast_df)

    # Optional drift check (uses training data as reference)
    drift_result: dict = {}
    if run_drift_check:
        reference_df = generate_training_data(n_days=30)
        drift_result = check_drift(reference_df, weather_df)

    return {
        "forecasts_saved": n_saved,
        "drift_detected": drift_result.get("drift_detected", False),
        "drifted_features": drift_result.get("drifted_features", []),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


@task(name="ensure-schema")
def ensure_schema_task() -> None:
    engine = create_db_engine()
    create_all_tables(engine)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = forecast_flow()
    print(result)
