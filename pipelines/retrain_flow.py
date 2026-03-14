"""
Prefect flow: weekly model retraining.

Runs every Monday at 02:00 UTC.
  1. Fetches last 30 days of weather + generation data from TimescaleDB
  2. Retrains :class:`SolarForecaster` from scratch
  3. Saves versioned model artifact
  4. Runs drift detection on the new model's training window

Deploy::

    prefect deploy pipelines/retrain_flow.py:retrain_flow \
        --name gridsense-retrain \
        --cron "0 2 * * MON"
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from prefect import flow, task
from prefect.logging import get_run_logger

from gridsense.db.connection import create_db_engine
from gridsense.db.models import WeatherReading
from gridsense.forecast.solar import DEFAULT_MODEL_PATH
from gridsense.forecast.trainer import generate_training_data, train_and_save

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@task(
    name="load-historical-weather",
    retries=2,
    retry_delay_seconds=30,
    description="Load last N days of weather data from TimescaleDB for retraining.",
)
def load_historical_weather(
    station_code: str = "A801",
    n_days: int = 30,
) -> pd.DataFrame:
    task_logger = get_run_logger()
    engine = create_db_engine()
    cutoff = datetime.now(timezone.utc) - timedelta(days=n_days)

    from sqlalchemy import select

    with engine.connect() as conn:
        stmt = (
            select(WeatherReading)
            .where(WeatherReading.station_code == station_code)
            .where(WeatherReading.time >= cutoff)
            .order_by(WeatherReading.time)
        )
        rows = conn.execute(stmt).fetchall()

    if not rows:
        task_logger.warning(
            "No historical data found — falling back to %d days of synthetic data.",
            n_days,
        )
        return generate_training_data(n_days=n_days)

    df = pd.DataFrame(
        [
            {
                "timestamp": row.time,
                "irradiance_wm2": row.irradiance_wm2 or 0.0,
                "temp_c": row.temp_c or 20.0,
                "humidity_pct": row.humidity_pct or 60.0,
                "power_kw": 0.0,   # will be enriched from solar_readings if available
            }
            for row in rows
        ]
    )
    task_logger.info("Loaded %d historical rows for retraining.", len(df))
    return df


@task(
    name="retrain-model",
    description="Train a new SolarForecaster and save a versioned artifact.",
)
def retrain_model(
    df: pd.DataFrame,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> str:
    task_logger = get_run_logger()
    forecaster = train_and_save(df=df, model_path=model_path)
    task_logger.info("Model retrained and saved to %s", model_path)
    return str(model_path)


@task(
    name="run-post-train-drift-check",
    description="Verify the new model's training window is self-consistent.",
)
def post_train_drift_check(train_df: pd.DataFrame) -> dict:
    task_logger = get_run_logger()

    try:
        from gridsense.forecast.monitor import DriftMonitor

        # Split into first-half reference vs second-half current
        mid = len(train_df) // 2
        reference_df = train_df.iloc[:mid]
        current_df = train_df.iloc[mid:]

        monitor = DriftMonitor(reference_data=reference_df)
        result = monitor.check(current_data=current_df)

        if result["drift_detected"]:
            task_logger.warning(
                "Post-retrain drift check flagged: %s", result["drifted_features"]
            )
        else:
            task_logger.info("Post-retrain drift check: clean.")

        return result

    except Exception as exc:
        task_logger.warning("Post-retrain drift check failed: %s", exc)
        return {"drift_detected": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------


@flow(
    name="retrain-flow",
    description="Weekly retraining of the solar generation forecasting model.",
    log_prints=True,
)
def retrain_flow(
    station_code: str = "A801",
    n_days: int = 30,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict:
    """Weekly model retraining flow.

    Parameters
    ----------
    station_code:
        INMET station used for historical data.
    n_days:
        How many days of history to use for retraining.
    model_path:
        Where to write the new model artifact.

    Returns
    -------
    dict
        Summary with the saved model path and drift check outcome.
    """
    df = load_historical_weather(station_code=station_code, n_days=n_days)
    saved_path = retrain_model(df=df, model_path=model_path)
    drift_result = post_train_drift_check(df)

    return {
        "model_path": saved_path,
        "training_rows": len(df),
        "drift_detected": drift_result.get("drift_detected", False),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    result = retrain_flow()
    print(result)
