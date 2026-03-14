"""
Solar model training utilities.

Provides :func:`generate_training_data` for creating a synthetic solar
dataset (used in tests and local development when no historical DB data
is available) and :func:`train_and_save` as the single entrypoint called
by the Prefect ``retrain_flow``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from gridsense.forecast.solar import DEFAULT_MODEL_PATH, SolarForecaster

logger = logging.getLogger(__name__)


def generate_training_data(
    n_days: int = 90,
    interval_minutes: int = 60,
    peak_kw: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic hourly solar generation dataset.

    Produces a realistic clear-sky curve modulated by random cloud cover
    and seasonal variation, covering ``n_days`` of hourly observations.

    Parameters
    ----------
    n_days:
        Number of days of history to generate.
    interval_minutes:
        Sampling interval in minutes (default 60 = hourly).
    peak_kw:
        Peak generation capacity in kW.
    seed:
        NumPy random seed.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp``, ``irradiance_wm2``, ``temp_c``,
        ``humidity_pct``, ``power_kw``.
    """
    rng = np.random.default_rng(seed)

    steps_per_day = 24 * 60 // interval_minutes
    n_steps = n_days * steps_per_day

    # Build timestamp index
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=n_days)
    timestamps = [
        start + timedelta(minutes=i * interval_minutes) for i in range(n_steps)
    ]

    hours = np.array([ts.hour + ts.minute / 60.0 for ts in timestamps])
    doys = np.array([ts.timetuple().tm_yday for ts in timestamps], dtype=float)

    # Seasonal irradiance envelope (stronger in summer)
    season_factor = 0.85 + 0.15 * np.cos(2 * np.pi * (doys - 172) / 365)

    # Solar elevation angle (simplified)
    solar_angle = np.pi * np.clip((hours - 6.0) / 12.0, 0.0, 1.0)
    clear_sky = 1000.0 * season_factor * np.sin(solar_angle) ** 2

    # Cloud cover attenuation (0.4–1.0 random per-hour)
    cloud_atten = rng.uniform(0.4, 1.0, n_steps)
    irradiance = np.clip(clear_sky * cloud_atten, 0.0, None)

    # Temperature: daily cycle 15–32 °C with random noise
    temp = 23.5 + 8.5 * np.sin(np.pi * (hours - 6) / 12) + rng.normal(0, 1.5, n_steps)

    # Humidity: inversely correlated with temperature
    humidity = np.clip(70.0 - 0.8 * (temp - 23.5) + rng.normal(0, 5, n_steps), 10, 100)

    # Power proportional to irradiance with panel efficiency + noise
    efficiency = 0.18 + rng.normal(0, 0.005, n_steps)
    area_m2 = peak_kw * 1000 / (1000 * 0.18)  # ~27 m² for 5 kW at std efficiency
    power_kw = np.clip(irradiance * efficiency * area_m2 / 1000.0, 0.0, peak_kw)
    power_kw += rng.normal(0, 0.02, n_steps)
    power_kw = np.clip(power_kw, 0.0, peak_kw)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "irradiance_wm2": irradiance.round(2),
            "temp_c": temp.round(2),
            "humidity_pct": humidity.round(2),
            "power_kw": power_kw.round(4),
        }
    )


def train_and_save(
    df: pd.DataFrame | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    n_days: int = 90,
    seed: int = 42,
) -> SolarForecaster:
    """Train a :class:`SolarForecaster` and persist it to disk.

    Parameters
    ----------
    df:
        Training data.  If ``None``, synthetic data is generated via
        :func:`generate_training_data`.
    model_path:
        Where to save the artifact.
    n_days:
        Used only when ``df`` is ``None``.
    seed:
        RNG seed for synthetic data generation.

    Returns
    -------
    SolarForecaster
        The trained forecaster (already saved to ``model_path``).
    """
    if df is None:
        logger.info("No training data provided — generating %d days of synthetic data.", n_days)
        df = generate_training_data(n_days=n_days, seed=seed)

    logger.info("Training SolarForecaster on %d rows.", len(df))
    forecaster = SolarForecaster()
    forecaster.train(df)
    forecaster.save(model_path)
    logger.info("Model saved to %s", model_path)
    return forecaster
