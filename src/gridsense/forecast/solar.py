"""
Solar generation forecasting model.

Wraps a scikit-learn gradient-boosting regressor trained on:
  - Global horizontal irradiance (W/m²)
  - Ambient temperature (°C)
  - Relative humidity (%)
  - Hour of day (sin/cos encoding)
  - Day of year (sin/cos encoding)
  - Lagged power readings (last 3 intervals, i.e. last 45 min at 15-min cadence)

Outputs 24 hourly predictions with a ±1 std bootstrap confidence interval.

Refactored from ``Solar-Generation-Forecasting-Pipeline``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("artifacts") / "solar_model.joblib"

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "irradiance_wm2",
    "temp_c",
    "humidity_pct",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "lag1_kw",
    "lag2_kw",
    "lag3_kw",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time encodings and lag features to a weather DataFrame.

    Parameters
    ----------
    df:
        Must contain at minimum: ``timestamp`` (tz-aware), ``irradiance_wm2``,
        ``temp_c``, ``humidity_pct``, and ``power_kw`` (for lag features).

    Returns
    -------
    pd.DataFrame
        Original columns plus the engineered features in :data:`FEATURE_COLS`.
    """
    df = df.copy()
    ts: pd.DatetimeIndex = pd.DatetimeIndex(df["timestamp"])

    # Cyclical hour encoding
    df["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)

    # Cyclical day-of-year encoding
    df["doy_sin"] = np.sin(2 * np.pi * ts.day_of_year / 365)
    df["doy_cos"] = np.cos(2 * np.pi * ts.day_of_year / 365)

    # Lag features (assume 15-min interval → 3 lags = 45 min look-back)
    if "power_kw" in df.columns:
        df["lag1_kw"] = df["power_kw"].shift(1).fillna(0.0)
        df["lag2_kw"] = df["power_kw"].shift(2).fillna(0.0)
        df["lag3_kw"] = df["power_kw"].shift(3).fillna(0.0)
    else:
        df["lag1_kw"] = 0.0
        df["lag2_kw"] = 0.0
        df["lag3_kw"] = 0.0

    return df


# ---------------------------------------------------------------------------
# Prediction result
# ---------------------------------------------------------------------------


@dataclass
class HourlyForecast:
    """One-hour generation forecast entry."""

    timestamp: datetime
    predicted_kw: float
    lower_bound: float
    upper_bound: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "predicted_kw": round(self.predicted_kw, 4),
            "lower_bound": round(self.lower_bound, 4),
            "upper_bound": round(self.upper_bound, 4),
        }


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------


class SolarForecaster:
    """24-hour solar generation forecaster.

    Parameters
    ----------
    n_estimators:
        Number of boosting stages.
    random_state:
        Seed for reproducibility.
    confidence_std_multiplier:
        How many standard deviations define the confidence interval (default 1).

    Usage — training::

        forecaster = SolarForecaster()
        forecaster.train(historical_df)          # DataFrame with power_kw column
        forecaster.save()

    Usage — inference::

        forecaster = SolarForecaster.load()
        forecast_df = forecaster.predict_next_24h(conditions_df)
    """

    MODEL_VERSION = "0.1.0"

    def __init__(
        self,
        n_estimators: int = 300,
        random_state: int = 42,
        confidence_std_multiplier: float = 1.0,
    ) -> None:
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "reg",
                    GradientBoostingRegressor(
                        n_estimators=n_estimators,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        self._std_mult = confidence_std_multiplier
        self._residual_std: float = 0.0
        self._trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> None:
        """Fit the model on historical weather + generation data.

        Parameters
        ----------
        df:
            DataFrame with columns: ``timestamp``, ``irradiance_wm2``,
            ``temp_c``, ``humidity_pct``, ``power_kw``.
        """
        df = engineer_features(df)
        X = df[FEATURE_COLS].values
        y = df["power_kw"].values

        self._pipeline.fit(X, y)

        # Estimate residual std for confidence intervals
        y_hat: NDArray[np.floating] = self._pipeline.predict(X)
        self._residual_std = float(np.std(y - y_hat))
        self._trained = True
        logger.info(
            "SolarForecaster trained. Residual std=%.4f kW", self._residual_std
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_next_24h(
        self,
        conditions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate an hourly forecast for the next 24 hours.

        Parameters
        ----------
        conditions:
            DataFrame with at least: ``timestamp``, ``irradiance_wm2``,
            ``temp_c``, ``humidity_pct``.  If ``power_kw`` is present, it
            is used for lag features; otherwise lags default to zero.

        Returns
        -------
        pd.DataFrame
            24-row DataFrame with columns: ``timestamp``, ``predicted_kw``,
            ``lower_bound``, ``upper_bound``.

        Raises
        ------
        RuntimeError
            If called before :meth:`train` or :meth:`load`.
        """
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call .train() or .load() first.")

        df = engineer_features(conditions.copy())

        # Use up to 24 rows; if fewer, pad with last row repeated
        if len(df) < 24:
            padding = pd.concat(
                [df.iloc[[-1]]] * (24 - len(df)), ignore_index=True
            )
            df = pd.concat([df, padding], ignore_index=True)

        df = df.iloc[:24].copy()

        # Build timestamps anchored to the next hour boundary
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        timestamps = [now + timedelta(hours=i + 1) for i in range(24)]
        df["timestamp"] = timestamps

        df = engineer_features(df)
        X = df[FEATURE_COLS].values
        predictions: NDArray[np.floating] = self._pipeline.predict(X)

        # Night-time clamp: irradiance == 0 → force prediction to 0
        night_mask = df["irradiance_wm2"].values <= 0
        predictions[night_mask] = 0.0

        predictions = np.clip(predictions, 0.0, None)
        std = self._residual_std * self._std_mult

        rows = []
        for i, (ts, pred) in enumerate(zip(timestamps, predictions)):
            lower = max(0.0, pred - std)
            upper = pred + std
            rows.append(
                {
                    "timestamp": ts,
                    "predicted_kw": float(pred),
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path = DEFAULT_MODEL_PATH) -> None:
        """Save the trained model to a joblib artifact.

        Parameters
        ----------
        path:
            Destination file.  Parent directories are created automatically.

        Raises
        ------
        RuntimeError
            If the model has not been trained.
        """
        if not self._trained:
            raise RuntimeError("Cannot save an untrained model.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self._pipeline,
                "residual_std": self._residual_std,
                "version": self.MODEL_VERSION,
                "std_mult": self._std_mult,
            },
            path,
        )
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL_PATH) -> SolarForecaster:
        """Load a pre-trained model from a joblib artifact.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Solar model artifact not found at '{path}'. "
                "Run the training pipeline first."
            )
        payload: dict = joblib.load(path)
        instance = cls(confidence_std_multiplier=payload.get("std_mult", 1.0))
        instance._pipeline = payload["pipeline"]
        instance._residual_std = payload.get("residual_std", 0.0)
        instance._trained = True
        return instance
