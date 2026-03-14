"""Solar forecast endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from gridsense.api.schemas import ForecastRequest, ForecastResponse, HourlyForecastItem
from gridsense.forecast.solar import DEFAULT_MODEL_PATH, SolarForecaster
from gridsense.forecast.trainer import generate_training_data

router = APIRouter(prefix="/predict", tags=["forecast"])


@router.post("/solar", response_model=ForecastResponse)
async def predict_solar(request: ForecastRequest) -> ForecastResponse:
    """Generate an hourly solar generation forecast.

    Loads the pre-trained model from disk and runs inference against the
    latest synthetic/real conditions.  Returns up to 48 hourly predictions.
    """
    try:
        forecaster = SolarForecaster.load(DEFAULT_MODEL_PATH)
    except FileNotFoundError:
        # Auto-train on first call if no artifact exists yet
        from gridsense.forecast.trainer import train_and_save
        forecaster = train_and_save()

    # Use synthetic conditions as a fallback when DB is not available
    conditions = generate_training_data(n_days=1).tail(request.horizon_hours)

    try:
        df = forecaster.predict_next_24h(conditions)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {exc}") from exc

    items = [
        HourlyForecastItem(
            timestamp=row["timestamp"],
            predicted_kw=row["predicted_kw"],
            lower_bound=row["lower_bound"],
            upper_bound=row["upper_bound"],
        )
        for _, row in df.head(request.horizon_hours).iterrows()
    ]

    return ForecastResponse(
        predictions=items,
        model_version=SolarForecaster.MODEL_VERSION,
        station_code=request.station_code,
        generated_at=datetime.now(timezone.utc),
    )
