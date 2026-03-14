"""Unit tests for gridsense.forecast.solar and gridsense.forecast.trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridsense.forecast.solar import SolarForecaster, engineer_features
from gridsense.forecast.trainer import generate_training_data, train_and_save


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    return generate_training_data(n_days=30, seed=0)


@pytest.fixture(scope="module")
def trained_forecaster(synthetic_df: pd.DataFrame) -> SolarForecaster:
    f = SolarForecaster(n_estimators=50, random_state=0)
    f.train(synthetic_df)
    return f


@pytest.fixture(scope="module")
def conditions_df(synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """Return a 24-row slice of conditions for predict_next_24h."""
    return synthetic_df.tail(24).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


class TestFeatureEngineering:
    def test_engineer_adds_expected_columns(self, synthetic_df: pd.DataFrame) -> None:
        df = engineer_features(synthetic_df.head(10).copy())
        for col in ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "lag1_kw", "lag2_kw", "lag3_kw"]:
            assert col in df.columns, f"Missing: {col}"

    def test_hour_sin_cos_range(self, synthetic_df: pd.DataFrame) -> None:
        df = engineer_features(synthetic_df.head(100).copy())
        assert df["hour_sin"].between(-1, 1).all()
        assert df["hour_cos"].between(-1, 1).all()

    def test_lag_columns_are_zero_without_power(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC"),
                "irradiance_wm2": [100.0] * 5,
                "temp_c": [25.0] * 5,
                "humidity_pct": [60.0] * 5,
            }
        )
        out = engineer_features(df)
        assert (out["lag1_kw"] == 0).all()


# ---------------------------------------------------------------------------
# Untrained forecaster
# ---------------------------------------------------------------------------


class TestUntrainedForecaster:
    def test_predict_raises_before_training(
        self, conditions_df: pd.DataFrame
    ) -> None:
        f = SolarForecaster()
        with pytest.raises(RuntimeError, match="trained"):
            f.predict_next_24h(conditions_df)

    def test_save_raises_before_training(self, tmp_path: Path) -> None:
        f = SolarForecaster()
        with pytest.raises(RuntimeError, match="untrained"):
            f.save(tmp_path / "model.joblib")


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


class TestPredictions:
    def test_predict_returns_dataframe(
        self,
        trained_forecaster: SolarForecaster,
        conditions_df: pd.DataFrame,
    ) -> None:
        result = trained_forecaster.predict_next_24h(conditions_df)
        assert isinstance(result, pd.DataFrame)

    def test_predict_returns_24_rows(
        self,
        trained_forecaster: SolarForecaster,
        conditions_df: pd.DataFrame,
    ) -> None:
        result = trained_forecaster.predict_next_24h(conditions_df)
        assert len(result) == 24

    def test_columns_present(
        self,
        trained_forecaster: SolarForecaster,
        conditions_df: pd.DataFrame,
    ) -> None:
        result = trained_forecaster.predict_next_24h(conditions_df)
        assert set(result.columns) >= {"timestamp", "predicted_kw", "lower_bound", "upper_bound"}

    def test_confidence_interval_valid(
        self,
        trained_forecaster: SolarForecaster,
        conditions_df: pd.DataFrame,
    ) -> None:
        """lower_bound <= predicted_kw <= upper_bound for every row."""
        result = trained_forecaster.predict_next_24h(conditions_df)
        assert (result["lower_bound"] <= result["predicted_kw"]).all()
        assert (result["predicted_kw"] <= result["upper_bound"]).all()

    def test_predictions_non_negative(
        self,
        trained_forecaster: SolarForecaster,
        conditions_df: pd.DataFrame,
    ) -> None:
        result = trained_forecaster.predict_next_24h(conditions_df)
        assert (result["predicted_kw"] >= 0).all()
        assert (result["lower_bound"] >= 0).all()

    def test_predict_zero_at_night(
        self, trained_forecaster: SolarForecaster
    ) -> None:
        """Rows with irradiance == 0 must produce predicted_kw == 0."""
        night_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01 20:00", periods=24, freq="h", tz="UTC"),
                "irradiance_wm2": [0.0] * 24,
                "temp_c": [18.0] * 24,
                "humidity_pct": [70.0] * 24,
                "power_kw": [0.0] * 24,
            }
        )
        result = trained_forecaster.predict_next_24h(night_df)
        assert (result["predicted_kw"] == 0.0).all()

    def test_fewer_than_24_conditions_still_returns_24(
        self, trained_forecaster: SolarForecaster, conditions_df: pd.DataFrame
    ) -> None:
        short = conditions_df.head(5)
        result = trained_forecaster.predict_next_24h(short)
        assert len(result) == 24

    def test_timestamps_are_future(
        self,
        trained_forecaster: SolarForecaster,
        conditions_df: pd.DataFrame,
    ) -> None:
        import datetime

        result = trained_forecaster.predict_next_24h(conditions_df)
        now = datetime.datetime.now(datetime.timezone.utc)
        assert all(ts > now for ts in result["timestamp"])


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_creates_file(
        self, trained_forecaster: SolarForecaster, tmp_path: Path
    ) -> None:
        path = tmp_path / "solar.joblib"
        trained_forecaster.save(path)
        assert path.exists()

    def test_load_roundtrip(
        self,
        trained_forecaster: SolarForecaster,
        conditions_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "solar.joblib"
        trained_forecaster.save(path)
        loaded = SolarForecaster.load(path)
        r1 = trained_forecaster.predict_next_24h(conditions_df)
        r2 = loaded.predict_next_24h(conditions_df)
        pd.testing.assert_frame_equal(r1, r2, check_like=False)

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            SolarForecaster.load(tmp_path / "no_model.joblib")


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------


class TestTrainingData:
    def test_shape(self) -> None:
        df = generate_training_data(n_days=7, interval_minutes=60)
        assert len(df) == 7 * 24
        assert set(df.columns) >= {
            "timestamp", "irradiance_wm2", "temp_c", "humidity_pct", "power_kw"
        }

    def test_power_non_negative(self) -> None:
        df = generate_training_data(n_days=7)
        assert (df["power_kw"] >= 0).all()

    def test_reproducibility(self) -> None:
        df1 = generate_training_data(n_days=5, seed=123)
        df2 = generate_training_data(n_days=5, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_train_and_save(self, tmp_path: Path) -> None:
        path = tmp_path / "solar.joblib"
        forecaster = train_and_save(n_days=10, model_path=path, seed=42)
        assert path.exists()
        assert forecaster._trained
