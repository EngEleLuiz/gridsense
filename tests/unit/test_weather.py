"""Unit tests for gridsense.ingest.weather."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from gridsense.ingest.weather import INMETClient


# ---------------------------------------------------------------------------
# Fixture: realistic INMET API response
# ---------------------------------------------------------------------------

INMET_FIXTURE = [
    {
        "DT_MEDICAO": "2025-12-01",
        "HR_MEDICAO": "1200",
        "RAD_GLO": "2160.0",   # kJ/m²/h → 600 W/m²
        "TEM_INS": "28.4",
        "UMD_INS": "65.0",
    },
    {
        "DT_MEDICAO": "2025-12-01",
        "HR_MEDICAO": "1300",
        "RAD_GLO": "2520.0",   # → 700 W/m²
        "TEM_INS": "29.1",
        "UMD_INS": "62.0",
    },
    {
        "DT_MEDICAO": "2025-12-01",
        "HR_MEDICAO": "1400",
        "RAD_GLO": None,        # missing value
        "TEM_INS": "29.5",
        "UMD_INS": "60.0",
    },
]


def _mock_session(json_data: list) -> MagicMock:
    """Return a fake requests.Session whose .get() returns json_data."""
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = json_data

    session = MagicMock()
    session.get.return_value = response
    return session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestINMETClientFetch:
    def test_fetch_returns_dataframe(self) -> None:
        client = INMETClient(station_code="A801", session=_mock_session(INMET_FIXTURE))
        df = client.fetch(date="2025-12-01")
        assert isinstance(df, pd.DataFrame)

    def test_fetch_dataframe_shape(self) -> None:
        client = INMETClient(station_code="A801", session=_mock_session(INMET_FIXTURE))
        df = client.fetch(date="2025-12-01")
        # 3 records in fixture
        assert len(df) == 3

    def test_fetch_column_names(self) -> None:
        client = INMETClient(station_code="A801", session=_mock_session(INMET_FIXTURE))
        df = client.fetch(date="2025-12-01")
        assert set(df.columns) >= {"timestamp", "irradiance_wm2", "temp_c", "humidity_pct"}

    def test_fetch_irradiance_conversion(self) -> None:
        """2160 kJ/m²/h should convert to 600 W/m²."""
        client = INMETClient(station_code="A801", session=_mock_session(INMET_FIXTURE))
        df = client.fetch(date="2025-12-01")
        assert df["irradiance_wm2"].iloc[0] == pytest.approx(600.0, abs=1.0)

    def test_fetch_temperature(self) -> None:
        client = INMETClient(station_code="A801", session=_mock_session(INMET_FIXTURE))
        df = client.fetch(date="2025-12-01")
        assert df["temp_c"].iloc[0] == pytest.approx(28.4)

    def test_handles_missing_irradiance(self) -> None:
        """NaN irradiance values must be filled (forward-fill or back-fill)."""
        client = INMETClient(station_code="A801", session=_mock_session(INMET_FIXTURE))
        df = client.fetch(date="2025-12-01")
        assert df["irradiance_wm2"].isna().sum() == 0

    def test_timestamp_is_tz_aware(self) -> None:
        client = INMETClient(station_code="A801", session=_mock_session(INMET_FIXTURE))
        df = client.fetch(date="2025-12-01")
        assert df["timestamp"].dt.tz is not None

    def test_empty_response_returns_empty_dataframe(self) -> None:
        client = INMETClient(station_code="A801", session=_mock_session([]))
        df = client.fetch(date="2025-12-01")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "timestamp" in df.columns


class TestINMETClientDateValidation:
    def test_single_date_works(self) -> None:
        client = INMETClient(station_code="A801", session=_mock_session(INMET_FIXTURE))
        df = client.fetch(date="2025-12-01")
        assert len(df) > 0

    def test_date_range_works(self) -> None:
        client = INMETClient(station_code="A801", session=_mock_session(INMET_FIXTURE))
        df = client.fetch(start_date="2025-12-01", end_date="2025-12-03")
        assert isinstance(df, pd.DataFrame)

    def test_no_date_raises(self) -> None:
        client = INMETClient(station_code="A801", session=_mock_session([]))
        with pytest.raises(ValueError, match="date"):
            client.fetch()

    def test_inverted_range_raises(self) -> None:
        client = INMETClient(station_code="A801", session=_mock_session([]))
        with pytest.raises(ValueError, match="start_date"):
            client.fetch(start_date="2025-12-10", end_date="2025-12-01")


class TestINMETClientHTTPError:
    def test_http_error_propagates(self) -> None:
        import requests

        response = MagicMock()
        response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        session = MagicMock()
        session.get.return_value = response

        client = INMETClient(station_code="XXXX", session=session)
        with pytest.raises(requests.HTTPError):
            client.fetch(date="2025-12-01")
