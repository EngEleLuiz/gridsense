"""
Integration tests for the FastAPI application.

These tests run against the actual ASGI app using httpx.AsyncClient —
no real database or Modbus hardware required.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from gridsense import __version__
from gridsense.api.main import app

# ---------------------------------------------------------------------------
# Shared async client fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_healthz_returns_200(client: AsyncClient) -> None:
    resp = await client.get("/healthz")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_healthz_returns_ok(client: AsyncClient) -> None:
    data = (await client.get("/healthz")).json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_healthz_contains_version(client: AsyncClient) -> None:
    data = (await client.get("/healthz")).json()
    assert data["version"] == __version__


# ---------------------------------------------------------------------------
# Solar forecast
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_solar_returns_200(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/predict/solar",
        json={"station_code": "A801", "horizon_hours": 24},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_predict_solar_returns_24_predictions(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/predict/solar",
        json={"station_code": "A801", "horizon_hours": 24},
    )
    data = resp.json()
    assert len(data["predictions"]) == 24


@pytest.mark.asyncio
async def test_predict_solar_prediction_fields(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/predict/solar",
        json={"station_code": "A801", "horizon_hours": 24},
    )
    item = resp.json()["predictions"][0]
    assert set(item.keys()) >= {"timestamp", "predicted_kw", "lower_bound", "upper_bound"}


@pytest.mark.asyncio
async def test_predict_solar_horizon_respected(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/predict/solar",
        json={"station_code": "A801", "horizon_hours": 12},
    )
    assert len(resp.json()["predictions"]) == 12


@pytest.mark.asyncio
async def test_predict_solar_invalid_station_returns_422(client: AsyncClient) -> None:
    """station_code must be a non-empty string — empty string → 422."""
    resp = await client.post(
        "/api/v1/predict/solar",
        json={"station_code": "", "horizon_hours": 24},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_predict_solar_invalid_horizon_returns_422(client: AsyncClient) -> None:
    """horizon_hours must be between 1 and 48."""
    resp = await client.post(
        "/api/v1/predict/solar",
        json={"station_code": "A801", "horizon_hours": 100},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# PQ events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_pq_events_returns_200(client: AsyncClient) -> None:
    resp = await client.get(
        "/api/v1/events/pq",
        params={"start": "2025-12-01T00:00:00Z", "end": "2025-12-02T00:00:00Z"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_get_pq_events_response_shape(client: AsyncClient) -> None:
    resp = await client.get(
        "/api/v1/events/pq",
        params={"start": "2025-12-01T00:00:00Z", "end": "2025-12-02T00:00:00Z"},
    )
    data = resp.json()
    assert "events" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_get_pq_events_invalid_range_returns_422(client: AsyncClient) -> None:
    """end before start must return 422."""
    resp = await client.get(
        "/api/v1/events/pq",
        params={"start": "2025-12-02T00:00:00Z", "end": "2025-12-01T00:00:00Z"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Battery SoC
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_soc_returns_200(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/battery/soc")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_get_soc_fields(client: AsyncClient) -> None:
    data = (await client.get("/api/v1/battery/soc")).json()
    assert set(data.keys()) >= {"soc", "soc_percent", "updated_at"}


@pytest.mark.asyncio
async def test_get_soc_range(client: AsyncClient) -> None:
    data = (await client.get("/api/v1/battery/soc")).json()
    assert 0.0 <= data["soc"] <= 1.0
    assert 0.0 <= data["soc_percent"] <= 100.0
