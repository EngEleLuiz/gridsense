"""
GridSense FastAPI application.

Endpoints
---------
POST /api/v1/predict/solar     — 24-hour solar generation forecast
GET  /api/v1/events/pq         — power quality event history
GET  /api/v1/battery/soc       — latest battery State-of-Charge
GET  /healthz                  — liveness probe

Run locally::

    uvicorn gridsense.api.main:app --reload --port 8000

Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gridsense import __version__
from gridsense.api.routers import battery, events, forecast
from gridsense.api.schemas import HealthResponse

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GridSense API",
    description=(
        "Real-time solar generation forecasting and power quality monitoring. "
        "Source: https://github.com/YOUR_USERNAME/gridsense"
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow the Streamlit dashboard (and local dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

API_PREFIX = "/api/v1"

app.include_router(forecast.router, prefix=API_PREFIX)
app.include_router(events.router, prefix=API_PREFIX)
app.include_router(battery.router, prefix=API_PREFIX)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/healthz", response_model=HealthResponse, tags=["meta"])
async def healthz() -> HealthResponse:
    """Liveness probe — always returns 200 OK if the process is running."""
    return HealthResponse(status="ok", version=__version__)


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "message": "GridSense API is running.",
        "docs": "/docs",
        "health": "/healthz",
        "version": __version__,
    }
