"""Power quality event endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from gridsense.api.schemas import PQEventItem, PQEventsResponse

router = APIRouter(prefix="/events", tags=["power-quality"])


@router.get("/pq", response_model=PQEventsResponse)
async def get_pq_events(
    start: datetime = Query(..., description="Start of query window (ISO 8601)."),
    end: datetime = Query(..., description="End of query window (ISO 8601)."),
) -> PQEventsResponse:
    """Return power quality disturbance events within a time window.

    Events are read from TimescaleDB when available, or an empty list is
    returned gracefully when the database is offline.
    """
    if end <= start:
        raise HTTPException(
            status_code=422,
            detail="'end' must be after 'start'.",
        )

    events: list[PQEventItem] = []

    try:
        from gridsense.db.connection import create_db_engine
        from gridsense.db.models import PQEvent
        from sqlalchemy import select

        engine = create_db_engine()
        with engine.connect() as conn:
            stmt = (
                select(PQEvent)
                .where(PQEvent.time >= start)
                .where(PQEvent.time < end)
                .order_by(PQEvent.time)
            )
            rows = conn.execute(stmt).fetchall()

        events = [
            PQEventItem(
                timestamp=row.time,
                label=row.label,
                confidence=row.confidence or 0.0,
                duration_ms=row.duration_ms,
            )
            for row in rows
        ]
    except Exception:
        # DB not available — return empty list, not a 500
        events = []

    return PQEventsResponse(
        events=events,
        total=len(events),
        start=start,
        end=end,
    )
