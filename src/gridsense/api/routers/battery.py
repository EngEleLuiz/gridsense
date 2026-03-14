"""Battery SoC endpoint."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from gridsense.api.schemas import SoCResponse

router = APIRouter(prefix="/battery", tags=["battery"])

# In-memory state — replaced by DB query when available
_LAST_SOC: float = 1.0
_LAST_UPDATED: datetime = datetime.now(timezone.utc)


@router.get("/soc", response_model=SoCResponse)
async def get_soc() -> SoCResponse:
    """Return the most recent battery State-of-Charge estimate.

    Reads from TimescaleDB when available; falls back to the last known
    in-memory value when the database is offline.
    """
    soc = _LAST_SOC
    updated_at = _LAST_UPDATED

    try:
        from sqlalchemy import select

        from gridsense.db.connection import create_db_engine
        from gridsense.db.models import BatterySoC

        engine = create_db_engine()
        with engine.connect() as conn:
            stmt = select(BatterySoC).order_by(BatterySoC.time.desc()).limit(1)
            row = conn.execute(stmt).fetchone()

        if row:
            soc = row.soc or _LAST_SOC
            updated_at = row.time
    except Exception:
        pass  # Return cached value if DB is unavailable

    return SoCResponse(
        soc=soc,
        soc_percent=soc * 100.0,
        updated_at=updated_at,
    )
