"""
Power Quality event management.

Provides :class:`PQEventLog` for accumulating :class:`PQEvent` records
produced by the classifier and querying them by time window or label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterator

from gridsense.pq.classifier import PQResult


@dataclass
class PQEvent:
    """Immutable record of a single PQ disturbance event."""

    label: str
    confidence: float
    timestamp: datetime
    duration_ms: int | None = None
    """Approximate duration of the disturbance in milliseconds, if known."""

    @classmethod
    def from_result(
        cls,
        result: PQResult,
        duration_ms: int | None = None,
    ) -> "PQEvent":
        """Construct a :class:`PQEvent` from a :class:`PQResult`."""
        return cls(
            label=result.label,
            confidence=result.confidence,
            timestamp=result.timestamp,
            duration_ms=duration_ms,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
        }


class PQEventLog:
    """In-memory log of PQ events with filtering helpers.

    Parameters
    ----------
    max_events:
        Maximum number of events retained (oldest are dropped on overflow).
    """

    def __init__(self, max_events: int = 10_000) -> None:
        self._events: list[PQEvent] = []
        self._max = max_events

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def append(self, event: PQEvent) -> None:
        """Add a new event, dropping the oldest if over capacity."""
        if len(self._events) >= self._max:
            self._events.pop(0)
        self._events.append(event)

    def clear(self) -> None:
        """Remove all events."""
        self._events.clear()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[PQEvent]:
        return iter(self._events)

    def recent(self, hours: float = 6.0) -> list[PQEvent]:
        """Return events from the last ``hours`` hours (UTC)."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [e for e in self._events if e.timestamp >= cutoff]

    def by_label(self, label: str) -> list[PQEvent]:
        """Return all events with a specific label."""
        return [e for e in self._events if e.label == label]

    def between(
        self,
        start: datetime,
        end: datetime,
    ) -> list[PQEvent]:
        """Return events whose timestamp falls in ``[start, end)``."""
        return [e for e in self._events if start <= e.timestamp < end]

    def latest(self, n: int = 10) -> list[PQEvent]:
        """Return the ``n`` most-recent events."""
        return self._events[-n:]

    def to_dicts(self) -> list[dict[str, object]]:
        """Serialise all events to a list of plain dicts."""
        return [e.to_dict() for e in self._events]
