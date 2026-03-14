"""
INMET (Instituto Nacional de Meteorologia) weather data client.

Fetches hourly meteorological observations from the public INMET REST API
for a given station and date range.  Returns a tidy pandas DataFrame with
columns: ``timestamp``, ``irradiance_wm2``, ``temp_c``, ``humidity_pct``.

API docs: https://portal.inmet.gov.br/manual/manual-de-uso-da-api-de-estações

Station codes used in GridSense
---------------------------------
A801 — Florianópolis, SC
A652 — São Paulo (Mirante de Santana), SP
A002 — Brasília, DF
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Base URL — no auth required
_BASE_URL = "https://apitempo.inmet.gov.br"
_TIMEOUT_S = 15


# ---------------------------------------------------------------------------
# Column mappings
# ---------------------------------------------------------------------------

# INMET field → our internal column name
_COLUMN_MAP: dict[str, str] = {
    "DT_MEDICAO": "_date",
    "HR_MEDICAO": "_hour",
    "RAD_GLO": "irradiance_wm2",      # Global radiation (kJ/m²) → will convert
    "TEM_INS": "temp_c",              # Instant temperature (°C)
    "UMD_INS": "humidity_pct",        # Instant relative humidity (%)
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class INMETClient:
    """Fetch hourly weather observations from INMET for a given station.

    Parameters
    ----------
    station_code:
        INMET automatic station code (e.g. ``"A801"`` for Florianópolis).
    session:
        Optional :class:`requests.Session` for connection reuse / mocking.
    timeout:
        HTTP request timeout in seconds.

    Examples
    --------
    >>> client = INMETClient(station_code="A801")
    >>> df = client.fetch(date="2025-12-01")
    >>> df.columns.tolist()
    ['timestamp', 'irradiance_wm2', 'temp_c', 'humidity_pct']
    """

    def __init__(
        self,
        station_code: str = "A801",
        session: requests.Session | None = None,
        timeout: int = _TIMEOUT_S,
    ) -> None:
        self.station_code = station_code
        self._session = session or requests.Session()
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch hourly weather data for a single day or a date range.

        Pass either ``date`` (single day) **or** ``start_date`` + ``end_date``.

        Parameters
        ----------
        date:
            ISO date string ``"YYYY-MM-DD"`` for a single day.
        start_date:
            ISO date string for the start of the range (inclusive).
        end_date:
            ISO date string for the end of the range (inclusive).

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp`` (tz-aware UTC), ``irradiance_wm2``,
            ``temp_c``, ``humidity_pct``.
            Missing values are forward-filled, then back-filled.

        Raises
        ------
        ValueError
            If neither ``date`` nor ``start_date``/``end_date`` are provided,
            or if the date range is invalid.
        requests.HTTPError
            If the INMET API returns a non-2xx status.
        """
        sd, ed = self._resolve_dates(date, start_date, end_date)
        raw = self._fetch_raw(sd, ed)
        return self._parse(raw)

    def fetch_last_n_days(self, n: int = 7) -> pd.DataFrame:
        """Convenience wrapper — fetch the last ``n`` days up to yesterday."""
        today = datetime.now(timezone.utc).date()
        end = today - timedelta(days=1)
        start = end - timedelta(days=n - 1)
        return self.fetch(start_date=str(start), end_date=str(end))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_dates(
        self,
        date_: str | None,
        start_date: str | None,
        end_date: str | None,
    ) -> tuple[str, str]:
        if date_ is not None:
            return date_, date_
        if start_date and end_date:
            if start_date > end_date:
                raise ValueError(
                    f"start_date ({start_date}) must be <= end_date ({end_date})."
                )
            return start_date, end_date
        raise ValueError(
            "Provide either 'date' (single day) or both 'start_date' and 'end_date'."
        )

    def _fetch_raw(self, start: str, end: str) -> list[dict[str, Any]]:
        url = f"{_BASE_URL}/estacao/{start}/{end}/{self.station_code}"
        logger.debug("GET %s", url)
        response = self._session.get(url, timeout=self._timeout)
        response.raise_for_status()

        body = response.text.strip()
        if not body:
            raise ValueError(
                f"INMET API returned an empty response for station "
                f"{self.station_code} on {start}. "
                f"The API may be temporarily unavailable or the station code is invalid."
            )

        try:
            data: list[dict[str, Any]] = response.json()
        except Exception as exc:
            raise ValueError(
                f"INMET API returned non-JSON for station {self.station_code}: "
                f"status={response.status_code}, body_preview={body[:200]!r}"
            ) from exc

        if not isinstance(data, list):
            raise ValueError(
                f"Unexpected INMET response format for station {self.station_code}."
            )
        return data

    def _parse(self, raw: list[dict[str, Any]]) -> pd.DataFrame:
        if not raw:
            return self._empty_df()

        rows = []
        for record in raw:
            row = self._parse_record(record)
            if row is not None:
                rows.append(row)

        if not rows:
            return self._empty_df()

        df = pd.DataFrame(rows)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Handle missing values: forward-fill, then back-fill
        numeric_cols = ["irradiance_wm2", "temp_c", "humidity_pct"]
        df[numeric_cols] = (
            df[numeric_cols]
            .ffill()
            .bfill()
        )

        return df

    def _parse_record(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Parse one INMET JSON record into a clean row dict."""
        try:
            date_str: str = record["DT_MEDICAO"]          # "2025-12-01"
            hour_str: str = str(record["HR_MEDICAO"]).zfill(4)  # "1400" → "1400"
            hour_int = int(hour_str[:2])

            ts = datetime.strptime(f"{date_str} {hour_int:02d}:00:00", "%Y-%m-%d %H:%M:%S")
            ts = ts.replace(tzinfo=timezone.utc)

            def _float(key: str) -> float | None:
                val = record.get(key)
                try:
                    return float(val) if val not in (None, "", "null") else None
                except (TypeError, ValueError):
                    return None

            # INMET reports global radiation in kJ/m²/h → convert to W/m²
            rad_kj = _float("RAD_GLO")
            irradiance = (rad_kj * 1000.0 / 3600.0) if rad_kj is not None else None

            return {
                "timestamp": ts,
                "irradiance_wm2": irradiance,
                "temp_c": _float("TEM_INS"),
                "humidity_pct": _float("UMD_INS"),
            }

        except (KeyError, ValueError) as exc:
            logger.warning("Skipping malformed INMET record: %s — %s", record, exc)
            return None

    @staticmethod
    def _empty_df() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["timestamp", "irradiance_wm2", "temp_c", "humidity_pct"]
        )
