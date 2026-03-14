"""
Modbus RTU/TCP reader for solar inverter data.

Connects to a real or simulated inverter and reads three holding registers:
  - Active power (W)
  - AC voltage (V)
  - AC current (A)

Register map follows the SMA Sunny Boy / Fronius Primo convention
(configurable via :class:`ModbusRegisterMap`).

Refactored from ``modbus-mqtt-iot-gateway``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Register map
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModbusRegisterMap:
    """Holding-register addresses for a single-phase inverter.

    All addresses are 0-based (pymodbus convention).
    Scale factors convert raw 16-bit integers to engineering units.
    """

    power_register: int = 40083      # Active power, scale 0.1 W
    power_scale: float = 0.1
    voltage_register: int = 40085    # AC voltage, scale 0.1 V
    voltage_scale: float = 0.1
    current_register: int = 40071    # AC current, scale 0.01 A
    current_scale: float = 0.01


# ---------------------------------------------------------------------------
# Reading result
# ---------------------------------------------------------------------------


@dataclass
class InverterReading:
    """A single snapshot from the inverter."""

    timestamp: datetime
    power_w: float
    voltage_v: float
    current_a: float
    station_id: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "station_id": self.station_id,
            "power_w": self.power_w,
            "voltage_v": self.voltage_v,
            "current_a": self.current_a,
        }


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class ModbusReader:
    """Read live data from a Modbus TCP inverter.

    Parameters
    ----------
    host:
        IP address or hostname of the Modbus TCP gateway.
    port:
        TCP port (default 502).
    unit_id:
        Modbus unit / slave ID (default 1).
    station_id:
        Human-readable name for this inverter, stored in readings.
    register_map:
        Custom register map; defaults to :class:`ModbusRegisterMap`.
    timeout:
        Socket timeout in seconds.

    Raises
    ------
    ImportError
        If ``pymodbus`` is not installed.

    Examples
    --------
    >>> reader = ModbusReader(host="192.168.1.100", port=502, unit_id=1)
    >>> reading = reader.read()
    >>> print(reading.power_w)
    1240.5
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 502,
        unit_id: int = 1,
        station_id: str = "default",
        register_map: ModbusRegisterMap | None = None,
        timeout: int = 5,
    ) -> None:
        try:
            from pymodbus.client import ModbusTcpClient  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pymodbus is required for ModbusReader. "
                "Install it with: pip install pymodbus>=3.5"
            ) from exc

        self._host = host
        self._port = port
        self._unit_id = unit_id
        self._station_id = station_id
        self._map = register_map or ModbusRegisterMap()
        self._timeout = timeout
        self._client_cls = ModbusTcpClient  # injectable for tests

    def read(self) -> InverterReading:
        """Connect, read registers, disconnect, and return an :class:`InverterReading`.

        Returns
        -------
        InverterReading
            Timestamped inverter snapshot.

        Raises
        ------
        ConnectionError
            If the TCP connection to the inverter cannot be established.
        RuntimeError
            If a Modbus register read returns an error response.
        """
        client = self._client_cls(host=self._host, port=self._port, timeout=self._timeout)

        if not client.connect():
            raise ConnectionError(
                f"Cannot connect to Modbus device at {self._host}:{self._port}. "
                "Check network connectivity and that the inverter is powered on."
            )

        try:
            power_w = self._read_register(client, self._map.power_register, self._map.power_scale)
            voltage_v = self._read_register(client, self._map.voltage_register, self._map.voltage_scale)
            current_a = self._read_register(client, self._map.current_register, self._map.current_scale)
        finally:
            client.close()

        return InverterReading(
            timestamp=datetime.now(timezone.utc),
            power_w=power_w,
            voltage_v=voltage_v,
            current_a=current_a,
            station_id=self._station_id,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_register(
        self,
        client: Any,
        address: int,
        scale: float,
    ) -> float:
        """Read a single holding register and apply the scale factor."""
        response = client.read_holding_registers(address=address, count=1, slave=self._unit_id)

        if response.isError():
            raise RuntimeError(
                f"Modbus error reading register {address} "
                f"from unit {self._unit_id}: {response}"
            )

        raw: int = response.registers[0]
        return float(raw) * scale


# ---------------------------------------------------------------------------
# Simulator (for local development without hardware)
# ---------------------------------------------------------------------------


class SimulatedModbusReader:
    """Drop-in replacement for :class:`ModbusReader` that generates synthetic data.

    Produces realistic solar-day curves using a simplified clear-sky model.
    Use this in development, CI, and the default Prefect flow when no real
    inverter is present.

    Parameters
    ----------
    station_id:
        Identifier stored in each :class:`InverterReading`.
    peak_power_w:
        Peak generation in Watts at solar noon (default 5000 W / 5 kW).
    noise_fraction:
        Random noise as a fraction of instantaneous power (default 0.03).
    """

    def __init__(
        self,
        station_id: str = "simulated",
        peak_power_w: float = 5000.0,
        noise_fraction: float = 0.03,
    ) -> None:
        import numpy as np

        self._station_id = station_id
        self._peak = peak_power_w
        self._noise = noise_fraction
        self._rng = np.random.default_rng()

    def read(self) -> InverterReading:
        """Return a synthetic reading for the current UTC time."""
        import numpy as np

        now = datetime.now(timezone.utc)
        hour = now.hour + now.minute / 60.0

        # Simple bell curve: max at hour=12, zero outside [6, 18]
        if 6.0 <= hour <= 18.0:
            angle = np.pi * (hour - 6.0) / 12.0
            base_power = self._peak * np.sin(angle) ** 2
        else:
            base_power = 0.0

        noise = self._rng.normal(0, self._noise * base_power) if base_power > 0 else 0.0
        power_w = float(np.clip(base_power + noise, 0.0, self._peak))

        # Derived voltage / current from power (nominal 230 V AC, PF=0.99)
        voltage_v = float(self._rng.normal(230.0, 1.5))
        current_a = power_w / (voltage_v * 0.99) if power_w > 0 else 0.0

        return InverterReading(
            timestamp=now,
            power_w=power_w,
            voltage_v=voltage_v,
            current_a=current_a,
            station_id=self._station_id,
        )
