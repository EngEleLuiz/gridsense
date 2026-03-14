"""Unit tests for gridsense.ingest.modbus."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from gridsense.ingest.modbus import InverterReading, ModbusReader, SimulatedModbusReader

# ---------------------------------------------------------------------------
# InverterReading
# ---------------------------------------------------------------------------


class TestInverterReading:
    def test_to_dict_keys(self) -> None:
        reading = InverterReading(
            timestamp=datetime.now(timezone.utc),
            power_w=1000.0,
            voltage_v=230.0,
            current_a=4.35,
        )
        d = reading.to_dict()
        assert set(d.keys()) == {"timestamp", "station_id", "power_w", "voltage_v", "current_a"}

    def test_to_dict_values(self) -> None:
        reading = InverterReading(
            timestamp=datetime.now(timezone.utc),
            power_w=500.0,
            voltage_v=228.5,
            current_a=2.19,
            station_id="test-01",
        )
        d = reading.to_dict()
        assert d["power_w"] == pytest.approx(500.0)
        assert d["station_id"] == "test-01"


# ---------------------------------------------------------------------------
# ModbusReader (mocked)
# ---------------------------------------------------------------------------


class TestModbusReader:
    def _make_mock_response(self, value: int) -> MagicMock:
        resp = MagicMock()
        resp.isError.return_value = False
        resp.registers = [value]
        return resp

    def test_read_returns_correct_keys(self) -> None:
        """read() must return an InverterReading with all fields set."""
        mock_client = MagicMock()
        mock_client.connect.return_value = True
        mock_client.read_holding_registers.return_value = self._make_mock_response(10000)

        reader = ModbusReader.__new__(ModbusReader)
        reader._host = "localhost"
        reader._port = 502
        reader._unit_id = 1
        reader._station_id = "test"
        reader._map = __import__(
            "gridsense.ingest.modbus", fromlist=["ModbusRegisterMap"]
        ).ModbusRegisterMap()
        reader._timeout = 5
        reader._client_cls = lambda **kwargs: mock_client

        reading = reader.read()
        assert isinstance(reading, InverterReading)
        assert reading.power_w is not None
        assert reading.voltage_v is not None
        assert reading.current_a is not None

    def test_read_with_mock_client(self) -> None:
        """Verify scaling: register value 1000 * scale 0.1 = 100.0."""
        mock_client = MagicMock()
        mock_client.connect.return_value = True
        mock_client.read_holding_registers.return_value = self._make_mock_response(1000)

        reader = ModbusReader.__new__(ModbusReader)
        from gridsense.ingest.modbus import ModbusRegisterMap
        reader._map = ModbusRegisterMap()
        reader._host = "localhost"
        reader._port = 502
        reader._unit_id = 1
        reader._station_id = "test"
        reader._timeout = 5
        reader._client_cls = lambda **kwargs: mock_client

        reading = reader.read()
        # power register scale = 0.1 → 1000 * 0.1 = 100.0 W
        assert reading.power_w == pytest.approx(100.0)

    def test_read_raises_on_connection_error(self) -> None:
        """ConnectionError must be raised with a helpful message when connect() fails."""
        mock_client = MagicMock()
        mock_client.connect.return_value = False  # <-- connection failure

        reader = ModbusReader.__new__(ModbusReader)
        from gridsense.ingest.modbus import ModbusRegisterMap
        reader._map = ModbusRegisterMap()
        reader._host = "10.0.0.99"
        reader._port = 502
        reader._unit_id = 1
        reader._station_id = "test"
        reader._timeout = 5
        reader._client_cls = lambda **kwargs: mock_client

        with pytest.raises(ConnectionError, match="10.0.0.99"):
            reader.read()

    def test_read_raises_on_modbus_error_response(self) -> None:
        """RuntimeError must be raised when a register read returns an error."""
        error_response = MagicMock()
        error_response.isError.return_value = True

        mock_client = MagicMock()
        mock_client.connect.return_value = True
        mock_client.read_holding_registers.return_value = error_response

        reader = ModbusReader.__new__(ModbusReader)
        from gridsense.ingest.modbus import ModbusRegisterMap
        reader._map = ModbusRegisterMap()
        reader._host = "localhost"
        reader._port = 502
        reader._unit_id = 1
        reader._station_id = "test"
        reader._timeout = 5
        reader._client_cls = lambda **kwargs: mock_client

        with pytest.raises(RuntimeError, match="Modbus error"):
            reader.read()


# ---------------------------------------------------------------------------
# SimulatedModbusReader
# ---------------------------------------------------------------------------


class TestSimulatedModbusReader:
    def test_read_returns_inverter_reading(self) -> None:
        reader = SimulatedModbusReader()
        reading = reader.read()
        assert isinstance(reading, InverterReading)

    def test_power_non_negative(self) -> None:
        reader = SimulatedModbusReader()
        for _ in range(20):
            assert reader.read().power_w >= 0.0

    def test_power_does_not_exceed_peak(self) -> None:
        peak = 3000.0
        reader = SimulatedModbusReader(peak_power_w=peak)
        for _ in range(20):
            assert reader.read().power_w <= peak + 1.0  # tiny float tolerance

    def test_station_id_propagated(self) -> None:
        reader = SimulatedModbusReader(station_id="my-station")
        assert reader.read().station_id == "my-station"

    def test_timestamp_is_set(self) -> None:
        reader = SimulatedModbusReader()
        reading = reader.read()
        assert isinstance(reading.timestamp, datetime)
