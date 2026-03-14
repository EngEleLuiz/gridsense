"""Unit tests for gridsense.battery.soc."""

from __future__ import annotations

import pytest

from gridsense.battery.soc import SoCEstimator, _ocv_to_soc, _soc_to_ocv

# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_valid_construction(self) -> None:
        est = SoCEstimator(capacity_ah=10.0, initial_soc=0.8)
        assert est.soc == pytest.approx(0.8)

    def test_zero_capacity_raises(self) -> None:
        with pytest.raises(ValueError, match="capacity_ah"):
            SoCEstimator(capacity_ah=0.0)

    def test_negative_capacity_raises(self) -> None:
        with pytest.raises(ValueError, match="capacity_ah"):
            SoCEstimator(capacity_ah=-5.0)

    def test_soc_above_1_raises(self) -> None:
        with pytest.raises(ValueError, match="initial_soc"):
            SoCEstimator(capacity_ah=10.0, initial_soc=1.1)

    def test_soc_below_0_raises(self) -> None:
        with pytest.raises(ValueError, match="initial_soc"):
            SoCEstimator(capacity_ah=10.0, initial_soc=-0.1)

    def test_invalid_eta_raises(self) -> None:
        with pytest.raises(ValueError, match="coulombic_efficiency"):
            SoCEstimator(capacity_ah=10.0, coulombic_efficiency=0.0)


# ---------------------------------------------------------------------------
# Update — basic Coulomb Counting
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_returns_float(self) -> None:
        est = SoCEstimator(capacity_ah=10.0, initial_soc=0.5)
        result = est.update(current_a=1.0, dt_seconds=1.0)
        assert isinstance(result, float)

    def test_zero_current_no_change(self) -> None:
        est = SoCEstimator(capacity_ah=10.0, initial_soc=0.5)
        soc_before = est.soc
        est.update(current_a=0.0, dt_seconds=60.0)
        assert est.soc == pytest.approx(soc_before)

    def test_discharge_decreases_soc(self) -> None:
        est = SoCEstimator(capacity_ah=10.0, initial_soc=0.5)
        est.update(current_a=-1.0, dt_seconds=3600.0)
        assert est.soc < 0.5

    def test_charge_increases_soc(self) -> None:
        est = SoCEstimator(capacity_ah=10.0, initial_soc=0.5)
        est.update(current_a=1.0, dt_seconds=3600.0)
        assert est.soc > 0.5

    def test_soc_never_exceeds_1(self) -> None:
        est = SoCEstimator(capacity_ah=10.0, initial_soc=0.99)
        for _ in range(100):
            est.update(current_a=100.0, dt_seconds=60.0)
        assert est.soc <= 1.0

    def test_full_discharge_reaches_zero(self) -> None:
        est = SoCEstimator(capacity_ah=10.0, initial_soc=1.0)
        # Discharge 10 Ah worth at 2 A → 5 hours = 18000 seconds
        for _ in range(1000):
            est.update(current_a=-2.0, dt_seconds=18.0)
        assert est.soc == pytest.approx(0.0, abs=0.01)

    def test_soc_never_goes_below_zero(self) -> None:
        est = SoCEstimator(capacity_ah=10.0, initial_soc=0.01)
        for _ in range(100):
            est.update(current_a=-100.0, dt_seconds=60.0)
        assert est.soc >= 0.0

    def test_invalid_dt_raises(self) -> None:
        est = SoCEstimator(capacity_ah=10.0)
        with pytest.raises(ValueError, match="dt_seconds"):
            est.update(current_a=1.0, dt_seconds=0.0)

    def test_negative_dt_raises(self) -> None:
        est = SoCEstimator(capacity_ah=10.0)
        with pytest.raises(ValueError, match="dt_seconds"):
            est.update(current_a=1.0, dt_seconds=-1.0)


# ---------------------------------------------------------------------------
# Coulomb Counting accuracy
# ---------------------------------------------------------------------------


class TestCoulombCounting:
    def test_half_charge_discharged(self) -> None:
        """Discharging half capacity should land near 0.5."""
        cap_ah = 20.0
        est = SoCEstimator(
            capacity_ah=cap_ah, initial_soc=1.0, coulombic_efficiency=1.0
        )
        # Discharge 10 Ah at 1 A → 10 hours = 36000 s
        est.update(current_a=-1.0, dt_seconds=36_000.0)
        assert est.soc == pytest.approx(0.5, abs=0.01)

    def test_coulombic_efficiency_reduces_charge_gain(self) -> None:
        est_ideal = SoCEstimator(
            capacity_ah=10.0, initial_soc=0.0, coulombic_efficiency=1.0
        )
        est_lossy = SoCEstimator(
            capacity_ah=10.0, initial_soc=0.0, coulombic_efficiency=0.95
        )
        for est in (est_ideal, est_lossy):
            est.update(current_a=1.0, dt_seconds=3600.0)
        assert est_lossy.soc < est_ideal.soc


# ---------------------------------------------------------------------------
# OCV correction
# ---------------------------------------------------------------------------


class TestOCVCorrection:
    def test_correct_with_ocv_changes_soc(self) -> None:
        est = SoCEstimator(capacity_ah=10.0, initial_soc=0.3)
        # OCV at 100% ≈ 12.6 V → corrects upward
        new_soc = est.correct_with_ocv(12.6)
        assert new_soc == pytest.approx(1.0, abs=0.05)

    def test_correct_with_fully_discharged_ocv(self) -> None:
        est = SoCEstimator(capacity_ah=10.0, initial_soc=0.8)
        new_soc = est.correct_with_ocv(9.0)   # OCV at 0 % SoC
        assert new_soc == pytest.approx(0.0, abs=0.05)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_to_specific_soc(self) -> None:
        est = SoCEstimator(capacity_ah=10.0, initial_soc=0.1)
        est.reset(0.9)
        assert est.soc == pytest.approx(0.9)

    def test_reset_out_of_range_raises(self) -> None:
        est = SoCEstimator(capacity_ah=10.0)
        with pytest.raises(ValueError):
            est.reset(1.5)


# ---------------------------------------------------------------------------
# OCV lookup helpers
# ---------------------------------------------------------------------------


class TestOCVHelpers:
    def test_ocv_to_soc_monotone(self) -> None:
        """Higher voltage → higher SoC."""
        assert _ocv_to_soc(9.0) < _ocv_to_soc(11.0) < _ocv_to_soc(12.6)

    def test_soc_to_ocv_monotone(self) -> None:
        """Higher SoC → higher OCV."""
        assert _soc_to_ocv(0.0) < _soc_to_ocv(0.5) < _soc_to_ocv(1.0)

    def test_soc_in_range(self) -> None:
        for v in (8.0, 9.0, 11.0, 12.6, 14.0):
            soc = _ocv_to_soc(v)
            assert 0.0 <= soc <= 1.0


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_contains_capacity(self) -> None:
        est = SoCEstimator(capacity_ah=15.0)
        assert "15.00" in repr(est)
