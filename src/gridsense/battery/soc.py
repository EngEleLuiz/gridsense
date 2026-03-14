"""
State-of-Charge (SoC) estimator for Li-Ion battery packs.

Implements Coulomb Counting as the primary integration method, with an
OCV (Open-Circuit Voltage) lookup table used for periodic drift correction.
Ported from the ``Digital-Twin-of-a-Li-Ion-Battery`` repository and adapted
for continuous streaming updates in the GridSense pipeline.

References
----------
* Plett, G.L. (2004) "Extended Kalman filtering for battery management
  systems of LiPB-based HEV battery packs."
* IEEE 1679.1-2017 — Guide for Evaluation of Li-Ion Batteries
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# OCV look-up table (3S Li-Ion pack, nominal 11.1 V)
# ---------------------------------------------------------------------------
# Each row is (soc_fraction, ocv_volts).
# Measured at 25 °C, 0.1C discharge rate.
# Interpolated linearly between points.

_OCV_TABLE: NDArray[np.float64] = np.array(
    [
        [0.00, 9.00],
        [0.05, 9.60],
        [0.10, 10.20],
        [0.20, 10.80],
        [0.30, 11.10],
        [0.40, 11.25],
        [0.50, 11.40],
        [0.60, 11.55],
        [0.70, 11.70],
        [0.80, 11.85],
        [0.90, 12.00],
        [1.00, 12.60],
    ],
    dtype=np.float64,
)


def _ocv_to_soc(voltage: float) -> float:
    """Map a measured OCV (V) to an SoC fraction via linear interpolation."""
    soc_pts = _OCV_TABLE[:, 0]
    ocv_pts = _OCV_TABLE[:, 1]
    return float(np.clip(np.interp(voltage, ocv_pts, soc_pts), 0.0, 1.0))


def _soc_to_ocv(soc: float) -> float:
    """Map an SoC fraction to its expected OCV (V) via linear interpolation."""
    soc_pts = _OCV_TABLE[:, 0]
    ocv_pts = _OCV_TABLE[:, 1]
    return float(np.interp(soc, soc_pts, ocv_pts))


# ---------------------------------------------------------------------------
# SoCEstimator
# ---------------------------------------------------------------------------


class SoCEstimator:
    """Coulomb-Counting SoC estimator with optional OCV correction.

    The estimator integrates current over time to track charge flow, and
    optionally applies a correction step when a measured OCV voltage is
    available (e.g. during rest periods).

    Parameters
    ----------
    capacity_ah:
        Nominal pack capacity in Ampere-hours.  Must be > 0.
    initial_soc:
        Starting SoC fraction in ``[0.0, 1.0]``.  Defaults to ``1.0``
        (fully charged).
    coulombic_efficiency:
        Charge efficiency factor ``η ∈ (0, 1]``.  Applied to charging
        current only.  Defaults to ``0.98``.

    Raises
    ------
    ValueError
        If ``capacity_ah`` ≤ 0 or ``initial_soc`` ∉ ``[0.0, 1.0]``.
    """

    def __init__(
        self,
        capacity_ah: float,
        initial_soc: float = 1.0,
        coulombic_efficiency: float = 0.98,
    ) -> None:
        if capacity_ah <= 0:
            raise ValueError(f"capacity_ah must be > 0, got {capacity_ah}.")
        if not (0.0 <= initial_soc <= 1.0):
            raise ValueError(
                f"initial_soc must be in [0, 1], got {initial_soc}."
            )
        if not (0.0 < coulombic_efficiency <= 1.0):
            raise ValueError(
                f"coulombic_efficiency must be in (0, 1], "
                f"got {coulombic_efficiency}."
            )

        self._capacity_as: float = capacity_ah * 3600.0  # convert to Amp-seconds
        self._soc: float = float(initial_soc)
        self._eta: float = coulombic_efficiency

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def soc(self) -> float:
        """Current SoC estimate in ``[0.0, 1.0]``."""
        return self._soc

    @property
    def soc_percent(self) -> float:
        """Current SoC expressed as a percentage ``[0.0, 100.0]``."""
        return self._soc * 100.0

    # ------------------------------------------------------------------
    # Update methods
    # ------------------------------------------------------------------

    def update(self, current_a: float, dt_seconds: float) -> float:
        """Integrate one current measurement and return the new SoC.

        Positive current = charging; negative current = discharging
        (load-convention sign).

        Parameters
        ----------
        current_a:
            Instantaneous current in Amperes.
        dt_seconds:
            Time elapsed since the last measurement, in seconds.  Must be > 0.

        Returns
        -------
        float
            Updated SoC in ``[0.0, 1.0]``.

        Raises
        ------
        ValueError
            If ``dt_seconds`` ≤ 0.
        """
        if dt_seconds <= 0:
            raise ValueError(f"dt_seconds must be > 0, got {dt_seconds}.")

        # Apply coulombic efficiency only during charging
        eta = self._eta if current_a > 0 else 1.0
        delta_soc = (eta * current_a * dt_seconds) / self._capacity_as
        self._soc = float(np.clip(self._soc + delta_soc, 0.0, 1.0))
        return self._soc

    def correct_with_ocv(self, measured_voltage_v: float) -> float:
        """Override the current SoC estimate using a resting OCV measurement.

        Call this during rest periods (|I| ≈ 0) to periodically correct
        Coulomb-Counting drift.

        Parameters
        ----------
        measured_voltage_v:
            Measured open-circuit voltage of the pack in Volts.

        Returns
        -------
        float
            Corrected SoC in ``[0.0, 1.0]``.
        """
        self._soc = _ocv_to_soc(measured_voltage_v)
        return self._soc

    def reset(self, soc: float = 1.0) -> None:
        """Reset the estimator to a known SoC.

        Parameters
        ----------
        soc:
            New SoC value in ``[0.0, 1.0]``.

        Raises
        ------
        ValueError
            If ``soc`` ∉ ``[0.0, 1.0]``.
        """
        if not (0.0 <= soc <= 1.0):
            raise ValueError(f"soc must be in [0, 1], got {soc}.")
        self._soc = float(soc)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cap_ah = self._capacity_as / 3600.0
        return (
            f"SoCEstimator(capacity_ah={cap_ah:.2f}, "
            f"soc={self._soc:.4f}, eta={self._eta:.3f})"
        )
