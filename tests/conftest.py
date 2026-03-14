"""Shared pytest fixtures for GridSense tests."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Waveform fixtures
# ---------------------------------------------------------------------------

FS = 6400.0   # Hz
F0 = 60.0     # fundamental frequency
SAMPLES = 1024


@pytest.fixture(scope="session")
def t() -> NDArray[np.float64]:
    """Time axis for all waveform fixtures."""
    return np.linspace(0, SAMPLES / FS, SAMPLES, endpoint=False)


@pytest.fixture(scope="session")
def clean_sine(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pure 60 Hz sinusoidal waveform — 'normal' class."""
    return np.sin(2 * np.pi * F0 * t)


@pytest.fixture(scope="session")
def sag_waveform(
    clean_sine: NDArray[np.float64],
    t: NDArray[np.float64],
) -> NDArray[np.float64]:
    """60 Hz sine with a 50 % voltage sag in the middle third."""
    wave = clean_sine.copy()
    n = len(wave)
    wave[n // 3: 2 * n // 3] *= 0.5
    return wave


@pytest.fixture(scope="session")
def swell_waveform(
    clean_sine: NDArray[np.float64],
    t: NDArray[np.float64],
) -> NDArray[np.float64]:
    """60 Hz sine with a 30 % voltage swell."""
    wave = clean_sine.copy()
    n = len(wave)
    wave[n // 3: 2 * n // 3] *= 1.3
    return wave


@pytest.fixture(scope="session")
def harmonic_waveform(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fundamental + 3rd + 5th harmonics (THD ~ 22 %)."""
    return (
        np.sin(2 * np.pi * F0 * t)
        + 0.15 * np.sin(2 * np.pi * 3 * F0 * t)
        + 0.10 * np.sin(2 * np.pi * 5 * F0 * t)
    )
