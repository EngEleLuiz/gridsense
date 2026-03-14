"""
DWT-based feature extraction for power quality waveform analysis.

Uses Discrete Wavelet Transform (PyWavelets) to decompose voltage waveforms
and compute statistical features at each decomposition level. These features
feed directly into the PQ disturbance classifier.

IEEE 1159-2019 reference disturbances supported:
  - Voltage sag / swell
  - Interruptions
  - Harmonics (THD > 5%)
  - Impulsive transients
"""

from __future__ import annotations

import numpy as np
import pywt
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Wavelet family recommended for power quality analysis (4th-order Daubechies).
DEFAULT_WAVELET: str = "db4"

#: Decomposition depth — 5 levels covers frequencies down to ~1 Hz at 6.4 kHz.
DEFAULT_LEVEL: int = 5

#: Minimum number of samples required in a waveform.
MIN_WAVEFORM_LENGTH: int = 64


# ---------------------------------------------------------------------------
# Core feature extractor
# ---------------------------------------------------------------------------


def extract_dwt_features(
    waveform: NDArray[np.floating],
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
) -> NDArray[np.floating]:
    """Extract a fixed-size feature vector from a voltage waveform via DWT.

    The waveform is decomposed into ``level`` detail bands plus one
    approximation band.  For each band the following statistics are computed:

    * RMS energy
    * Mean absolute value
    * Standard deviation
    * Maximum absolute value
    * Shannon entropy (normalised)

    This yields ``(level + 1) × 5`` features — always the same length for a
    given ``level``, regardless of the input waveform length.

    Parameters
    ----------
    waveform:
        1-D array of voltage samples (arbitrary units, e.g. volts or per-unit).
        Must have at least ``MIN_WAVEFORM_LENGTH`` samples.
    wavelet:
        PyWavelets wavelet identifier (default ``"db4"``).
    level:
        Number of decomposition levels (default ``5``).

    Returns
    -------
    NDArray[np.floating]
        1-D feature vector of shape ``((level + 1) * 5,)``.

    Raises
    ------
    ValueError
        If ``waveform`` is empty, not 1-D, or shorter than
        ``MIN_WAVEFORM_LENGTH`` samples.
    ValueError
        If ``level`` is less than 1.
    """
    waveform = np.asarray(waveform, dtype=np.float64)

    # --- Input validation ---------------------------------------------------
    if waveform.ndim != 1:
        raise ValueError(
            f"waveform must be 1-D, got shape {waveform.shape}."
        )
    if len(waveform) < MIN_WAVEFORM_LENGTH:
        raise ValueError(
            f"waveform must have at least {MIN_WAVEFORM_LENGTH} samples, "
            f"got {len(waveform)}."
        )
    if level < 1:
        raise ValueError(f"level must be >= 1, got {level}.")

    # --- Decompose ----------------------------------------------------------
    coeffs = pywt.wavedec(waveform, wavelet=wavelet, level=level)
    # coeffs = [cA_N, cD_N, cD_{N-1}, ..., cD_1]  (length = level + 1)

    # --- Per-band statistics ------------------------------------------------
    features: list[float] = []
    for band in coeffs:
        features.extend(_band_stats(band))

    return np.array(features, dtype=np.float64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _band_stats(band: NDArray[np.floating]) -> list[float]:
    """Return [rms, mean_abs, std, max_abs, entropy] for one DWT sub-band."""
    band = np.asarray(band, dtype=np.float64)

    rms = float(np.sqrt(np.mean(band ** 2)))
    mean_abs = float(np.mean(np.abs(band)))
    std = float(np.std(band))
    max_abs = float(np.max(np.abs(band)))
    entropy = _shannon_entropy(band)

    return [rms, mean_abs, std, max_abs, entropy]


def _shannon_entropy(band: NDArray[np.floating]) -> float:
    """Normalised Shannon entropy of the squared DWT coefficients."""
    energy = band ** 2
    total = float(np.sum(energy))
    if total == 0.0:
        return 0.0
    prob = energy / total
    # Avoid log(0) — replace zeros with 1 so log term vanishes
    prob = np.where(prob > 0, prob, 1.0)
    return float(-np.sum(prob * np.log2(prob)))


def feature_names(level: int = DEFAULT_LEVEL) -> list[str]:
    """Return human-readable names for each feature, in extraction order.

    Useful for building DataFrames or logging feature importances.

    Parameters
    ----------
    level:
        The decomposition level used during extraction.

    Returns
    -------
    list[str]
        List of ``(level + 1) * 5`` feature name strings.
    """
    stats = ["rms", "mean_abs", "std", "max_abs", "entropy"]
    names: list[str] = []

    # Approximation band comes first in pywt.wavedec output
    names += [f"cA{level}_{s}" for s in stats]

    # Then detail bands from highest to lowest level
    for lv in range(level, 0, -1):
        names += [f"cD{lv}_{s}" for s in stats]

    return names
