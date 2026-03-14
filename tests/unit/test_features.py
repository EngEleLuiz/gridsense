"""Unit tests for gridsense.pq.features."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from gridsense.pq.features import (
    DEFAULT_LEVEL,
    MIN_WAVEFORM_LENGTH,
    extract_dwt_features,
    feature_names,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_N_FEATURES = (DEFAULT_LEVEL + 1) * 5  # 30 at level=5


# ---------------------------------------------------------------------------
# Shape and consistency tests
# ---------------------------------------------------------------------------


class TestOutputShape:
    def test_output_shape_is_fixed(self, clean_sine: NDArray[np.float64]) -> None:
        """Feature vector length must always be (level+1)*5."""
        features = extract_dwt_features(clean_sine)
        assert features.shape == (EXPECTED_N_FEATURES,)

    def test_output_shape_independent_of_input_length(self) -> None:
        """Different waveform lengths must still yield same-size feature vector."""
        short = np.sin(2 * np.pi * 60 * np.linspace(0, 1, 512))
        long_ = np.sin(2 * np.pi * 60 * np.linspace(0, 2, 4096))
        assert extract_dwt_features(short).shape == extract_dwt_features(long_).shape

    def test_output_dtype_is_float(self, clean_sine: NDArray[np.float64]) -> None:
        features = extract_dwt_features(clean_sine)
        assert np.issubdtype(features.dtype, np.floating)

    def test_custom_level_changes_feature_count(
        self, clean_sine: NDArray[np.float64]
    ) -> None:
        for lvl in (2, 3, 4):
            feats = extract_dwt_features(clean_sine, level=lvl)
            assert feats.shape == ((lvl + 1) * 5,), f"Failed for level={lvl}"


# ---------------------------------------------------------------------------
# Physical / signal tests
# ---------------------------------------------------------------------------


class TestSignalSemantics:
    def test_clean_sine_detail_energy_is_low(
        self, clean_sine: NDArray[np.float64]
    ) -> None:
        """A pure sine should produce near-zero detail RMS at fine scales."""
        import pywt

        coeffs = pywt.wavedec(clean_sine, "db4", level=DEFAULT_LEVEL)
        # cD1 (finest detail) should have much lower energy than cA (approx.)
        approx_rms = float(np.sqrt(np.mean(coeffs[0] ** 2)))
        detail_rms = float(np.sqrt(np.mean(coeffs[-1] ** 2)))
        assert detail_rms < approx_rms * 0.5

    def test_disturbance_changes_features(
        self,
        clean_sine: NDArray[np.float64],
        sag_waveform: NDArray[np.float64],
    ) -> None:
        """Features of a sag waveform must differ from clean sine features."""
        f_clean = extract_dwt_features(clean_sine)
        f_sag = extract_dwt_features(sag_waveform)
        assert not np.allclose(f_clean, f_sag)

    def test_identical_waveforms_produce_identical_features(
        self, clean_sine: NDArray[np.float64]
    ) -> None:
        """Extraction must be deterministic."""
        f1 = extract_dwt_features(clean_sine)
        f2 = extract_dwt_features(clean_sine.copy())
        np.testing.assert_array_equal(f1, f2)

    def test_harmonic_distortion_increases_detail_energy(
        self,
        clean_sine: NDArray[np.float64],
        harmonic_waveform: NDArray[np.float64],
    ) -> None:
        """Harmonics should increase energy in mid-frequency detail bands."""
        import pywt

        c_clean = pywt.wavedec(clean_sine, "db4", level=DEFAULT_LEVEL)
        c_harm = pywt.wavedec(harmonic_waveform, "db4", level=DEFAULT_LEVEL)

        # At least one detail band should have higher energy
        detail_rms_clean = [float(np.sqrt(np.mean(c ** 2))) for c in c_clean[1:]]
        detail_rms_harm = [float(np.sqrt(np.mean(c ** 2))) for c in c_harm[1:]]
        assert any(h > c for h, c in zip(detail_rms_harm, detail_rms_clean))


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_array_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="samples"):
            extract_dwt_features(np.array([]))

    def test_too_short_raises_value_error(self) -> None:
        short = np.ones(MIN_WAVEFORM_LENGTH - 1)
        with pytest.raises(ValueError, match="samples"):
            extract_dwt_features(short)

    def test_2d_array_raises_value_error(self) -> None:
        bad = np.ones((64, 64))
        with pytest.raises(ValueError, match="1-D"):
            extract_dwt_features(bad)

    def test_invalid_level_raises_value_error(
        self, clean_sine: NDArray[np.float64]
    ) -> None:
        with pytest.raises(ValueError, match="level"):
            extract_dwt_features(clean_sine, level=0)

    def test_non_float_input_is_coerced(self) -> None:
        """Integer arrays should be silently coerced to float."""
        wave = np.ones(256, dtype=np.int32)
        feats = extract_dwt_features(wave)
        assert feats.shape == (EXPECTED_N_FEATURES,)


# ---------------------------------------------------------------------------
# feature_names helper
# ---------------------------------------------------------------------------


class TestFeatureNames:
    def test_names_length_matches_feature_count(self) -> None:
        names = feature_names(DEFAULT_LEVEL)
        assert len(names) == EXPECTED_N_FEATURES

    def test_names_are_strings(self) -> None:
        for name in feature_names():
            assert isinstance(name, str)

    def test_names_are_unique(self) -> None:
        names = feature_names()
        assert len(set(names)) == len(names)
