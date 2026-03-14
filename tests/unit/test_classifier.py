"""Unit tests for gridsense.pq.classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from gridsense.pq.classifier import (
    DISTURBANCE_CLASSES,
    PQClassifier,
    PQResult,
    generate_synthetic_dataset,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_clf() -> PQClassifier:
    """Return a classifier trained on enough data to be reliable."""
    X, y = generate_synthetic_dataset(n_per_class=300, seed=42)
    clf = PQClassifier(n_estimators=150, random_state=42)
    clf.train(X, y)
    return clf


# ---------------------------------------------------------------------------
# PQResult dataclass
# ---------------------------------------------------------------------------


class TestPQResult:
    def test_to_dict_keys(self) -> None:
        result = PQResult(label="normal", confidence=0.9)
        d = result.to_dict()
        assert set(d.keys()) == {"label", "confidence", "timestamp"}

    def test_confidence_stored(self) -> None:
        result = PQResult(label="harmonics", confidence=0.75)
        assert result.confidence == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Untrained model
# ---------------------------------------------------------------------------


class TestUntrainedClassifier:
    def test_predict_raises_before_training(
        self, clean_sine: NDArray[np.float64]
    ) -> None:
        clf = PQClassifier()
        with pytest.raises(RuntimeError, match="trained"):
            clf.predict(clean_sine)

    def test_save_raises_before_training(self, tmp_path: Path) -> None:
        clf = PQClassifier()
        with pytest.raises(RuntimeError, match="untrained"):
            clf.save(tmp_path / "model.joblib")


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


class TestPredictions:
    def test_predict_returns_pqresult(
        self,
        trained_clf: PQClassifier,
        clean_sine: NDArray[np.float64],
    ) -> None:
        result = trained_clf.predict(clean_sine)
        assert isinstance(result, PQResult)

    def test_label_is_valid_class(
        self,
        trained_clf: PQClassifier,
        clean_sine: NDArray[np.float64],
    ) -> None:
        result = trained_clf.predict(clean_sine)
        assert result.label in DISTURBANCE_CLASSES

    def test_confidence_between_0_and_1(
        self,
        trained_clf: PQClassifier,
        clean_sine: NDArray[np.float64],
    ) -> None:
        result = trained_clf.predict(clean_sine)
        assert 0.0 <= result.confidence <= 1.0

    def test_predict_normal_signal(
        self,
        trained_clf: PQClassifier,
    ) -> None:
        """Normal waveform (same distribution as training) → 'normal'."""
        from gridsense.pq.classifier import _waveform_normal

        rng = np.random.default_rng(0)
        t = np.linspace(0, 1024 / 6400.0, 1024, endpoint=False)
        fundamental = np.sin(2 * np.pi * 60.0 * t)
        wave = _waveform_normal(fundamental, rng)
        assert trained_clf.predict(wave).label == "normal"

    def test_predict_sag_signal(
        self,
        trained_clf: PQClassifier,
    ) -> None:
        """Sag waveform (same distribution as training) → 'voltage_sag'."""
        from gridsense.pq.classifier import _waveform_sag

        rng = np.random.default_rng(1)
        t = np.linspace(0, 1024 / 6400.0, 1024, endpoint=False)
        fundamental = np.sin(2 * np.pi * 60.0 * t)
        wave = _waveform_sag(fundamental, t, 60.0, rng)
        assert trained_clf.predict(wave).label == "voltage_sag"

    def test_predict_swell_signal(
        self,
        trained_clf: PQClassifier,
    ) -> None:
        """Swell waveform (same distribution as training) → 'voltage_swell'."""
        from gridsense.pq.classifier import _waveform_swell

        rng = np.random.default_rng(2)
        t = np.linspace(0, 1024 / 6400.0, 1024, endpoint=False)
        fundamental = np.sin(2 * np.pi * 60.0 * t)
        wave = _waveform_swell(fundamental, t, 60.0, rng)
        assert trained_clf.predict(wave).label == "voltage_swell"

    def test_timestamp_is_set(
        self,
        trained_clf: PQClassifier,
        clean_sine: NDArray[np.float64],
    ) -> None:
        result = trained_clf.predict(clean_sine)
        assert result.timestamp is not None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load_roundtrip(
        self,
        trained_clf: PQClassifier,
        clean_sine: NDArray[np.float64],
        tmp_path: Path,
    ) -> None:
        model_path = tmp_path / "pq_model.joblib"
        trained_clf.save(model_path)

        loaded = PQClassifier.load(model_path)
        original_result = trained_clf.predict(clean_sine)
        loaded_result = loaded.predict(clean_sine)

        assert loaded_result.label == original_result.label
        assert loaded_result.confidence == pytest.approx(
            original_result.confidence, abs=1e-6
        )

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            PQClassifier.load(tmp_path / "nonexistent.joblib")

    def test_save_creates_parent_dirs(
        self,
        trained_clf: PQClassifier,
        tmp_path: Path,
    ) -> None:
        deep_path = tmp_path / "nested" / "dir" / "model.joblib"
        trained_clf.save(deep_path)
        assert deep_path.exists()


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------


class TestSyntheticDataset:
    def test_dataset_shape(self) -> None:
        X, y = generate_synthetic_dataset(n_per_class=10, seed=0)
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] == 10 * len(DISTURBANCE_CLASSES)

    def test_all_classes_present(self) -> None:
        _, y = generate_synthetic_dataset(n_per_class=5, seed=0)
        assert set(y.tolist()) == set(range(len(DISTURBANCE_CLASSES)))

    def test_reproducibility(self) -> None:
        X1, y1 = generate_synthetic_dataset(n_per_class=10, seed=99)
        X2, y2 = generate_synthetic_dataset(n_per_class=10, seed=99)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
