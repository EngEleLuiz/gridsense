"""
Power Quality disturbance classifier.

Wraps a scikit-learn Random Forest trained on a synthetic IEEE 1159-2019
dataset.  The classifier accepts raw voltage waveforms, extracts DWT features
internally, and returns a label + confidence score.

Supported disturbance classes
------------------------------
normal          — clean sinusoidal signal
voltage_sag     — amplitude < 0.9 pu for 0.5–30 cycles
voltage_swell   — amplitude > 1.1 pu for 0.5–30 cycles
interruption    — amplitude < 0.1 pu for > 0.5 cycles
harmonics       — THD > 5 % (3rd / 5th harmonic injection)
transient       — impulsive spike superimposed on the fundamental
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gridsense.pq.features import DEFAULT_LEVEL, DEFAULT_WAVELET, extract_dwt_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISTURBANCE_CLASSES: list[str] = [
    "normal",
    "voltage_sag",
    "voltage_swell",
    "interruption",
    "harmonics",
    "transient",
]

#: Default artifact location — ``artifacts/pq_model.joblib`` relative to CWD.
DEFAULT_MODEL_PATH: Path = Path("artifacts") / "pq_model.joblib"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PQResult:
    """Prediction result returned by :class:`PQClassifier`."""

    label: str
    """Disturbance class label, one of :data:`DISTURBANCE_CLASSES`."""

    confidence: float
    """Probability of the predicted class in ``[0.0, 1.0]``."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """UTC timestamp at prediction time."""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class PQClassifier:
    """Random Forest classifier for IEEE 1159 power quality disturbances.

    Usage — loading a pre-trained model::

        clf = PQClassifier.load()           # from default path
        result = clf.predict(waveform)      # NDArray → PQResult

    Usage — training from scratch (e.g. in tests or offline scripts)::

        clf = PQClassifier()
        clf.train(X_train, y_train)
        clf.save(Path("artifacts/pq_model.joblib"))

    Parameters
    ----------
    wavelet:
        PyWavelets wavelet used for feature extraction.
    level:
        DWT decomposition depth.
    n_estimators:
        Number of trees in the Random Forest.
    random_state:
        Seed for reproducibility.
    """

    #: Classes this classifier can predict, in label-encoder order.
    CLASSES: ClassVar[list[str]] = DISTURBANCE_CLASSES

    def __init__(
        self,
        wavelet: str = DEFAULT_WAVELET,
        level: int = DEFAULT_LEVEL,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        self.wavelet = wavelet
        self.level = level
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        self._trained: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, waveform: NDArray[np.float64]) -> PQResult:
        """Classify a single voltage waveform.

        Parameters
        ----------
        waveform:
            1-D array of voltage samples.

        Returns
        -------
        PQResult
            Predicted label and confidence.

        Raises
        ------
        RuntimeError
            If called before the model has been trained or loaded.
        """
        if not self._trained:
            raise RuntimeError(
                "Model has not been trained. Call .train() or .load() first."
            )
        features = extract_dwt_features(
            waveform, wavelet=self.wavelet, level=self.level
        )
        X = features.reshape(1, -1)

        label_idx: int = int(self._pipeline.predict(X)[0])
        proba: NDArray[np.float64] = self._pipeline.predict_proba(X)[0]
        confidence = float(np.clip(proba[label_idx], 0.0, 1.0))
        label = self.CLASSES[label_idx]

        return PQResult(label=label, confidence=confidence)

    def train(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.intp],
    ) -> None:
        """Fit the pipeline on pre-extracted feature matrix ``X`` and labels ``y``.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)``.
        y:
            Integer label array of shape ``(n_samples,)``,
            values in ``range(len(CLASSES))``.
        """
        self._pipeline.fit(X, y)
        self._trained = True

    def save(self, path: Path = DEFAULT_MODEL_PATH) -> None:
        """Persist the fitted pipeline to disk.

        Parameters
        ----------
        path:
            Destination ``.joblib`` file.  Parent directories are created
            automatically.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if not self._trained:
            raise RuntimeError("Cannot save an untrained model.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self._pipeline,
                "wavelet": self.wavelet,
                "level": self.level,
                "classes": self.CLASSES,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL_PATH) -> PQClassifier:
        """Load a pre-trained classifier from a ``.joblib`` artifact.

        Parameters
        ----------
        path:
            Path to the ``.joblib`` file written by :meth:`save`.

        Returns
        -------
        PQClassifier
            Ready-to-use classifier instance.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{path}'. "
                "Run the training pipeline first, or pass the correct path."
            )
        payload: dict[str, Any] = joblib.load(path)
        instance = cls(wavelet=payload["wavelet"], level=payload["level"])
        instance._pipeline = payload["pipeline"]
        instance._trained = True
        return instance


# ---------------------------------------------------------------------------
# Training data generator (synthetic IEEE 1159 waveforms)
# ---------------------------------------------------------------------------


def generate_synthetic_dataset(
    n_per_class: int = 300,
    samples: int = 1024,
    fs: float = 6400.0,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Generate a balanced synthetic PQ dataset for training / testing.

    Each waveform is a 60 Hz fundamental with class-specific disturbances
    applied per IEEE 1159-2019 definitions.

    Parameters
    ----------
    n_per_class:
        Number of waveforms per disturbance class.
    samples:
        Number of samples per waveform.
    fs:
        Sampling frequency in Hz.
    seed:
        NumPy random seed for reproducibility.

    Returns
    -------
    X : NDArray, shape ``(n_total, n_features)``
        Feature matrix ready for :meth:`PQClassifier.train`.
    y : NDArray, shape ``(n_total,)``
        Integer labels corresponding to :data:`DISTURBANCE_CLASSES`.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, samples / fs, samples, endpoint=False)
    f0 = 60.0
    fundamental = np.sin(2 * np.pi * f0 * t)

    generators = [
        lambda: _waveform_normal(fundamental, rng),
        lambda: _waveform_sag(fundamental, t, f0, rng),
        lambda: _waveform_swell(fundamental, t, f0, rng),
        lambda: _waveform_interruption(fundamental, t, f0, rng),
        lambda: _waveform_harmonics(t, f0, rng),
        lambda: _waveform_transient(fundamental, t, rng),
    ]

    X_list: list[NDArray[np.float64]] = []
    y_list: list[int] = []

    for class_idx, gen in enumerate(generators):
        for _ in range(n_per_class):
            wave = gen()
            feats = extract_dwt_features(wave)
            X_list.append(feats)
            y_list.append(class_idx)

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.intp)

    # Shuffle
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Synthetic waveform builders (private)
# ---------------------------------------------------------------------------


def _waveform_normal(
    fundamental: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    noise = rng.normal(0, 0.02, fundamental.shape)
    return fundamental + noise


def _waveform_sag(
    fundamental: NDArray[np.float64],
    t: NDArray[np.float64],
    f0: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    wave = fundamental.copy()
    signal_len = t[-1]
    depth = rng.uniform(0.1, 0.5)          # sag to 50–90 % of nominal
    max_duration = signal_len * 0.4        # at most 40 % of the signal
    duration = min(rng.uniform(0.5, 10) / f0, max_duration)
    start = rng.uniform(0.0, max(0.0, signal_len - duration))
    mask = (t >= start) & (t < start + duration)
    wave[mask] *= (1.0 - depth)
    return wave


def _waveform_swell(
    fundamental: NDArray[np.float64],
    t: NDArray[np.float64],
    f0: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    wave = fundamental.copy()
    signal_len = t[-1]
    boost = rng.uniform(0.1, 0.4)          # swell to 110–140 % of nominal
    max_duration = signal_len * 0.4
    duration = min(rng.uniform(0.5, 10) / f0, max_duration)
    start = rng.uniform(0.0, max(0.0, signal_len - duration))
    mask = (t >= start) & (t < start + duration)
    wave[mask] *= (1.0 + boost)
    return wave


def _waveform_interruption(
    fundamental: NDArray[np.float64],
    t: NDArray[np.float64],
    f0: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    wave = fundamental.copy()
    signal_len = t[-1]
    max_duration = signal_len * 0.4
    duration = min(rng.uniform(1.0, 5.0) / f0, max_duration)
    start = rng.uniform(0.0, max(0.0, signal_len - duration))
    mask = (t >= start) & (t < start + duration)
    wave[mask] *= rng.uniform(0.0, 0.1)    # < 10 % residual
    return wave


def _waveform_harmonics(
    t: NDArray[np.float64],
    f0: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    h3 = rng.uniform(0.05, 0.20) * np.sin(2 * np.pi * 3 * f0 * t)
    h5 = rng.uniform(0.05, 0.15) * np.sin(2 * np.pi * 5 * f0 * t)
    h7 = rng.uniform(0.01, 0.08) * np.sin(2 * np.pi * 7 * f0 * t)
    return np.sin(2 * np.pi * f0 * t) + h3 + h5 + h7


def _waveform_transient(
    fundamental: NDArray[np.float64],
    t: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    wave = fundamental.copy()
    idx = rng.integers(len(t) // 4, 3 * len(t) // 4)
    width = rng.integers(2, 8)
    amplitude = rng.uniform(1.5, 4.0)
    spike = np.zeros_like(wave)
    spike[max(0, idx - width): idx + width] = amplitude * np.sign(
        rng.choice([-1, 1])
    )
    return wave + spike
