#!/usr/bin/env python
"""
Train and save all GridSense ML model artifacts.

Run this once before starting the API or running integration tests
that depend on pre-trained models.

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --days 180 --estimators 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def train_solar(n_days: int, n_estimators: int) -> None:
    from gridsense.forecast.trainer import train_and_save

    print(f"  Training solar forecaster on {n_days} days of synthetic data...")
    forecaster = train_and_save(
        n_days=n_days,
        model_path=Path("artifacts/solar_model.joblib"),
        seed=42,
    )
    print(f"  ✓ Saved. Residual std = {forecaster._residual_std:.4f} kW")


def train_pq(n_per_class: int, n_estimators: int) -> None:
    from gridsense.pq.classifier import (
        DISTURBANCE_CLASSES,
        PQClassifier,
        generate_synthetic_dataset,
    )

    print(f"  Generating PQ dataset ({n_per_class} samples × {len(DISTURBANCE_CLASSES)} classes)...")
    X, y = generate_synthetic_dataset(n_per_class=n_per_class, seed=42)

    print(f"  Training Random Forest ({n_estimators} trees)...")
    clf = PQClassifier(n_estimators=n_estimators, random_state=42)
    clf.train(X, y)
    clf.save(Path("artifacts/pq_model.joblib"))

    # Quick accuracy check
    from sklearn.metrics import accuracy_score
    preds = [clf._pipeline.predict(X[i].reshape(1, -1))[0] for i in range(min(200, len(X)))]
    acc = accuracy_score(y[: len(preds)], preds)
    print(f"  ✓ Saved. Train accuracy = {acc:.1%}")


def verify_artifacts() -> None:
    from gridsense.forecast.solar import SolarForecaster
    from gridsense.forecast.trainer import generate_training_data
    from gridsense.pq.classifier import PQClassifier
    import numpy as np

    print("\nVerifying artifacts...")

    # Solar
    f = SolarForecaster.load(Path("artifacts/solar_model.joblib"))
    conditions = generate_training_data(n_days=1).tail(24)
    df = f.predict_next_24h(conditions)
    assert len(df) == 24
    print("  ✓ Solar model: loaded and produces 24-row forecast")

    # PQ
    clf = PQClassifier.load(Path("artifacts/pq_model.joblib"))
    wave = np.sin(2 * np.pi * 60 * np.linspace(0, 1, 1024))
    result = clf.predict(wave)
    assert result.label in ["normal", "voltage_sag", "voltage_swell",
                             "interruption", "harmonics", "transient"]
    print(f"  ✓ PQ model: loaded, predicts '{result.label}' (confidence={result.confidence:.2%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all GridSense ML artifacts.")
    parser.add_argument("--days", type=int, default=90,
                        help="Days of synthetic solar data (default: 90)")
    parser.add_argument("--estimators", type=int, default=300,
                        help="Number of trees in each forest (default: 300)")
    parser.add_argument("--pq-samples", type=int, default=500,
                        help="PQ training samples per class (default: 500)")
    args = parser.parse_args()

    Path("artifacts").mkdir(exist_ok=True)

    print("\n═══════════════════════════════════════════")
    print("  GridSense — Model Training")
    print("═══════════════════════════════════════════\n")

    print("1. Solar generation forecaster")
    train_solar(n_days=args.days, n_estimators=args.estimators)

    print("\n2. Power quality classifier")
    train_pq(n_per_class=args.pq_samples, n_estimators=args.estimators)

    verify_artifacts()

    print("\n═══════════════════════════════════════════")
    print("  ✓ All artifacts saved to ./artifacts/")
    print("═══════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
