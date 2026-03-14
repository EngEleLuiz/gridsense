#!/usr/bin/env python
"""
GridSense smoke test — runs in under 30 seconds, no Docker needed.

Exercises the full import chain, trains tiny models, runs predictions,
and calls the FastAPI ASGI app. Use this as a fast sanity check
before pushing or after making changes.

Usage:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PASS = "✓"
FAIL = "✗"
results: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        fn()
        results.append((name, True, ""))
        print(f"  {PASS}  {name}")
    except Exception:
        tb = traceback.format_exc(limit=3)
        results.append((name, False, tb))
        print(f"  {FAIL}  {name}")
        print(f"      {tb.splitlines()[-1]}")


# ── 1. Imports ───────────────────────────────────────────────────────────────

print("\n[1/6] Imports")


def _imports():
    import gridsense  # noqa: F401
    from gridsense.pq.features import extract_dwt_features  # noqa: F401
    from gridsense.pq.classifier import PQClassifier  # noqa: F401
    from gridsense.battery.soc import SoCEstimator  # noqa: F401
    from gridsense.forecast.solar import SolarForecaster  # noqa: F401
    from gridsense.forecast.trainer import generate_training_data  # noqa: F401
    from gridsense.ingest.modbus import SimulatedModbusReader  # noqa: F401
    from gridsense.api.main import app  # noqa: F401


check("All modules importable", _imports)

# ── 2. PQ features ───────────────────────────────────────────────────────────

print("\n[2/6] PQ feature extraction")

import numpy as np


def _features_shape():
    from gridsense.pq.features import extract_dwt_features
    wave = np.sin(2 * np.pi * 60 * np.linspace(0, 1, 1024))
    f = extract_dwt_features(wave)
    assert f.shape == (30,), f"Expected (30,), got {f.shape}"


def _features_sag_differs():
    from gridsense.pq.features import extract_dwt_features
    clean = np.sin(2 * np.pi * 60 * np.linspace(0, 1, 1024))
    sag = clean.copy()
    sag[300:700] *= 0.5
    assert not np.allclose(extract_dwt_features(clean), extract_dwt_features(sag))


check("Feature vector shape (30,)", _features_shape)
check("Sag produces different features from clean", _features_sag_differs)

# ── 3. PQ classifier ─────────────────────────────────────────────────────────

print("\n[3/6] PQ classifier (quick train)")

from gridsense.pq.classifier import PQClassifier, generate_synthetic_dataset


def _classifier_train_predict():
    X, y = generate_synthetic_dataset(n_per_class=30, seed=0)
    clf = PQClassifier(n_estimators=20, random_state=0)
    clf.train(X, y)
    wave = np.sin(2 * np.pi * 60 * np.linspace(0, 1, 1024))
    result = clf.predict(wave)
    assert result.label in ["normal", "voltage_sag", "voltage_swell",
                             "interruption", "harmonics", "transient"]
    assert 0.0 <= result.confidence <= 1.0


def _classifier_save_load(tmp=Path("/tmp/pq_smoke.joblib")):
    X, y = generate_synthetic_dataset(n_per_class=30, seed=0)
    clf = PQClassifier(n_estimators=20, random_state=0)
    clf.train(X, y)
    clf.save(tmp)
    loaded = PQClassifier.load(tmp)
    wave = np.sin(2 * np.pi * 60 * np.linspace(0, 1, 1024))
    r1 = clf.predict(wave)
    r2 = loaded.predict(wave)
    assert r1.label == r2.label
    tmp.unlink(missing_ok=True)


check("Train + predict returns valid result", _classifier_train_predict)
check("Save → load → predict is consistent", _classifier_save_load)

# ── 4. Battery SoC ───────────────────────────────────────────────────────────

print("\n[4/6] Battery SoC estimator")

from gridsense.battery.soc import SoCEstimator


def _soc_discharge():
    est = SoCEstimator(capacity_ah=10.0, initial_soc=1.0, coulombic_efficiency=1.0)
    # Discharge 10 Ah at 2 A → 5 h = 18000 s in 1000 steps
    for _ in range(1000):
        est.update(current_a=-2.0, dt_seconds=18.0)
    assert est.soc < 0.02, f"Expected ~0, got {est.soc}"


def _soc_clamp():
    est = SoCEstimator(capacity_ah=10.0, initial_soc=0.99)
    for _ in range(100):
        est.update(current_a=100.0, dt_seconds=60.0)
    assert est.soc == 1.0


check("Full discharge reaches ~0", _soc_discharge)
check("Overcharge is clamped to 1.0", _soc_clamp)

# ── 5. Solar forecaster ──────────────────────────────────────────────────────

print("\n[5/6] Solar forecaster (quick train)")

from gridsense.forecast.trainer import generate_training_data, train_and_save
from gridsense.forecast.solar import SolarForecaster


def _forecast_train_predict():
    df = generate_training_data(n_days=10, seed=0)
    f = SolarForecaster(n_estimators=20, random_state=0)
    f.train(df)
    conditions = df.tail(24).reset_index(drop=True)
    result = f.predict_next_24h(conditions)
    assert len(result) == 24
    assert (result["predicted_kw"] >= 0).all()
    assert (result["lower_bound"] <= result["predicted_kw"]).all()
    assert (result["predicted_kw"] <= result["upper_bound"]).all()


def _forecast_save_load():
    tmp = Path("/tmp/solar_smoke.joblib")
    df = generate_training_data(n_days=10, seed=0)
    f = train_and_save(df=df, model_path=tmp, seed=0)
    loaded = SolarForecaster.load(tmp)
    conditions = df.tail(24).reset_index(drop=True)
    r1 = f.predict_next_24h(conditions)
    r2 = loaded.predict_next_24h(conditions)
    import pandas as pd
    pd.testing.assert_frame_equal(r1, r2)
    tmp.unlink(missing_ok=True)


check("Train + predict returns 24-row DataFrame", _forecast_train_predict)
check("Save → load → predict is consistent", _forecast_save_load)

# ── 6. FastAPI ───────────────────────────────────────────────────────────────

print("\n[6/6] FastAPI endpoints (ASGI, no server)")

import asyncio


async def _run_api_checks():
    from httpx import ASGITransport, AsyncClient
    from gridsense.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:

        def check_async(name, coro):
            results.append((name, True, ""))  # placeholder — updated below
            return coro

        # /healthz
        r = await client.get("/healthz")
        assert r.status_code == 200 and r.json()["status"] == "ok"
        print(f"  {PASS}  GET /healthz → 200 ok")

        # POST /predict/solar
        r = await client.post(
            "/api/v1/predict/solar",
            json={"station_code": "A801", "horizon_hours": 24},
        )
        assert r.status_code == 200
        assert len(r.json()["predictions"]) == 24
        print(f"  {PASS}  POST /api/v1/predict/solar → 200, 24 predictions")

        # Validation error — empty station_code
        r = await client.post(
            "/api/v1/predict/solar",
            json={"station_code": "", "horizon_hours": 24},
        )
        assert r.status_code == 422
        print(f"  {PASS}  POST /api/v1/predict/solar (empty station) → 422")

        # GET /battery/soc
        r = await client.get("/api/v1/battery/soc")
        assert r.status_code == 200
        data = r.json()
        assert 0.0 <= data["soc"] <= 1.0
        print(f"  {PASS}  GET /api/v1/battery/soc → 200, soc={data['soc']}")

        # GET /events/pq
        r = await client.get(
            "/api/v1/events/pq",
            params={"start": "2025-12-01T00:00:00Z", "end": "2025-12-02T00:00:00Z"},
        )
        assert r.status_code == 200
        print(f"  {PASS}  GET /api/v1/events/pq → 200")


try:
    asyncio.run(_run_api_checks())
except Exception:
    tb = traceback.format_exc(limit=3)
    print(f"  {FAIL}  API checks failed")
    print(f"      {tb.splitlines()[-1]}")
    results.append(("API checks", False, tb))

# ── Summary ──────────────────────────────────────────────────────────────────

total = len(results)
passed = sum(1 for _, ok, _ in results if ok)

print("\n═══════════════════════════════════════════")
print(f"  Smoke test: {passed}/{total} passed")
if passed == total:
    print("  ✓ All checks passed — project is healthy!")
else:
    print("  ✗ Some checks failed — see output above.")
    for name, ok, tb in results:
        if not ok:
            print(f"\n  FAILED: {name}")
            print(tb)
print("═══════════════════════════════════════════\n")

sys.exit(0 if passed == total else 1)
