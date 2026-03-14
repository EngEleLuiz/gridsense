# GridSense — Complete Testing Guide

Everything you need to run, validate, and verify the project before releasing
to PyPI and making the dashboard public.

---

## Prerequisites

```bash
# Python 3.10 or 3.11 (check yours)
python --version

# Install the package in editable mode with all dev dependencies
cd gridsense
pip install -e ".[dev,dashboard]"

# Verify the install
python -c "import gridsense; print(gridsense.__version__)"
```

---

## Step 1 — Linting (catch style and import errors)

```bash
# Run ruff on all source and test code
ruff check src/ tests/

# Auto-fix what can be fixed automatically
ruff check src/ tests/ --fix

# Type checking with mypy
mypy src/gridsense --ignore-missing-imports
```

**Expected output:** no errors. Any `E`, `F`, or `I` codes mean something
needs fixing before you push.

---

## Step 2 — Unit Tests (no database, no network)

These tests run entirely in memory with mocks. They should pass on any machine.

```bash
# Run all unit tests with verbose output
pytest tests/unit/ -v

# Run with coverage report in the terminal
pytest tests/unit/ -v \
  --cov=gridsense \
  --cov-report=term-missing \
  --cov-fail-under=80

# Generate an HTML coverage report (open in browser)
pytest tests/unit/ \
  --cov=gridsense \
  --cov-report=html
open htmlcov/index.html   # macOS
# xdg-open htmlcov/index.html   # Linux
```

### Run individual test modules

```bash
# Power Quality feature extraction
pytest tests/unit/test_features.py -v

# PQ disturbance classifier
pytest tests/unit/test_classifier.py -v

# Battery SoC estimator
pytest tests/unit/test_soc.py -v

# Solar forecaster
pytest tests/unit/test_forecast.py -v

# Modbus reader (mocked hardware)
pytest tests/unit/test_modbus.py -v

# INMET weather client (mocked HTTP)
pytest tests/unit/test_weather.py -v
```

### Run a single test by name

```bash
pytest tests/unit/test_soc.py::TestUpdate::test_zero_current_no_change -v
```

**Expected:** all ~80 tests pass, coverage ≥ 80 %.

---

## Step 3 — Integration Tests (needs the API running in-process)

The integration tests use `httpx.AsyncClient` against the ASGI app directly —
no real server needed, but the app must import cleanly.

```bash
pytest tests/integration/ -v
```

If the tests fail with import errors, make sure you installed with `pip install -e ".[dev]"`.

---

## Step 4 — Test the FastAPI locally (manual smoke test)

```bash
# Start the API
uvicorn gridsense.api.main:app --reload --port 8000
```

Open a second terminal and run:

```bash
# Health check
curl http://localhost:8000/healthz
# Expected: {"status":"ok","version":"0.1.0"}

# Solar forecast
curl -X POST http://localhost:8000/api/v1/predict/solar \
  -H "Content-Type: application/json" \
  -d '{"station_code": "A801", "horizon_hours": 24}'
# Expected: {"predictions":[...24 items...],"model_version":"0.1.0",...}

# PQ events (empty is fine — no DB yet)
curl "http://localhost:8000/api/v1/events/pq?start=2025-12-01T00:00:00Z&end=2025-12-02T00:00:00Z"
# Expected: {"events":[],"total":0,...}

# Battery SoC
curl http://localhost:8000/api/v1/battery/soc
# Expected: {"soc":1.0,"soc_percent":100.0,...}
```

Or open the interactive docs at **http://localhost:8000/docs** and test
everything from the browser UI.

---

## Step 5 — Test the Streamlit Dashboard locally

```bash
# In a separate terminal (API must be running from Step 4)
streamlit run dashboard/app.py
```

Open **http://localhost:8501** and verify:

- [ ] Header shows "● API Online" (green)
- [ ] Solar forecast chart renders with 24 data points
- [ ] Confidence band (orange shading) is visible
- [ ] Battery SoC gauge shows a value
- [ ] No Python tracebacks in the terminal

If the API is not running, the dashboard falls back to demo data automatically —
you should still see charts, but the status pill will say "● API Offline".

---

## Step 6 — Test the full stack with Docker Compose

This is the closest thing to a production environment.

```bash
# Build and start all services
docker-compose up --build

# Wait until you see:
#   api        | INFO:     Application startup complete.
#   dashboard  | You can now view your Streamlit app in your browser.

# In a second terminal — run the smoke tests against Docker
curl http://localhost:8000/healthz
curl http://localhost:8501   # dashboard HTML
```

### Verify the database is reachable

```bash
# Connect to TimescaleDB inside Docker
# Note: host port is 5433 (not 5432) to avoid conflicts with local PostgreSQL
docker exec -it gridsense-timescaledb-1 \
  psql -U gridsense -d gridsense -c "\dt"
# Expected: list of 5 tables (solar_readings, weather_readings, etc.)
```

If you need to connect from your host machine (e.g. with DBeaver or psql):
- Host: `localhost`
- Port: `5433` ← not 5432
- User: `gridsense`
- Password: `gridsense`
- Database: `gridsense`

### Run the ingest pipeline manually inside Docker

```bash
docker exec -it gridsense-pipeline-1 \
  python -m pipelines.ingest_flow
# Expected: logs showing weather fetch + inverter reading + DB save
```

### Tear down cleanly

```bash
docker-compose down -v   # -v removes the DB volume too
```

---

## Step 7 — Train and save the model artifact

The API auto-trains on first call, but for a real release you want a
pre-trained artifact committed (or uploaded to a release asset).

```bash
# Train on synthetic data and save artifact
python - <<'EOF'
from gridsense.forecast.trainer import train_and_save
from pathlib import Path

forecaster = train_and_save(
    n_days=90,
    model_path=Path("artifacts/solar_model.joblib"),
    seed=42,
)
print("Model saved. Residual std:", forecaster._residual_std)
EOF

# Verify it loads cleanly
python - <<'EOF'
from gridsense.forecast.solar import SolarForecaster
from pathlib import Path

f = SolarForecaster.load(Path("artifacts/solar_model.joblib"))
print("Loaded OK. Version:", f.MODEL_VERSION)
EOF
```

Also train the PQ classifier:

```bash
python - <<'EOF'
from gridsense.pq.classifier import PQClassifier, generate_synthetic_dataset
from pathlib import Path

X, y = generate_synthetic_dataset(n_per_class=500, seed=42)
clf = PQClassifier(n_estimators=300, random_state=42)
clf.train(X, y)
clf.save(Path("artifacts/pq_model.joblib"))
print("PQ model saved.")
EOF
```

---

## Step 8 — Verify the package builds cleanly

```bash
pip install build twine

# Build source distribution and wheel
python -m build

# Check the distribution for common packaging errors
twine check dist/*

# List the files that will be shipped
python -m zipfile -l dist/gridsense-0.1.0-py3-none-any.whl
```

Expected: no warnings from `twine check`.

---

## Step 9 — Test install from the built wheel

This proves `pip install gridsense` will work after publishing.

```bash
# Create a fresh virtual environment
python -m venv /tmp/gridsense-test-env
source /tmp/gridsense-test-env/bin/activate

# Install from the local wheel (not PyPI)
pip install dist/gridsense-0.1.0-py3-none-any.whl

# Verify it works
python - <<'EOF'
from gridsense.pq.features import extract_dwt_features
from gridsense.battery.soc import SoCEstimator
import numpy as np

wave = np.sin(2 * np.pi * 60 * np.linspace(0, 1, 1024))
feats = extract_dwt_features(wave)
print(f"Features shape: {feats.shape}")   # (30,)

est = SoCEstimator(capacity_ah=10.0, initial_soc=0.8)
print(f"Initial SoC: {est.soc_percent:.1f}%")   # 80.0%
EOF

deactivate
rm -rf /tmp/gridsense-test-env
```

---

## Step 10 — Full pre-release checklist

Run everything in sequence with one command:

```bash
# 1. Lint
ruff check src/ tests/ && echo "✓ Lint passed"

# 2. Type check
mypy src/gridsense --ignore-missing-imports && echo "✓ Types OK"

# 3. Unit tests + coverage gate
pytest tests/unit/ \
  --cov=gridsense \
  --cov-report=term-missing \
  --cov-fail-under=80 \
  -q && echo "✓ Unit tests passed"

# 4. Integration tests
pytest tests/integration/ -q && echo "✓ Integration tests passed"

# 5. Package build
python -m build && twine check dist/* && echo "✓ Package builds cleanly"
```

If all five echo lines print, you are ready to release. 🎉

---

## Common Failures and Fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: gridsense` | Package not installed | `pip install -e ".[dev]"` |
| `FileNotFoundError: pq_model.joblib` | Artifact not trained yet | Run Step 7 above |
| `ConnectionError: timescaledb` | Docker not running | `docker-compose up timescaledb` |
| `pytest: no tests ran` | Wrong directory | Run from the `gridsense/` root |
| `ruff: E501 line too long` | Line > 88 chars | Shorten the line or add `# noqa: E501` |
| `mypy: Missing return type` | Missing annotation | Add `-> None` or correct type |
| Coverage below 80 % | New code missing tests | Add tests for the new module |
| `twine check` warning | Missing metadata | Fill in `pyproject.toml` fields |
