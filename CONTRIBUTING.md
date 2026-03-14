# Contributing to GridSense

Thank you for your interest in contributing! GridSense is an open-source
platform for solar generation forecasting and power quality monitoring, built
by electrical engineers transitioning into data/ML engineering.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Coding Standards](#coding-standards)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Good First Issues](#good-first-issues)
- [Architecture Overview](#architecture-overview)

---

## Code of Conduct

Be respectful, constructive, and patient. We welcome contributors of all
experience levels.

---

## How to Contribute

You can help in several ways:

- **Bug reports** — open an issue with a minimal reproducible example
- **Feature requests** — open an issue describing the use case first
- **Pull requests** — see the workflow below
- **Documentation** — improve docstrings, examples, or this guide
- **Real data** — share INMET station data or ANEEL generation datasets
- **Domain knowledge** — improve the IEEE 1159 disturbance models or the
  battery SoC estimator physics

---

## Development Setup

### 1. Fork and clone

```bash
git clone https://github.com/YOUR_USERNAME/gridsense.git
cd gridsense
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install in editable mode with dev dependencies

```bash
pip install -e ".[dev,dashboard]"
```

### 4. Verify the setup

```bash
pytest tests/unit/ -q
# Expected: all tests pass
```

---

## Running Tests

```bash
# Fastest — unit tests only (no DB, no Docker)
make test-unit

# All tests
make test-all

# Just lint
make lint

# Full pre-release check (what CI runs)
make release
```

See [TESTING.md](TESTING.md) for the complete step-by-step testing guide,
including how to run with Docker Compose and how to smoke-test the API.

---

## Coding Standards

### Style

- Formatter / linter: **ruff** (`make lint`)
- Line length: **88** characters
- Type hints: **required** on all public functions and methods
- Docstrings: **Google style**, required on all public classes and functions

### Structure rules

| Rule | Rationale |
|------|-----------|
| No business logic in `api/routers/` | Routers delegate to domain modules |
| No DB calls in `pq/` or `battery/` | Domain modules must be DB-agnostic |
| Every public interface must have a unit test | Keeps coverage above 80 % |
| New ML models go in `artifacts/` (gitignored) | Keep repo size small |
| External HTTP calls must be mockable | Pass `session=` or use dependency injection |

### Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add harmonic distortion severity score to PQResult
fix: handle None irradiance in INMETClient when station is offline
test: add edge case for SoCEstimator.reset() with soc=0.0
docs: document ModbusRegisterMap fields with unit annotations
refactor: extract _band_stats helper from extract_dwt_features
```

---

## Submitting a Pull Request

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```

2. **Write your code** following the standards above.

3. **Add or update tests** — new features must include tests; bug fixes
   should include a regression test.

4. **Run the full check locally:**
   ```bash
   make release
   ```
   All checks must pass before opening a PR.

5. **Push and open a PR** against `main`.

6. **Fill in the PR template** — describe what changed, why, and how to test it.

CI will run automatically. A maintainer will review within a few days.

---

## Good First Issues

Look for issues labelled
[`good first issue`](https://github.com/YOUR_USERNAME/gridsense/issues?q=label%3A%22good+first+issue%22)
on the issue tracker. Good starting points:

- Adding more synthetic waveform types (notch, flicker, DC offset) to
  `pq/classifier.py::generate_synthetic_dataset`
- Extending `INMETClient` to support additional meteorological variables
  (wind speed, cloud cover)
- Adding a `--station` CLI flag to `pipelines/ingest_flow.py`
- Writing tests for `pq/events.py`
- Improving the Streamlit dashboard with a date-range picker

---

## Architecture Overview

```
Data Sources
  Solar array (Modbus) ──► gridsense.ingest.modbus
  INMET Weather API    ──► gridsense.ingest.weather
  Battery sensors      ──► gridsense.ingest.battery
         │
         ▼
  Prefect DAG (15-min cadence)
         │
         ▼
  TimescaleDB (time-series storage)
         │
    ┌────┴────┐
    ▼         ▼
  Forecast   PQ classifier + Battery SoC
    │              │
    └──────┬────────┘
           ▼
     FastAPI REST API
           │
           ▼
     Streamlit Dashboard
```

Each sub-package (`pq`, `battery`, `forecast`, `ingest`) is independent and
can be used standalone:

```python
# Use the PQ classifier without any database or API
from gridsense.pq.classifier import PQClassifier
import numpy as np

clf = PQClassifier.load()
result = clf.predict(np.sin(2 * np.pi * 60 * np.linspace(0, 1, 1024)))
```

---

## Questions?

Open a [GitHub Discussion](https://github.com/YOUR_USERNAME/gridsense/discussions)
or file an issue. We're happy to help.
