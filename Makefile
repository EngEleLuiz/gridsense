.PHONY: install lint lint-fix typecheck test test-unit test-integration test-all coverage \
        build check-build clean docker-up docker-down train-models smoke help

# ── Variables ──────────────────────────────────────────────────────────────
PYTHON   := python
SRC      := src/
TESTS    := tests/
COVERAGE := 80

# ── Help ───────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "GridSense — available make targets"
	@echo "─────────────────────────────────────────────────────────"
	@echo "  make smoke            Quick end-to-end sanity check (no Docker needed)"
	@echo "  make install          Install package + all dev dependencies"
	@echo "  make lint             Run ruff linter"
	@echo "  make typecheck        Run mypy type checker"
	@echo "  make test-unit        Run unit tests (no DB needed)"
	@echo "  make test-integration Run integration tests (ASGI, no DB needed)"
	@echo "  make test-all         Run all tests with coverage report"
	@echo "  make coverage         Open HTML coverage report in browser"
	@echo "  make build            Build source dist + wheel"
	@echo "  make check-build      Build + run twine check"
	@echo "  make train-models     Train and save ML model artifacts"
	@echo "  make docker-up        Start full stack with Docker Compose"
	@echo "  make docker-down      Stop and remove containers + volumes"
	@echo "  make clean            Remove build artifacts and caches"
	@echo "  make release          Full pre-release check (lint+types+tests+build)"
	@echo "─────────────────────────────────────────────────────────"
	@echo ""

# ── Install ────────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev,dashboard]"

# ── Lint ───────────────────────────────────────────────────────────────────
lint:
	ruff check $(SRC) $(TESTS)

lint-fix:
	ruff check $(SRC) $(TESTS) --fix

# ── Type check ─────────────────────────────────────────────────────────────
typecheck:
	mypy $(SRC)gridsense --ignore-missing-imports

# ── Tests ──────────────────────────────────────────────────────────────────
test-unit:
	pytest $(TESTS)unit/ -v \
		--cov=gridsense \
		--cov-report=term-missing \
		--cov-fail-under=$(COVERAGE)

test-integration:
	pytest $(TESTS)integration/ -v

test-all:
	pytest $(TESTS) -v \
		--cov=gridsense \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-fail-under=$(COVERAGE)

coverage:
	pytest $(TESTS)unit/ \
		--cov=gridsense \
		--cov-report=html \
		-q
	@echo "Opening coverage report..."
	@python -m webbrowser htmlcov/index.html 2>/dev/null || \
		xdg-open htmlcov/index.html 2>/dev/null || \
		open htmlcov/index.html 2>/dev/null || \
		echo "Open htmlcov/index.html in your browser."

# ── Build ──────────────────────────────────────────────────────────────────
build:
	pip install --quiet build
	$(PYTHON) -m build

check-build: build
	pip install --quiet twine
	twine check dist/*

# ── Model training ─────────────────────────────────────────────────────────
train-models:
	@echo "Training solar forecasting model..."
	$(PYTHON) -c "\
from gridsense.forecast.trainer import train_and_save; \
from pathlib import Path; \
f = train_and_save(n_days=90, model_path=Path('artifacts/solar_model.joblib'), seed=42); \
print(f'Solar model saved. Residual std={f._residual_std:.4f} kW')"
	@echo "Training PQ classifier..."
	$(PYTHON) -c "\
from gridsense.pq.classifier import PQClassifier, generate_synthetic_dataset; \
from pathlib import Path; \
X, y = generate_synthetic_dataset(n_per_class=500, seed=42); \
clf = PQClassifier(n_estimators=300, random_state=42); \
clf.train(X, y); \
clf.save(Path('artifacts/pq_model.joblib')); \
print('PQ model saved.')"
	@echo "✓ All model artifacts saved to ./artifacts/"

# ── Docker ─────────────────────────────────────────────────────────────────
docker-up:
	docker-compose up --build

docker-down:
	docker-compose down -v

# ── Full release check ─────────────────────────────────────────────────────
release: lint typecheck test-all check-build
	@echo ""
	@echo "═══════════════════════════════════════════"
	@echo "  ✓ All checks passed — ready to release!"
	@echo "═══════════════════════════════════════════"
	@echo ""
	@echo "  Next steps:"
	@echo "  1. git tag v0.1.0"
	@echo "  2. git push origin v0.1.0"
	@echo "  3. GitHub Actions will publish to PyPI automatically."
	@echo ""

# ── Smoke test ─────────────────────────────────────────────────────────────
smoke:
	$(PYTHON) scripts/smoke_test.py

# ── Clean ──────────────────────────────────────────────────────────────────
clean:
	rm -rf dist/ build/ htmlcov/ .coverage coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned."
