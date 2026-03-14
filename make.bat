@echo off
REM GridSense — Windows command runner (replacement for make)
REM Usage: make.bat <target>
REM Example: make.bat test-unit

set TARGET=%1

if "%TARGET%"=="" goto help
if "%TARGET%"=="help" goto help
if "%TARGET%"=="install" goto install
if "%TARGET%"=="lint" goto lint
if "%TARGET%"=="lint-fix" goto lint-fix
if "%TARGET%"=="typecheck" goto typecheck
if "%TARGET%"=="test-unit" goto test-unit
if "%TARGET%"=="test-integration" goto test-integration
if "%TARGET%"=="test-all" goto test-all
if "%TARGET%"=="coverage" goto coverage
if "%TARGET%"=="smoke" goto smoke
if "%TARGET%"=="train-models" goto train-models
if "%TARGET%"=="build" goto build
if "%TARGET%"=="check-build" goto check-build
if "%TARGET%"=="release" goto release
if "%TARGET%"=="clean" goto clean
if "%TARGET%"=="docker-up" goto docker-up
if "%TARGET%"=="docker-down" goto docker-down

echo Unknown target: %TARGET%
goto help

REM ── Help ──────────────────────────────────────────────────────────────────
:help
echo.
echo GridSense — available targets
echo ─────────────────────────────────────────────────────────
echo   make.bat install          Install package + all dev dependencies
echo   make.bat smoke            Quick end-to-end sanity check (no Docker)
echo   make.bat lint             Run ruff linter
echo   make.bat lint-fix         Run ruff and auto-fix issues
echo   make.bat typecheck        Run mypy type checker
echo   make.bat test-unit        Run unit tests (no DB needed)
echo   make.bat test-integration Run integration tests
echo   make.bat test-all         Run all tests with coverage report
echo   make.bat coverage         Run tests and open HTML coverage report
echo   make.bat train-models     Train and save ML model artifacts
echo   make.bat build            Build source dist + wheel
echo   make.bat check-build      Build + run twine check
echo   make.bat release          Full pre-release check
echo   make.bat docker-up        Start full stack with Docker Compose
echo   make.bat docker-down      Stop containers + remove volumes
echo   make.bat clean            Remove build artifacts and caches
echo ─────────────────────────────────────────────────────────
echo.
goto end

REM ── Install ───────────────────────────────────────────────────────────────
:install
pip install -e ".[dev,dashboard]"
goto end

REM ── Lint ──────────────────────────────────────────────────────────────────
:lint
ruff check src/ tests/
goto end

:lint-fix
ruff check src/ tests/ --fix
goto end

REM ── Type check ────────────────────────────────────────────────────────────
:typecheck
mypy src/gridsense --ignore-missing-imports
goto end

REM ── Tests ─────────────────────────────────────────────────────────────────
:test-unit
echo Running unit tests (API/DB omitted from coverage - use test-all for full picture)
pytest tests/unit/ -v --cov=gridsense --cov-report=term-missing --cov-fail-under=80
goto end

:test-integration
pytest tests/integration/ -v
goto end

:test-all
echo Running ALL tests with full coverage (unit + integration combined)
pytest tests/ -v --cov=gridsense --cov-report=term-missing --cov-report=html --cov-fail-under=70
goto end
goto end

:test-all
pytest tests/ -v --cov=gridsense --cov-report=term-missing --cov-report=html --cov-fail-under=80
goto end

:coverage
pytest tests/unit/ --cov=gridsense --cov-report=html -q
start htmlcov\index.html
goto end

REM ── Smoke test ────────────────────────────────────────────────────────────
:smoke
python scripts/smoke_test.py
goto end

REM ── Model training ────────────────────────────────────────────────────────
:train-models
python scripts/train_models.py
goto end

REM ── Build ─────────────────────────────────────────────────────────────────
:build
pip install --quiet build
python -m build
goto end

:check-build
pip install --quiet build twine
python -m build
twine check dist/*
goto end

REM ── Full release check ────────────────────────────────────────────────────
:release
echo.
echo [1/4] Lint...
ruff check src/ tests/
if errorlevel 1 (echo FAILED: lint & goto end)

echo [2/4] Type check...
mypy src/gridsense --ignore-missing-imports
if errorlevel 1 (echo FAILED: typecheck & goto end)

echo [3/4] Tests...
pytest tests/ -v --cov=gridsense --cov-report=term-missing --cov-fail-under=80
if errorlevel 1 (echo FAILED: tests & goto end)

echo [4/4] Build check...
pip install --quiet build twine
python -m build
twine check dist/*
if errorlevel 1 (echo FAILED: build & goto end)

echo.
echo ═══════════════════════════════════════════
echo   All checks passed — ready to release!
echo ═══════════════════════════════════════════
echo.
echo   Next steps:
echo   1. git tag v0.1.0
echo   2. git push origin v0.1.0
echo   3. GitHub Actions will publish to PyPI automatically.
echo.
goto end

REM ── Docker ────────────────────────────────────────────────────────────────
:docker-up
docker-compose up --build
goto end

:docker-down
docker-compose down -v
goto end

REM ── Clean ─────────────────────────────────────────────────────────────────
:clean
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
if exist htmlcov rmdir /s /q htmlcov
if exist .coverage del .coverage
if exist coverage.xml del coverage.xml
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
for /d /r . %%d in (*.egg-info) do @if exist "%%d" rmdir /s /q "%%d"
for /d /r . %%d in (.pytest_cache) do @if exist "%%d" rmdir /s /q "%%d"
for /d /r . %%d in (.mypy_cache) do @if exist "%%d" rmdir /s /q "%%d"
for /d /r . %%d in (.ruff_cache) do @if exist "%%d" rmdir /s /q "%%d"
echo Cleaned.
goto end

:end
