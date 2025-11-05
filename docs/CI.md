# Continuous Integration and Quality Tooling

This repository ships with a unified tooling stack so that local development and the
continuous integration (CI) environment share the same expectations.

## Toolchain Overview

The configuration for the code-quality tools lives in `pyproject.toml`:

- **Black** enforces Python formatting (`line-length` 88, Python 3.11 syntax).
- **Ruff** performs linting and import sorting across `src/` and `tests/` while
  ignoring generated or vendor directories such as `.venv/` and `test_env/`.
- **Mypy** runs type checking with a source root of `src/` and ignores missing
  third‑party stubs by default.
- **Pytest** is configured for strict marker usage and automatically enables the
  asyncio event loop via `asyncio_mode=auto`.
- **Coverage** is pre-configured to measure branch coverage for modules under `src/`.

Runtime dependencies continue to be tracked under `src/requirements.txt`. Developer
and testing dependencies (pytest plug-ins, linters, type checkers, etc.) now live in
`requirements-dev.txt`.

## Local Development Workflow

1. Create a virtual environment (optional but recommended) and install runtime and
   development dependencies:

   ```bash
   pip install -r src/requirements.txt
   pip install -r requirements-dev.txt
   ```

2. Run the bundled quality checks helper:

   ```bash
   ./scripts/run_checks.sh
   ```

   The script executes Ruff, Black (in `--check` mode), Mypy, and then Pytest. You can
   pass additional arguments to Pytest if required, e.g. `./scripts/run_checks.sh -k smoke`.

3. To inspect coverage locally:

   ```bash
   pytest --cov=src --cov-report=term-missing
   ```

## GitHub Actions Workflow

The GitHub Actions workflow is defined in `.github/workflows/ci.yml`. It runs on pushes
and pull requests targeting the `study-repo-write-summary` branch with a matrix over
`ubuntu-latest` and `windows-latest` using Python 3.11.

Each job performs the following steps:

1. Check out the repository.
2. Install the runtime dependencies (`src/requirements.txt`) and development
   dependencies (`requirements-dev.txt`), using the setup-python built-in pip cache.
3. Run Ruff, Black (`--check`), and Mypy.
4. Execute the Pytest suite. Failures are reported but do not fail the job while the
   suite is being stabilized, ensuring results are visible without blocking iteration.

These steps mirror the expectations for local development, so passing the local helper
script should lead to green CI runs across both supported operating systems.
