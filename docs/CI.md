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

## Change Summary and CI Reports

The repository includes automated tooling to generate change summaries and CI status
reports for every task or feature branch.

### Generate Change Summary

The `scripts/generate_change_summary.py` script creates a detailed Markdown report
showing:

- List of changed files with their status (Added/Modified/Deleted)
- File changes categorized by directory (src/, tests/, docs/, .github/, etc.)
- Line-by-line diff statistics
- CI status summary (lint, format, type check, tests)
- Links to available test reports and artifacts

Reports are saved to `reports/changes/${ISO_DATETIME}_change_summary.md`.

**Usage:**

```bash
# Use default base branch (study-repo-write-summary)
python3 scripts/generate_change_summary.py

# Specify a custom base branch
BASE_BRANCH=main python3 scripts/generate_change_summary.py
```

**Environment Variables:**

- `BASE_BRANCH` - The base branch to compare against (default: `study-repo-write-summary`)

The script will automatically try to resolve the branch name by checking:
1. The exact branch name
2. The branch name with `origin/` prefix
3. Common fallbacks like `main`, `origin/main`, `master`, `origin/master`

### Post-Task Summary

The `scripts/post_task_summary.sh` wrapper script combines change summary generation
with test report collection. It provides a comprehensive view of all changes and
quality checks for the current branch.

**Usage:**

```bash
# Generate full post-task summary
./scripts/post_task_summary.sh

# With custom base branch
BASE_BRANCH=main ./scripts/post_task_summary.sh

# Skip specific steps
SKIP_CHANGE_SUMMARY=1 ./scripts/post_task_summary.sh
SKIP_TEST_REPORT=1 ./scripts/post_task_summary.sh
```

**Environment Variables:**

- `BASE_BRANCH` - Base branch for comparison
- `SKIP_CHANGE_SUMMARY` - Set to `1` to skip change summary generation
- `SKIP_TEST_REPORT` - Set to `1` to skip test report collection

### CI Status Artifacts

The GitHub Actions workflow now generates a CI status summary after running all
quality checks. This summary is uploaded as an artifact and includes the status
of each check (lint, format, type check, tests).

The artifact is named `ci-status-{os}-py{version}` and contains a `ci_status.md`
file with the formatted results. This can be downloaded from the GitHub Actions
run page or referenced in pull request reviews.

### Integration with Development Workflow

You can integrate the post-task summary script into your development workflow:

1. **After completing a feature:** Run `./scripts/post_task_summary.sh` to generate
   a comprehensive summary of your changes before creating a pull request.

2. **Before merge decisions:** Review the generated change summary in
   `reports/changes/` to quickly understand the scope and impact of changes.

3. **CI integration:** The GitHub Actions workflow automatically generates status
   summaries, making it easy to see which checks passed or failed without
   reviewing individual job logs.
