# Baseline documentation vs. implementation gaps

## Repository summary input
- `REPOSITORY_SUMMARY.md` is absent from the repository. For this baseline pass the top-level `README.md` serves as a provisional source of truth, but the missing summary document should be restored or recreated for future audits.

## Test discovery baseline (2025-11-05)
- Command: `pytest --collect-only > reports/baseline/test_collection.txt`
  - Pytest enumerated **111** test nodes but emitted **55 collection errors** before execution. The collector failures align with the dependency gaps documented below and demonstrate that the advertised test layout in the README cannot currently be exercised.

## Targeted test execution
- Command: `pytest -m 'not slow and not integration' -vv --maxfail=0 --durations=20 --junitxml=reports/baseline/pytest_results.xml`
  - Result: **55 collection errors**, matching the collector-only run. The generated JUnit report confirms two systemic blockers:
    - `ModuleNotFoundError` for `pydantic.version` across API, orchestrator, memory, and subsystem tests—FastAPI expects a Pydantic v1-compatible package but the environment lacks it (or has an incompatible v2 install shadowing the namespace).
    - `TypeError: Cannot create a consistent method resolution order (MRO)` while importing `chromadb`, triggered by `BaseModelJSONSerializable`. This points to a Chromadb build compiled for Pydantic v1 while the runtime exposes Pydantic v2, producing the incompatible base-class hierarchy.
  - No tests executed; the taxonomy in `reports/baseline/failures_taxonomy.json` records **55 erroring test cases**, split between the two dependency gaps above.

## Coverage attempt
- Command: `pytest --cov=src --cov-report=xml:reports/baseline/coverage.xml --cov-report=term-missing > reports/baseline/coverage.txt`
  - Result: Pytest aborted before collection with `error: unrecognized arguments: --cov …`, indicating that `pytest-cov` (or equivalent coverage plugin) is not installed in the baseline environment. `coverage.txt` and `coverage.xml` contain diagnostic placeholders instead of metrics.

## Contract vs. implementation findings from README.md
- **Testability claim:** README advertises a straightforward `pytest tests/ -v` workflow with 90%+ coverage. In practice, even the constrained marker run fails during import, so no automated assurance is produced.
- **Dependency guidance:** README instructs `pip install -r src/requirements.txt`, but the runtime dependency lock does not install a Pydantic variant compatible with FastAPI/Chromadb. Development extras (`pytest-cov`, tooling) referenced in the documentation are also absent.
- **Test tree references:** README sections still reference directories such as `tests/test_memory_system/`; the actual repository consolidates these suites under `tests/test_memory_manager.py`, `tests/test_memory_persistence.py`, and `src/tests/…`. Documentation should be realigned once the suites are runnable.

## Suggested remediation (outside current scope)
1. Align dependency versions so FastAPI, Pydantic, and Chromadb agree on a common major version (likely pinning Pydantic v1.x until the codebase is ported to v2 APIs).
2. Add the missing coverage plugin (`pytest-cov`) to the managed dependencies to satisfy coverage workflows.
3. Reinstate or recreate `REPOSITORY_SUMMARY.md`, and update README/testing sections to match the operative folder structure and validated dependency setup once the above issues are resolved.
