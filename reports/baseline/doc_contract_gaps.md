# Baseline documentation vs. implementation gaps

## Source of repository summary
- `REPOSITORY_SUMMARY.md` is not present in the tree. For the baseline review, `README.md` is being treated as the authoritative reference.
- README states the project is "Production Ready" with 90%+ automated test coverage and highlights a frictionless `pytest tests/ -v` workflow. The current environment cannot satisfy those guarantees due to dependency and import failures (see below).

## Test execution baseline (2025-11-05)
- Command: `pytest tests src/tests -vv --maxfail=0 --durations=20 --junitxml=reports/baseline/pytest_results.xml`
  - Result: **29 collection errors**, no tests executed. Generated report stored at `reports/baseline/pytest_results.xml`.
  - Failure taxonomy (`reports/baseline/failures_taxonomy.json`) shows:
    - 18 occurrences of `ModuleNotFoundError` stemming from FastAPI importing `pydantic.version` (`'pydantic' is not a package`). A working Pydantic installation is missing or shadowed in the virtualenv, blocking every API-facing test module.
    - 11 occurrences of `TypeError` during `chromadb` initialization: "Cannot create a consistent method resolution order (MRO) for bases object, BaseModelJSONSerializable". This is consistent with installing `chromadb` alongside Pydantic v2, while the library still expects Pydantic v1 classes.
  - Impacted suites include: multi-agent factory, memory manager layers, KAG integration tests, orchestrator, metrics, policy engine, and ingestion pipelines—effectively the entire documented surface area.

- Command: `pytest --collect-only > reports/baseline/test_collection.txt`
  - Result: Pytest enumerated **111 test items** but reported **55 collection errors**, matching the two dependency failure modes above. The tree walk confirms the suites advertised in README, but none progress past import time.

## Coverage attempt
- Command: `pytest tests src/tests --cov=src --cov-report=xml:reports/baseline/coverage.xml --cov-report=term-missing > reports/baseline/coverage.txt`
  - Result: Pytest aborts with `error: unrecognized arguments: --cov ...`, because the `pytest-cov` plugin is absent from the environment. Coverage artifacts were captured as diagnostic placeholders:
    - `reports/baseline/coverage.txt` records the CLI failure.
    - `reports/baseline/coverage.xml` documents that coverage data could not be produced pending plugin installation.

## README contract mismatches observed so far
- **Coverage claims:** README advertises 90%+ coverage across unit, integration, performance, and E2E tiers, but the baseline run cannot execute any tests. Achieving the documented coverage requires restoring compatible versions of Pydantic (v1.x) and Chromadb (or pinning an interop shim) plus installing `pytest-cov`.
- **Operational readiness:** Documentation labels the release "Production Ready"; however, the shipping virtualenv lacks critical runtime dependencies (`pydantic`/`pytest-cov`) and exhibits library incompatibilities, preventing even import-time validation.
- **Test layout references:** README refers to `tests/test_memory_system/` and other subtrees; the actual repository anchors memory tests under `tests/test_memory_manager.py` and `tests/test_memory_persistence.py`, while additional suites live under `src/tests/…`. The documented structure partially aligns but still points to non-existent `tests/test_memory_system/` paths, indicating stale documentation.
- **Run instructions:** README suggests `pip install -r src/requirements.txt`, yet the current `.venv` depends on packages (FastAPI, Chromadb) whose transitive requirements are not harmonised. Additional dependency locks or overrides are required to make the quick start commands succeed.

## Next steps for remediation (outside the scope of this baseline task)
1. Repair the Python environment to provide a valid Pydantic v1 installation (or update dependent code to Pydantic v2 APIs) and reconcile Chromadb's requirements.
2. Add `pytest-cov` (or alternative coverage tooling) to the dev dependencies so that coverage artefacts can be generated as documented.
3. Align documentation with the actual test tree and operational prerequisites once the above issues are resolved.
