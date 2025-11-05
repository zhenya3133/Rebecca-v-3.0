# Change Summary Reports

This directory contains automatically generated change summary reports created by
`scripts/generate_change_summary.py`.

## About These Reports

Each report provides:

- List of changed files with their status (Added/Modified/Deleted)
- File changes categorized by directory (src/, tests/, docs/, .github/, etc.)
- Line-by-line diff statistics
- CI status summary (lint, format, type check, tests)
- Links to available test reports and artifacts

## File Naming Convention

Reports are named using the format: `YYYYMMDD_HHMMSS_change_summary.md`

The timestamp represents when the report was generated.

## Usage

Generate a new report:

```bash
# Use default base branch (study-repo-write-summary)
python3 scripts/generate_change_summary.py

# Specify a custom base branch
BASE_BRANCH=main python3 scripts/generate_change_summary.py
```

For more information, see `docs/CI.md`.
