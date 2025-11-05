#!/usr/bin/env python3
"""Summarize pytest failure taxonomy from a JUnit XML report.

This utility ingests pytest's ``--junitxml`` output and produces a structured
JSON summary that highlights failing modules, error types, and inferred
third-party dependency gaps. Run it after generating ``pytest_results.xml``::

    python scripts/summarize_failures.py \
        reports/baseline/pytest_results.xml \
        reports/baseline/failures_taxonomy.json

The output path argument is optional—omit it to stream the JSON summary to
stdout, which makes shell redirection straightforward::

    python scripts/summarize_failures.py reports/baseline/pytest_results.xml \
        > reports/baseline/failures_taxonomy.json

The resulting JSON is intended to support baseline documentation and triage
before remediation work begins.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Set


@dataclass
class FailureRecord:
    """Container for structured information about a single failing testcase."""

    module: str
    name: str
    error_type: str | None
    message: str
    details: str
    dependency_tags: Sequence[str]


TAG_DESCRIPTIONS = {
    "pydantic_missing_package": "Environment lacks a functional pydantic installation (pytest reports ModuleNotFoundError).",
    "pydantic_incompatibility": "Pydantic import succeeded but produced compatibility errors (likely version mismatch with FastAPI/Chromadb).",
    "chromadb_dependency": "Chromadb initialization failed, indicating vector store dependencies are unsatisfied or incompatible.",
}


ERROR_TYPE_PATTERN = re.compile(r"E\s+([A-Za-z0-9_]+(?:Error|Exception)):")


def discover_failure_records(testsuites: ET.Element) -> List[FailureRecord]:
    records: List[FailureRecord] = []

    for testcase in testsuites.iter("testcase"):
        failure_node = testcase.find("failure") or testcase.find("error")
        if failure_node is None:
            continue

        text = (failure_node.text or "").strip()
        message = failure_node.get("message") or ""
        module_name = testcase.get("classname") or testcase.get("name") or ""
        testcase_name = testcase.get("name") or module_name
        error_type = extract_error_type(text, message)
        dependency_tags = sorted(infer_dependency_tags(text, message))

        records.append(
            FailureRecord(
                module=module_name or testcase_name,
                name=testcase_name,
                error_type=error_type,
                message=message.strip(),
                details=text,
                dependency_tags=dependency_tags,
            )
        )

    return records


def extract_error_type(details: str, message: str) -> str | None:
    """Best-effort extraction of the reported error type."""

    combined = "\n".join(filter(None, [details, message]))
    match = None
    for match in ERROR_TYPE_PATTERN.finditer(combined):
        pass
    if match:
        return match.group(1)

    lowered = combined.lower()
    if "modulenotfounderror" in lowered:
        return "ModuleNotFoundError"
    if "typeerror" in lowered:
        return "TypeError"
    if "importerror" in lowered:
        return "ImportError"
    return None


def infer_dependency_tags(details: str, message: str) -> Set[str]:
    """Infer dependency-related tags from the failure payload."""

    combined = " ".join([details or "", message or ""]).lower()
    tags: Set[str] = set()

    if "pydantic" in combined:
        if "not a package" in combined or "no module named 'pydantic" in combined:
            tags.add("pydantic_missing_package")
        else:
            tags.add("pydantic_incompatibility")

    if "chromadb" in combined or "basemodeljsonserializable" in combined:
        tags.add("chromadb_dependency")

    return tags


def summarize_by_module(records: Sequence[FailureRecord]):
    grouped = defaultdict(list)
    for record in records:
        grouped[record.module].append(record)

    summary = []
    for module, entries in sorted(grouped.items()):
        summary.append(
            {
                "module": module,
                "occurrences": len(entries),
                "error_types": dict(Counter(e.error_type for e in entries if e.error_type)),
                "dependency_tags": sorted({tag for e in entries for tag in e.dependency_tags}),
                "representative_detail": entries[0].details.splitlines()[:5],
            }
        )
    return summary


def summarize_by_error_type(records: Sequence[FailureRecord]):
    grouped = defaultdict(list)
    for record in records:
        key = record.error_type or "Unknown"
        grouped[key].append(record)

    summary = []
    for error_type, entries in sorted(grouped.items()):
        summary.append(
            {
                "error_type": error_type,
                "occurrences": len(entries),
                "modules": sorted({e.module for e in entries}),
                "dependency_tags": sorted({tag for e in entries for tag in e.dependency_tags}),
            }
        )
    return summary


def summarize_dependencies(records: Sequence[FailureRecord]):
    dependency_summary = []
    for tag, description in TAG_DESCRIPTIONS.items():
        affected = [r for r in records if tag in r.dependency_tags]
        if not affected:
            continue
        dependency_summary.append(
            {
                "tag": tag,
                "description": description,
                "occurrences": len(affected),
                "affected_error_types": sorted({r.error_type or "Unknown" for r in affected}),
                "affected_modules": sorted({r.module for r in affected}),
                "representative_detail": affected[0].details.splitlines()[:5],
            }
        )
    return dependency_summary


def build_report(records: Sequence[FailureRecord]):
    total_cases = len(records)
    distinct_errors = sorted({r.error_type or "Unknown" for r in records})
    distinct_modules = sorted({r.module for r in records})

    return {
        "summary": {
            "total_error_testcases": total_cases,
            "distinct_error_types": distinct_errors,
            "distinct_modules": len(distinct_modules),
        },
        "by_module": summarize_by_module(records),
        "by_error_type": summarize_by_error_type(records),
        "dependency_gaps": summarize_dependencies(records),
    }


def parse_junit(xml_path: Path) -> ET.Element:
    tree = ET.parse(xml_path)
    return tree.getroot()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize pytest failure taxonomy from a JUnit XML report.")
    parser.add_argument("junit_xml", type=Path, help="Path to pytest --junitxml output.")
    parser.add_argument(
        "output_json",
        type=Path,
        nargs="?",
        help="Optional destination path for the failure taxonomy JSON (defaults to stdout).",
    )
    args = parser.parse_args(argv)

    testsuites = parse_junit(args.junit_xml)
    records = discover_failure_records(testsuites)
    report = build_report(records)
    report_text = json.dumps(report, indent=2, sort_keys=True)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(report_text + "\n")
    else:
        print(report_text)


if __name__ == "__main__":
    main()
