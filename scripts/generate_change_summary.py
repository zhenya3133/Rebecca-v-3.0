#!/usr/bin/env python3
"""
Generate a change summary report for the current branch.

This script creates a Markdown report showing:
- Changed files with their status (Added/Modified/Deleted)
- Line statistics per file category
- CI status summary (lint/format/type/tests)
- Links to artifacts if available

The report is saved to reports/changes/${ISO_DATETIME}_change_summary.md
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def run_git_command(cmd: List[str]) -> Tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def get_base_branch() -> str:
    """Get the base branch from environment or use default."""
    return os.environ.get("BASE_BRANCH", "study-repo-write-summary")


def resolve_base_branch(base_branch: str) -> str:
    """Resolve base branch name, trying origin/ prefix if needed."""
    # Try as-is first
    returncode, _, _ = run_git_command(["git", "rev-parse", "--verify", base_branch])
    if returncode == 0:
        return base_branch

    # Try with origin/ prefix
    origin_branch = f"origin/{base_branch}"
    returncode, _, _ = run_git_command(["git", "rev-parse", "--verify", origin_branch])
    if returncode == 0:
        return origin_branch

    # Try to find merge-base with main
    for fallback in ["main", "origin/main", "master", "origin/master"]:
        returncode, _, _ = run_git_command(["git", "rev-parse", "--verify", fallback])
        if returncode == 0:
            return fallback

    # Return original if nothing worked
    return base_branch


def get_changed_files(base_branch: str) -> List[Tuple[str, str]]:
    """
    Get list of changed files with their status.

    Returns list of tuples: (status, filepath)
    Status codes: A=Added, M=Modified, D=Deleted, R=Renamed, etc.
    """
    base_branch = resolve_base_branch(base_branch)

    returncode, stdout, stderr = run_git_command(
        ["git", "diff", "--name-status", f"{base_branch}...HEAD"]
    )

    if returncode != 0:
        print(f"Warning: Could not get changed files: {stderr}", file=sys.stderr)
        return []

    files = []
    for line in stdout.split("\n"):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) >= 2:
            status, filepath = parts[0], parts[1]
            files.append((status, filepath))

    return files


def get_diff_stats(base_branch: str) -> str:
    """Get git diff statistics."""
    base_branch = resolve_base_branch(base_branch)

    returncode, stdout, stderr = run_git_command(
        ["git", "diff", "--stat", f"{base_branch}...HEAD"]
    )

    if returncode != 0:
        return f"Could not retrieve diff stats: {stderr}"

    return stdout if stdout else "No changes detected"


def categorize_files(
    files: List[Tuple[str, str]],
) -> Dict[str, List[Tuple[str, str]]]:
    """Categorize files by directory prefix."""
    categories: Dict[str, List[Tuple[str, str]]] = {
        "src/": [],
        "tests/": [],
        "docs/": [],
        ".github/": [],
        "scripts/": [],
        "config/": [],
        "other": [],
    }

    for status, filepath in files:
        categorized = False
        for category in categories.keys():
            if category != "other" and filepath.startswith(category):
                categories[category].append((status, filepath))
                categorized = True
                break
        if not categorized:
            categories["other"].append((status, filepath))

    return categories


def get_ci_status() -> Dict[str, str]:
    """
    Get CI status from available sources.

    Checks for:
    1. GitHub Actions workflow runs (if GITHUB_RUN_ID is set)
    2. Local test results
    3. Returns UNKNOWN if status cannot be determined
    """
    status = {
        "lint": "UNKNOWN",
        "format": "UNKNOWN",
        "type_check": "UNKNOWN",
        "tests": "UNKNOWN",
    }

    # Check if we're running in GitHub Actions
    github_run_id = os.environ.get("GITHUB_RUN_ID")
    if github_run_id:
        status["lint"] = "See GitHub Actions"
        status["format"] = "See GitHub Actions"
        status["type_check"] = "See GitHub Actions"
        status["tests"] = "See GitHub Actions"
        return status

    # Try to determine status from local checks
    project_root = Path(__file__).parent.parent

    # Check for pytest results
    test_reports = list(project_root.glob("reports/**/pytest*.xml"))
    if test_reports:
        status["tests"] = "COMPLETED (see artifacts)"

    # For local runs, we can try running checks to get status
    # But we'll just mark as UNKNOWN for now to avoid side effects

    return status


def get_artifact_links() -> List[Tuple[str, str]]:
    """Get links to available artifacts."""
    artifacts = []
    project_root = Path(__file__).parent.parent

    # Check for GitHub Actions artifacts
    github_run_id = os.environ.get("GITHUB_RUN_ID")
    github_repo = os.environ.get("GITHUB_REPOSITORY")
    if github_run_id and github_repo:
        artifacts.append(
            (
                "GitHub Actions Run",
                f"https://github.com/{github_repo}/actions/runs/{github_run_id}",
            )
        )

    # Check for local test reports
    reports_dir = project_root / "reports"
    if reports_dir.exists():
        for report_file in reports_dir.rglob("*.xml"):
            rel_path = report_file.relative_to(project_root)
            artifacts.append((str(rel_path), str(rel_path)))

        for report_file in reports_dir.rglob("*.md"):
            rel_path = report_file.relative_to(project_root)
            artifacts.append((str(rel_path), str(rel_path)))

    return artifacts


def format_status_badge(status: str) -> str:
    """Format status as a badge-like string."""
    if status.upper() in ["OK", "PASS", "PASSED", "SUCCESS"]:
        return "✅ PASS"
    elif status.upper() in ["FAIL", "FAILED", "ERROR"]:
        return "❌ FAIL"
    elif status.upper() == "COMPLETED (SEE ARTIFACTS)":
        return "📊 COMPLETED"
    elif "GITHUB" in status.upper():
        return "🔗 CI"
    else:
        return "❓ UNKNOWN"


def generate_report(base_branch: str) -> str:
    """Generate the complete change summary report."""
    files = get_changed_files(base_branch)
    diff_stats = get_diff_stats(base_branch)
    categories = categorize_files(files)
    ci_status = get_ci_status()
    artifacts = get_artifact_links()

    current_branch = run_git_command(["git", "branch", "--show-current"])[1]

    report_lines = [
        "# Change Summary Report",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        f"**Branch:** `{current_branch}`",
        f"**Base:** `{base_branch}`",
        f"**Total Files Changed:** {len(files)}",
        "",
        "---",
        "",
    ]

    # CI Status Summary
    report_lines.extend(
        [
            "## CI Status Summary",
            "",
            "| Check | Status |",
            "|-------|--------|",
            f"| Lint | {format_status_badge(ci_status['lint'])} |",
            f"| Format | {format_status_badge(ci_status['format'])} |",
            f"| Type Check | {format_status_badge(ci_status['type_check'])} |",
            f"| Tests | {format_status_badge(ci_status['tests'])} |",
            "",
        ]
    )

    # Changed Files by Category
    report_lines.extend(
        [
            "## Changed Files by Category",
            "",
        ]
    )

    for category, cat_files in categories.items():
        if not cat_files:
            continue

        report_lines.append(f"### {category}")
        report_lines.append("")

        for status, filepath in cat_files:
            status_symbol = {
                "A": "➕",
                "M": "📝",
                "D": "➖",
                "R": "🔄",
            }.get(status, "❔")
            report_lines.append(f"- {status_symbol} `{filepath}` ({status})")

        report_lines.append("")

    # Diff Statistics
    report_lines.extend(
        [
            "## Diff Statistics",
            "",
            "```",
            diff_stats,
            "```",
            "",
        ]
    )

    # Artifacts
    if artifacts:
        report_lines.extend(
            [
                "## Available Artifacts & Reports",
                "",
            ]
        )

        for name, link in artifacts:
            if link.startswith("http"):
                report_lines.append(f"- [{name}]({link})")
            else:
                report_lines.append(f"- `{link}`")

        report_lines.append("")

    return "\n".join(report_lines)


def save_report(report_content: str) -> Path:
    """Save the report to reports/changes/ directory."""
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "reports" / "changes"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{timestamp}_change_summary.md"

    output_file.write_text(report_content)
    return output_file


def main():
    """Main entry point."""
    print("Generating change summary report...")

    base_branch = get_base_branch()
    print(f"Base branch: {base_branch}")

    # Check if we're in a git repository
    returncode, _, stderr = run_git_command(["git", "rev-parse", "--git-dir"])
    if returncode != 0:
        print(f"Error: Not in a git repository: {stderr}", file=sys.stderr)
        sys.exit(1)

    # Generate report
    report = generate_report(base_branch)

    # Save report
    output_file = save_report(report)

    print(f"✅ Report saved to: {output_file}")
    print(f"   Relative path: {output_file.relative_to(Path.cwd())}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
