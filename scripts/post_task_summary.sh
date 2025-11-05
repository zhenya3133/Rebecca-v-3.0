#!/usr/bin/env bash
#
# Post-task summary script
#
# This script combines:
# 1. Change summary generation (via generate_change_summary.py)
# 2. Test report collection (if available)
#
# Environment variables:
#   BASE_BRANCH - Base branch for comparison (default: study-repo-write-summary)
#   SKIP_CHANGE_SUMMARY - Set to 1 to skip change summary generation
#   SKIP_TEST_REPORT - Set to 1 to skip test report collection

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "================================================"
echo "Post-Task Summary"
echo "================================================"
echo ""

# Step 1: Generate change summary
if [ "${SKIP_CHANGE_SUMMARY:-0}" != "1" ]; then
    echo "📊 Generating change summary..."
    python3 "$SCRIPT_DIR/generate_change_summary.py"
    echo ""
else
    echo "⏭️  Skipping change summary (SKIP_CHANGE_SUMMARY=1)"
    echo ""
fi

# Step 2: Collect test reports
if [ "${SKIP_TEST_REPORT:-0}" != "1" ]; then
    echo "📋 Checking for test reports..."
    
    REPORTS_DIR="$PROJECT_ROOT/reports"
    FOUND_REPORTS=0
    
    if [ -d "$REPORTS_DIR" ]; then
        # Look for pytest XML results
        if find "$REPORTS_DIR" -name "pytest*.xml" -o -name "*test*.xml" | grep -q .; then
            echo "✅ Found pytest XML reports:"
            find "$REPORTS_DIR" -name "pytest*.xml" -o -name "*test*.xml" | while read -r file; do
                echo "   - $(realpath --relative-to="$PROJECT_ROOT" "$file")"
            done
            FOUND_REPORTS=1
        fi
        
        # Look for coverage reports
        if find "$REPORTS_DIR" -name "coverage*.xml" -o -name ".coverage" | grep -q .; then
            echo "✅ Found coverage reports:"
            find "$REPORTS_DIR" -name "coverage*.xml" -o -name ".coverage" | while read -r file; do
                echo "   - $(realpath --relative-to="$PROJECT_ROOT" "$file")"
            done
            FOUND_REPORTS=1
        fi
        
        # Look for markdown reports
        if find "$REPORTS_DIR" -name "*.md" | grep -q .; then
            echo "✅ Found markdown reports:"
            find "$REPORTS_DIR" -name "*.md" | while read -r file; do
                echo "   - $(realpath --relative-to="$PROJECT_ROOT" "$file")"
            done
            FOUND_REPORTS=1
        fi
    fi
    
    if [ "$FOUND_REPORTS" = "0" ]; then
        echo "ℹ️  No test reports found in $REPORTS_DIR"
    fi
    echo ""
else
    echo "⏭️  Skipping test report collection (SKIP_TEST_REPORT=1)"
    echo ""
fi

echo "================================================"
echo "Summary complete!"
echo "================================================"
echo ""
echo "📁 Check the reports/changes/ directory for the change summary"
