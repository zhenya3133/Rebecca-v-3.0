import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_PATH)

import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from orchestrator.main_workflow import main_workflow


def test_workflow_full():
    result = main_workflow("integration smoke")
    assert "context" in result
    print("INTEGRATION-TEST OK:", result["result"])


if __name__ == "__main__":
    test_workflow_full()
