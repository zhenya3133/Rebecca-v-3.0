import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from orchestrator.main_workflow import main_workflow


def test_main_workflow_smoke():
    result = main_workflow({"task": "smoke"})
    assert "result" in result
    print("Main workflow result:", result["result"])
