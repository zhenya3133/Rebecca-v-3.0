"""End-to-end scenario ensuring agent pipeline composes correctly."""

import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


from orchestrator.main_workflow import main_workflow


def test_agent_army_flow():
    payload = {"task": "oss agent landscape"}
    result = main_workflow(payload)
    assert "context" in result
    context = result["context"]
    memory = context["memory"]
    assert memory.semantic.get_concept("blueprint_snapshot") is not None
    assert memory.security.get_alerts(), "SecOps should register alerts"
    assert memory.procedural.get_workflow("ops_commander") is not None
    assert memory.procedural.get_workflow("deployment_ops") is not None


if __name__ == "__main__":
    test_agent_army_flow()
