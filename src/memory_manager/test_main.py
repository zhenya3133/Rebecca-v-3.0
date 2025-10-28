import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memory_manager.memory_manager import MemoryManager

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from main import run_agent


def test_memory_manager():
    log_path = os.path.join(os.path.dirname(__file__), "agent_log.txt")
    if os.path.exists(log_path):
        os.remove(log_path)
    memory = MemoryManager()
    context = {"memory": memory}
    result = run_agent(context, "MMgr")
    assert memory.core.get_fact("memory access") is True
    assert memory.procedural.get_workflow("memory_manager") == ["init", "manage", "terminate"]
    assert os.path.exists(log_path)
    with open(log_path, "r", encoding="utf-8") as f:
        logs = f.read()
    assert "started with data" in logs
    assert "completed successfully" in logs or "error" in logs
    print("memory_manager OK:", result["result"])


if __name__ == "__main__":
    test_memory_manager()
