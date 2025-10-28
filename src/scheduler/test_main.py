import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memory_manager.memory_manager import MemoryManager
from main import run_agent


def test_scheduler():
    log_path = os.path.join(os.path.dirname(__file__), "agent_log.txt")
    if os.path.exists(log_path):
        os.remove(log_path)
    memory = MemoryManager()
    context = {"memory": memory}
    result = run_agent(context, "plan123")
    assert memory.procedural.get_workflow("schedule") == ["plan", "allocate", "execute"]
    assert "scheduler: new round" in memory.episodic.get_events()
    assert os.path.exists(log_path)
    with open(log_path, "r", encoding="utf-8") as f:
        logs = f.read()
    assert "started with data" in logs
    assert "completed successfully" in logs or "error" in logs
    print("scheduler OK:", result["result"])


if __name__ == "__main__":
    test_scheduler()
