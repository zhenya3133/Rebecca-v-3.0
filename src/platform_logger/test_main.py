import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from memory_manager.memory_manager import MemoryManager
from .platform_logger_main import run_agent


def test_logger():
    log_path = os.path.join(os.path.dirname(__file__), "agent_log.txt")
    if os.path.exists(log_path):
        os.remove(log_path)
    memory = MemoryManager()
    context = {"memory": memory}
    result = run_agent(context, "log it")
    assert "logger event" in memory.episodic.get_events()
    assert any("logging audit" == item for item in memory.security.get_audits())
    assert os.path.exists(log_path)
    with open(log_path, "r", encoding="utf-8") as f:
        logs = f.read()
    assert "started with data" in logs
    assert "completed successfully" in logs or "error" in logs
    print("logger OK:", result["result"])


if __name__ == "__main__":
    test_logger()
