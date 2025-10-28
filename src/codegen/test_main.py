import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memory_manager.memory_manager import MemoryManager

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from main import run_agent


def test_codegen():
    log_path = os.path.join(os.path.dirname(__file__), "agent_log.txt")
    if os.path.exists(log_path):
        os.remove(log_path)
    memory = MemoryManager()
    context = {"memory": memory}
    result = run_agent(context, "Тестовый input")
    assert memory.procedural.get_workflow("deploy") == ["build", "test", "deploy"]
    assert memory.vault.get_secret("api_key") == "example_api_key"
    assert os.path.exists(log_path)
    with open(log_path, "r", encoding="utf-8") as f:
        logs = f.read()
    assert "started with data" in logs
    assert "completed successfully" in logs or "error" in logs
    print("codegen OK:", result["result"])


if __name__ == "__main__":
    test_codegen()
