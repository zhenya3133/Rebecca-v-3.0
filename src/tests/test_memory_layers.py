import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from memory_manager.memory_manager import MemoryManager


def test_memory_layers_basic():
    memory = MemoryManager()
    memory.core.store_fact("rule", "enabled")
    memory.semantic.store_concept("concept", {"info": "data"})
    memory.episodic.store_event({"id": "1"})
    memory.procedural.store_workflow("workflow", ["a", "b"])
    memory.vault.store_secret("secret", "value")
    memory.security.store_audit({"action": "read"})

    assert memory.core.get_fact("rule") == "enabled"
    assert memory.semantic.get_concept("concept")
    assert memory.episodic.get_events()
    assert memory.procedural.get_workflow("workflow")
    assert memory.vault.get_secret("secret") == "value"
    assert memory.security.get_audits()
    print("Memory layers OK")
