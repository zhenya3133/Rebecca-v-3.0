"""Memory layer smoke tests covering episodic, semantic, procedural, vault, and blueprint tracker."""

import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


from memory_manager.memory_manager import MemoryManager


def test_memory_layers_persist():
    manager = MemoryManager()
    manager.core.store_fact("key", "value")
    manager.semantic.store_concept("concept", "details")
    manager.episodic.store_event({"type": "event", "payload": 1})
    manager.procedural.store_workflow("test", ["a", "b"])
    manager.vault.store_secret("secret", "token")
    manager.security.store_audit("check")
    manager.security.register_alert("test", {"x": 1})
    manager.blueprint_tracker.record_blueprint({"plan": "initial"})

    assert manager.core.get_fact("key") == "value"
    assert manager.semantic.get_concept("concept") == "details"
    assert manager.episodic.get_events()
    assert manager.procedural.get_workflow("test") == ["a", "b"]
    assert manager.vault.get_secret("secret") == "token"
    assert manager.security.get_audits() == ["check"]
    assert manager.security.get_alerts()
    assert manager.blueprint_tracker.latest() == {"plan": "initial"}


if __name__ == "__main__":
    test_memory_layers_persist()
