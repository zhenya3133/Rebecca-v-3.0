"""Cross-modal linking for ingested artifacts."""

from typing import Protocol


class SemanticMemoryProtocol(Protocol):
    def store_concept(self, key: str, value): ...


class MemoryProtocol(Protocol):
    semantic: SemanticMemoryProtocol


def link_artifacts(memory_manager: MemoryProtocol, link_id: str, items):
    record = {"link_id": link_id, "items": list(items)}
    memory_manager.semantic.store_concept(f"cross_modal::{link_id}", record)
    return record
