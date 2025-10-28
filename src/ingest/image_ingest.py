"""Image ingest pipeline with in-memory storage."""

from typing import Protocol


class VaultMemoryProtocol(Protocol):
    def store_secret(self, name: str, value): ...


class MemoryProtocol(Protocol):
    vault: VaultMemoryProtocol


def ingest_image(memory_manager: MemoryProtocol, image_path: str):
    metadata = {"path": image_path, "status": "ingested"}
    memory_manager.vault.store_secret(f"image::{image_path}", metadata)
    return metadata
