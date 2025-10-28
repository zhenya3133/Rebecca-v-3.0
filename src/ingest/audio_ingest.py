"""Audio ingest pipeline with in-memory storage integration."""

from typing import Protocol


class ProceduralMemoryProtocol(Protocol):
    def store_workflow(self, name: str, steps): ...


class MemoryProtocol(Protocol):
    procedural: ProceduralMemoryProtocol


def ingest_audio(memory_manager: MemoryProtocol, audio_path: str):
    steps = ["load", "transcribe", "embed"]
    memory_manager.procedural.store_workflow(f"audio::{audio_path}", steps)
    return {"path": audio_path, "status": "ingested", "steps": steps}
