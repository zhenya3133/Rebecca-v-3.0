"""Bridges Rebecca services for DROId."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

from .config import CoreConfig


class CoreConnectionError(RuntimeError):
    """Raised when the adapter cannot reach Rebecca core."""


@dataclass
class ContextBridge:
    config: CoreConfig

    def fetch_context(self, trace_id: str) -> Dict[str, Any]:
        return {"trace_id": trace_id, "metadata": {"source": "droid"}}


@dataclass
class MemoryBridge:
    config: CoreConfig

    def sync_blueprint(self, blueprint: Dict[str, Any]) -> None:
        _ = blueprint

    def link_resource(self, identifier: str, resource: Dict[str, Any]) -> None:
        _ = (identifier, resource)

    def register_documents(self, documents: Iterable[Dict[str, Any]]) -> None:
        _ = list(documents)


@dataclass
class EventBridge:
    config: CoreConfig

    def emit_event(self, name: str, payload: Dict[str, Any]) -> None:
        _ = (name, payload)


@dataclass
class RebeccaCoreAdapter:
    config: CoreConfig

    def __post_init__(self) -> None:
        self.context_bridge = ContextBridge(self.config)
        self.memory_bridge = MemoryBridge(self.config)
        self.event_bridge = EventBridge(self.config)

    @classmethod
    def from_config(cls, config: CoreConfig | None = None) -> "RebeccaCoreAdapter":
        return cls(config or CoreConfig.load())

    def connectivity_check(self) -> bool:
        return bool(self.config.endpoint)

    def fetch_context(self, trace_id: str) -> Dict[str, Any]:
        return self.context_bridge.fetch_context(trace_id)

    def sync_blueprint(self, blueprint: Dict[str, Any]) -> None:
        self.memory_bridge.sync_blueprint(blueprint)

    def emit_event(self, name: str, payload: Dict[str, Any]) -> None:
        self.event_bridge.emit_event(name, payload)
