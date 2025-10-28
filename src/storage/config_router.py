"""Storage adapter routing based on configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from configuration import get_storage_config


@dataclass
class StorageAdapter:
    name: str
    type: str
    config: Dict[str, object]

    def connect(self) -> None:
        # TODO: implement real connection logic for postgres/mongo/qdrant/local_files
        pass


class StorageRouter:
    def __init__(self) -> None:
        self.adapters: Dict[str, StorageAdapter] = {}
        self._load_from_config()

    def _load_from_config(self) -> None:
        storage_cfg = get_storage_config()
        for name, cfg in storage_cfg.items():
            adapter = StorageAdapter(name=name, type=cfg.get("type", "unknown"), config=cfg)
            self.adapters[name] = adapter

    def get(self, name: str) -> Optional[StorageAdapter]:
        return self.adapters.get(name)

    def all(self) -> Dict[str, StorageAdapter]:
        return dict(self.adapters)
