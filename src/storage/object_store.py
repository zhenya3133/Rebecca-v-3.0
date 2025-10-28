"""Object storage facade with in-memory bucket implementation."""

from typing import Dict


class InMemoryObjectStore:
    def __init__(self) -> None:
        self._store: Dict[str, bytes] = {}

    def put(self, key: str, data: bytes) -> None:
        self._store[key] = data

    def get(self, key: str) -> bytes | None:
        return self._store.get(key)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)
