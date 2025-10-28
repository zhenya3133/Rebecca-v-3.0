"""Document ingestion workflow inspired by mem0 pipeline patterns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


@dataclass
class DocumentIngestor:
    """Stub for preprocessing documents before vectorization/storage.

    TODO: Connect to VectorStoreClient for embedding + storage.
    TODO: Provide hooks for layer-specific normalization (e.g., Procedural vs.
    Vault security requirements).
    """

    vector_client: Optional["VectorStoreClient"] = None

    def load_sources(self, sources: Iterable[Any]) -> None:
        """Placeholder for ingesting new data sources."""

    def preprocess(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize/clean a record based on target memory layer (stub)."""

    def embed_and_store(self, record: Dict[str, Any], layer: str) -> None:
        """Stub for embedding and storing a record via vector client."""


from .vector_store_client import VectorStoreClient  # noqa: E402  # pylint: disable=wrong-import-position
