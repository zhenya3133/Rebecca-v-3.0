"""
Pytest configuration for Rebecca-Platform tests.

This conftest automatically enables offline mode for all tests,
seeds randomness for deterministic behavior, and provides fixtures
for dummy connectors and services.
"""

import os
import random
import pytest
from typing import Dict, Any, List


@pytest.fixture(scope="session", autouse=True)
def enable_offline_mode():
    """Automatically enable offline mode for all tests."""
    os.environ["REBECCA_OFFLINE_MODE"] = "1"
    os.environ["REBECCA_TEST_MODE"] = "1"
    yield
    # Cleanup
    os.environ.pop("REBECCA_OFFLINE_MODE", None)
    os.environ.pop("REBECCA_TEST_MODE", None)


@pytest.fixture(scope="session", autouse=True)
def seed_randomness():
    """Seed randomness for deterministic tests."""
    random.seed(42)
    try:
        import numpy as np
        np.random.seed(42)
    except ImportError:
        pass


@pytest.fixture
def dummy_vector_store_config():
    """Provides a dummy vector store configuration for testing."""
    from memory_manager.vector_store_client import VectorStoreConfig
    
    return VectorStoreConfig(
        provider="memory",
        vector_size=384,
        collection_name="test_vectors",
        embedding_provider="local",
        fallback_enabled=True
    )


@pytest.fixture
def dummy_embedding_provider(dummy_vector_store_config):
    """Provides a dummy embedding provider for testing."""
    from memory_manager.vector_store_client import EmbeddingProvider
    
    return EmbeddingProvider(dummy_vector_store_config)


@pytest.fixture
def in_memory_vector_store(dummy_vector_store_config):
    """Provides an in-memory vector store for testing."""
    from memory_manager.vector_store_client import MemoryVectorStore
    
    return MemoryVectorStore(dummy_vector_store_config)


@pytest.fixture
def dummy_vector_client(dummy_vector_store_config):
    """Provides a dummy vector store client for testing."""
    from memory_manager.vector_store_client import VectorStoreClient
    
    return VectorStoreClient(dummy_vector_store_config)


@pytest.fixture
def dummy_bm25_index():
    """Provides a dummy BM25 index for testing."""
    from retrieval.indexes import InMemoryBM25Index
    
    return InMemoryBM25Index()


@pytest.fixture
def dummy_vector_index():
    """Provides a dummy vector index for testing."""
    from retrieval.indexes import InMemoryVectorIndex
    
    return InMemoryVectorIndex()


@pytest.fixture
def dummy_graph_index():
    """Provides a dummy graph index for testing."""
    from retrieval.indexes import InMemoryGraphIndex
    
    return InMemoryGraphIndex()


@pytest.fixture
def dummy_concept_extractor():
    """Provides a dummy concept extractor for testing."""
    from knowledge_graph.concept_extractor import ConceptExtractor
    
    return ConceptExtractor(
        memory_manager=None,
        enable_semantic_grouping=False
    )


@pytest.fixture
def sample_documents() -> Dict[str, str]:
    """Provides sample documents for testing."""
    return {
        "doc1": "Rebecca Platform is a multi-agent system with knowledge augmented generation.",
        "doc2": "The system uses 6 layers of memory including core, semantic, and episodic.",
        "doc3": "Knowledge graphs help in concept extraction and relationship mapping.",
        "doc4": "Offline mode enables deterministic testing without network dependencies.",
    }


@pytest.fixture
def sample_concepts() -> List[Dict[str, Any]]:
    """Provides sample concepts for testing."""
    return [
        {
            "id": "concept1",
            "text": "Rebecca Platform",
            "label": "SYSTEM",
            "confidence": 0.95
        },
        {
            "id": "concept2",
            "text": "multi-agent system",
            "label": "CONCEPT",
            "confidence": 0.90
        },
        {
            "id": "concept3",
            "text": "knowledge graph",
            "label": "CONCEPT",
            "confidence": 0.88
        }
    ]


@pytest.fixture
def mock_llm_response():
    """Provides a deterministic mock LLM response."""
    def _response(query: str, context: str = "") -> str:
        # Deterministic response based on query hash
        import hashlib
        hash_val = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
        responses = [
            "Based on the provided information, here is the answer.",
            "The system processes this request deterministically.",
            "In offline mode, responses are generated without external APIs.",
        ]
        return responses[hash_val % len(responses)]
    
    return _response


@pytest.fixture
def mock_config_data() -> Dict[str, Any]:
    """Provides mock configuration data for testing."""
    return {
        "storage": {
            "type": "memory",
            "path": ":memory:"
        },
        "llm_adapters": {
            "default": "offline_stub"
        },
        "agents": {
            "max_workers": 4,
            "timeout": 30
        },
        "ingest": {
            "batch_size": 100,
            "max_file_size": "10MB"
        },
        "misc": {
            "log_level": "INFO",
            "offline_mode": True
        }
    }


@pytest.fixture
def offline_mode_check():
    """Fixture to verify offline mode is enabled."""
    from configuration import is_offline_mode
    
    assert is_offline_mode(), "Offline mode should be enabled in tests"
    return True
