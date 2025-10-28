import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from retrieval.indexes import InMemoryBM25Index, InMemoryVectorIndex, InMemoryGraphIndex


def test_bm25_index():
    idx = InMemoryBM25Index()
    idx.upsert("1", "privacy law gdpr")
    results = idx.search("gdpr", 5)
    assert results and results[0][0] == "1"


def test_vector_index():
    idx = InMemoryVectorIndex()
    idx.upsert("1", [0.1, 0.2])
    idx.upsert("2", [0.2, 0.3])
    results = idx.search("query", 2)
    assert len(results) == 2


def test_graph_index():
    idx = InMemoryGraphIndex()
    idx.set_neighbors("node", ["a", "b"])
    results = idx.search_related("node", 2)
    assert len(results) == 2
