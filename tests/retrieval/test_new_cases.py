import os
import sys


CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir, "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


from retrieval.hybrid_retriever import HybridRetriever


class DummyIdx:
    def search(self, query, k):
        return [("1", 0.8), ("2", 0.6)]

    def search_related(self, query, k):
        return [("1", 0.5), ("2", 0.4)]


class DummyDao:
    def __init__(self):
        self.nodes = {
            "1": {"id": "1", "text": "GDPR stands for General Data Protection Regulation."},
            "2": {"id": "2", "text": "AI helps automate compliance."},
        }

    def fetch_node(self, idx):
        return self.nodes.get(idx)


def test_edge_cases():
    hr = HybridRetriever(DummyDao(), DummyIdx(), DummyIdx(), DummyIdx())
    res = hr.retrieve("GDPR abbreviation, AI & privacy", k=5, use_llm_eval=False)
    print("Edge result:", res)
    assert res, "Expected retrieval results for edge case query"



if __name__ == "__main__":
    test_edge_cases()
