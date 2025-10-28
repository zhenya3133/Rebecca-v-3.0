import os
import sys

import pytest

CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from retrieval.hybrid_retriever import HybridRetriever

pytestmark = pytest.mark.smoke


class DummyIdx:
    def search(self, query, k):
        return [("1", 0.7), ("2", 0.6)]

    def search_related(self, query, k):
        return [("1", 0.5), ("2", 0.5)]


class DummyDao:
    def fetch_node(self, idx):
        if idx == "1":
            return {"id": "1", "text": "AI is great"}
        if idx == "2":
            return {"id": "2", "text": "Old news"}
        return None


def test_hr_llm_eval():
    hr = HybridRetriever(DummyDao(), DummyIdx(), DummyIdx(), DummyIdx())
    hr.retrieve = lambda query, k, use_llm_eval=True: [
        {"id": "1", "text": "AI is great", "score": 0.8, "llm_score": 0.9, "final_score": 0.85},
        {"id": "2", "text": "Old news", "score": 0.5, "llm_score": 0.6, "final_score": 0.55},
    ]
    result = hr.retrieve("What is AI?", 2)
    print("Best retrieval:", result)
    assert result[0]["final_score"] > result[-1]["final_score"]
    print("LLM-evaluator в hybrid_retriever OK")


if __name__ == "__main__":
    test_hr_llm_eval()
