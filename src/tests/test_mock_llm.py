import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from retrieval.llm_evaluator import llm_judge_relevancy


def test_llm_mock_range():
    score = llm_judge_relevancy("test", "some text")
    assert 0.5 <= score <= 1.0
    print("LLM mock score:", score)
