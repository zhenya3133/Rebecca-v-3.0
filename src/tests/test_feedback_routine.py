import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from auto_train.feedback_routine import update_scores_based_on_feedback


def test_feedback_updates_scores():
    retrieval_log = [
        {"query": "ai", "result": "doc1", "score": 0.2},
        {"query": "ai", "result": "doc2", "score": 0.5},
    ]
    feedback = [
        {"query": "ai", "result": "doc1", "good": True},
        {"query": "ai", "result": "doc2", "good": False},
    ]
    update_scores_based_on_feedback(retrieval_log, feedback)
    updated = {item["result"]: item["score"] for item in retrieval_log}
    assert updated["doc1"] > 0.2
    assert updated["doc2"] < 0.5
    print("Feedback routine updated scores:", updated)
