import os
import sys


CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


from retrieval.hybrid_retriever import HybridRetriever
from observability.metrics import coverage_at_k, contradiction_rate


def run_nightly_eval():
    queries = ["What is AI?", "Explain RL", "Data privacy"]
    ground_truth = [["AI is smart", "Artificial Intelligence"], ["RL is ..."], ["privacy", "GDPR"]]
    dummy_results = [["AI is smart", "Noise"], ["RL is ...", "Mistake"], ["privacy"]]
    for query, gt, retrieved in zip(queries, ground_truth, dummy_results):
        cov = coverage_at_k(retrieved, gt, k=2)
        print(f"Q: {query} | coverage@2 = {cov:.2f}")
    contr = contradiction_rate([item for results in dummy_results for item in results])
    print("Total contradiction rate:", contr)
    print("Nightly eval done.")


if __name__ == "__main__":
    run_nightly_eval()
