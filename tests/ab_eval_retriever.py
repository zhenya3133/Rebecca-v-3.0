import random

from retrieval.hybrid_retriever import HybridRetriever


def ab_eval(hrA, hrB, queries, ground_truth):
    better_A, better_B = 0, 0
    for query, gt in zip(queries, ground_truth):
        resultA = hrA.retrieve(query, 3)
        resultB = hrB.retrieve(query, 3)
        topA = {item.get("text") for item in resultA}
        topB = {item.get("text") for item in resultB}
        scoreA = len(set(gt) & topA)
        scoreB = len(set(gt) & topB)
        if scoreA > scoreB:
            better_A += 1
        elif scoreB > scoreA:
            better_B += 1
    print(f"A/B eval: A better: {better_A}, B better: {better_B}")


# Моки для демонстрации, подключи свои HybridRetriever настроенные по-разному
# ab_eval(HybridRetriever(...), HybridRetriever(...), ["q1", ...], [["gt1", ...], ...])
