def fusion_score(bm25: float, vec: float, graph: float, w=(0.35, 0.45, 0.20)) -> float:
    return w[0] * bm25 + w[1] * vec + w[2] * graph
