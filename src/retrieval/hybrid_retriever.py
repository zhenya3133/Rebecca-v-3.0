from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from .scorers import fusion_score
from .llm_evaluator import llm_judge_relevancy


class HybridRetriever:
    def __init__(self, dao, graph_view, bm25_idx, vec_idx):
        self.dao = dao
        self.graph = graph_view
        self.bm25 = bm25_idx
        self.vec = vec_idx

    def _as_dict(self, items: Iterable[Tuple[str, float]]) -> Dict[str, float]:
        scores: Dict[str, float] = defaultdict(float)
        for idx, score in items:
            scores[idx] = max(scores[idx], score)
        return scores

    def retrieve(self, query: str, k: int = 40, use_llm_eval: bool = True):
        bm = self._as_dict(self.bm25.search(query, k * 2))
        ve = self._as_dict(self.vec.search(query, k * 2))
        gr = self._as_dict(self.graph.search_related(query, k * 2))

        all_ids = set(bm) | set(ve) | set(gr)
        fused: List[Tuple[str, float]] = []
        for idx in all_ids:
            score = fusion_score(bm.get(idx, 0.0), ve.get(idx, 0.0), gr.get(idx, 0.0))
            fused.append((idx, score))

        fused.sort(key=lambda x: x[1], reverse=True)
        candidates = []
        for idx, score in fused:
            node = self.dao.fetch_node(idx)
            if not node:
                continue
            payload = node.model_dump() if hasattr(node, "model_dump") else (
                node.dict() if hasattr(node, "dict") else node
            )
            if not isinstance(payload, dict):
                payload = {"text": str(payload)}
            candidate = {
                "id": idx,
                "text": payload.get("text") or payload.get("attrs", {}).get("text", str(node)),
                "score": score,
                **payload,
            }
            candidates.append(candidate)

        if use_llm_eval:
            for candidate in candidates:
                llm_score = llm_judge_relevancy(query, candidate.get("text", ""))
                candidate["llm_score"] = llm_score
                candidate["final_score"] = 0.5 * candidate["score"] + 0.5 * llm_score
            candidates.sort(key=lambda c: c.get("final_score", c["score"]), reverse=True)
        else:
            candidates.sort(key=lambda c: c["score"], reverse=True)

        return candidates[:k]
