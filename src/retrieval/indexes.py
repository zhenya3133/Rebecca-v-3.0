"""In-memory search indexes used by HybridRetriever for testing."""

from typing import Dict, Iterable, List, Tuple


class InMemoryBM25Index:
    def __init__(self, documents: Dict[str, str] | None = None) -> None:
        self.documents = documents or {}

    def upsert(self, doc_id: str, text: str) -> None:
        self.documents[doc_id] = text

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        matches: List[Tuple[str, float]] = []
        for doc_id, text in self.documents.items():
            score = text.lower().count(query.lower())
            if score:
                matches.append((doc_id, float(score)))
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[:k]


class InMemoryVectorIndex:
    def __init__(self, embeddings: Dict[str, List[float]] | None = None) -> None:
        self.embeddings = embeddings or {}

    def upsert(self, doc_id: str, embedding: List[float]) -> None:
        self.embeddings[doc_id] = embedding

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        # simplistic cosine-like similarity with dummy query embedding
        dummy = [1.0] * len(next(iter(self.embeddings.values()), [1.0]))
        matches: List[Tuple[str, float]] = []
        for doc_id, emb in self.embeddings.items():
            score = sum(a * b for a, b in zip(dummy, emb))
            matches.append((doc_id, score))
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[:k]


class InMemoryGraphIndex:
    def __init__(self, neighbors: Dict[str, List[str]] | None = None) -> None:
        self.neighbors = neighbors or {}

    def set_neighbors(self, node_id: str, neigh_ids: Iterable[str]) -> None:
        self.neighbors[node_id] = list(neigh_ids)

    def search_related(self, query: str, k: int) -> List[Tuple[str, float]]:
        # Graph index is query-agnostic for MVP; return pre-linked neighbors.
        results: List[Tuple[str, float]] = []
        for node_id, neighs in self.neighbors.items():
            score = 1.0 if query.lower() in node_id.lower() else 0.5
            for neigh in neighs:
                results.append((neigh, score))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:k]
