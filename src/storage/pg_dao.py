"""DAO facade with in-memory document store for retrieval."""

from typing import Dict, Iterable, List, Tuple

from schema.nodes import Event, Fact, NodeBase, Procedure


class InMemoryDAO:
    def __init__(self) -> None:
        self._nodes: Dict[str, dict] = {}

    def upsert_node(self, node: NodeBase | dict) -> None:
        payload = node.dict() if isinstance(node, NodeBase) else node
        self._nodes[payload["id"]] = payload

    def fetch_node(self, node_id: str) -> NodeBase | None:
        raw = self._nodes.get(node_id)
        if not raw:
            return None
        ntype = raw.get("ntype")
        if ntype == "Event":
            return Event(**raw)
        if ntype == "Fact":
            return Fact(**raw)
        if ntype == "Procedure":
            return Procedure(**raw)
        return NodeBase(**raw)

    def search_text(self, query: str, k: int) -> List[Tuple[str, float]]:
        matches: List[Tuple[str, float]] = []
        for node_id, node in self._nodes.items():
            text = node.get("text") or node.get("attrs", {}).get("text", "")
            score = text.lower().count(query.lower())
            if score:
                matches.append((node_id, float(score)))
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[:k]

    def all_nodes(self) -> Iterable[NodeBase]:
        return [self.fetch_node(node_id) for node_id in self._nodes]