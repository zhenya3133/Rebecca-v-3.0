"""Graph view/projection layer backed by in-memory stores."""

from typing import Iterable, List

from event_graph.event_graph import InMemoryEventGraph
from schema.edges import Edge
from schema.nodes import Event


class InMemoryGraphView:
    def __init__(self, graph: InMemoryEventGraph | None = None) -> None:
        self.graph = graph or InMemoryEventGraph()

    def upsert_event(self, event: Event) -> None:
        self.graph.upsert_event(event)

    def search_related(self, query: str, k: int) -> List[tuple[str, float]]:
        matches: List[tuple[str, float]] = []
        for event in self.graph.list_events():
            score = 0.0
            if query.lower() in event.attrs.get("text", "").lower():
                score += 1.0
            if query.lower() in event.ntype.lower():
                score += 0.5
            if score:
                matches.append((event.id, score))
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[:k]

    def link(self, edge: Edge) -> None:
        self.graph.link(edge)

    def neighbors(self, source_id: str) -> Iterable[Event]:
        return self.graph.neighbors(source_id)
