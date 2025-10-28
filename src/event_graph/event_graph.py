"""In-memory event graph for ingest and consolidation flows."""

from typing import Dict, Iterable, List

from schema.edges import Edge
from schema.nodes import Event


class InMemoryEventGraph:
    def __init__(self) -> None:
        self._events: Dict[str, Event] = {}
        self._edges: List[Edge] = []

    def upsert_event(self, event: Event) -> None:
        self._events[event.id] = event

    def get_event(self, event_id: str) -> Event | None:
        return self._events.get(event_id)

    def list_events(self) -> List[Event]:
        return list(self._events.values())

    def link(self, edge: Edge) -> None:
        self._edges.append(edge)

    def list_edges(self) -> List[Edge]:
        return list(self._edges)

    def neighbors(self, source_id: str) -> Iterable[Event]:
        ids = [edge.target for edge in self._edges if edge.source == source_id]
        return [self._events[_id] for _id in ids if _id in self._events]
