from typing import Any, Dict, Iterable, List


class EpisodicMemory:
    def __init__(self) -> None:
        self._events: List[Dict[str, Any]] = []

    def store_event(self, event: Dict[str, Any]) -> None:
        self._events.append(event)

    def store_events(self, events: Iterable[Dict[str, Any]]) -> None:
        for event in events:
            self.store_event(event)

    def get_events(self) -> List[Dict[str, Any]]:
        return list(self._events)

    def find_events(self, predicate) -> List[Dict[str, Any]]:
        return [event for event in self._events if predicate(event)]

    def clear_events(self) -> None:
        self._events.clear()
