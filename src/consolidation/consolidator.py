"""Consolidation strategies for episodic → semantic promotion."""

from collections import Counter
from typing import Dict, Iterable, List


class MemoryConsolidator:
    def __init__(self, memory_manager):
        self.memory = memory_manager

    def summarize_events(self, events: Iterable[Dict]) -> Dict[str, List[Dict]]:
        grouped: Dict[str, List[Dict]] = {}
        for event in events:
            category = event.get("category", "general")
            grouped.setdefault(category, []).append(event)
        return grouped

    def consolidate(self) -> Dict[str, Dict[str, float]]:
        events = self.memory.episodic.get_events()
        grouped = self.summarize_events(events)
        summaries: Dict[str, Dict[str, float]] = {}
        for category, records in grouped.items():
            counter = Counter(record.get("sentiment", "neutral") for record in records)
            total = sum(counter.values()) or 1
            stats = {k: v / total for k, v in counter.items()}
            self.memory.semantic.store_concept(
                f"summary::{category}",
                {
                    "count": len(records),
                    "sentiment_breakdown": stats,
                },
            )
            summaries[category] = stats
        self.memory.episodic.clear_events()
        return summaries
