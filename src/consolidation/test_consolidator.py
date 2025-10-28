import os
import sys


CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


from consolidation.consolidator import MemoryConsolidator


class DummyEpisodic:
    def __init__(self):
        self.events = ["I loved the result!", "Work failed", "It is a day."]

    def get_events(self):
        return self.events

    def clear_events(self):
        self.events = []


class DummySemantic:
    def __init__(self):
        self.data = []

    def store_concept(self, key, value):
        self.data.append((key, value))

    def get_all(self):
        return self.data


class DummyMemory:
    def __init__(self):
        self.episodic = DummyEpisodic()
        self.semantic = DummySemantic()


def test_consolidator_emotions():
    memory = DummyMemory()
    consolidator = MemoryConsolidator(memory)
    consolidator.consolidate()
    data = memory.semantic.get_all()
    print("Emotional summaries:", data)
    assert any("positive" in value for _, value in data)
    assert any("negative" in value for _, value in data)
    assert any("neutral" in value for _, value in data)
    print("test_consolidator_emotions OK")


if __name__ == "__main__":
    test_consolidator_emotions()
