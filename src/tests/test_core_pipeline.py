import os
import sys
from datetime import datetime

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ingest.loader import IngestPipeline
from ingest.audio_ingest import ingest_audio
from ingest.image_ingest import ingest_image
from ingest.cross_modal_links import link_artifacts
from memory_manager.memory_manager import MemoryManager
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.indexes import InMemoryBM25Index, InMemoryVectorIndex, InMemoryGraphIndex
from storage.pg_dao import InMemoryDAO
from storage.object_store import InMemoryObjectStore
from storage.graph_view import InMemoryGraphView
from event_graph.event_graph import InMemoryEventGraph
from reconstruction.context_packer import build_context_pack
from schema.nodes import Fact
from policy_engine.policy_engine import PolicyEngine
from policy_engine.evaluators import allow_all


def test_ingest_consolidate_retrieve_reconstruct():
    memory = MemoryManager()
    dao = InMemoryDAO()
    bm25 = InMemoryBM25Index()
    vec = InMemoryVectorIndex()
    graph_idx = InMemoryGraphIndex()
    graph_view = InMemoryGraphView(InMemoryEventGraph())
    object_store = InMemoryObjectStore()

    pipeline = IngestPipeline(memory, dao, bm25, vec, graph_idx, graph_view, object_store)
    # --- ingest layer ---
    event = pipeline.ingest_pdf("dummy.pdf")
    ingest_audio(memory, "call.wav")
    ingest_image(memory, "diagram.png")
    link_artifacts(memory, "pdf-audio", ["pdf::dummy", "call.wav"])
    fact = Fact(
        id="fact::gdpr",
        ntype="Fact",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        owner="system",
        privacy="team",
        confidence=0.8,
        subject="GDPR",
        predicate="relates_to",
        object="privacy",
        evidence=[],
    )
    pipeline.ingest_facts([fact])

    # --- consolidate ---
    memory.episodic.store_event({"id": event.id, "category": "ingest", "sentiment": "positive"})
    from consolidation.consolidator import MemoryConsolidator

    consolidator = MemoryConsolidator(memory)
    summaries = consolidator.consolidate()
    assert summaries

    # --- retrieve ---
    retriever = HybridRetriever(dao, graph_idx, bm25, vec)
    results = retriever.retrieve("privacy", k=5, use_llm_eval=False)
    assert results, "Expected retrieval results"

    # --- policy & reconstruction ---
    policy = PolicyEngine([allow_all])
    pack = build_context_pack(
        query="privacy",
        nodes=[dao.fetch_node(item["id"]) for item in results],
        policy=policy,
        actor="test",
        budget_tokens=1024,
    )
    assert pack.nodes, "Context pack must contain nodes"
    assert memory.semantic.get_concept("cross_modal::pdf-audio")
    print("Core pipeline context pack:", pack.model_dump())
