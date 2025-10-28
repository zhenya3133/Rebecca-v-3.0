"""
Глобальные фикстуры для KAG Testing Framework
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Any
import json
import os


@pytest.fixture
def base_config():
    """Базовая конфигурация для KAG системы"""
    return {
        "memory_layers": {
            "semantic": {"max_concepts": 10000, "embedding_dim": 768},
            "episodic": {"max_events": 5000, "retention_days": 30},
            "procedural": {"max_workflows": 1000, "max_steps": 100},
            "vault": {"max_secrets": 100, "encryption": "AES-256"},
            "security": {"max_policies": 500, "audit_level": "high"},
            "vector_store": {"dimension": 768, "metric": "cosine", "index_type": "hnsw"}
        },
        "graph_config": {
            "max_nodes": 50000,
            "max_edges": 200000,
            "traversal_depth": 10,
            "pruning_threshold": 0.1
        },
        "retrieval_config": {
            "bm25_k": 40,
            "vector_k": 40,
            "graph_k": 40,
            "fusion_weights": [0.4, 0.4, 0.2],
            "llm_evaluation": True
        },
        "performance_config": {
            "max_concurrent_graph_traversals": 10,
            "timeout_seconds": 30,
            "memory_limit_mb": 512
        }
    }


@pytest.fixture
def strict_config():
    """Строгая конфигурация для stress тестов"""
    return {
        "memory_layers": {
            "semantic": {"max_concepts": 50000, "embedding_dim": 1024},
            "episodic": {"max_events": 25000, "retention_days": 7},
            "procedural": {"max_workflows": 5000, "max_steps": 200},
            "vault": {"max_secrets": 500, "encryption": "AES-256"},
            "security": {"max_policies": 1000, "audit_level": "critical"},
            "vector_store": {"dimension": 1024, "metric": "cosine", "index_type": "hnsw"}
        },
        "graph_config": {
            "max_nodes": 100000,
            "max_edges": 500000,
            "traversal_depth": 15,
            "pruning_threshold": 0.05
        },
        "retrieval_config": {
            "bm25_k": 80,
            "vector_k": 80,
            "graph_k": 80,
            "fusion_weights": [0.3, 0.4, 0.3],
            "llm_evaluation": True
        },
        "performance_config": {
            "max_concurrent_graph_traversals": 20,
            "timeout_seconds": 15,
            "memory_limit_mb": 256
        }
    }


@pytest.fixture
def memory_manager_mock():
    """Mock объект для MemoryManager"""
    memory_manager = Mock()
    
    # Настройка семантической памяти
    memory_manager.semantic = Mock()
    memory_manager.semantic.store_concept = Mock()
    memory_manager.semantic.get_concept = Mock(return_value="mock_concept")
    memory_manager.semantic.concepts = {
        "artificial_intelligence": "Complex AI system for reasoning",
        "machine_learning": "Subset of AI focused on pattern recognition",
        "neural_networks": "Deep learning architecture",
        "cognitive_bias": "Systematic error in thinking patterns",
        "decision_making": "Process of selecting optimal choices"
    }
    
    # Настройка эпизодической памяти
    memory_manager.episodic = Mock()
    memory_manager.episodic.store_event = Mock()
    memory_manager.episodic.get_events = Mock(return_value=[
        {"type": "knowledge_ingest", "timestamp": "2025-10-28T06:56:00Z"},
        {"type": "concept_extraction", "timestamp": "2025-10-28T06:55:00Z"},
        {"type": "graph_update", "timestamp": "2025-10-28T06:54:00Z"}
    ])
    memory_manager.episodic.events = []
    
    # Настройка процедурной памяти
    memory_manager.procedural = Mock()
    memory_manager.procedural.store_workflow = Mock()
    memory_manager.procedural.get_workflow = Mock(return_value=["ingest", "normalize", "link"])
    memory_manager.procedural.workflows = {}
    
    # Настройка хранилища секретов
    memory_manager.vault = Mock()
    memory_manager.vault.store_secret = Mock()
    memory_manager.vault.get_secret = Mock(return_value="mock_secret")
    memory_manager.vault.secrets = {}
    
    # Настройка безопасной памяти
    memory_manager.security = Mock()
    memory_manager.security.check_policy = Mock(return_value=True)
    memory_manager.security.store_policy = Mock()
    memory_manager.security.policies = {}
    
    # Настройка векторного хранилища
    memory_manager.vector_store = Mock()
    memory_manager.vector_store.add_vector = Mock()
    memory_manager.vector_store.search = Mock(return_value=[
        {"id": "vec1", "score": 0.95, "text": "artificial intelligence definition"},
        {"id": "vec2", "score": 0.87, "text": "machine learning algorithms"}
    ])
    memory_manager.vector_store.vectors = []
    
    return memory_manager


@pytest.fixture
def kag_graph_mock():
    """Mock объект для KAG Graph"""
    graph = Mock()
    
    # Настройка узлов графа
    graph.nodes = {
        "ai_concept": {
            "id": "ai_concept",
            "type": "concept",
            "label": "Artificial Intelligence",
            "properties": {
                "definition": "Computer system capable of intelligent behavior",
                "confidence": 0.95,
                "source": "wikipedia"
            }
        },
        "ml_concept": {
            "id": "ml_concept",
            "type": "concept", 
            "label": "Machine Learning",
            "properties": {
                "definition": "Subset of AI focused on data-driven learning",
                "confidence": 0.92,
                "source": "research_papers"
            }
        },
        "bias_concept": {
            "id": "bias_concept",
            "type": "concept",
            "label": "Cognitive Bias",
            "properties": {
                "definition": "Systematic error in human thinking",
                "confidence": 0.89,
                "source": "psychology_studies"
            }
        }
    }
    
    # Настройка рёбер графа
    graph.edges = [
        {"source": "ml_concept", "target": "ai_concept", "relation": "is_subset_of", "weight": 0.95},
        {"source": "ai_concept", "target": "bias_concept", "relation": "can_have", "weight": 0.7},
        {"source": "ml_concept", "target": "bias_concept", "relation": "can_exhibit", "weight": 0.6}
    ]
    
    # Mock методы графа
    graph.add_node = Mock()
    graph.add_edge = Mock()
    graph.remove_node = Mock()
    graph.remove_edge = Mock()
    graph.get_neighbors = Mock(return_value=[{"id": "ml_concept", "relation": "is_subset_of", "weight": 0.95}])
    graph.search_related = Mock(return_value=[
        {"id": "ai_concept", "score": 0.95, "relation": "related"},
        {"id": "ml_concept", "score": 0.87, "relation": "subset"}
    ])
    graph.traverse = Mock(return_value=["ai_concept", "ml_concept", "bias_concept"])
    graph.get_path = Mock(return_value=["ai_concept", "ml_concept"])
    
    return graph


@pytest.fixture
def concept_extractor_mock():
    """Mock объект для ConceptExtractor"""
    extractor = Mock()
    
    # Настройка извлекателя концептов
    extractor.extract_concepts = Mock(return_value=[
        {
            "name": "Artificial Intelligence",
            "type": "concept",
            "confidence": 0.95,
            "description": "Computer system capable of intelligent behavior",
            "context": "AI research and applications",
            "source": "text_analysis"
        },
        {
            "name": "Machine Learning",
            "type": "concept", 
            "confidence": 0.92,
            "description": "Subset of AI focused on data-driven learning",
            "context": "algorithms and models",
            "source": "domain_knowledge"
        }
    ])
    
    extractor.validate_concept = Mock(return_value=True)
    extractor.link_concepts = Mock(return_value=[
        {"source": "Machine Learning", "target": "Artificial Intelligence", "relation": "is_subset_of"}
    ])
    extractor.score_concept_quality = Mock(return_value=0.85)
    extractor.normalize_concept = Mock(return_value={
        "name": "artificial_intelligence",
        "type": "concept",
        "normalized": True
    })
    
    return extractor


@pytest.fixture
def context_engine_mock():
    """Mock объект для ContextEngine"""
    engine = Mock()
    
    engine.build_context = Mock(return_value={
        "semantic_context": {
            "concepts": ["AI", "ML", "cognitive_bias"],
            "relationships": ["ML is_subset_of AI", "AI can_have bias"],
            "confidence": 0.87
        },
        "episodic_context": {
            "recent_events": ["knowledge_ingest", "concept_extraction"],
            "temporal_order": ["2025-10-28T06:54:00Z", "2025-10-28T06:55:00Z", "2025-10-28T06:56:00Z"]
        },
        "procedural_context": {
            "active_workflow": "knowledge_processing",
            "steps_completed": ["ingest", "extract", "validate"],
            "next_step": "link"
        }
    })
    
    engine.get_context_for_query = Mock(return_value={
        "query_context": "What is the relationship between AI and cognitive bias?",
        "relevant_concepts": ["artificial_intelligence", "cognitive_bias", "decision_making"],
        "context_score": 0.92
    })
    
    engine.merge_contexts = Mock(return_value={
        "merged_context": "Combined semantic and episodic context",
        "confidence": 0.89,
        "sources": ["semantic", "episodic", "procedural"]
    })
    
    engine.rank_context_relevance = Mock(return_value=[
        {"context": "AI definition", "relevance": 0.95},
        {"context": "ML algorithms", "relevance": 0.87},
        {"context": "bias examples", "relevance": 0.75}
    ])
    
    return engine


@pytest.fixture
def hybrid_retriever_mock():
    """Mock объект для HybridRetriever"""
    retriever = Mock()
    
    retriever.retrieve = Mock(return_value=[
        {
            "id": "doc_ai_1",
            "text": "Artificial Intelligence (AI) is intelligence demonstrated by machines",
            "score": 0.95,
            "llm_score": 0.89,
            "final_score": 0.92,
            "source": "wikipedia",
            "metadata": {"topic": "AI", "confidence": 0.95}
        },
        {
            "id": "doc_ml_1", 
            "text": "Machine Learning is a subset of artificial intelligence",
            "score": 0.87,
            "llm_score": 0.92,
            "final_score": 0.89,
            "source": "research_paper",
            "metadata": {"topic": "ML", "confidence": 0.87}
        }
    ])
    
    retriever.fusion_score = Mock(return_value=0.85)
    retriever.evaluate_relevance = Mock(return_value={"relevant": True, "confidence": 0.83})
    
    return retriever


@pytest.fixture
def cognitive_bias_data():
    """Тестовые данные по когнитивным искажениям"""
    return {
        "confirmation_bias": {
            "definition": "Tendency to search for, interpret, favor, and recall information that confirms one's preexisting beliefs",
            "examples": [
                "Ignoring contradictory evidence when making investment decisions",
                "Only reading news sources that align with political views",
                "Dismissing expert opinions that challenge existing beliefs"
            ],
            "related_concepts": ["selective_perception", "biased_interpretation", "belief_persistence"],
            "severity": "high",
            "contexts": ["decision_making", "information_processing", "belief_systems"]
        },
        "availability_heuristic": {
            "definition": "Overestimating likelihood of events based on ease of recall",
            "examples": [
                "Fearing plane crashes more than car accidents",
                "Overestimating crime rates after watching crime shows",
                "Believing smoking is less dangerous because you know smokers who are healthy"
            ],
            "related_concepts": ["recency_bias", "frequency_estimation", "emotional_response"],
            "severity": "medium",
            "contexts": ["risk_assessment", "probability_judgment", "memory_recall"]
        },
        "anchoring_bias": {
            "definition": "Over-relying on first piece of information received when making decisions",
            "examples": [
                "Negotiations heavily influenced by initial price offers",
                "First estimate strongly affects final valuation",
                "Initial impression colors subsequent judgments"
            ],
            "related_concepts": ["first_impression", "reference_point", "adjustment_bias"],
            "severity": "medium",
            "contexts": ["negotiation", "estimation", "judgment"]
        },
        "overconfidence_bias": {
            "definition": "Excessive confidence in one's own answers to questions",
            "examples": [
                "Overestimating accuracy of predictions",
                "Underestimating time needed to complete tasks",
                "Taking on projects beyond actual capability"
            ],
            "related_concepts": ["dunning_kruger_effect", "self_attribution_bias", "planning_fallacy"],
            "severity": "high",
            "contexts": ["skill_assessment", "planning", "performance_evaluation"]
        },
        "sunk_cost_fallacy": {
            "definition": "Continuing a behavior or endeavor as a result of previously invested resources",
            "examples": [
                "Continuing to play losing poker game to 'win back' losses",
                "Sticking with bad investment due to initial investment",
                "Continuing bad relationship due to time already invested"
            ],
            "related_concepts": ["loss_aversion", "escalation_of_commitment", "investment_bias"],
            "severity": "high",
            "contexts": ["economic_decisions", "relationship_management", "project_management"]
        }
    }


@pytest.fixture
def kag_test_data():
    """Общие тестовые данные для KAG системы"""
    return {
        "sample_texts": [
            "Artificial Intelligence (AI) represents the simulation of human intelligence processes by machines, especially computer systems.",
            "Machine Learning is a subset of artificial intelligence that focuses on the use of data and algorithms to imitate the way that humans learn.",
            "Cognitive biases are systematic patterns of deviation from norm or rationality in judgment.",
            "Decision-making in complex environments often involves heuristic processing that can lead to systematic errors.",
            "Neural networks are computing systems inspired by biological neural networks that constitute animal brains."
        ],
        "sample_queries": [
            "What is the relationship between AI and machine learning?",
            "How do cognitive biases affect decision making?",
            "What are the main types of cognitive biases in data analysis?",
            "How can neural networks be used to reduce bias in AI systems?",
            "What is the difference between artificial intelligence and machine intelligence?"
        ],
        "expected_concepts": [
            "artificial_intelligence",
            "machine_learning", 
            "cognitive_bias",
            "decision_making",
            "neural_networks",
            "heuristic_processing",
            "systematic_errors",
            "data_analysis",
            "bias_reduction",
            "ai_systems"
        ],
        "expected_relationships": [
            "machine_learning is_subset_of artificial_intelligence",
            "cognitive_bias affects decision_making",
            "neural_networks used_in artificial_intelligence",
            "heuristic_processing can_cause systematic_errors",
            "data_analysis requires bias_awareness"
        ]
    }