# Rebecca-Platform Architecture Blueprint (v2)

## Overview

Rebecca-Platform implements a multi-layered memory architecture backed by
graph-oriented storage and policy-aware retrieval. The blueprint below captures
the primary modules, storage contracts, and processing flow introduced in
`src/`.

## Memory Layers

| Layer        | Purpose                                                | Backing Modules                                         |
|--------------|--------------------------------------------------------|---------------------------------------------------------|
| Core         | Persistent system rules, charters, manifests           | `memory_manager/core_memory.py`, `schema/nodes.py`      |
| Semantic     | Long-term knowledge, ontologies, patterns              | `semantic_network/semantic_graph.py`, `schema/nodes.py` |
| Episodic     | Recent events and interactions                         | `event_graph/event_graph.py`, `ingest/*`                |
| Procedural   | Playbooks, workflows, promotion decisions              | `rules/promotion.py`, `consolidation/consolidator.py`   |
| Vault        | Sensitive artifacts and raw references                 | `storage/object_store.py`, `schema/nodes.Event.raw_ref` |
| Security     | Audit logs, access decisions, policy enforcement       | `policy_engine/*`, `observability/audit_log.py`         |

Each memory layer stores references to graph nodes (facts, events,
procedures). Promotion and decay rules (`rules/`) evolve data across layers.

## Architectural Components

- **Schema (`src/schema/`)** — Pydantic definitions for nodes, edges, and
  `ContextPack` envelopes (used for retrieval responses).
- **Storage (`src/storage/`)** — DAO and view facades for relational/vector
  storage and object stores.
- **Policy Engine (`src/policy_engine/`)** — Rule-based RBAC/ABAC enforcement
  with transformers for redaction.
- **Retrieval (`src/retrieval/`)** — Hybrid retriever performing BM25 + vector
  + graph fusion scoring.
- **Rules (`src/rules/`)** — Promotion and decay primitives governing data
  lifecycle across memory layers.
- **Reconstruction (`src/reconstruction/`)** — `context_packer` assembles
  policy-compliant context packs with rationale and cache-friendly budgets.
- **Adaptive Memory (`src/adaptive_mem/`)** — Forgetting agent orchestrates
  decay and self-evolving strategies via LLM judgement.
- **Observability (`src/observability/`)** — Append-only audit log и набор
  метрик (`metrics.py`) для отслеживания coverage, contradictions,
  token-efficiency, drift и privacy violations.
- **Consolidation, Event Graph, Semantic Network, Ingest** — Supporting
  modules for graph normalization, ontology management, and modality-specific
  ingest pipelines.

## Execution Flow (High Level)

1. **Ingest** raw artifacts (PDF, audio, image) and annotate with privacy
   metadata; store raw references in vault object storage.
2. **Event Graph** normalizes inputs into events with timelines and actor
   relationships; links evidence references.
3. **Consolidation** applies semantic or preference-based strategies, invokes
   promotion rules, and records decisions via the audit log.
4. **Retrieval** использует гибридный retriever (BM25 + vector + graph
   fusion) для нахождения релевантных узлов, применяет policy-проверки,
   дополняет LLM-оценкой и формирует `ContextPack` с rationales и trace IDs.
5. **Reconstruction** consumes context packs for LLM planners or downstream
   agents.
6. **Adaptive Memory** periodically runs decay and self-evolving strategies to
   distill lessons from failures into semantic memory.
7. **Observability** агрегирует coverage, drift, contradictions,
   token-efficiency и privacy violation rate; nightly отчёт фиксирует
   отклонения.

## Directory Schematic

```
src/
 ├─ schema/              # node/edge/context models
 ├─ storage/             # DAO, graph views, object store
 ├─ policy_engine/       # RBAC/ABAC rules and redactors
 ├─ retrieval/           # hybrid retriever and scorers
 ├─ rules/               # promotion/decay logic
 ├─ consolidation/       # fact consolidation strategies
 ├─ event_graph/         # causal event modelling
 ├─ semantic_network/    # ontology & associations
 ├─ adaptive_mem/        # forgetting agent
 ├─ ingest/              # modality-specific ingestors
 ├─ reconstruction/      # context pack assembly
 ├─ observability/       # metrics & audit log
 └─ multi_agent/         # experimental shared memory & replication
```

This document should serve as the anchor reference for the blueprint
introduced in commit `b3af5ea`.
