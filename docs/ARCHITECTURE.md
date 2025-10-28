# Rebecca-Platform Architecture Overview

## Core Principles
- Многоуровневая память (Core, Semantic, Episodic, Procedural, Vault, Security) управляется через `MemoryManager`.
- Оркестрация заданий осуществляется Meta-Orchestrator'ом; агенты из `src/<module>` взаимодействуют через унифицированные конверты (`ContextPack`).
- Retrieval-стек основан на гибридном слиянии BM25, векторного поиска и графовых связей.

## Потоки данных
1. **Ingest**: модули `src/ingest/*` принимают артефакты, обогащают метаданными, сохраняют ссылки в Vault.
2. **Event Graph & Semantic Network**: нормализуют события, строят связи, обновляют онтологии.
3. **Consolidation & Rules**: применяют стратегии промоушена/декея, фиксируют решения в аудит-логе.
4. **Retrieval**: `HybridRetriever` агрегирует результаты нескольких индексов, затем применяет policy и LLM-оценку релевантности.
5. **Reconstruction**: `context_packer` формирует итоговые контексты для агентов/LLM.
6. **Observability**: `audit_log` и `metrics` контролируют drift, privacy-нарушения и качество выдачи.

## Ключевые файлы
- `docs/architecture_blueprint.md` — подробный blueprint (v2) с описанием слоёв и модулей.
- `src/retrieval/hybrid_retriever.py` — реализация гибридного ретривера.
- `src/observability/metrics.py` — метрики coverage, contradictions, token efficiency, drift, privacy.
- `src/auto_train/feedback_routine.py` — обновление скорингов на основе обратной связи.
- `tests/retrieval/test_new_cases.py` — регрессионные сценарии для retrieval/ingest.

## CI/Automation Hooks
- GitHub Actions (`.github/workflows/tests.yml`) выполняет smoke, integration и nightly-метрики.
- Nightly задачи (`tests/nightly_eval.py`) фиксируют метрики и сигнализируют о дрейфе.
