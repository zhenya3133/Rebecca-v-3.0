# AGENTS.md

## Справочник агентов системы Rebecca-Platform

| Агент             | Роль                                              | Используемые слои памяти              | Краткое описание доступа |
|-------------------|---------------------------------------------------|---------------------------------------|--------------------------|
| Meta-Orchestrator | Управление системой, распределение задач, SLA      | Core, Semantic, Procedural, Security  | Глобальное состояние, правила управления, аудиты |
| Architect         | Проектирование архитектуры системы                | Core, Semantic                        | Запись архитектурных фактов и паттернов |
| CodeGen           | Генерация и сборка кода                           | Procedural, Vault                     | Хранит pipeline workflow и деплой-секреты |
| QA                | Тестирование, создание и запуск тестов            | Episodic, Procedural                  | Сохраняет события прогонов тестов и чеклисты |
| Educator          | Подготовка обучающих материалов                   | Core, Semantic                        | Фиксация уроков, концептуальных активов |
| Researcher        | Сбор знаний, анализ внешних источников            | Core, Semantic                        | Хранит результаты исследований, long-term знания |
| Memory Manager    | Управление слоями памяти                          | Core, Procedural                      | Аудит доступа, описание внутренних memory flows |
| Idea Generator    | Генерация идей, альтернативных решений            | Semantic, Episodic                    | Хранит новые концепты, историю идей |
| Security Agent    | Моделирование угроз, политика безопасности        | Security, Vault                       | Запись аудитов безопасности, управление секретами |
| UI/UX             | Проектирование пользовательских сценариев         | Semantic, Episodic                    | Карта UI-концептов, история действий пользователя |
| Integration       | Связь сервисов, поддержка API контрактов          | Vault, Procedural                     | Управление токенами интеграций и workflow |
| Feedback          | Обработка обратной связи                          | Episodic, Semantic                     | Сессии обратной связи и контекстные инсайты |
| Scheduler         | Оркестрация таймингов, распределение ресурсов     | Procedural, Episodic                  | Планирование и история запуска |
| Logger            | Логирование событий, структурная телеметрия       | Episodic, Security                    | Сессии логов и сопутствующие аудиты |

---

## Описание связи с памятью (примеры для main.py и тестов)

- Architect: `core.store_fact`, `semantic.store_concept`
- CodeGen: `procedural.store_workflow`, `vault.store_secret`
- QA: `episodic.store_event`, `procedural.store_workflow`
- Security Agent: `security.store_audit`, `vault.store_secret`
- UI/UX: `semantic.store_concept`, `episodic.store_event`
- И т.д. — см. шаблоны агентских main.py/test_main.py

---

## Принципы работы агентов

- Каждый агент получает доступ только к нужным слоям памяти через объект MemoryManager.
- Вся логика взаимодействия с памятью реализуется через Python-классы core_memory, semantic_memory, и пр.
- Smoke-тесты обеспечивают проверку корректности доступа агента к слоям памяти.
- Логирование обращений и ошибок реализовано через event_logger.

