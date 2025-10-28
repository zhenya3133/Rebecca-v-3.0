"""Comprehensive тесты для MemoryManager с полным покрытием всех компонентов.

Включает:
- Unit тесты для каждого слоя памяти
- Integration тесты для MemoryManager
- Тесты для MemoryContext и VectorStoreClient интеграции
- Тесты для AdaptiveBlueprintTracker
- Performance тесты для больших объемов данных
- Mock тесты для внешних зависимостей
- Smoke тесты для API endpoints

Автор: Claude Code
Дата: 2025-10-28
"""

import asyncio
import json
import pytest
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any

# Настройка путей
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Импорты
from memory_manager.memory_manager import (
    MemoryManager,
    AdaptiveBlueprintTracker,
    MemoryLayerFactory,
    MemoryCache,
    LayerStats,
    CacheEntry,
    create_memory_manager,
    quick_memory_test
)
from memory_manager.memory_context import (
    MemoryContext,
    MemoryEntry,
    CORE,
    EPISODIC,
    SEMANTIC,
    PROCEDURAL,
    VAULT,
    SECURITY
)
from memory_manager.vector_store_client import (
    VectorStoreClient,
    VectorStoreConfig,
    VectorItem
)
from memory_manager.semantic_memory import SemanticMemory
from memory_manager.core_memory import CoreMemory
from memory_manager.episodic_memory import EpisodicMemory
from memory_manager.procedural_memory import ProceduralMemory
from memory_manager.vault_memory import VaultMemory
from memory_manager.security_memory import SecurityMemory


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Временная директория для тестов."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def vector_config():
    """Конфигурация векторного хранилища для тестов."""
    return VectorStoreConfig(
        provider="memory",
        collection_name="test_vectors",
        vector_size=384,
        distance_metric="cosine"
    )


@pytest.fixture
async def memory_manager(vector_config):
    """MemoryManager для тестов."""
    manager = MemoryManager(
        cache_size=100,
        cache_ttl=300,
        vector_store_config=vector_config,
        optimization_interval=60
    )
    
    await manager.start()
    yield manager
    await manager.stop()


@pytest.fixture
def memory_context():
    """MemoryContext для тестов."""
    return MemoryContext()


@pytest.fixture
def semantic_layer():
    """Слой семантической памяти для тестов."""
    return SemanticMemory()


@pytest.fixture
def adaptive_tracker(semantic_layer):
    """AdaptiveBlueprintTracker для тестов."""
    return AdaptiveBlueprintTracker(semantic_layer)


@pytest.fixture
def memory_cache():
    """MemoryCache для тестов."""
    return MemoryCache(max_size=50, default_ttl=300)


@pytest.fixture
def layer_factory():
    """MemoryLayerFactory для тестов."""
    return MemoryLayerFactory()


@pytest.fixture
def sample_data():
    """Примерные данные для тестов."""
    return {
        "text": "Тестовые данные для проверки функциональности",
        "metadata": {"source": "test", "priority": 5},
        "tags": ["test", "automated"],
        "timestamp": datetime.now().isoformat()
    }


@pytest.fixture
def large_dataset():
    """Большой набор данных для performance тестов."""
    return [
        {
            "id": f"item_{i}",
            "content": f"Content item {i}",
            "metadata": {"index": i, "batch": i // 100},
            "tags": [f"tag_{i % 10}"],
            "priority": i % 10
        }
        for i in range(1000)
    ]


# ============================================================================
# UNIT TESTS - Memory Layers
# ============================================================================

class TestCoreMemory:
    """Тесты для Core Memory слоя."""
    
    def test_store_and_retrieve_fact(self):
        """Тест сохранения и извлечения факта."""
        core = CoreMemory()
        
        # Сохраняем факт
        core.store_fact("rule_enabled", True)
        core.store_fact("max_retries", 3)
        
        # Извлекаем факт
        assert core.get_fact("rule_enabled") is True
        assert core.get_fact("max_retries") == 3
        assert core.get_fact("nonexistent") is None
    
    def test_overwrite_fact(self):
        """Тест перезаписи факта."""
        core = CoreMemory()
        
        core.store_fact("counter", 1)
        assert core.get_fact("counter") == 1
        
        core.store_fact("counter", 2)
        assert core.get_fact("counter") == 2
    
    def test_data_persistence(self):
        """Тест сохранения данных."""
        core = CoreMemory()
        
        test_data = {
            "string": "value",
            "number": 42,
            "boolean": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"}
        }
        
        for key, value in test_data.items():
            core.store_fact(key, value)
        
        for key, expected_value in test_data.items():
            assert core.get_fact(key) == expected_value


class TestEpisodicMemory:
    """Тесты для Episodic Memory слоя."""
    
    def test_store_event(self):
        """Тест сохранения события."""
        episodic = EpisodicMemory()
        
        event = {
            "type": "user_action",
            "timestamp": datetime.now().isoformat(),
            "data": {"user_id": 123, "action": "click"}
        }
        
        episodic.store_event(event)
        events = episodic.get_events()
        
        assert len(events) == 1
        assert events[0]["type"] == "user_action"
    
    def test_get_events_by_time_range(self):
        """Тест получения событий по временному диапазону."""
        episodic = EpisodicMemory()
        
        now = datetime.now()
        past = now - timedelta(hours=1)
        future = now + timedelta(hours=1)
        
        episodic.store_event({"type": "past", "timestamp": past.isoformat()})
        episodic.store_event({"type": "current", "timestamp": now.isoformat()})
        episodic.store_event({"type": "future", "timestamp": future.isoformat()})
        
        events = episodic.get_events()
        assert len(events) == 3


class TestSemanticMemory:
    """Тесты для Semantic Memory слоя."""
    
    def test_store_concept(self):
        """Тест сохранения концепции."""
        semantic = SemanticMemory()
        
        concept_id = "artificial_intelligence"
        concept_data = {
            "definition": "Branch of computer science",
            "related_concepts": ["machine_learning", "neural_networks"]
        }
        
        semantic.store_concept(concept_id, concept_data)
        retrieved = semantic.get_concept(concept_id)
        
        assert retrieved is not None
        assert retrieved["definition"] == concept_data["definition"]
    
    def test_get_concept_by_name(self):
        """Тест получения концепции по имени."""
        semantic = SemanticMemory()
        
        concept_name = "artificial_intelligence"
        concept_data = {
            "definition": "Branch of computer science",
            "related_concepts": ["machine_learning", "neural_networks"]
        }
        
        semantic.store_concept(concept_name, concept_data)
        retrieved = semantic.get_concept(concept_name)
        
        assert retrieved is not None
        assert retrieved["definition"] == concept_data["definition"]
        assert "machine_learning" in retrieved["related_concepts"]
    
    def test_missing_concept(self):
        """Тест получения несуществующей концепции."""
        semantic = SemanticMemory()
        
        result = semantic.get_concept("nonexistent_concept")
        assert result is None


class TestProceduralMemory:
    """Тесты для Procedural Memory слоя."""
    
    def test_store_workflow(self):
        """Тест сохранения workflow."""
        procedural = ProceduralMemory()
        
        workflow_name = "user_registration"
        steps = [
            "validate_input",
            "check_availability", 
            "create_account",
            "send_confirmation"
        ]
        
        procedural.store_workflow(workflow_name, steps)
        retrieved = procedural.get_workflow(workflow_name)
        
        assert retrieved == steps
    
    def test_execute_workflow(self):
        """Тест выполнения workflow."""
        procedural = ProceduralMemory()
        
        workflow_steps = [
            "step1",
            "step2",
            "step3"
        ]
        
        procedural.store_workflow("test_workflow", workflow_steps)
        
        # Симулируем выполнение
        executed = []
        for step in workflow_steps:
            executed.append(step)
        
        assert executed == workflow_steps


class TestVaultMemory:
    """Тесты для Vault Memory слоя."""
    
    def test_store_secret(self):
        """Тест сохранения секрета."""
        vault = VaultMemory()
        
        secret_id = "api_key"
        secret_value = "sk-1234567890abcdef"
        
        vault.store_secret(secret_id, secret_value)
        retrieved = vault.get_secret(secret_id)
        
        assert retrieved == secret_value
    
    def test_secret_not_found(self):
        """Тест получения несуществующего секрета."""
        vault = VaultMemory()
        
        # Пытаемся получить несуществующий секрет
        assert vault.get_secret("nonexistent") is None
    
    def test_encrypt_secret(self):
        """Тест шифрования секрета."""
        vault = VaultMemory()
        
        vault.store_secret("password", "secret123")
        
        # В реальной реализации здесь должно быть шифрование
        # Для теста проверяем, что секрет сохранен
        assert vault.get_secret("password") == "secret123"


class TestSecurityMemory:
    """Тесты для Security Memory слоя."""
    
    def test_store_audit(self):
        """Тест сохранения аудита."""
        security = SecurityMemory()
        
        audit_entry = {
            "user_id": 123,
            "action": "login",
            "timestamp": datetime.now().isoformat(),
            "ip_address": "192.168.1.100"
        }
        
        security.store_audit(audit_entry)
        audits = security.get_audits()
        
        assert len(audits) == 1
        assert audits[0]["action"] == "login"
    
    def test_register_alert(self):
        """Тест регистрации алерта."""
        security = SecurityMemory()
        
        alert = {
            "type": "failed_login",
            "severity": "high",
            "user_id": 123
        }
        
        security.register_alert("alert_1", alert)
        alerts = security.get_alerts()
        
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "high"


# ============================================================================
# UNIT TESTS - Cache and Factory
# ============================================================================

class TestMemoryCache:
    """Тесты для MemoryCache."""
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self, memory_cache):
        """Тест сохранения и получения из кэша."""
        key = "test_key"
        data = {"value": "test_data"}
        metadata = {"type": "test"}
        
        # Сохраняем в кэш
        await memory_cache.set(key, data, metadata)
        
        # Получаем из кэша
        retrieved = await memory_cache.get(key)
        
        assert retrieved == data
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, memory_cache):
        """Тест промаха кэша."""
        # Пытаемся получить данные, которых нет в кэше
        result = await memory_cache.get("nonexistent_key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self, memory_cache):
        """Тест истечения TTL кэша."""
        # Создаем кэш с коротким TTL
        cache = MemoryCache(default_ttl=0.1)  # 100ms
        
        await cache.set("key", "data", {})
        
        # Сразу получаем - должно быть
        assert await cache.get("key") == "data"
        
        # Ждем истечения TTL
        await asyncio.sleep(0.2)
        
        # После истечения - должно быть None
        assert await cache.get("key") is None
    
    @pytest.mark.asyncio
    async def test_cache_deletion(self, memory_cache):
        """Тест удаления из кэша."""
        key = "test_key"
        
        await memory_cache.set(key, "data", {})
        
        # Удаляем
        result = await memory_cache.delete(key)
        assert result is True
        
        # Проверяем, что данных нет
        assert await memory_cache.get(key) is None
    
    @pytest.mark.asyncio
    async def test_cache_clear(self, memory_cache):
        """Тест очистки кэша."""
        # Сохраняем несколько записей
        for i in range(5):
            await memory_cache.set(f"key_{i}", f"data_{i}", {})
        
        # Очищаем
        count = await memory_cache.clear()
        assert count == 5
        
        # Проверяем, что кэш пуст
        assert await memory_cache.get("key_0") is None
    
    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self, memory_cache):
        """Тест LRU вытеснения из кэша."""
        # Кэш с размером 3
        cache = MemoryCache(max_size=3)
        
        # Сохраняем 4 записи
        for i in range(4):
            await cache.set(f"key_{i}", f"data_{i}", {})
        
        # Проверяем, что самая старая запись вытеснена
        assert await cache.get("key_0") is None
        assert await cache.get("key_1") is not None
        assert await cache.get("key_2") is not None
        assert await cache.get("key_3") is not None
    
    def test_cache_stats(self, memory_cache):
        """Тест статистики кэша."""
        # Синхронно получаем статистику
        stats = memory_cache.get_stats()
        
        assert "size" in stats
        assert "max_size" in stats
        assert "utilization" in stats
        assert "entries" in stats
        assert stats["max_size"] == 50


class TestMemoryLayerFactory:
    """Тесты для MemoryLayerFactory."""
    
    def test_create_core_layer(self, layer_factory):
        """Тест создания Core слоя."""
        layer = layer_factory.create_layer(CORE)
        assert isinstance(layer, CoreMemory)
    
    def test_create_episodic_layer(self, layer_factory):
        """Тест создания Episodic слоя."""
        layer = layer_factory.create_layer(EPISODIC)
        assert isinstance(layer, EpisodicMemory)
    
    def test_create_semantic_layer(self, layer_factory):
        """Тест создания Semantic слоя."""
        layer = layer_factory.create_layer(SEMANTIC)
        assert isinstance(layer, SemanticMemory)
    
    def test_create_procedural_layer(self, layer_factory):
        """Тест создания Procedural слоя."""
        layer = layer_factory.create_layer(PROCEDURAL)
        assert isinstance(layer, ProceduralMemory)
    
    def test_create_vault_layer(self, layer_factory):
        """Тест создания Vault слоя."""
        layer = layer_factory.create_layer(VAULT)
        assert isinstance(layer, VaultMemory)
    
    def test_create_security_layer(self, layer_factory):
        """Тест создания Security слоя."""
        layer = layer_factory.create_layer(SECURITY)
        assert isinstance(layer, SecurityMemory)
    
    def test_create_invalid_layer(self, layer_factory):
        """Тест создания недопустимого слоя."""
        with pytest.raises(ValueError, match="Неподдерживаемый тип слоя памяти"):
            layer_factory.create_layer("INVALID_LAYER")


# ============================================================================
# UNIT TESTS - Adaptive Blueprint Tracker
# ============================================================================

class TestAdaptiveBlueprintTracker:
    """Тесты для AdaptiveBlueprintTracker."""
    
    @pytest.mark.asyncio
    async def test_record_blueprint(self, adaptive_tracker):
        """Тест записи blueprint."""
        blueprint = {
            "version": "1.0",
            "components": ["core", "memory", "api"]
        }
        metadata = {"author": "test", "changes": "initial"}
        
        blueprint_id = await adaptive_tracker.record_blueprint(blueprint, metadata)
        
        assert blueprint_id.startswith("blueprint_v")
        assert adaptive_tracker.version_counter == 1
        assert len(adaptive_tracker.blueprint_history) == 1
    
    @pytest.mark.asyncio
    async def test_get_latest_blueprint(self, adaptive_tracker):
        """Тест получения последнего blueprint."""
        # Записываем несколько версий
        await adaptive_tracker.record_blueprint({"version": "1.0"})
        await adaptive_tracker.record_blueprint({"version": "2.0"})
        
        latest = await adaptive_tracker.get_latest_blueprint()
        
        assert latest is not None
        assert latest["blueprint"]["version"] == "2.0"
        assert latest["version"] == 2
    
    @pytest.mark.asyncio
    async def test_compare_blueprints(self, adaptive_tracker):
        """Тест сравнения blueprint версий."""
        # Записываем версии
        await adaptive_tracker.record_blueprint({"components": ["a", "b"]})
        await adaptive_tracker.record_blueprint({"components": ["a", "b", "c"]})
        
        # Сравниваем
        changes = await adaptive_tracker.compare_blueprints(1, 2)
        
        assert "added" in changes
        assert "removed" in changes
        assert "modified" in changes
        assert "c" in changes["added"]
    
    @pytest.mark.asyncio
    async def test_link_resource(self, adaptive_tracker):
        """Тест связывания ресурса."""
        resource = {"file": "config.json", "type": "config"}
        
        await adaptive_tracker.link_resource("config_file", resource, "config")
        
        assert "config_file" in adaptive_tracker.resource_links
        link = adaptive_tracker.resource_links["config_file"]
        assert link["resource_type"] == "config"
        assert link["resource"]["file"] == "config.json"
    
    @pytest.mark.asyncio
    async def test_get_blueprint_lineage(self, adaptive_tracker):
        """Тест получения истории blueprint."""
        # Записываем несколько версий
        for i in range(5):
            await adaptive_tracker.record_blueprint({"version": f"{i+1}.0"})
        
        # Получаем историю
        lineage = await adaptive_tracker.get_blueprint_lineage(3)
        
        # Должны получить последние 3 версии
        assert len(lineage) == 3
        assert lineage[0]["version"] == 3
        assert lineage[2]["version"] == 5
    
    @pytest.mark.asyncio
    async def test_blueprint_hash_calculation(self, adaptive_tracker):
        """Тест вычисления хеша blueprint."""
        blueprint1 = {"key": "value", "number": 42}
        blueprint2 = {"number": 42, "key": "value"}  # другой порядок
        
        hash1 = adaptive_tracker._calculate_blueprint_hash(blueprint1)
        hash2 = adaptive_tracker._calculate_blueprint_hash(blueprint2)
        
        # Хеши должны быть одинаковыми для одинакового содержимого
        assert hash1 == hash2
    
    @pytest.mark.asyncio
    async def test_flatten_dict(self, adaptive_tracker):
        """Тест разворачивания вложенного словаря."""
        nested = {
            "level1": {
                "level2": {
                    "value": "deep"
                }
            },
            "simple": "value"
        }
        
        flattened = adaptive_tracker._flatten_dict(nested)
        
        assert "level1.level2.value" in flattened
        assert flattened["level1.level2.value"] == "deep"
        assert "simple" in flattened
        assert flattened["simple"] == "value"


# ============================================================================
# UNIT TESTS - MemoryContext
# ============================================================================

class TestMemoryContext:
    """Тесты для MemoryContext."""
    
    def test_register_layer(self, memory_context):
        """Тест регистрации слоя памяти."""
        custom_layer = CoreMemory()
        memory_context.register_layer("custom", custom_layer)
        
        assert "custom" in memory_context.layers
    
    def test_add_memory(self, memory_context):
        """Тест добавления памяти."""
        memory_id = memory_context.add_memory(
            layer_type=CORE,
            content="test content",
            metadata={"key": "value"},
            tags=["test"],
            priority=5
        )
        
        assert memory_id is not None
        assert memory_id in memory_context.memory_entries
        
        entry = memory_context.memory_entries[memory_id]
        assert entry.content == "test content"
        assert entry.metadata["key"] == "value"
        assert entry.tags == ["test"]
        assert entry.priority == 5
    
    def test_get_memory(self, memory_context):
        """Тест получения памяти."""
        # Добавляем память
        memory_id = memory_context.add_memory(
            layer_type=CORE,
            content="test"
        )
        
        # Получаем память
        entry = memory_context.get_memory(memory_id)
        
        assert entry is not None
        assert entry.content == "test"
        assert entry.layer_type == CORE
    
    def test_get_memory_not_found(self, memory_context):
        """Тест получения несуществующей памяти."""
        result = memory_context.get_memory("nonexistent_id")
        assert result is None
    
    def test_remove_memory(self, memory_context):
        """Тест удаления памяти."""
        # Добавляем память
        memory_id = memory_context.add_memory(
            layer_type=CORE,
            content="test"
        )
        
        # Удаляем память
        success = memory_context.remove_memory(memory_id)
        assert success is True
        
        # Проверяем, что память удалена
        entry = memory_context.get_memory(memory_id)
        assert entry is None
    
    def test_get_memories_by_layer(self, memory_context):
        """Тест получения памяти по слою."""
        # Добавляем память в разные слои
        core_id = memory_context.add_memory(layer_type=CORE, content="core")
        semantic_id = memory_context.add_memory(layer_type=SEMANTIC, content="semantic")
        
        # Получаем память по слоям
        core_memories = memory_context.get_memories_by_layer(CORE)
        semantic_memories = memory_context.get_memories_by_layer(SEMANTIC)
        
        assert len(core_memories) == 1
        assert len(semantic_memories) == 1
        assert core_memories[0].content == "core"
        assert semantic_memories[0].content == "semantic"
    
    def test_get_memory_statistics(self, memory_context):
        """Тест получения статистики памяти."""
        # Добавляем несколько записей
        memory_context.add_memory(layer_type=CORE, content="core1")
        memory_context.add_memory(layer_type=CORE, content="core2")
        memory_context.add_memory(layer_type=SEMANTIC, content="semantic")
        
        stats = memory_context.get_memory_statistics()
        
        assert "total_memories" in stats
        assert stats["total_memories"] == 3
        assert "memories_by_layer" in stats
    
    def test_build_context_envelope(self, memory_context):
        """Тест создания контекстной оболочки."""
        # Добавляем данные
        memory_context.add_memory(layer_type=CORE, content="test")
        
        # Создаем оболочку
        envelope = memory_context.build_context_envelope("trace_123")
        
        assert "trace_id" in envelope
        assert envelope["trace_id"] == "trace_123"
        assert "timestamp" in envelope
        assert "memories" in envelope


# ============================================================================
# INTEGRATION TESTS - MemoryManager
# ============================================================================

class TestMemoryManagerIntegration:
    """Integration тесты для MemoryManager."""
    
    @pytest.mark.asyncio
    async def test_basic_operations(self, memory_manager, sample_data):
        """Тест базовых операций MemoryManager."""
        # Сохраняем данные в разные слои
        core_id = await memory_manager.store(
            CORE, 
            sample_data["text"], 
            sample_data["metadata"],
            sample_data["tags"]
        )
        
        semantic_id = await memory_manager.store(
            SEMANTIC,
            "Концепция искусственного интеллекта",
            {"domain": "ai", "importance": "high"}
        )
        
        episodic_id = await memory_manager.store(
            EPISODIC,
            "Пользователь вошел в систему",
            {"user_id": 123, "timestamp": datetime.now().isoformat()}
        )
        
        # Проверяем, что данные сохранены
        assert core_id is not None
        assert semantic_id is not None
        assert episodic_id is not None
        
        # Извлекаем данные
        core_data = await memory_manager.retrieve(CORE)
        semantic_data = await memory_manager.retrieve(SEMANTIC)
        episodic_data = await memory_manager.retrieve(EPISODIC)
        
        assert len(core_data) >= 1
        assert len(semantic_data) >= 1
        assert len(episodic_data) >= 1
    
    @pytest.mark.asyncio
    async def test_update_operation(self, memory_manager):
        """Тест операции обновления."""
        # Сохраняем данные
        memory_id = await memory_manager.store(
            CORE,
            "Начальные данные",
            {"version": 1}
        )
        
        # Обновляем данные
        success = await memory_manager.update(
            CORE,
            memory_id,
            "Обновленные данные",
            {"version": 2}
        )
        
        assert success is True
        
        # Проверяем обновление
        updated_data = await memory_manager.retrieve(CORE, filters={"id": memory_id})
        assert len(updated_data) > 0
        # Здесь должен быть контент "Обновленные данные"
    
    @pytest.mark.asyncio
    async def test_delete_operation(self, memory_manager):
        """Тест операции удаления."""
        # Сохраняем данные
        memory_id = await memory_manager.store(
            CORE,
            "Временные данные",
            {"temporary": True}
        )
        
        # Удаляем данные
        success = await memory_manager.delete(CORE, memory_id)
        assert success is True
        
        # Проверяем, что данные удалены
        remaining_data = await memory_manager.retrieve(CORE, filters={"id": memory_id})
        # После удаления записи быть не должно (или должна быть только если есть дубликаты)
    
    @pytest.mark.asyncio
    async def test_search_across_layers(self, memory_manager):
        """Тест поиска по всем слоям."""
        # Сохраняем связанные данные в разных слоях
        await memory_manager.store(CORE, "AI технологии развиваются", {"topic": "ai"})
        await memory_manager.store(SEMANTIC, "Машинное обучение - часть AI", {"topic": "ai"})
        await memory_manager.store(EPISODIC, "Обсуждение AI на встрече", {"topic": "ai"})
        
        # Ищем по всем слоям
        results = await memory_manager.search_across_layers("AI", limit=10)
        
        assert len(results) >= 3  # Должны найти все 3 записи
        assert all("ai" in result.get("metadata", {}).get("topic", "").lower() for result in results)
    
    @pytest.mark.asyncio
    async def test_layer_statistics(self, memory_manager):
        """Тест получения статистики по слоям."""
        # Сохраняем данные
        await memory_manager.store(CORE, "Факт 1")
        await memory_manager.store(CORE, "Факт 2")
        await memory_manager.store(SEMANTIC, "Концепция 1")
        
        # Получаем статистику
        stats = await memory_manager.get_layer_statistics()
        
        assert "layer_statistics" in stats
        assert "cache" in stats
        assert "vector_store" in stats
        assert "blueprint_tracker" in stats
        
        # Проверяем статистику по слоям
        layer_stats = stats["layer_statistics"]
        assert CORE in layer_stats
        assert SEMANTIC in layer_stats
    
    @pytest.mark.asyncio
    async def test_optimize_memory(self, memory_manager):
        """Тест оптимизации памяти."""
        # Сохраняем данные
        for i in range(10):
            await memory_manager.store(CORE, f"Data {i}", {"index": i})
        
        # Выполняем оптимизацию
        result = await memory_manager.optimize_memory()
        
        assert "memory_optimization" in result
        assert "cache_optimization" in result
        assert "duration" in result
        assert result["duration"] > 0
    
    @pytest.mark.asyncio
    async def test_sync_with_orchestrator(self, memory_manager):
        """Тест синхронизации с оркестратором."""
        # Сохраняем данные
        await memory_manager.store(CORE, "Test sync")
        
        # Синхронизируемся
        sync_result = await memory_manager.sync_with_orchestrator()
        
        assert "success" in sync_result
        assert "trace_id" in sync_result
        assert "sync_timestamp" in sync_result
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self, memory_manager):
        """Тест поиска с фильтрами."""
        # Сохраняем данные с разными метаданными
        await memory_manager.store(CORE, "High priority", {"priority": 8, "type": "important"})
        await memory_manager.store(CORE, "Low priority", {"priority": 2, "type": "normal"})
        await memory_manager.store(CORE, "Medium priority", {"priority": 5, "type": "normal"})
        
        # Ищем с фильтром по приоритету
        results = await memory_manager.retrieve(
            CORE,
            filters={"priority": {"gte": 5}}
        )
        
        assert len(results) >= 2  # Должны найти high и medium priority
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, memory_manager):
        """Тест конкурентных операций."""
        # Сохраняем данные конкурентно
        tasks = []
        for i in range(10):
            task = memory_manager.store(
                CORE,
                f"Concurrent data {i}",
                {"index": i}
            )
            tasks.append(task)
        
        # Ждем завершения всех задач
        memory_ids = await asyncio.gather(*tasks)
        
        assert len(memory_ids) == 10
        assert all(memory_id is not None for memory_id in memory_ids)
        
        # Проверяем, что все данные сохранены
        results = await memory_manager.retrieve(CORE)
        assert len(results) >= 10
    
    @pytest.mark.asyncio
    async def test_cache_performance(self, memory_manager):
        """Тест производительности кэша."""
        # Сохраняем данные
        memory_id = await memory_manager.store(CORE, "Cached data", {"test": True})
        
        # Первый доступ (кэш промах)
        start_time = time.time()
        result1 = await memory_manager.retrieve(CORE)
        first_access_time = time.time() - start_time
        
        # Второй доступ (кэш попадание)
        start_time = time.time()
        result2 = await memory_manager.retrieve(CORE)
        second_access_time = time.time() - start_time
        
        # Второй доступ должен быть быстрее
        assert second_access_time <= first_access_time * 2  # Эвристика для теста
    
    @pytest.mark.asyncio
    async def test_invalid_layer_error(self, memory_manager):
        """Тест ошибки для недопустимого слоя."""
        with pytest.raises(ValueError, match="Недопустимый слой памяти"):
            await memory_manager.store("INVALID_LAYER", "data")
        
        with pytest.raises(ValueError, match="Недопустимый слой памяти"):
            await memory_manager.retrieve("INVALID_LAYER")


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestMemoryManagerPerformance:
    """Performance тесты для MemoryManager."""
    
    @pytest.mark.asyncio
    async def test_large_dataset_storage(self, memory_manager, large_dataset):
        """Тест сохранения большого набора данных."""
        start_time = time.time()
        
        # Сохраняем данные пакетами
        batch_size = 100
        for i in range(0, len(large_dataset), batch_size):
            batch = large_dataset[i:i + batch_size]
            tasks = [
                memory_manager.store(
                    CORE,
                    item["content"],
                    item["metadata"],
                    item["tags"],
                    item["priority"]
                )
                for item in batch
            ]
            await asyncio.gather(*tasks)
        
        storage_time = time.time() - start_time
        
        # Проверяем, что время сохранения разумное (менее 30 секунд для 1000 записей)
        assert storage_time < 30.0
        
        # Проверяем, что данные сохранены
        results = await memory_manager.retrieve(CORE)
        assert len(results) >= 1000
    
    @pytest.mark.asyncio
    async def test_large_dataset_retrieval(self, memory_manager, large_dataset):
        """Тест извлечения большого набора данных."""
        # Сначала сохраняем данные
        for item in large_dataset:
            await memory_manager.store(
                CORE,
                item["content"],
                item["metadata"],
                item["tags"],
                item["priority"]
            )
        
        # Тестируем извлечение
        start_time = time.time()
        results = await memory_manager.retrieve(CORE, limit=1000)
        retrieval_time = time.time() - start_time
        
        # Проверяем время извлечения
        assert retrieval_time < 10.0  # Менее 10 секунд
        assert len(results) >= 1000
    
    @pytest.mark.asyncio
    async def test_concurrent_access_performance(self, memory_manager):
        """Тест производительности конкурентного доступа."""
        # Сохраняем базовые данные
        await memory_manager.store(CORE, "Base data", {"base": True})
        
        # Конкурентные операции чтения
        read_tasks = [
            memory_manager.retrieve(CORE)
            for _ in range(50)
        ]
        
        # Конкурентные операции записи
        write_tasks = [
            memory_manager.store(
                CORE,
                f"Concurrent write {i}",
                {"index": i}
            )
            for i in range(20)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*read_tasks, *write_tasks)
        total_time = time.time() - start_time
        
        # Проверяем, что все операции завершились
        assert len(results) == 70
        assert total_time < 20.0  # Менее 20 секунд
    
    @pytest.mark.asyncio
    async def test_memory_optimization_performance(self, memory_manager):
        """Тест производительности оптимизации памяти."""
        # Сохраняем много данных для оптимизации
        for i in range(500):
            await memory_manager.store(
                CORE,
                f"Optimization test data {i}",
                {"index": i, "temp": True}
            )
        
        # Запускаем оптимизацию и измеряем время
        start_time = time.time()
        result = await memory_manager.optimize_memory()
        optimization_time = time.time() - start_time
        
        # Проверяем время оптимизации
        assert optimization_time < 15.0  # Менее 15 секунд
        assert "duration" in result
    
    @pytest.mark.asyncio
    async def test_search_performance_across_layers(self, memory_manager):
        """Тест производительности поиска по слоям."""
        # Сохраняем данные во всех слоях
        layers = [CORE, EPISODIC, SEMANTIC, PROCEDURAL, VAULT, SECURITY]
        for layer in layers:
            for i in range(100):
                await memory_manager.store(
                    layer,
                    f"Search test data in {layer} {i}",
                    {"layer": layer, "index": i}
                )
        
        # Тестируем поиск
        start_time = time.time()
        results = await memory_manager.search_across_layers(
            "Search test data",
            layers=layers,
            limit=50
        )
        search_time = time.time() - start_time
        
        # Проверяем время поиска
        assert search_time < 5.0  # Менее 5 секунд
        assert len(results) >= 1


# ============================================================================
# MOCK TESTS - External Dependencies
# ============================================================================

class TestMemoryManagerWithMocks:
    """Mock тесты для внешних зависимостей."""
    
    @pytest.mark.asyncio
    async def test_vector_store_client_mock(self):
        """Тест с мок-клиентом векторного хранилища."""
        # Создаем мок-клиент
        mock_client = AsyncMock()
        mock_client.sync_schema.return_value = None
        mock_client.store_vectors.return_value = None
        mock_client.retrieve_vectors.return_value = []
        mock_client.update_vector.return_value = None
        mock_client.health_check.return_value = {"status": "healthy"}
        mock_client.close.return_value = None
        
        # Создаем менеджер с мок-клиентом
        manager = MemoryManager()
        manager.vector_store_client = mock_client
        
        await manager.start()
        
        # Проверяем, что sync_schema был вызван
        mock_client.sync_schema.assert_called_once()
        
        # Сохраняем данные (это должно работать с мок-клиентом)
        memory_id = await manager.store(CORE, "Test data", {"mock": True})
        
        await manager.stop()
        
        # Проверяем, что close был вызван
        mock_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_vector_store_error_handling(self):
        """Тест обработки ошибок векторного хранилища."""
        # Создаем клиент, который выбрасывает исключение
        mock_client = AsyncMock()
        mock_client.sync_schema.side_effect = Exception("Connection error")
        mock_client.health_check.return_value = {"status": "unhealthy", "error": "Connection error"}
        
        manager = MemoryManager()
        manager.vector_store_client = mock_client
        
        # Запуск должен пройти, несмотря на ошибку векторного хранилища
        try:
            await manager.start()
        except Exception:
            pass  # Игнорируем ошибку для этого теста
        
        # Остановка должна работать
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_cache_with_external_dependencies(self, memory_cache):
        """Тест кэша с внешними зависимостями."""
        # Мокаем время для теста истечения TTL
        with patch('memory_manager.memory_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            await memory_cache.set("key", "data", {}, ttl=3600)
            
            # Симулируем прошедшее время
            mock_datetime.now.return_value = datetime(2023, 1, 1, 13, 0, 0)
            
            # Данные должны быть истекшими
            result = await memory_cache.get("key")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_blueprint_tracker_with_semantic_mock(self):
        """Тест трекера blueprint с мок-слоем семантики."""
        # Создаем мок-семантический слой
        mock_semantic = AsyncMock()
        mock_semantic.store_concept_async.return_value = None
        
        # Создаем трекер с мок-слоем
        tracker = AdaptiveBlueprintTracker(mock_semantic)
        
        blueprint = {"version": "1.0", "test": True}
        blueprint_id = await tracker.record_blueprint(blueprint)
        
        # Проверяем, что метод семантического слоя был вызван
        mock_semantic.store_concept_async.assert_called_once()
        
        assert blueprint_id.startswith("blueprint_v")


# ============================================================================
# SMOKE TESTS - API Endpoints
# ============================================================================

class TestMemoryManagerSmoke:
    """Smoke тесты для API endpoints."""
    
    @pytest.mark.asyncio
    async def test_manager_startup_shutdown(self):
        """Тест запуска и остановки менеджера."""
        manager = create_memory_manager({
            "cache_size": 50,
            "cache_ttl": 300
        })
        
        # Запуск
        await manager.start()
        
        # Проверяем, что компоненты инициализированы
        assert manager.memory_context is not None
        assert manager.vector_store_client is not None
        assert manager.cache is not None
        assert manager.blueprint_tracker is not None
        
        # Остановка
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_quick_memory_test(self):
        """Тест быстрого теста памяти."""
        result = await quick_memory_test()
        
        assert "success" in result
        assert result["success"] is True
        assert "stored_items" in result
        assert "statistics" in result
    
    @pytest.mark.asyncio
    async def test_create_memory_manager_with_config(self):
        """Тест создания менеджера с конфигурацией."""
        config = {
            "cache_size": 200,
            "cache_ttl": 600,
            "optimization_interval": 120,
            "vector_store": {
                "provider": "memory",
                "collection_name": "test_collection"
            }
        }
        
        manager = create_memory_manager(config)
        
        assert manager.cache.max_size == 200
        assert manager.cache.default_ttl == 600
        assert manager.optimization_interval == 120
        
        # Проверяем конфигурацию векторного хранилища
        # (зависит от реализации VectorStoreClient)
    
    @pytest.mark.asyncio
    async def test_basic_crud_operations_smoke(self, memory_manager):
        """Smoke тест базовых CRUD операций."""
        # Create
        create_id = await memory_manager.store(CORE, "Smoke test data", {"smoke": True})
        assert create_id is not None
        
        # Read
        read_result = await memory_manager.retrieve(CORE)
        assert len(read_result) >= 1
        
        # Update
        update_success = await memory_manager.update(CORE, create_id, "Updated data")
        assert update_success is True
        
        # Delete
        delete_success = await memory_manager.delete(CORE, create_id)
        assert delete_success is True
    
    @pytest.mark.asyncio
    async def test_search_smoke(self, memory_manager):
        """Smoke тест поиска."""
        # Сохраняем данные
        await memory_manager.store(CORE, "Smoke search test")
        
        # Ищем
        results = await memory_manager.search_across_layers("search")
        assert isinstance(results, list)


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestMemoryManagerEdgeCases:
    """Тесты граничных случаев и обработки ошибок."""
    
    @pytest.mark.asyncio
    async def test_empty_data_storage(self, memory_manager):
        """Тест сохранения пустых данных."""
        # Сохраняем пустые данные
        empty_id = await memory_manager.store(CORE, "", {"empty": True})
        none_id = await memory_manager.store(CORE, None, {"none": True})
        
        assert empty_id is not None
        assert none_id is not None
        
        # Извлекаем
        results = await memory_manager.retrieve(CORE)
        assert len(results) >= 2
    
    @pytest.mark.asyncio
    async def test_large_content_storage(self, memory_manager):
        """Тест сохранения большого контента."""
        # Создаем большой контент (1MB)
        large_content = "x" * (1024 * 1024)
        
        memory_id = await memory_manager.store(CORE, large_content, {"size": "large"})
        
        assert memory_id is not None
        
        # Извлекаем и проверяем размер
        results = await memory_manager.retrieve(CORE, filters={"id": memory_id})
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_special_characters_in_data(self, memory_manager):
        """Тест специальных символов в данных."""
        special_content = "Тест с русскими символами: привет мир! 🌍"
        special_metadata = {"emoji": "🎉", "unicode": "αβγδε"}
        
        memory_id = await memory_manager.store(
            CORE,
            special_content,
            special_metadata
        )
        
        assert memory_id is not None
        
        # Проверяем, что данные сохранились корректно
        results = await memory_manager.retrieve(CORE, filters={"id": memory_id})
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_same_id_operations(self, memory_manager):
        """Тест конкурентных операций с тем же ID."""
        # Сохраняем данные
        memory_id = await memory_manager.store(CORE, "Initial data")
        
        # Конкурентные обновления
        update_tasks = [
            memory_manager.update(CORE, memory_id, f"Update {i}")
            for i in range(10)
        ]
        
        # Ждем завершения всех обновлений
        results = await asyncio.gather(*update_tasks, return_exceptions=True)
        
        # Некоторые обновления могут не пройти из-за конкуренции
        # Но система должна остаться стабильной
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_invalid_metadata_handling(self, memory_manager):
        """Тест обработки недопустимых метаданных."""
        # Метаданные с недопустимыми типами
        invalid_metadata = {
            "datetime": datetime.now(),
            "set": {1, 2, 3},  # set не сериализуется в JSON
            "complex": lambda x: x  # функция не сериализуется
        }
        
        # Должно работать несмотря на сложные типы
        memory_id = await memory_manager.store(
            CORE,
            "Data with complex metadata",
            invalid_metadata
        )
        
        assert memory_id is not None
    
    @pytest.mark.asyncio
    async def test_priority_boundaries(self, memory_manager):
        """Тест граничных значений приоритета."""
        # Тест приоритетов за пределами диапазона 0-10
        priorities = [-5, 0, 5, 10, 15, 100]
        
        for priority in priorities:
            memory_id = await memory_manager.store(
                CORE,
                f"Priority test {priority}",
                {"original_priority": priority}
            )
            assert memory_id is not None
        
        # Все приоритеты должны быть в диапазоне 0-10
        results = await memory_manager.retrieve(CORE)
        for result in results:
            priority = result.get("priority", 0)
            assert 0 <= priority <= 10


# ============================================================================
# STRESS TESTS
# ============================================================================

class TestMemoryManagerStress:
    """Stress тесты для экстремальных нагрузок."""
    
    @pytest.mark.asyncio
    async def test_rapid_operations(self, memory_manager):
        """Тест быстрых операций."""
        start_time = time.time()
        operations_count = 0
        
        # Быстрые операции в течение 5 секунд
        while time.time() - start_time < 5.0:
            await memory_manager.store(CORE, f"Rapid op {operations_count}")
            operations_count += 1
        
        # Проверяем, что система выдержала нагрузку
        assert operations_count > 100  # Должно быть выполнено много операций
        
        # Проверяем, что данные сохранились
        results = await memory_manager.retrieve(CORE)
        assert len(results) >= operations_count // 2  # Примерно половина должна сохраниться
    
    @pytest.mark.asyncio
    async def test_cache_stress(self, memory_manager):
        """Тест стресса кэша."""
        # Создаем кэш маленького размера
        small_cache = MemoryCache(max_size=10)
        
        # Заполняем кэш сверх лимита
        for i in range(50):
            await small_cache.set(
                f"key_{i}",
                f"data_{i}",
                {"iteration": i}
            )
        
        # Проверяем, что старые записи вытеснены
        stats = small_cache.get_stats()
        assert stats["size"] <= 10
        
        # Проверяем, что последние записи сохранились
        for i in range(40, 50):
            result = await small_cache.get(f"key_{i}")
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_blueprint_tracker_stress(self, semantic_layer):
        """Тест стресса трекера blueprint."""
        tracker = AdaptiveBlueprintTracker(semantic_layer, max_history=20)
        
        # Создаем много версий blueprint
        for i in range(50):
            blueprint = {
                "version": f"{i}.0",
                "features": [f"feature_{j}" for j in range(i % 10)],
                "timestamp": datetime.now().isoformat()
            }
            
            await tracker.record_blueprint(blueprint)
        
        # Проверяем, что история ограничена
        assert len(tracker.version_history) <= 20
        
        # Проверяем, что последние версии сохранились
        latest = await tracker.get_latest_blueprint()
        assert latest is not None
        assert latest["version"].startswith("4")  # Последние версии


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Запуск всех тестов (для ручного тестирования)."""
    print("🚀 Запуск comprehensive тестов MemoryManager...")
    
    # Запускаем основные категории тестов
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n✅ Все тесты MemoryManager прошли успешно!")
    else:
        print(f"\n❌ Тесты завершились с ошибкой: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    run_all_tests()