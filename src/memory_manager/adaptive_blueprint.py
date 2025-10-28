"""Расширенный AdaptiveBlueprintTracker для отслеживания изменений архитектуры.

Предоставляет комплексное отслеживание изменений архитектуры с:
- Версионированием blueprint
- Сравнением версий
- Связыванием ресурсов
- Анализом изменений
- Сохранением истории
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict

import logging

logger = logging.getLogger(__name__)


@dataclass
class BlueprintVersion:
    """Версия архитектуры."""
    version: int
    blueprint: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: str
    hash: str
    parent_version: Optional[int] = None
    change_type: str = "initial"
    change_description: str = ""


@dataclass
class ResourceLink:
    """Связь ресурса с архитектурой."""
    identifier: str
    resource_type: str
    resource: Dict[str, Any]
    linked_at: str
    linked_to_version: int
    hash: str
    dependency_level: int = 1


@dataclass
class ChangeAnalysis:
    """Анализ изменений между версиями."""
    from_version: int
    to_version: int
    changes: Dict[str, Any]
    impact_score: float
    risk_assessment: str
    recommendations: List[str]


class AdaptiveBlueprintTracker:
    """Расширенный трекер изменений архитектуры.
    
    Отслеживает изменения в архитектуре системы, хранит историю версий,
    анализирует изменения и обеспечивает связывание ресурсов.
    """
    
    def __init__(self, semantic_layer: Any, max_history: int = 100):
        """Инициализирует трекер.
        
        Args:
            semantic_layer: Слой семантической памяти
            max_history: Максимальное количество версий в истории
        """
        self.semantic_layer = semantic_layer
        self.max_history = max_history
        
        # История версий
        self.version_history: List[BlueprintVersion] = []
        self.version_counter = 0
        
        # Связи ресурсов
        self.resource_links: Dict[str, ResourceLink] = {}
        
        # Анализ изменений
        self.change_analysis_cache: Dict[str, ChangeAnalysis] = {}
        
        # Конфигурация анализа
        self.change_thresholds = {
            "low_impact": 0.2,
            "medium_impact": 0.5,
            "high_impact": 0.8
        }
        
        # Зависимости
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        
        logger.info(f"AdaptiveBlueprintTracker инициализирован (max_history={max_history})")
    
    async def record_blueprint(self, blueprint: Dict[str, Any], 
                             metadata: Optional[Dict[str, Any]] = None,
                             change_type: str = "update",
                             change_description: str = "",
                             auto_analyze: bool = True) -> int:
        """Записывает новое состояние архитектуры.
        
        Args:
            blueprint: Данные архитектуры
            metadata: Дополнительные метаданные
            change_type: Тип изменения
            change_description: Описание изменения
            auto_analyze: Автоматически анализировать изменения
            
        Returns:
            Номер версии blueprint
        """
        try:
            self.version_counter += 1
            
            # Вычисляем хеш blueprint
            blueprint_hash = self._calculate_blueprint_hash(blueprint)
            
            # Определяем родительскую версию
            parent_version = None
            if self.version_history:
                parent_version = self.version_history[-1].version
            
            # Создаем версию
            version = BlueprintVersion(
                version=self.version_counter,
                blueprint=blueprint,
                metadata=metadata or {},
                timestamp=datetime.now().isoformat(),
                hash=blueprint_hash,
                parent_version=parent_version,
                change_type=change_type,
                change_description=change_description
            )
            
            # Добавляем в историю
            self.version_history.append(version)
            
            # Ограничиваем размер истории
            if len(self.version_history) > self.max_history:
                old_versions = self.version_history[:-self.max_history]
                self.version_history = self.version_history[-self.max_history:]
                logger.info(f"Удалено {len(old_versions)} старых версий из истории")
            
            # Сохраняем в семантической памяти
            blueprint_key = f"blueprint_v{self.version_counter}"
            await self._store_in_semantic_memory(blueprint_key, version.__dict__)
            
            # Автоматический анализ изменений
            if auto_analyze and parent_version:
                await self._analyze_changes(parent_version, self.version_counter)
            
            logger.info(f"Записан blueprint версии {self.version_counter}, тип: {change_type}")
            return self.version_counter
            
        except Exception as e:
            logger.error(f"Ошибка записи blueprint: {e}")
            raise RuntimeError(f"Не удалось записать blueprint: {e}")
    
    async def get_latest_blueprint(self) -> Optional[BlueprintVersion]:
        """Возвращает последнее состояние архитектуры."""
        if self.version_history:
            return self.version_history[-1]
        return None
    
    async def get_blueprint_version(self, version: int) -> Optional[BlueprintVersion]:
        """Возвращает конкретную версию архитектуры.
        
        Args:
            version: Номер версии
            
        Returns:
            Версия blueprint или None
        """
        for v in self.version_history:
            if v.version == version:
                return v
        return None
    
    async def compare_blueprints(self, version1: int, version2: int,
                               detailed: bool = True) -> Dict[str, Any]:
        """Сравнивает две версии архитектуры.
        
        Args:
            version1: Первая версия
            version2: Вторая версия
            detailed: Подробное сравнение
            
        Returns:
            Результаты сравнения
        """
        try:
            bp1 = await self.get_blueprint_version(version1)
            bp2 = await self.get_blueprint_version(version2)
            
            if not bp1 or not bp2:
                raise ValueError(f"Версии {version1} или {version2} не найдены")
            
            # Базовое сравнение
            basic_changes = {
                "version1": version1,
                "version2": version2,
                "time_diff": self._calculate_time_diff(bp1.timestamp, bp2.timestamp),
                "change_type": bp2.change_type,
                "change_description": bp2.change_description
            }
            
            if detailed:
                # Подробный анализ изменений
                changes = await self._detailed_comparison(bp1.blueprint, bp2.blueprint)
                basic_changes["detailed_changes"] = changes
            
            logger.info(f"Сравнены версии {version1} и {version2}")
            return basic_changes
            
        except Exception as e:
            logger.error(f"Ошибка сравнения версий {version1} и {version2}: {e}")
            raise RuntimeError(f"Не удалось сравнить версии: {e}")
    
    async def link_resource(self, identifier: str, resource: Dict[str, Any],
                          resource_type: str = "unknown", 
                          dependency_level: int = 1,
                          linked_version: Optional[int] = None) -> None:
        """Связывает ресурс с архитектурой.
        
        Args:
            identifier: Идентификатор ресурса
            resource: Данные ресурса
            resource_type: Тип ресурса
            dependency_level: Уровень зависимости (1-5)
            linked_version: Версия blueprint для связывания
        """
        try:
            # Определяем версию для связывания
            if linked_version is None:
                if self.version_history:
                    linked_version = self.version_history[-1].version
                else:
                    linked_version = 1
            
            # Создаем связь
            resource_hash = self._calculate_blueprint_hash(resource)
            
            link = ResourceLink(
                identifier=identifier,
                resource_type=resource_type,
                resource=resource,
                linked_at=datetime.now().isoformat(),
                linked_to_version=linked_version,
                hash=resource_hash,
                dependency_level=max(1, min(5, dependency_level))
            )
            
            self.resource_links[identifier] = link
            
            # Добавляем в семантическую память
            link_key = f"resource::{identifier}"
            await self._store_in_semantic_memory(link_key, link.__dict__)
            
            # Обновляем зависимости
            self.dependencies[linked_version].add(identifier)
            
            logger.info(f"Связан ресурс {identifier} (тип: {resource_type}) "
                       f"с версией {linked_version}, уровень зависимости: {dependency_level}")
            
        except Exception as e:
            logger.error(f"Ошибка связывания ресурса {identifier}: {e}")
            raise RuntimeError(f"Не удалось связать ресурс: {e}")
    
    async def get_resource_links(self, version: Optional[int] = None,
                               resource_type: Optional[str] = None) -> List[ResourceLink]:
        """Возвращает связи ресурсов.
        
        Args:
            version: Фильтр по версии
            resource_type: Фильтр по типу ресурса
            
        Returns:
            Список связей ресурсов
        """
        links = list(self.resource_links.values())
        
        # Фильтруем по версии
        if version is not None:
            links = [link for link in links if link.linked_to_version == version]
        
        # Фильтруем по типу
        if resource_type is not None:
            links = [link for link in links if link.resource_type == resource_type]
        
        return sorted(links, key=lambda x: x.linked_at, reverse=True)
    
    async def get_blueprint_lineage(self, max_versions: Optional[int] = None) -> List[BlueprintVersion]:
        """Возвращает историю изменений архитектуры.
        
        Args:
            max_versions: Максимальное количество версий
            
        Returns:
            Список версий архитектуры
        """
        history = self.version_history
        
        if max_versions:
            history = history[-max_versions:]
        
        return history
    
    async def analyze_impact(self, from_version: int, to_version: int) -> ChangeAnalysis:
        """Анализирует влияние изменений.
        
        Args:
            from_version: Версия до изменений
            to_version: Версия после изменений
            
        Returns:
            Анализ влияния изменений
        """
        try:
            # Проверяем кэш
            cache_key = f"{from_version}_{to_version}"
            if cache_key in self.change_analysis_cache:
                return self.change_analysis_cache[cache_key]
            
            bp1 = await self.get_blueprint_version(from_version)
            bp2 = await self.get_blueprint_version(to_version)
            
            if not bp1 or not bp2:
                raise ValueError(f"Версии {from_version} или {to_version} не найдены")
            
            # Вычисляем изменения
            changes = await self._detailed_comparison(bp1.blueprint, bp2.blueprint)
            
            # Оцениваем влияние
            impact_score = self._calculate_impact_score(changes)
            
            # Определяем уровень риска
            risk_assessment = self._assess_risk(impact_score, changes)
            
            # Формируем рекомендации
            recommendations = self._generate_recommendations(changes, impact_score)
            
            # Создаем анализ
            analysis = ChangeAnalysis(
                from_version=from_version,
                to_version=to_version,
                changes=changes,
                impact_score=impact_score,
                risk_assessment=risk_assessment,
                recommendations=recommendations
            )
            
            # Кэшируем
            self.change_analysis_cache[cache_key] = analysis
            
            logger.info(f"Проанализировано влияние изменений от версии {from_version} "
                       f"к версии {to_version}, оценка: {impact_score:.2f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Ошибка анализа влияния: {e}")
            raise RuntimeError(f"Не удалось проанализировать влияние: {e}")
    
    async def find_blueprint_by_hash(self, blueprint_hash: str) -> Optional[BlueprintVersion]:
        """Находит версию blueprint по хешу.
        
        Args:
            blueprint_hash: Хеш blueprint
            
        Returns:
            Версия blueprint или None
        """
        for version in self.version_history:
            if version.hash == blueprint_hash:
                return version
        return None
    
    async def validate_blueprint_integrity(self, version: int) -> Dict[str, Any]:
        """Проверяет целостность blueprint.
        
        Args:
            version: Версия для проверки
            
        Returns:
            Результат проверки целостности
        """
        try:
            bp = await self.get_blueprint_version(version)
            if not bp:
                return {"valid": False, "error": "Version not found"}
            
            # Проверяем хеш
            calculated_hash = self._calculate_blueprint_hash(bp.blueprint)
            hash_valid = calculated_hash == bp.hash
            
            # Проверяем связанные ресурсы
            linked_resources = await self.get_resource_links(version=version)
            resources_valid = all(
                link.hash == self._calculate_blueprint_hash(link.resource)
                for link in linked_resources
            )
            
            return {
                "valid": hash_valid and resources_valid,
                "version": version,
                "hash_valid": hash_valid,
                "resources_valid": resources_valid,
                "linked_resources_count": len(linked_resources),
                "timestamp": bp.timestamp
            }
            
        except Exception as e:
            logger.error(f"Ошибка проверки целостности версии {version}: {e}")
            return {"valid": False, "error": str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику трекера."""
        return {
            "total_versions": len(self.version_history),
            "current_version": self.version_counter,
            "total_resource_links": len(self.resource_links),
            "resource_types": list(set(link.resource_type for link in self.resource_links.values())),
            "dependency_levels": list(set(link.dependency_level for link in self.resource_links.values())),
            "cached_analyses": len(self.change_analysis_cache),
            "history_truncation": len(self.version_history) == self.max_history
        }
    
    # Вспомогательные методы
    
    def _calculate_blueprint_hash(self, blueprint: Dict[str, Any]) -> str:
        """Вычисляет хеш для blueprint."""
        blueprint_str = json.dumps(blueprint, sort_keys=True)
        return hashlib.sha256(blueprint_str.encode()).hexdigest()
    
    def _calculate_time_diff(self, timestamp1: str, timestamp2: str) -> float:
        """Вычисляет разницу во времени между timestamp."""
        try:
            t1 = datetime.fromisoformat(timestamp1)
            t2 = datetime.fromisoformat(timestamp2)
            return abs((t2 - t1).total_seconds())
        except:
            return 0.0
    
    async def _store_in_semantic_memory(self, key: str, data: Dict[str, Any]) -> None:
        """Сохраняет данные в семантической памяти."""
        try:
            if hasattr(self.semantic_layer, 'store_concept_async'):
                await self.semantic_layer.store_concept_async(key, data)
            elif hasattr(self.semantic_layer, 'store_concept'):
                self.semantic_layer.store_concept(key, data)
        except Exception as e:
            logger.warning(f"Не удалось сохранить в семантическую память {key}: {e}")
    
    async def _detailed_comparison(self, bp1: Dict[str, Any], bp2: Dict[str, Any]) -> Dict[str, Any]:
        """Подробное сравнение blueprint."""
        changes = {
            "added": [],
            "removed": [],
            "modified": [],
            "structure_changes": []
        }
        
        # Флаttен словари для сравнения
        flat1 = self._flatten_dict(bp1)
        flat2 = self._flatten_dict(bp2)
        
        keys1 = set(flat1.keys())
        keys2 = set(flat2.keys())
        
        # Добавленные ключи
        changes["added"] = list(keys2 - keys1)
        
        # Удаленные ключи
        changes["removed"] = list(keys1 - keys2)
        
        # Измененные ключи
        common_keys = keys1 & keys2
        for key in common_keys:
            if flat1[key] != flat2[key]:
                changes["modified"].append({
                    "key": key,
                    "old_value": flat1[key],
                    "new_value": flat2[key]
                })
        
        # Структурные изменения (добавление/удаление объектов)
        changes["structure_changes"] = self._detect_structure_changes(bp1, bp2)
        
        return changes
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Разворачивает вложенный словарь."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _detect_structure_changes(self, bp1: Dict[str, Any], bp2: Dict[str, Any]) -> List[str]:
        """Определяет структурные изменения."""
        changes = []
        
        # Простое определение структурных изменений
        # (добавление/удаление основных секций)
        
        if set(bp1.keys()) != set(bp2.keys()):
            added_sections = set(bp2.keys()) - set(bp1.keys())
            removed_sections = set(bp1.keys()) - set(bp2.keys())
            
            changes.extend([f"Added section: {s}" for s in added_sections])
            changes.extend([f"Removed section: {s}" for s in removed_sections])
        
        return changes
    
    def _calculate_impact_score(self, changes: Dict[str, Any]) -> float:
        """Вычисляет оценку влияния изменений."""
        score = 0.0
        
        # Вес различных типов изменений
        weights = {
            "added": 0.3,
            "removed": 0.5,
            "modified": 0.4,
            "structure_changes": 0.6
        }
        
        # Подсчитываем изменения
        for change_type, weight in weights.items():
            count = len(changes.get(change_type, []))
            score += count * weight
        
        # Нормализуем оценку (0-1)
        return min(1.0, score / 10.0)
    
    def _assess_risk(self, impact_score: float, changes: Dict[str, Any]) -> str:
        """Оценивает риск изменений."""
        if impact_score >= self.change_thresholds["high_impact"]:
            return "high"
        elif impact_score >= self.change_thresholds["medium_impact"]:
            return "medium"
        elif impact_score >= self.change_thresholds["low_impact"]:
            return "low"
        else:
            return "minimal"
    
    def _generate_recommendations(self, changes: Dict[str, Any], impact_score: float) -> List[str]:
        """Генерирует рекомендации на основе изменений."""
        recommendations = []
        
        if impact_score >= 0.8:
            recommendations.append("Рекомендуется тщательное тестирование")
            recommendations.append("Рассмотрите поэтапное развертывание")
        
        if changes.get("removed"):
            recommendations.append("Убедитесь в резервном копировании удаленных компонентов")
            recommendations.append("Проверьте совместимость с существующими интеграциями")
        
        if changes.get("structure_changes"):
            recommendations.append("Проверьте документацию API")
            recommendations.append("Обновите зависимые системы")
        
        if not recommendations:
            recommendations.append("Изменения выглядят безопасными")
        
        return recommendations
    
    async def _analyze_changes(self, from_version: int, to_version: int) -> None:
        """Автоматически анализирует изменения."""
        try:
            await self.analyze_impact(from_version, to_version)
        except Exception as e:
            logger.warning(f"Не удалось автоматически проанализировать изменения: {e}")


# Экспорт основных классов
__all__ = [
    "AdaptiveBlueprintTracker",
    "BlueprintVersion",
    "ResourceLink", 
    "ChangeAnalysis"
]
