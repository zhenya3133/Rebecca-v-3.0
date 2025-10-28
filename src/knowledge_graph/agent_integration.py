"""
Интеграция Context Engine с BaseAgent для context-aware решений

Реализует:
1. Адаптацию BaseAgent для работы с контекстуальными знаниями
2. Автоматическое обогащение задач контекстом
3. Контекстуально-осведомленное выполнение задач
4. Обучение и адаптацию на основе контекста
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
import uuid
import json

from .context_engine import (
    ContextEngine, 
    ContextRequest, 
    KnowledgeDomain, 
    create_context_engine,
    integrate_context_awareness
)

# Импорты из Rebecca-Platform
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from multi_agent.base_agent import (
    BaseAgent, 
    AgentType, 
    TaskRequest, 
    TaskResult, 
    TaskStatus,
    ContextAwareBaseAgent
)

from memory_manager.memory_manager import MemoryManager


# =============================================================================
# Расширенный агент с контекстуальными возможностями
# =============================================================================

class ContextAwareAgent(BaseAgent):
    """
    Расширенный BaseAgent с поддержкой контекстуальной интеграции знаний
    
    Добавляет:
    - Автоматическое обогащение контекстом
    - Контекстуально-осведомленное выполнение задач
    - Адаптивное поведение на основе знаний
    - Интеграцию с психологическими доменами
    """
    
    def __init__(
        self,
        agent_type: AgentType,
        capabilities,
        memory_manager: Optional[MemoryManager] = None,
        context_engine: Optional[ContextEngine] = None,
        **kwargs
    ):
        super().__init__(
            agent_type=agent_type,
            capabilities=capabilities,
            memory_manager=memory_manager,
            **kwargs
        )
        
        self.context_engine = context_engine
        self.context_aware = False
        
        # Контекстуальные настройки
        self.context_config = {
            "auto_enrich_context": True,
            "reasoning_depth": 2,
            "freshness_threshold": 0.7,
            "cross_domain_links": True,
            "temporal_validation": True,
            "learning_enabled": True
        }
        
        # Специализация агента по доменам
        self.domain_specializations = self._infer_domain_specializations()
        
        # Статистика контекстуального выполнения
        self.context_stats = {
            "total_context_requests": 0,
            "successful_enrichments": 0,
            "knowledge_gaps_identified": 0,
            "cross_domain_connections_used": 0,
            "average_confidence_score": 0.0
        }
        
        self.logger.info(f"ContextAwareAgent {agent_type.value} инициализирован")
    
    def _infer_domain_specializations(self) -> List[KnowledgeDomain]:
        """Определение доменных специализаций агента"""
        specializations = []
        
        # Матрица специализаций агентов по типам
        agent_specializations = {
            AgentType.RESEARCH: [KnowledgeDomain.PSYCHOLOGY, KnowledgeDomain.SCIENCE, KnowledgeDomain.EDUCATION],
            AgentType.BACKEND: [KnowledgeDomain.TECHNOLOGY],
            AgentType.FRONTEND: [KnowledgeDomain.TECHNOLOGY, KnowledgeDomain.EDUCATION],
            AgentType.ML_ENGINEER: [KnowledgeDomain.TECHNOLOGY, KnowledgeDomain.SCIENCE],
            AgentType.QA_ANALYST: [KnowledgeDomain.TECHNOLOGY, KnowledgeDomain.EDUCATION],
            AgentType.DEVOPS: [KnowledgeDomain.TECHNOLOGY],
            AgentType.WRITER: [KnowledgeDomain.EDUCATION, KnowledgeDomain.GENERAL],
            AgentType.BUSINESS: [KnowledgeDomain.BUSINESS],
            AgentType.COORDINATOR: [KnowledgeDomain.GENERAL] + [d for d in KnowledgeDomain]
        }
        
        return agent_specializations.get(self.agent_type, [KnowledgeDomain.GENERAL])
    
    async def execute_task_with_context(self, task: TaskRequest) -> TaskResult:
        """
        Выполнение задачи с автоматическим обогащением контекстом
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            self.context_stats["total_context_requests"] += 1
            
            # 1. Анализ необходимости контекстуального обогащения
            needs_context = await self._analyze_context_needs(task)
            
            enhanced_task = task
            context_enriched = False
            confidence_score = 0.0
            
            if needs_context and self.context_engine and self.context_config["auto_enrich_context"]:
                # 2. Обогащение задачи контекстом
                enhanced_task, context_info = await self._enrich_task_with_context(task)
                context_enriched = True
                confidence_score = context_info.get("confidence_score", 0.0)
                
                self.context_stats["successful_enrichments"] += 1
                
                # Сохранение контекстной информации в результат
                task.context = task.context or {}
                task.context["context_engine_info"] = context_info
            
            # 3. Выполнение задачи
            result = await self.execute_task(enhanced_task)
            
            # 4. Пост-обработка с учетом контекста
            if context_enriched:
                result = await self._post_process_with_context(result, context_info)
            
            # 5. Обновление статистики
            await self._update_context_statistics(confidence_score, context_info)
            
            self.logger.info(
                f"Задача {task.task_id} выполнена с контекстом: "
                f"обогащена={context_enriched}, уверенность={confidence_score:.2f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка выполнения задачи с контекстом: {str(e)}")
            # Fallback к обычному выполнению
            return await self.execute_task(task)
    
    async def _analyze_context_needs(self, task: TaskRequest) -> bool:
        """Анализ необходимости контекстуального обогащения"""
        
        # Критерии для включения контекста
        criteria = {
            "high_priority": task.priority <= 2,
            "complex_task": len(task.description) > 100 or len(task.dependencies) > 2,
            "cross_domain_potential": await self._has_cross_domain_potential(task),
            "knowledge_intensive": await self._is_knowledge_intensive(task),
            "time_critical": task.timeout and task.timeout > 1800  # больше 30 минут
        }
        
        # Если выполняется хотя бы один критерий - используем контекст
        needs_context = any(criteria.values())
        
        self.logger.debug(f"Анализ контекстных потребностей для {task.task_id}: {criteria} -> {needs_context}")
        
        return needs_context
    
    async def _has_cross_domain_potential(self, task: TaskRequest) -> bool:
        """Проверка потенциала междоменных связей"""
        task_text = f"{task.task_type} {task.description}".lower()
        
        # Ключевые слова, указывающие на междоменный потенциал
        cross_domain_keywords = [
            "психолог", "медицин", "образовател", "организацион",
            "когнитивн", "поведен", "клиническ", "терапевт",
            "развит", "обучен", "стресс", "выгоран"
        ]
        
        return any(keyword in task_text for keyword in cross_domain_keywords)
    
    async def _is_knowledge_intensive(self, task: TaskRequest) -> bool:
        """Определение интенсивности знаний в задаче"""
        knowledge_intensive_keywords = [
            "анализ", "оценка", "исследован", "диагностик",
            "планирован", "разработка", "стратег", "концепт"
        ]
        
        task_text = task.description.lower()
        return any(keyword in task_text for keyword in knowledge_intensive_keywords)
    
    async def _enrich_task_with_context(
        self, 
        task: TaskRequest
    ) -> tuple[TaskRequest, Dict[str, Any]]:
        """Обогащение задачи контекстом"""
        
        # Определение релевантных доменов
        target_domains = await self._determine_target_domains(task)
        
        # Создание запроса контекста
        context_request = ContextRequest(
            current_task=task,
            active_context={
                "agent_type": self.agent_type.value,
                "specializations": [d.value for d in self.domain_specializations],
                "agent_capabilities": self.capabilities.supported_tasks
            },
            target_domains=target_domains,
            reasoning_depth=self.context_config["reasoning_depth"],
            freshness_threshold=self.context_config["freshness_threshold"],
            include_controversial=False,
            cross_domain_links=self.context_config["cross_domain_links"],
            temporal_validation=self.context_config["temporal_validation"]
        )
        
        # Получение обогащенного контекста
        context_info = await self.context_engine.enhance_agent_context(
            agent_id=f"{self.agent_type.value}_agent",
            task=task,
            domains=target_domains
        )
        
        # Обогащение задачи
        enhanced_task = task.copy()
        enhanced_task.context = enhanced_task.context or {}
        enhanced_task.context["enhanced_context"] = context_info
        
        return enhanced_task, context_info
    
    async def _determine_target_domains(self, task: TaskRequest) -> List[KnowledgeDomain]:
        """Определение целевых доменов для задачи"""
        target_domains = self.domain_specializations.copy()
        
        task_text = f"{task.task_type} {task.description}".lower()
        
        # Дополнительные домены на основе содержания задачи
        domain_indicators = {
            KnowledgeDomain.PSYCHOLOGY: ["психолог", "поведен", "когнитивн", "эмоц", "терапия", "клиническ"],
            KnowledgeDomain.MEDICINE: ["медицин", "здоровье", "пациент", "диагностик", "лечен"],
            KnowledgeDomain.TECHNOLOGY: ["программ", "система", "api", "сервер", "технолог"],
            KnowledgeDomain.BUSINESS: ["бизнес", "маркетинг", "продаж", "управлен", "организац"],
            KnowledgeDomain.EDUCATION: ["образован", "обучен", "учен", "курс", "студент"]
        }
        
        for domain, indicators in domain_indicators.items():
            if any(indicator in task_text for indicator in indicators):
                if domain not in target_domains:
                    target_domains.append(domain)
        
        return target_domains
    
    async def _post_process_with_context(
        self, 
        result: TaskResult, 
        context_info: Dict[str, Any]
    ) -> TaskResult:
        """Пост-обработка результата с учетом контекста"""
        
        # Добавление контекстуальных метрик в результат
        if "context_result" in context_info:
            context_result = context_info["context_result"]
            
            # Метрики из контекстного анализа
            result.metrics.update({
                "context_confidence": context_result.get("confidence_score", 0.0),
                "relevant_concepts_count": len(context_result.get("relevant_concepts", [])),
                "reasoning_chains_count": len(context_result.get("reasoning_chains", [])),
                "cross_domain_connections_count": len(context_result.get("cross_domain_connections", [])),
                "processing_time": context_result.get("processing_time", 0.0)
            })
            
            # Контекстуальные инсайты
            actionable_insights = context_info.get("actionable_insights", [])
            if actionable_insights:
                result.next_actions.extend(actionable_insights[:3])  # Первые 3 инсайта
            
            # Рекомендации
            recommendations = context_info.get("recommended_actions", [])
            if recommendations:
                result.output["contextual_recommendations"] = recommendations
        
        # Обновление статуса на основе контекста
        confidence_score = result.metrics.get("context_confidence", 0.0)
        if confidence_score < 0.3 and result.status == TaskStatus.COMPLETED:
            result.warnings.append("Низкая уверенность в контексте - рекомендуется дополнительная проверка")
        
        return result
    
    async def _update_context_statistics(
        self, 
        confidence_score: float, 
        context_info: Dict[str, Any]
    ):
        """Обновление статистики контекстуального выполнения"""
        
        # Обновление среднего показателя уверенности
        current_avg = self.context_stats["average_confidence_score"]
        total_requests = self.context_stats["total_context_requests"]
        
        if total_requests > 1:
            self.context_stats["average_confidence_score"] = (
                (current_avg * (total_requests - 1) + confidence_score) / total_requests
            )
        else:
            self.context_stats["average_confidence_score"] = confidence_score
        
        # Подсчет пробелов в знаниях
        knowledge_gaps = context_info.get("knowledge_gaps", [])
        self.context_stats["knowledge_gaps_identified"] += len(knowledge_gaps)
        
        # Подсчет использованных междоменных связей
        cross_domain_connections = context_info.get("context_result", {}).get("cross_domain_connections", [])
        self.context_stats["cross_domain_connections_used"] += len(cross_domain_connections)
    
    def get_context_capabilities(self) -> Dict[str, Any]:
        """Получение контекстуальных возможностей агента"""
        return {
            "context_engine_enabled": self.context_engine is not None,
            "domain_specializations": [d.value for d in self.domain_specializations],
            "context_config": self.context_config,
            "context_statistics": self.context_stats,
            "average_confidence": self.context_stats["average_confidence_score"],
            "enrichment_success_rate": (
                self.context_stats["successful_enrichments"] / 
                max(1, self.context_stats["total_context_requests"])
            )
        }
    
    def update_context_config(self, config_updates: Dict[str, Any]):
        """Обновление конфигурации контекстуальной обработки"""
        self.context_config.update(config_updates)
        self.logger.info(f"Обновлена конфигурация контекста: {config_updates}")
    
    async def learn_from_context(self, task_id: str, outcome_feedback: Dict[str, Any]):
        """Обучение на основе контекстного результата"""
        if not self.context_config["learning_enabled"]:
            return
        
        try:
            # Анализ обратной связи для улучшения контекстной обработки
            confidence_score = outcome_feedback.get("confidence_rating", 0.5)
            task_completion_quality = outcome_feedback.get("completion_quality", 0.5)
            
            # Адаптация параметров на основе результатов
            if confidence_score < 0.4:
                # Увеличиваем глубину рассуждений для лучшего понимания
                self.context_config["reasoning_depth"] = min(5, self.context_config["reasoning_depth"] + 1)
            
            if task_completion_quality > 0.8:
                # Если задача выполнена хорошо, можно снизить глубину для скорости
                self.context_config["reasoning_depth"] = max(1, self.context_config["reasoning_depth"] - 0.2)
            
            # Сохранение обучения в памяти
            if self.memory_manager:
                learning_data = {
                    "task_id": task_id,
                    "agent_type": self.agent_type.value,
                    "feedback": outcome_feedback,
                    "config_state": self.context_config.copy(),
                    "learning_timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                await self.memory_manager.store(
                    layer="SEMANTIC",
                    key=f"context_learning_{task_id}",
                    value=learning_data,
                    metadata={
                        "type": "context_learning",
                        "agent_type": self.agent_type.value
                    }
                )
            
            self.logger.info(f"Контекстуальное обучение выполнено для задачи {task_id}")
            
        except Exception as e:
            self.logger.error(f"Ошибка контекстуального обучения: {str(e)}")


# =============================================================================
# Фабрика контекстуальных агентов
# =============================================================================

class ContextAwareAgentFactory:
    """Фабрика для создания контекстуально-осведомленных агентов"""
    
    def __init__(self, memory_manager: MemoryManager, context_engine: ContextEngine):
        self.memory_manager = memory_manager
        self.context_engine = context_engine
        self.logger = logging.getLogger("agent_factory")
        
        # Стандартные конфигурации для типов агентов
        self.agent_configs = {
            AgentType.RESEARCH: {
                "reasoning_depth": 3,
                "freshness_threshold": 0.8,
                "cross_domain_links": True,
                "temporal_validation": True,
                "learning_enabled": True
            },
            AgentType.BACKEND: {
                "reasoning_depth": 2,
                "freshness_threshold": 0.6,
                "cross_domain_links": False,
                "temporal_validation": False,
                "learning_enabled": True
            },
            AgentType.FRONTEND: {
                "reasoning_depth": 2,
                "freshness_threshold": 0.6,
                "cross_domain_links": False,
                "temporal_validation": False,
                "learning_enabled": True
            },
            AgentType.QA_ANALYST: {
                "reasoning_depth": 3,
                "freshness_threshold": 0.9,
                "cross_domain_links": True,
                "temporal_validation": True,
                "learning_enabled": True
            }
        }
    
    def create_agent(
        self,
        agent_type: AgentType,
        capabilities,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> ContextAwareAgent:
        """Создание контекстуально-осведомленного агента"""
        
        # Получение базовой конфигурации
        base_config = self.agent_configs.get(agent_type, {})
        
        # Применение кастомной конфигурации
        if custom_config:
            base_config.update(custom_config)
        
        # Создание агента
        agent = ContextAwareAgent(
            agent_type=agent_type,
            capabilities=capabilities,
            memory_manager=self.memory_manager,
            context_engine=self.context_engine,
            context_config=base_config
        )
        
        self.logger.info(f"Создан контекстуально-осведомленный агент: {agent_type.value}")
        
        return agent
    
    def create_psychology_specialist_agent(self) -> ContextAwareAgent:
        """Создание специализированного агента для психологических задач"""
        
        # Специализированные возможности для психологии
        from multi_agent.base_agent import AgentCapabilities
        
        psychology_capabilities = AgentCapabilities(
            agent_type=AgentType.RESEARCH,
            name="Psychology Specialist Agent",
            description="Специализированный агент для психологических задач и исследований",
            supported_tasks=[
                "cognitive_assessment",
                "therapy_planning", 
                "developmental_assessment",
                "psychological_analysis",
                "behavioral_intervention"
            ],
            specializations=[
                "clinical_psychology",
                "neuropsychology", 
                "developmental_psychology",
                "educational_psychology"
            ],
            integrations=["context_engine", "psychology_knowledge_base"]
        )
        
        # Специализированная конфигурация для психологии
        psychology_config = {
            "reasoning_depth": 4,  # Глубокий анализ для психологии
            "freshness_threshold": 0.8,
            "cross_domain_links": True,  # Важны связи с медициной, образованием
            "temporal_validation": True,
            "learning_enabled": True,
            "auto_enrich_context": True
        }
        
        return self.create_agent(
            AgentType.RESEARCH,
            psychology_capabilities,
            custom_config=psychology_config
        )


# =============================================================================
# Утилиты для интеграции
# =============================================================================

async def create_context_aware_ecosystem(
    memory_manager: MemoryManager,
    agent_types: List[AgentType] = None
) -> Dict[str, ContextAwareAgent]:
    """Создание экосистемы контекстуально-осведомленных агентов"""
    
    if agent_types is None:
        agent_types = [
            AgentType.RESEARCH,
            AgentType.BACKEND,
            AgentType.FRONTEND,
            AgentType.QA_ANALYST
        ]
    
    # Создание контекстного движка
    context_engine = await create_context_engine(memory_manager)
    
    # Создание фабрики агентов
    agent_factory = ContextAwareAgentFactory(memory_manager, context_engine)
    
    # Создание агентов
    agents = {}
    
    for agent_type in agent_types:
        try:
            agent = agent_factory.create_agent(agent_type, None)  # capabilities будут загружены из конфигурации
            agents[agent_type.value] = agent
        except Exception as e:
            logging.error(f"Ошибка создания агента {agent_type}: {str(e)}")
    
    # Добавление специализированного психологического агента
    try:
        psychology_agent = agent_factory.create_psychology_specialist_agent()
        agents["psychology_specialist"] = psychology_agent
    except Exception as e:
        logging.error(f"Ошибка создания психологического агента: {str(e)}")
    
    ecosystem = {
        "agents": agents,
        "context_engine": context_engine,
        "agent_factory": agent_factory,
        "memory_manager": memory_manager
    }
    
    logging.info(f"Создана экосистема из {len(agents)} контекстуально-осведомленных агентов")
    
    return ecosystem


def get_agent_context_summary(agent: ContextAwareAgent) -> Dict[str, Any]:
    """Получение сводки контекстуальных возможностей агента"""
    
    capabilities = agent.get_context_capabilities()
    
    return {
        "agent_info": {
            "type": agent.agent_type.value,
            "name": agent.capabilities.name,
            "is_context_aware": agent.context_engine is not None
        },
        "context_capabilities": capabilities,
        "performance_metrics": {
            "total_context_requests": agent.context_stats["total_context_requests"],
            "success_rate": (
                agent.context_stats["successful_enrichments"] / 
                max(1, agent.context_stats["total_context_requests"])
            ),
            "average_confidence": agent.context_stats["average_confidence_score"]
        },
        "domain_specializations": [d.value for d in agent.domain_specializations],
        "config": agent.context_config
    }


# =============================================================================
# Демонстрация интеграции
# =============================================================================

async def demonstrate_agent_integration():
    """Демонстрация интеграции агентов с контекстным движком"""
    
    print("🔗 ДЕМОНСТРАЦИЯ ИНТЕГРАЦИИ AGENT-CONTEXT ENGINE")
    print("=" * 60)
    
    try:
        # Создание экосистемы (заглушка для демонстрации)
        ecosystem = await create_context_aware_ecosystem(memory_manager=None)  # В реальности здесь был бы MemoryManager
        
        agents = ecosystem["agents"]
        context_engine = ecosystem["context_engine"]
        
        print(f"✅ Создано агентов: {len(agents)}")
        print(f"✅ Context Engine: {context_engine is not None}")
        
        # Демонстрация возможностей агентов
        for agent_type, agent in agents.items():
            print(f"\n🤖 АГЕНТ: {agent_type}")
            summary = get_agent_context_summary(agent)
            
            print(f"  • Контекстуально-осведомленный: {summary['agent_info']['is_context_aware']}")
            print(f"  • Доменные специализации: {summary['domain_specializations']}")
            print(f"  • Контекстных запросов: {summary['performance_metrics']['total_context_requests']}")
            print(f"  • Средняя уверенность: {summary['performance_metrics']['average_confidence']:.2f}")
        
        print(f"\n🎯 Интеграция успешно продемонстрирована!")
        
    except Exception as e:
        print(f"❌ Ошибка демонстрации: {str(e)}")


async def run_integration_example():
    """Запуск примера интеграции"""
    await demonstrate_agent_integration()


if __name__ == "__main__":
    asyncio.run(run_integration_example())
