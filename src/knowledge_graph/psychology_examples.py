"""
Примеры использования Context Engine для психологического домена

Демонстрирует:
1. Анализ психологических задач
2. Извлечение релевантных психологических знаний
3. Multi-hop рассуждения в психологии
4. Временная валидация психологических концепций
5. Междоменные связи с медициной и образованием
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any
import json

from context_engine import (
    ContextEngine, 
    ContextRequest, 
    KnowledgeDomain, 
    TaskRequest,
    AgentType,
    create_context_engine
)

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Примеры психологических задач и анализа
# =============================================================================

class PsychologyContextExamples:
    """Примеры контекстуального анализа для психологических задач"""
    
    def __init__(self, context_engine: ContextEngine):
        self.context_engine = context_engine
        self.logger = logger
    
    async def example_1_cognitive_assessment_analysis(self) -> Dict[str, Any]:
        """
        Пример 1: Анализ задачи когнитивной оценки
        """
        print("\n" + "="*60)
        print("ПРИМЕР 1: КОГНИТИВНАЯ ОЦЕНКА ПАЦИЕНТА")
        print("="*60)
        
        # Создание задачи когнитивной оценки
        task = TaskRequest(
            agent_type=AgentType.RESEARCH,
            task_type="cognitive_assessment",
            description="Провести комплексную когнитивную оценку пациента с подозрением на раннюю стадию болезни Альцгеймера, включая тестирование памяти, внимания и исполнительных функций",
            inputs={
                "patient_age": 68,
                "assessment_tools": ["MMSE", "MoCA", "clock_drawing_test"],
                "focus_areas": ["episodic_memory", "working_memory", "executive_function"],
                "previous_conditions": ["hypertension", "mild_diabetes"]
            },
            priority=2,
            timeout=1800  # 30 минут
        )
        
        print(f"Задача: {task.description}")
        print(f"Приоритет: {task.priority}")
        print(f"Инструменты оценки: {task.inputs['assessment_tools']}")
        
        # Анализ контекста
        enhanced_context = await self.context_engine.enhance_agent_context(
            agent_id="psychology_assessment_agent",
            task=task,
            domains=[KnowledgeDomain.PSYCHOLOGY, KnowledgeDomain.MEDICINE]
        )
        
        # Вывод результатов анализа
        self._print_analysis_results(enhanced_context, "Когнитивная оценка")
        
        return enhanced_context
    
    async def example_2_therapy_session_planning(self) -> Dict[str, Any]:
        """
        Пример 2: Планирование терапевтической сессии
        """
        print("\n" + "="*60)
        print("ПРИМЕР 2: ПЛАНИРОВАНИЕ ТЕРАПЕВТИЧЕСКОЙ СЕССИИ")
        print("="*60)
        
        task = TaskRequest(
            agent_type=AgentType.RESEARCH,
            task_type="therapy_planning",
            description="Разработать план терапевтической работы с пациентом, страдающим генерализованным тревожным расстройством, включающий когнитивно-поведенческие техники и методы управления стрессом",
            inputs={
                "patient_diagnosis": "GAD",
                "therapy_approach": "CBT",
                "session_number": 5,
                "focus_issues": ["excessive_worry", "somatic_symptoms", "avoidance_behavior"],
                "previous_treatments": ["medication", "mindfulness_based_stress_reduction"]
            },
            priority=1,
            timeout=1200  # 20 минут
        )
        
        print(f"Задача: {task.description}")
        print(f"Подход: {task.inputs['therapy_approach']}")
        print(f"Номер сессии: {task.inputs['session_number']}")
        
        enhanced_context = await self.context_engine.enhance_agent_context(
            agent_id="therapy_planning_agent",
            task=task,
            domains=[KnowledgeDomain.PSYCHOLOGY, KnowledgeDomain.MEDICINE, KnowledgeDomain.EDUCATION]
        )
        
        self._print_analysis_results(enhanced_context, "Планирование терапии")
        
        return enhanced_context
    
    async def example_3_child_development_assessment(self) -> Dict[str, Any]:
        """
        Пример 3: Оценка развития ребенка
        """
        print("\n" + "="*60)
        print("ПРИМЕР 3: ОЦЕНКА РАЗВИТИЯ РЕБЕНКА")
        print("="*60)
        
        task = TaskRequest(
            agent_type=AgentType.RESEARCH,
            task_type="child_development_assessment",
            description="Провести оценку психического развития 5-летнего ребенка с подозрением на задержку речевого развития, включая анализ когнитивных, социальных и эмоциональных навыков",
            inputs={
                "child_age": 5,
                "assessment_area": "language_development",
                "observed_symptoms": ["delayed_speech", "difficulty_understanding", "limited_vocabulary"],
                "family_history": ["no_family_history", "supportive_environment"],
                "referral_source": "pediatrician"
            },
            priority=2,
            timeout=1500  # 25 минут
        )
        
        print(f"Задача: {task.description}")
        print(f"Возраст ребенка: {task.inputs['child_age']} лет")
        print(f"Область оценки: {task.inputs['assessment_area']}")
        
        enhanced_context = await self.context_engine.enhance_agent_context(
            agent_id="child_development_agent",
            task=task,
            domains=[KnowledgeDomain.PSYCHOLOGY, KnowledgeDomain.EDUCATION, KnowledgeDomain.MEDICINE]
        )
        
        self._print_analysis_results(enhanced_context, "Развитие ребенка")
        
        return enhanced_context
    
    async def example_4_workplace_stress_analysis(self) -> Dict[str, Any]:
        """
        Пример 4: Анализ стресса на рабочем месте
        """
        print("\n" + "="*60)
        print("ПРИМЕР 4: АНАЛИЗ СТРЕССА НА РАБОЧЕМ МЕСТЕ")
        print("="*60)
        
        task = TaskRequest(
            agent_type=AgentType.RESEARCH,
            task_type="workplace_stress_analysis",
            description="Проанализировать источники стресса в IT-компании и разработать программу снижения профессионального выгорания сотрудников, учитывая организационные и индивидуальные факторы",
            inputs={
                "industry": "IT",
                "company_size": "medium",
                "stress_factors": ["tight_deadlines", "high_workload", "unclear_expectations"],
                "employee_concerns": ["work_life_balance", "career_development", "communication"],
                "previous_interventions": ["employee_assistance_program", "flexible_hours"]
            },
            priority=2,
            timeout=2400  # 40 минут
        )
        
        print(f"Задача: {task.description}")
        print(f"Отрасль: {task.inputs['industry']}")
        print(f"Размер компании: {task.inputs['company_size']}")
        
        enhanced_context = await self.context_engine.enhance_agent_context(
            agent_id="workplace_psychology_agent",
            task=task,
            domains=[KnowledgeDomain.PSYCHOLOGY, KnowledgeDomain.BUSINESS, KnowledgeDomain.TECHNOLOGY]
        )
        
        self._print_analysis_results(enhanced_context, "Стресс на работе")
        
        return enhanced_context
    
    async def example_5_cross_domain_psychology_medicine(self) -> Dict[str, Any]:
        """
        Пример 5: Междоменный анализ психология-медицина
        """
        print("\n" + "="*60)
        print("ПРИМЕР 5: МЕЖДОМЕННЫЙ АНАЛИЗ (ПСИХОЛОГИЯ-МЕДИЦИНА)")
        print("="*60)
        
        task = TaskRequest(
            agent_type=AgentType.RESEARCH,
            task_type="psychosomatic_medicine_analysis",
            description="Исследовать психосоматические аспекты хронических заболеваний и разработать интегрированный подход к лечению, учитывающий психологические факторы в терапии соматических расстройств",
            inputs={
                "condition_type": "chronic_disease",
                "specific_conditions": ["irritable_bowel_syndrome", "chronic_fatigue_syndrome"],
                "psychological_factors": ["stress", "anxiety", "depression"],
                "treatment_focus": "integrated_care",
                "collaboration_areas": ["psychotherapy", "behavioral_medicine", "stress_management"]
            },
            priority=1,
            timeout=3000  # 50 минут
        )
        
        print(f"Задача: {task.description}")
        print(f"Тип состояния: {task.inputs['condition_type']}")
        print(f"Психологические факторы: {task.inputs['psychological_factors']}")
        
        enhanced_context = await self.context_engine.enhance_agent_context(
            agent_id="psychosomatic_medicine_agent",
            task=task,
            domains=[KnowledgeDomain.PSYCHOLOGY, KnowledgeDomain.MEDICINE, KnowledgeDomain.SCIENCE]
        )
        
        self._print_analysis_results(enhanced_context, "Психосоматическая медицина")
        
        return enhanced_context
    
    def _print_analysis_results(self, enhanced_context: Dict[str, Any], analysis_type: str):
        """Печать результатов анализа"""
        context_result = enhanced_context.get("context_result", {})
        
        print(f"\n🔍 РЕЗУЛЬТАТЫ АНАЛИЗА: {analysis_type}")
        print("-" * 40)
        
        # Основные метрики
        print(f"Уверенность в контексте: {context_result.get('confidence_score', 0):.2%}")
        print(f"Релевантных концептов найдено: {len(context_result.get('relevant_concepts', []))}")
        print(f"Цепочек рассуждений: {len(context_result.get('reasoning_chains', []))}")
        print(f"Междоменных связей: {len(context_result.get('cross_domain_connections', []))}")
        
        # Временные инсайты
        temporal_insights = context_result.get("temporal_insights", {})
        if temporal_insights:
            print(f"\nВременные инсайты:")
            print(f"  - Согласованность знаний: {temporal_insights.get('consistency_score', 0):.2%}")
            print(f"  - Валидных единиц: {temporal_insights.get('valid_units', 0)}")
            print(f"  - Устаревших единиц: {temporal_insights.get('expired_units', 0)}")
        
        # Действенные инсайты
        actionable_insights = enhanced_context.get("actionable_insights", [])
        if actionable_insights:
            print(f"\nДейственные инсайты:")
            for insight in actionable_insights[:3]:
                print(f"  • {insight}")
        
        # Рекомендации
        recommendations = enhanced_context.get("recommended_actions", [])
        if recommendations:
            print(f"\nРекомендации:")
            for rec in recommendations[:3]:
                print(f"  → {rec}")
        
        # Пробелы в знаниях
        knowledge_gaps = enhanced_context.get("knowledge_gaps", [])
        if knowledge_gaps:
            print(f"\nВыявленные пробелы:")
            for gap in knowledge_gaps:
                print(f"  ⚠️ {gap}")
        
        print(f"\nВремя обработки: {context_result.get('processing_time', 0):.3f}s")
    
    async def run_all_examples(self) -> List[Dict[str, Any]]:
        """Запуск всех примеров"""
        print("🚀 ЗАПУСК ВСЕХ ПРИМЕРОВ КОНТЕКСТУАЛЬНОЙ ИНТЕГРАЦИИ")
        print("=" * 80)
        
        results = []
        
        try:
            # Пример 1: Когнитивная оценка
            result1 = await self.example_1_cognitive_assessment_analysis()
            results.append(result1)
            
            # Пример 2: Планирование терапии
            result2 = await self.example_2_therapy_session_planning()
            results.append(result2)
            
            # Пример 3: Оценка развития ребенка
            result3 = await self.example_3_child_development_assessment()
            results.append(result3)
            
            # Пример 4: Стресс на работе
            result4 = await self.example_4_workplace_stress_analysis()
            results.append(result4)
            
            # Пример 5: Междоменный анализ
            result5 = await self.example_5_cross_domain_psychology_medicine()
            results.append(result5)
            
            # Сводная статистика
            await self._print_summary_statistics(results)
            
        except Exception as e:
            self.logger.error(f"Ошибка выполнения примеров: {str(e)}")
            raise
        
        return results
    
    async def _print_summary_statistics(self, results: List[Dict[str, Any]]):
        """Печать сводной статистики"""
        print("\n" + "="*60)
        print("СВОДНАЯ СТАТИСТИКА")
        print("="*60)
        
        total_processing_time = 0
        total_concepts = 0
        total_chains = 0
        total_connections = 0
        avg_confidence = 0
        
        confidence_scores = []
        
        for i, result in enumerate(results, 1):
            context_result = result.get("context_result", {})
            
            processing_time = context_result.get('processing_time', 0)
            concepts_count = len(context_result.get('relevant_concepts', []))
            chains_count = len(context_result.get('reasoning_chains', []))
            connections_count = len(context_result.get('cross_domain_connections', []))
            confidence = context_result.get('confidence_score', 0)
            
            total_processing_time += processing_time
            total_concepts += concepts_count
            total_chains += chains_count
            total_connections += connections_count
            confidence_scores.append(confidence)
            
            print(f"Пример {i}:")
            print(f"  Время обработки: {processing_time:.3f}s")
            print(f"  Концептов: {concepts_count}")
            print(f"  Цепочек: {chains_count}")
            print(f"  Связей: {connections_count}")
            print(f"  Уверенность: {confidence:.2%}")
            print()
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        print(f"ОБЩАЯ СТАТИСТИКА:")
        print(f"  Всего примеров: {len(results)}")
        print(f"  Общее время: {total_processing_time:.3f}s")
        print(f"  Среднее время на пример: {total_processing_time/len(results):.3f}s")
        print(f"  Всего концептов: {total_concepts}")
        print(f"  Всего цепочек: {total_chains}")
        print(f"  Всего связей: {total_connections}")
        print(f"  Средняя уверенность: {avg_confidence:.2%}")


# =============================================================================
# Дополнительные утилиты для психологического домена
# =============================================================================

class PsychologyKnowledgeBase:
    """База знаний для психологических концепций"""
    
    # Базовые психологические концепты
    PSYCHOLOGY_CONCEPTS = {
        "cognitive_assessment": {
            "description": "Когнитивная оценка - систематическое измерение когнитивных функций",
            "related_concepts": ["memory", "attention", "executive_function", "processing_speed"],
            "assessment_tools": ["MMSE", "MoCA", "WAIS", "Rey_Osterrieth"],
            "domains": ["neuropsychology", "clinical_psychology"]
        },
        "anxiety_disorders": {
            "description": "Тревожные расстройства - группа психических расстройств",
            "related_concepts": ["GAD", "panic_disorder", "social_anxiety", "phobias"],
            "treatment_approaches": ["CBT", "exposure_therapy", "medication", "mindfulness"],
            "domains": ["clinical_psychology", "behavioral_therapy"]
        },
        "child_development": {
            "description": "Развитие ребенка - процессы роста и изменения в детском возрасте",
            "related_concepts": ["language_development", "cognitive_development", "social_development"],
            "assessment_areas": ["motor_skills", "language", "social_skills", "cognitive_abilities"],
            "domains": ["developmental_psychology", "educational_psychology"]
        },
        "workplace_stress": {
            "description": "Стресс на рабочем месте - психологические и физиологические реакции",
            "related_concepts": ["burnout", "work_life_balance", "job_satisfaction", "organizational_stress"],
            "interventions": ["stress_management", "workplace_wellness", "employee_assistance"],
            "domains": ["industrial_psychology", "organizational_psychology"]
        },
        "psychosomatic_medicine": {
            "description": "Психосоматическая медицина - изучение взаимосвязи психологических и соматических факторов",
            "related_concepts": ["stress_illness_connection", "behavioral_medicine", "mind_body_connection"],
            "applications": ["chronic_illness", "functional_disorders", "stress_related_conditions"],
            "domains": ["health_psychology", "behavioral_medicine"]
        }
    }
    
    @classmethod
    def get_concept_info(cls, concept_id: str) -> Dict[str, Any]:
        """Получение информации о психологическом концепте"""
        return cls.PSYCHOLOGY_CONCEPTS.get(concept_id, {})
    
    @classmethod
    def get_related_concepts(cls, concept_id: str) -> List[str]:
        """Получение связанных концептов"""
        concept_info = cls.get_concept_info(concept_id)
        return concept_info.get("related_concepts", [])
    
    @classmethod
    def get_domains(cls, concept_id: str) -> List[str]:
        """Получение доменов концепта"""
        concept_info = cls.get_concept_info(concept_id)
        return concept_info.get("domains", [])


class PsychologyTaskTemplates:
    """Шаблоны психологических задач для Context Engine"""
    
    COGNITIVE_ASSESSMENT_TEMPLATE = {
        "task_type": "cognitive_assessment",
        "description_template": "Провести когнитивную оценку {patient_age}-летнего пациента с фокусом на {assessment_areas}",
        "inputs_template": {
            "patient_age": "<int: возраст>",
            "assessment_areas": ["<list: области оценки>"],
            "assessment_tools": ["<list: инструменты оценки>"],
            "clinical_history": "<dict: клинический анамнез>"
        },
        "target_domains": [KnowledgeDomain.PSYCHOLOGY, KnowledgeDomain.MEDICINE],
        "reasoning_depth": 3,
        "temporal_validation": True,
        "cross_domain_links": True
    }
    
    THERAPY_PLANNING_TEMPLATE = {
        "task_type": "therapy_planning",
        "description_template": "Разработать план терапевтической работы с пациентом с {diagnosis} используя {approach}",
        "inputs_template": {
            "diagnosis": "<str: диагноз>",
            "therapy_approach": "<str: терапевтический подход>",
            "session_number": "<int: номер сессии>",
            "clinical_presentation": "<dict: клиническая картина>"
        },
        "target_domains": [KnowledgeDomain.PSYCHOLOGY, KnowledgeDomain.MEDICINE, KnowledgeDomain.EDUCATION],
        "reasoning_depth": 2,
        "temporal_validation": True,
        "cross_domain_links": True
    }
    
    DEVELOPMENTAL_ASSESSMENT_TEMPLATE = {
        "task_type": "developmental_assessment",
        "description_template": "Оценить развитие ребенка {age} лет в области {development_areas}",
        "inputs_template": {
            "age": "<int: возраст ребенка>",
            "development_areas": ["<list: области развития>"],
            "observed_concerns": ["<list: наблюдаемые проблемы>"],
            "family_context": "<dict: семейный контекст>"
        },
        "target_domains": [KnowledgeDomain.PSYCHOLOGY, KnowledgeDomain.EDUCATION],
        "reasoning_depth": 2,
        "temporal_validation": True,
        "cross_domain_links": True
    }
    
    @classmethod
    def get_template(cls, task_type: str) -> Dict[str, Any]:
        """Получение шаблона задачи"""
        templates = {
            "cognitive_assessment": cls.COGNITIVE_ASSESSMENT_TEMPLATE,
            "therapy_planning": cls.THERAPY_PLANNING_TEMPLATE,
            "developmental_assessment": cls.DEVELOPMENTAL_ASSESSMENT_TEMPLATE
        }
        return templates.get(task_type, {})
    
    @classmethod
    def customize_task(cls, template: Dict[str, Any], custom_values: Dict[str, Any]) -> Dict[str, Any]:
        """Кастомизация шаблона задачи"""
        if not template:
            return {}
        
        customized = template.copy()
        
        # Подстановка значений в описание
        if "description_template" in customized and "description" in custom_values:
            customized["description"] = custom_values["description"]
        
        # Подстановка значений в inputs
        if "inputs_template" in customized and "inputs" in custom_values:
            customized["inputs"] = custom_values["inputs"]
        
        # Переопределение других полей
        for key, value in custom_values.items():
            if key not in ["description_template", "inputs_template", "description", "inputs"]:
                customized[key] = value
        
        return customized


# =============================================================================
# Главная функция демонстрации
# =============================================================================

async def demonstrate_psychology_context_integration():
    """Демонстрация контекстуальной интеграции знаний для психологии"""
    
    print("🧠 ДЕМОНСТРАЦИЯ КОНТЕКСТУАЛЬНОЙ ИНТЕГРАЦИИ ЗНАНИЙ")
    print("📚 Специализация: ПСИХОЛОГИЧЕСКИЙ ДОМЕН")
    print("=" * 80)
    
    # Инициализация (в реальном приложении здесь был бы полноценный MemoryManager)
    try:
        # Создание контекстного движка (заглушка для демонстрации)
        memory_manager = None  # В реальном приложении здесь был бы экземпляр MemoryManager
        context_engine = ContextEngine(memory_manager)  # Без hybrid_retriever для упрощения
        
        print(f"✅ Context Engine инициализирован")
        
        # Создание примеров
        examples = PsychologyContextExamples(context_engine)
        
        # Запуск всех примеров
        results = await examples.run_all_examples()
        
        print(f"\n🎉 Все примеры выполнены успешно!")
        print(f"📊 Обработано задач: {len(results)}")
        
        return results
        
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")
        return []


async def demonstrate_knowledge_base_integration():
    """Демонстрация интеграции с базой знаний психологии"""
    
    print("\n🔗 ДЕМОНСТРАЦИЯ ИНТЕГРАЦИИ С БАЗОЙ ЗНАНИЙ")
    print("=" * 60)
    
    # Демонстрация работы с базой знаний
    kb = PsychologyKnowledgeBase()
    templates = PsychologyTaskTemplates()
    
    print("\n📖 ДОСТУПНЫЕ КОНЦЕПТЫ ПСИХОЛОГИИ:")
    for concept_id, concept_info in kb.PSYCHOLOGY_CONCEPTS.items():
        print(f"  • {concept_id}")
        print(f"    Описание: {concept_info['description']}")
        print(f"    Домены: {', '.join(concept_info['domains'])}")
        print()
    
    print("\n📋 ДОСТУПНЫЕ ШАБЛОНЫ ЗАДАЧ:")
    for task_type in ["cognitive_assessment", "therapy_planning", "developmental_assessment"]:
        template = templates.get_template(task_type)
        if template:
            print(f"  • {task_type}")
            print(f"    Домены: {', '.join(template['target_domains'])}")
            print(f"    Глубина рассуждений: {template['reasoning_depth']}")
            print()
    
    print("✅ Демонстрация базы знаний завершена")


# =============================================================================
# Точка входа
# =============================================================================

async def main():
    """Главная функция"""
    try:
        # Демонстрация контекстуальной интеграции
        results = await demonstrate_psychology_context_integration()
        
        # Демонстрация базы знаний
        await demonstrate_knowledge_base_integration()
        
        print(f"\n🎯 ЗАКЛЮЧЕНИЕ:")
        print(f"Все компоненты контекстуальной интеграции знаний успешно продемонстрированы!")
        print(f"Система готова к интеграции с Rebecca-Platform.")
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
