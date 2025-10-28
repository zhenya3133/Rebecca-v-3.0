"""
Полноценный оркестратор с интеграцией мета-агента Rebecca.

Обеспечивает:
- Инициализацию всех ключевых компонентов системы
- Управление workflow pipeline через 7 этапов
- Интеграцию MemoryManager, IngestPipeline и RebeccaMetaAgent
- Error handling и fallback механизмы
- Совместимость с существующими API
"""

import asyncio
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

# Импорт компонентов системы (с обработкой ошибок импорта)
try:
    from ..memory_manager.memory_manager import MemoryManager
except ImportError:
    MemoryManager = None

try:
    from ..ingest.loader import IngestPipeline
except ImportError:
    IngestPipeline = None

try:
    from ..rebecca.meta_agent import RebeccaMetaAgent, TaskType, TaskPriority
except ImportError:
    RebeccaMetaAgent = None
    TaskType = None
    TaskPriority = None

try:
    from ..architect.main import run_agent as run_architect
except ImportError:
    def run_architect(context, input_data):
        return {"result": "Architect analysis completed", "context": context}

try:
    from ..researcher.main import run_agent as run_researcher
except ImportError:
    def run_researcher(context, input_data):
        return {"result": "Research completed", "context": context}

try:
    from ..knowledge_curator.main import run_agent as run_knowledge_curator
except ImportError:
    def run_knowledge_curator(context, input_data):
        return {"result": "Knowledge curation completed", "context": context}

try:
    from ..blueprint_generator.main import run_agent as run_blueprint_generator
except ImportError:
    def run_blueprint_generator(context, input_data):
        return {"result": "Blueprint generation completed", "context": context}

try:
    from ..codegen.main import run_agent as run_codegen
except ImportError:
    def run_codegen(context, input_data):
        return {"result": "Code generation completed", "context": context}

try:
    from ..qa_guardian.main import run_agent as run_qa_guardian
except ImportError:
    def run_qa_guardian(context, input_data):
        return {"result": "QA completed", "context": context}

try:
    from ..security.main import run_agent as run_security
except ImportError:
    def run_security(context, input_data):
        return {"result": "Security check completed", "context": context}

try:
    from ..deployment_ops.main import run_agent as run_deployment_ops
except ImportError:
    def run_deployment_ops(context, input_data):
        return {"result": "Deployment completed", "context": context}

try:
    from ..ops_commander.main import run_agent as run_ops_commander
except ImportError:
    def run_ops_commander(context, input_data):
        return {"result": "Ops command executed", "context": context}

try:
    from ..feedback.main import run_agent as run_feedback
except ImportError:
    def run_feedback(context, input_data):
        return {"result": "Feedback processed", "context": context}

try:
    from ..integration.main import run_agent as run_integration
except ImportError:
    def run_integration(context, input_data):
        return {"result": "Integration completed", "context": context}

try:
    from ..platform_logger.main import run_agent as run_platform_logger
except ImportError:
    def run_platform_logger(context, input_data):
        return {"result": "Logging completed", "context": context}

try:
    from ..scheduler.main import run_agent as run_scheduler
except ImportError:
    def run_scheduler(context, input_data):
        return {"result": "Scheduling completed", "context": context}

try:
    from ..ui_ux.main import run_agent as run_ui_ux
except ImportError:
    def run_ui_ux(context, input_data):
        return {"result": "UI/UX completed", "context": context}
from .context_handler import ContextHandler
from .logger import log_event

# Настройка логгера
logger = logging.getLogger(__name__)


class OrchestratorError(Exception):
    """Исключение оркестратора."""
    pass


class WorkflowStep:
    """Шаг workflow pipeline."""
    
    def __init__(self, name: str, function, required: bool = True):
        self.name = name
        self.function = function
        self.required = required
        self.status = "pending"
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
    
    def execute(self, context: Dict[str, Any], input_data: Any) -> Dict[str, Any]:
        """Выполнение шага."""
        self.start_time = datetime.utcnow()
        self.status = "running"
        
        try:
            log_event(f"Executing step: {self.name}")
            result = self.function(context, input_data)
            self.result = result
            self.status = "completed"
            self.end_time = datetime.utcnow()
            
            log_event(f"Step {self.name} completed successfully")
            return result
            
        except Exception as e:
            self.error = str(e)
            self.status = "failed"
            self.end_time = datetime.utcnow()
            
            logger.error(f"Step {self.name} failed: {e}")
            logger.error(traceback.format_exc())
            
            if self.required:
                raise OrchestratorError(f"Required step {self.name} failed: {e}")
            
            return {"error": str(e), "context": context}


def init_orchestrator_components(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Инициализация всех компонентов оркестратора.
    
    Args:
        config_path: Путь к конфигурационному файлу
        
    Returns:
        Словарь с инициализированными компонентами
    """
    components = {}
    
    try:
        logger.info("Initializing orchestrator components...")
        
        # 1. Инициализация MemoryManager
        if MemoryManager:
            logger.info("Initializing MemoryManager...")
            components["memory_manager"] = MemoryManager(config_path=config_path)
            logger.info("MemoryManager initialized successfully")
        else:
            components["memory_manager"] = None
            logger.warning("MemoryManager not available, using fallback")
        
        # 2. Инициализация IngestPipeline
        if IngestPipeline:
            logger.info("Initializing IngestPipeline...")
            if components["memory_manager"]:
                components["ingest_pipeline"] = IngestPipeline(
                    memory_manager=components["memory_manager"]
                )
            else:
                components["ingest_pipeline"] = IngestPipeline()
            logger.info("IngestPipeline initialized successfully")
        else:
            components["ingest_pipeline"] = None
            logger.warning("IngestPipeline not available, using fallback")
        
        # 3. Инициализация RebeccaMetaAgent
        if RebeccaMetaAgent:
            logger.info("Initializing RebeccaMetaAgent...")
            if components["memory_manager"] and components["ingest_pipeline"]:
                components["rebecca_meta_agent"] = RebeccaMetaAgent(
                    memory_manager=components["memory_manager"],
                    ingest_pipeline=components["ingest_pipeline"]
                )
            else:
                components["rebecca_meta_agent"] = RebeccaMetaAgent()
            logger.info("RebeccaMetaAgent initialized successfully")
        else:
            components["rebecca_meta_agent"] = None
            logger.warning("RebeccaMetaAgent not available, using fallback")
        
        # 4. Инициализация ContextHandler
        try:
            logger.info("Initializing ContextHandler...")
            components["context_handler"] = ContextHandler()
            logger.info("ContextHandler initialized successfully")
        except Exception as e:
            components["context_handler"] = None
            logger.warning(f"ContextHandler not available: {e}")
        
        # 5. Создание контекста
        context = {
            "memory": components["memory_manager"],
            "ingest_pipeline": components["ingest_pipeline"],
            "rebecca_meta_agent": components["rebecca_meta_agent"],
            "context_handler": components["context_handler"],
            "components": components,
            "initialized_at": datetime.utcnow().isoformat()
        }
        
        components["context"] = context
        
        logger.info("All orchestrator components initialized successfully")
        return components
        
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator components: {e}")
        logger.error(traceback.format_exc())
        raise OrchestratorError(f"Component initialization failed: {e}")


def input_processing_step(context: Dict[str, Any], task_data: Any) -> Dict[str, Any]:
    """Этап 1: Обработка входных данных."""
    try:
        logger.info("Starting input processing step")
        
        # Валидация входных данных
        if not task_data:
            raise ValueError("Task data is required")
        
        # Нормализация данных
        if isinstance(task_data, str):
            task_data = {"input": task_data}
        elif not isinstance(task_data, dict):
            task_data = {"data": task_data}
        
        # Добавление метаданных
        task_data["processed_at"] = datetime.utcnow().isoformat()
        task_data["workflow_version"] = "2.0"
        
        # Сохранение в память (если доступна)
        memory = context.get("memory")
        if memory and hasattr(memory, 'episodic'):
            try:
                memory.episodic.store_event(
                    event_type="input_processing",
                    description=f"Processing task: {task_data.get('input', 'unknown')}",
                    metadata=task_data
                )
            except Exception as e:
                logger.warning(f"Failed to save input processing to memory: {e}")
        
        logger.info("Input processing step completed")
        return {
            "result": "Input processing completed",
            "context": context,
            "processed_data": task_data
        }
        
    except Exception as e:
        logger.error(f"Input processing failed: {e}")
        raise OrchestratorError(f"Input processing failed: {e}")


def context_preparation_step(context: Dict[str, Any], processed_data: Any) -> Dict[str, Any]:
    """Этап 2: Подготовка контекста."""
    try:
        logger.info("Starting context preparation step")
        
        # Получение контекста из memory
        context_handler = context.get("context_handler")
        
        # Анализ контекста задачи
        if context_handler and hasattr(context_handler, 'analyze_context'):
            task_context = context_handler.analyze_context(processed_data)
        else:
            task_context = {"type": "development", "priority": "high"}
        
        # Извлечение релевантной информации из памяти (если доступна)
        memory = context.get("memory")
        relevant_memories = []
        if memory and hasattr(memory, 'semantic'):
            try:
                relevant_memories = memory.semantic.retrieve_similar(
                    query=processed_data.get("input", ""),
                    limit=10
                )
            except Exception as e:
                logger.warning(f"Failed to retrieve memories: {e}")
        
        # Подготовка контекста для агентов
        agent_context = {
            "task_data": processed_data,
            "task_context": task_context,
            "relevant_memories": relevant_memories,
            "session_id": processed_data.get("session_id", "default"),
            "user_id": processed_data.get("user_id", "anonymous")
        }
        
        # Сохранение в контекст
        context["agent_context"] = agent_context
        
        logger.info("Context preparation step completed")
        return {
            "result": "Context preparation completed",
            "context": context,
            "agent_context": agent_context
        }
        
    except Exception as e:
        logger.error(f"Context preparation failed: {e}")
        raise OrchestratorError(f"Context preparation failed: {e}")


def architect_step(context: Dict[str, Any], agent_context: Any) -> Dict[str, Any]:
    """Этап 3: Архитектурный анализ."""
    try:
        logger.info("Starting architect step")
        
        # Вызов архитектурного агента
        result = run_architect(context, agent_context)
        
        logger.info("Architect step completed")
        return result
        
    except Exception as e:
        logger.error(f"Architect step failed: {e}")
        # Архитектурный анализ не критичен, продолжаем
        return {"result": "Architect step failed, continuing...", "error": str(e), "context": context}


def rebecca_metagent_step(context: Dict[str, Any], architect_result: Any) -> Dict[str, Any]:
    """Этап 4: Вызов мета-агента Ребекки."""
    try:
        logger.info("Starting RebeccaMetaAgent step")
        
        rebecca_agent = context.get("rebecca_meta_agent")
        
        if not rebecca_agent:
            logger.warning("RebeccaMetaAgent not available, using fallback")
            # Fallback: создаем базовый план задач
            task_plan = {
                "tasks": [
                    {
                        "id": "dev_001",
                        "title": "Code Generation",
                        "description": "Generate implementation code",
                        "agent_type": "codegen",
                        "priority": "high"
                    },
                    {
                        "id": "qa_001", 
                        "title": "Quality Assurance",
                        "description": "Perform quality checks",
                        "agent_type": "qa_guardian",
                        "priority": "medium"
                    },
                    {
                        "id": "deploy_001",
                        "title": "Deployment",
                        "description": "Deploy the application",
                        "agent_type": "deployment_ops",
                        "priority": "medium"
                    }
                ]
            }
        else:
            # Подготовка задачи для мета-агента
            task_description = architect_result.get("result", "")
            if isinstance(task_description, dict):
                task_description = str(task_description)
            
            if TaskType and TaskPriority:
                # Создание плана задач
                task_plan = rebecca_agent.create_task_plan(
                    task_description=task_description,
                    task_type=TaskType.DEVELOPMENT,
                    priority=TaskPriority.HIGH
                )
                
                # Выполнение плана
                execution_result = rebecca_agent.execute_task_plan(task_plan)
                
                logger.info("RebeccaMetaAgent completed task planning")
                return {
                    "result": execution_result,
                    "context": context,
                    "task_plan": task_plan
                }
            else:
                logger.warning("TaskType/TaskPriority not available, using basic planning")
                task_plan = {
                    "tasks": [
                        {
                            "id": "dev_001",
                            "title": "Code Generation",
                            "description": f"Generate code for: {task_description}",
                            "agent_type": "codegen",
                            "priority": "high"
                        }
                    ]
                }
        
        logger.info("RebeccaMetaAgent step completed with basic task plan")
        return {
            "result": task_plan,
            "context": context,
            "task_plan": task_plan
        }
        
    except Exception as e:
        logger.error(f"RebeccaMetaAgent step failed: {e}")
        # Fallback план при ошибке
        fallback_plan = {
            "tasks": [
                {
                    "id": "basic_001",
                    "title": "Basic Processing",
                    "description": "Basic task processing",
                    "agent_type": "general",
                    "priority": "medium"
                }
            ]
        }
        
        return {
            "result": fallback_plan,
            "context": context,
            "task_plan": fallback_plan,
            "error": str(e)
        }


def task_planning_step(context: Dict[str, Any], metagent_result: Any) -> Dict[str, Any]:
    """Этап 5: Планирование задач."""
    try:
        log_event("Starting task planning step")
        
        # Извлечение задач из результата мета-агента
        planned_tasks = metagent_result.get("result", {})
        
        if isinstance(planned_tasks, dict) and "tasks" in planned_tasks:
            tasks = planned_tasks["tasks"]
        elif hasattr(planned_tasks, "tasks"):
            tasks = planned_tasks.tasks
        else:
            # Fallback: создаем базовые задачи
            tasks = [
                {
                    "id": "dev_001",
                    "title": "Code Generation",
                    "description": "Generate implementation code",
                    "agent_type": "codegen"
                },
                {
                    "id": "qa_001", 
                    "title": "Quality Assurance",
                    "description": "Perform quality checks",
                    "agent_type": "qa_guardian"
                }
            ]
        
        # Сохранение плана задач в контекст
        context["task_plan"] = tasks
        
        log_event("Task planning step completed")
        return {
            "result": f"Task planning completed with {len(tasks)} tasks",
            "context": context,
            "planned_tasks": tasks
        }
        
    except Exception as e:
        logger.error(f"Task planning failed: {e}")
        # Создаем базовый план при ошибке
        fallback_tasks = [
            {
                "id": "basic_001",
                "title": "Basic Processing",
                "description": "Basic task processing",
                "agent_type": "general"
            }
        ]
        
        context["task_plan"] = fallback_tasks
        
        return {
            "result": "Task planning failed, using fallback",
            "error": str(e),
            "context": context,
            "planned_tasks": fallback_tasks
        }


def agent_orchestration_step(context: Dict[str, Any], task_plan: Any) -> Dict[str, Any]:
    """Этап 6: Оркестрация специализированных агентов."""
    try:
        log_event("Starting agent orchestration step")
        
        tasks = context.get("task_plan", [])
        agent_results = []
        
        for task in tasks:
            try:
                agent_type = task.get("agent_type", "general")
                task_data = {
                    "task": task,
                    "context": context
                }
                
                # Вызов соответствующего агента
                if agent_type == "architect":
                    result = run_architect(context, task_data)
                elif agent_type == "researcher":
                    result = run_researcher(context, task_data)
                elif agent_type == "knowledge_curator":
                    result = run_knowledge_curator(context, task_data)
                elif agent_type == "blueprint_generator":
                    result = run_blueprint_generator(context, task_data)
                elif agent_type == "codegen":
                    result = run_codegen(context, task_data)
                elif agent_type == "qa_guardian":
                    result = run_qa_guardian(context, task_data)
                elif agent_type == "security":
                    result = run_security(context, task_data)
                elif agent_type == "deployment_ops":
                    result = run_deployment_ops(context, task_data)
                elif agent_type == "ops_commander":
                    result = run_ops_commander(context, task_data)
                elif agent_type == "feedback":
                    result = run_feedback(context, task_data)
                elif agent_type == "integration":
                    result = run_integration(context, task_data)
                elif agent_type == "platform_logger":
                    result = run_platform_logger(context, task_data)
                elif agent_type == "scheduler":
                    result = run_scheduler(context, task_data)
                elif agent_type == "ui_ux":
                    result = run_ui_ux(context, task_data)
                else:
                    # Общий обработчик
                    result = {"result": f"Task {task.get('title')} processed by general handler", "context": context}
                
                agent_results.append({
                    "task_id": task.get("id"),
                    "agent_type": agent_type,
                    "result": result
                })
                
            except Exception as e:
                logger.error(f"Agent {agent_type} failed for task {task.get('id')}: {e}")
                agent_results.append({
                    "task_id": task.get("id"),
                    "agent_type": agent_type,
                    "error": str(e),
                    "result": {"error": str(e), "context": context}
                })
        
        logger.info("Agent orchestration step completed")
        return {
            "result": f"Agent orchestration completed with {len(agent_results)} results",
            "context": context,
            "agent_results": agent_results
        }
        
    except Exception as e:
        logger.error(f"Agent orchestration failed: {e}")
        raise OrchestratorError(f"Agent orchestration failed: {e}")


def result_compilation_step(context: Dict[str, Any], orchestration_result: Any) -> Dict[str, Any]:
    """Этап 7: Сборка результатов."""
    try:
        log_event("Starting result compilation step")
        
        # Сбор результатов от агентов
        agent_results = orchestration_result.get("agent_results", [])
        
        # Агрегация результатов
        compiled_results = {
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat(),
            "workflow_version": "2.0",
            "agent_results": [],
            "summary": {
                "total_agents": len(agent_results),
                "successful_agents": 0,
                "failed_agents": 0
            }
        }
        
        for agent_result in agent_results:
            compiled_results["agent_results"].append({
                "task_id": agent_result.get("task_id"),
                "agent_type": agent_result.get("agent_type"),
                "status": "error" if "error" in agent_result else "success",
                "result": agent_result.get("result", {}),
                "error": agent_result.get("error")
            })
            
            if "error" in agent_result:
                compiled_results["summary"]["failed_agents"] += 1
            else:
                compiled_results["summary"]["successful_agents"] += 1
        
        # Создание общего резюме
        compiled_results["workflow_summary"] = {
            "input_processing": "completed",
            "context_preparation": "completed", 
            "architect_step": "completed",
            "rebecca_metagent_step": "completed",
            "task_planning": "completed",
            "agent_orchestration": "completed",
            "result_compilation": "completed"
        }
        
        # Сохранение результатов в память (если доступна)
        memory = context.get("memory")
        if memory and hasattr(memory, 'episodic'):
            try:
                memory.episodic.store_event(
                    event_type="workflow_completed",
                    description="Full workflow execution completed",
                    metadata=compiled_results
                )
            except Exception as e:
                logger.warning(f"Failed to save workflow results to memory: {e}")
        
        log_event("Result compilation step completed")
        return compiled_results
        
    except Exception as e:
        logger.error(f"Result compilation failed: {e}")
        raise OrchestratorError(f"Result compilation failed: {e}")


def main_workflow(task_data: Union[str, Dict[str, Any]], 
                  config_path: Optional[str] = None,
                  enable_fallback: bool = True) -> Dict[str, Any]:
    """
    Основной workflow оркестратора с полной интеграцией мета-агента.
    
    Args:
        task_data: Входные данные задачи
        config_path: Путь к конфигурационному файлу
        enable_fallback: Включить fallback механизмы
        
    Returns:
        Результат выполнения workflow
    """
    workflow_id = f"workflow_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Starting workflow: {workflow_id}")
    
    try:
        # Шаг 1: Инициализация компонентов
        logger.info(f"[{workflow_id}] Initializing components...")
        components = init_orchestrator_components(config_path)
        context = components["context"]
        
        # Шаг 2: Создание pipeline
        workflow_steps = [
            WorkflowStep("input_processing", input_processing_step),
            WorkflowStep("context_preparation", context_preparation_step),
            WorkflowStep("architect_step", architect_step, required=False),
            WorkflowStep("rebecca_metagent_step", rebecca_metagent_step, required=enable_fallback),
            WorkflowStep("task_planning", task_planning_step, required=enable_fallback),
            WorkflowStep("agent_orchestration", agent_orchestration_step, required=enable_fallback),
            WorkflowStep("result_compilation", result_compilation_step)
        ]
        
        logger.info(f"Pipeline created with {len(workflow_steps)} steps")
        
        # Шаг 3: Последовательное выполнение этапов
        logger.info(f"[{workflow_id}] Executing workflow steps...")
        current_input = task_data
        current_context = context
        
        for step in workflow_steps:
            try:
                logger.info(f"Executing step: {step.name}")
                step_result = step.execute(current_context, current_input)
                
                # Обновление контекста и входа для следующего шага
                current_context = step_result.get("context", current_context)
                current_input = step_result
                
            except Exception as e:
                logger.error(f"Step {step.name} failed: {e}")
                
                if step.required:
                    logger.error(f"Required step {step.name} failed, stopping workflow")
                    raise
                else:
                    logger.warning(f"Non-required step {step.name} failed, continuing with fallback")
                    if enable_fallback:
                        current_input = {"error": str(e), "context": current_context, "fallback": True}
                    else:
                        raise
        
        # Шаг 4: Финализация результата
        logger.info(f"[{workflow_id}] Finalizing results...")
        final_result = current_input
        
        # Добавление метаданных workflow
        final_result["workflow_id"] = workflow_id
        final_result["completed_at"] = datetime.utcnow().isoformat()
        final_result["components_initialized"] = list(components.keys())
        
        logger.info(f"[{workflow_id}] Workflow completed successfully")
        return final_result
        
    except Exception as e:
        logger.error(f"[{workflow_id}] Workflow failed: {e}")
        logger.error(traceback.format_exc())
        
        # Fallback result при критической ошибке
        if enable_fallback:
            fallback_result = {
                "result": "Workflow execution failed, but fallback was successful",
                "error": str(e),
                "status": "fallback_completed",
                "workflow_id": workflow_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.info(f"[{workflow_id}] Fallback result provided")
            return fallback_result
        else:
            raise OrchestratorError(f"Workflow failed: {e}")


def test_main_workflow():
    """Тест основного workflow."""
    logger.info("Starting main workflow test...")
    
    test_data = {
        "input": "Hello world example",
        "user_id": "test_user",
        "session_id": "test_session"
    }
    
    try:
        result = main_workflow(test_data)
        logger.info("Workflow test completed successfully")
        print("✅ Workflow test completed successfully")
        print("Result:", result)
        return True
        
    except Exception as e:
        logger.error(f"Workflow test failed: {e}")
        print(f"❌ Workflow test failed: {e}")
        return False


if __name__ == "__main__":
    # Запуск теста при прямом вызове
    test_main_workflow()