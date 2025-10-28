"""Meta-Orchestrator scaffold package for Rebecca-Platform."""

from .main_loop import MetaOrchestratorLoop
from .task_manager import TaskManager
from .context_handler import ContextHandler
from .messaging import MessagingClient

# Основной workflow с интеграцией мета-агента
from .main_workflow import (
    main_workflow,
    init_orchestrator_components,
    input_processing_step,
    context_preparation_step,
    architect_step,
    rebecca_metagent_step,
    task_planning_step,
    agent_orchestration_step,
    result_compilation_step,
    WorkflowStep,
    OrchestratorError
)

__all__ = [
    "MetaOrchestratorLoop",
    "TaskManager", 
    "ContextHandler",
    "MessagingClient",
    # Новые функции workflow
    "main_workflow",
    "init_orchestrator_components",
    "input_processing_step",
    "context_preparation_step", 
    "architect_step",
    "rebecca_metagent_step",
    "task_planning_step",
    "agent_orchestration_step",
    "result_compilation_step",
    "WorkflowStep",
    "OrchestratorError"
]
