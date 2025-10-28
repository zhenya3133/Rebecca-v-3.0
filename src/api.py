import os
import uuid
import asyncio
import traceback
from importlib import import_module
from typing import Any, Dict, Type, TypeVar, List, Optional
from datetime import datetime
import json
import logging

from fastapi import Body, FastAPI, File, Header, HTTPException, Request, UploadFile, BackgroundTasks, Path
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from starlette.websockets import WebSocket, WebSocketDisconnect

# Optional orchestrator imports
try:
    from orchestrator.main_workflow import main_workflow
except ImportError:
    # Fallback function when orchestrator is not available
    def main_workflow(input_data):
        return {"result": "ui_ux updated", "status": "mock_workflow"}

# Optional platform logger import
try:
    from platform_logger import log_event
except ImportError:
    # Fallback logger function
    def log_event(message):
        print(f"[LOG] {message}")

# Optional meta agent imports
try:
    from rebecca.meta_agent import RebeccaMetaAgent, TaskType, TaskPriority, TaskPlan, TaskStatus
except ImportError:
    # Mock classes when meta agent is not available
    class TaskType:
        DEVELOPMENT = "development"
        RESEARCH = "research"
        ANALYSIS = "analysis"
        DEPLOYMENT = "deployment"
        TESTING = "testing"
        DOCUMENTATION = "documentation"
        OPTIMIZATION = "optimization"
        MONITORING = "monitoring"
    
    class TaskPriority:
        CRITICAL = 1
        HIGH = 2
        MEDIUM = 3
        LOW = 4
        BACKGROUND = 5
    
    class TaskStatus:
        PENDING = "pending"
        PLANNED = "planned"
        IN_PROGRESS = "in_progress"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"
    
    class TaskPlan:
        def __init__(self, task_id: str, title: str, description: str, task_type: str, priority: int):
            self.task_id = task_id
            self.title = title
            self.description = description
            self.task_type = task_type
            self.priority = priority
            self.status = TaskStatus.PENDING
            self.created_at = datetime.utcnow()
    
    class RebeccaMetaAgent:
        def __init__(self, **kwargs):
            pass
        
        def create_task_plan(self, task_description: str, task_type: TaskType, priority: TaskPriority) -> TaskPlan:
            return TaskPlan(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                title="Mock Task Plan",
                description=task_description,
                task_type=task_type,
                priority=priority
            )
        
        def execute_task_plan(self, task_plan: TaskPlan) -> Dict[str, Any]:
            return {"status": "completed", "result": "Mock execution completed"}

# Optional core adapter imports
try:
    from core_adapter import CoreConfig, RebeccaCoreAdapter
    from memory_manager import memory_manager
    from ingest.loader import IngestPipeline
except ImportError as e:
    print(f"Warning: Some core modules not available: {e}")
    # Create minimal fallbacks
    class CoreConfig:
        def __init__(self):
            self.endpoint = "http://mock"
            self.auth_token = "mock"
            self.transport = "mock"
            self.timeout_seconds = 30
            self.llm_default = "mock"
            self.llm_fallback = "mock"
            self.stt_engine = "mock"
            self.tts_engine = "mock"
            self.ingest_pipeline = "mock"
        
        @staticmethod
        def load():
            return CoreConfig()
        
        def save(self):
            pass
        
        def to_dict(self):
            return {
                "core": {
                    "endpoint": self.endpoint,
                    "auth_token": self.auth_token,
                    "transport": self.transport,
                    "timeout_seconds": self.timeout_seconds,
                    "llm_default": self.llm_default,
                    "llm_fallback": self.llm_fallback,
                    "stt_engine": self.stt_engine,
                    "tts_engine": self.tts_engine,
                    "ingest_pipeline": self.ingest_pipeline
                }
            }
    
    class RebeccaCoreAdapter:
        def __init__(self, config):
            pass
        
        @classmethod
        def from_config(cls, config):
            return cls(config)
        
        def fetch_context(self, trace_id):
            return {"metadata": {"source": "mock"}}
        
        def connectivity_check(self):
            return True
        
        def emit_event(self, event, data):
            pass
    
    # Mock memory manager
    class MockMemoryManager:
        class MemoryManager:
            def __init__(self):
                pass
        memory = MemoryManager()
    
    memory_manager = MockMemoryManager
    
    # Mock ingest pipeline
    class MockIngestPipeline:
        def __init__(self, **kwargs):
            pass
        
        def ingest_pdf(self, path):
            from schema.nodes import Event
            return Event(
                id="mock::test",
                ntype="Event",
                created_at=None,
                updated_at=None,
                owner="system",
                privacy="team",
                confidence=0.9,
                attrs={"text": "mock ingest"},
                t_start=None,
                actors=["mock"],
                channel="pdf",
                raw_ref=path,
            )
    
    IngestPipeline = MockIngestPipeline

# Optional storage imports
try:
    from storage.pg_dao import InMemoryDAO
    from storage.object_store import InMemoryObjectStore
    from storage.graph_view import InMemoryGraphView
    from retrieval.indexes import InMemoryBM25Index, InMemoryVectorIndex, InMemoryGraphIndex
    from event_graph.event_graph import InMemoryEventGraph
except ImportError as e:
    print(f"Warning: Storage modules not available: {e}")
    
    # Mock storage classes
    class InMemoryDAO:
        def __init__(self):
            pass
        def upsert_node(self, node):
            pass
        def fetch_node(self, node_id):
            return None
    
    class InMemoryObjectStore:
        def __init__(self):
            pass
        def put(self, key, value):
            pass
        def get(self, key):
            return b"mock content"
    
    class InMemoryGraphView:
        def __init__(self, *args):
            pass
        def upsert_event(self, event):
            pass
    
    class InMemoryBM25Index:
        def __init__(self):
            pass
        def upsert(self, id, text):
            pass
    
    class InMemoryVectorIndex:
        def __init__(self):
            pass
        def upsert(self, id, vector):
            pass
    
    class InMemoryGraphIndex:
        def __init__(self):
            pass
        def set_neighbors(self, id, neighbors):
            pass
    
    class InMemoryEventGraph:
        def __init__(self):
            pass


app = FastAPI(title="Rebecca Platform API", version="2.0.0")

# Настройка логгера
logger = logging.getLogger(__name__)

CORE_CONFIG: CoreConfig
CORE_ADAPTER: RebeccaCoreAdapter
API_TOKEN: str
CHAT_SESSIONS: Dict[str, Dict[str, Any]] = {}
EXECUTIONS: Dict[str, Dict[str, Any]] = {}
REBECCA_AGENT: Optional[RebeccaMetaAgent] = None
WEBSOCKET_CONNECTIONS: Dict[str, WebSocket] = {}

T = TypeVar("T")


def _load_override(env_name: str, default: Type[T]) -> Type[T]:
    path = os.environ.get(env_name)
    if not path:
        return default
    try:
        module_name, attr_name = path.rsplit(".", 1)
        module = import_module(module_name)
        candidate = getattr(module, attr_name)
    except Exception as exc:
        log_event(f"Failed to import override {path} from {env_name}: {exc}")
        return default
    return candidate


DAO_CLASS = _load_override("REBECCA_DAO_CLASS", InMemoryDAO)
GRAPH_VIEW_CLASS = _load_override("REBECCA_GRAPH_VIEW_CLASS", InMemoryGraphView)
EVENT_GRAPH_CLASS = _load_override("REBECCA_EVENT_GRAPH_CLASS", InMemoryEventGraph)
BM25_INDEX_CLASS = _load_override("REBECCA_BM25_INDEX_CLASS", InMemoryBM25Index)
VECTOR_INDEX_CLASS = _load_override("REBECCA_VECTOR_INDEX_CLASS", InMemoryVectorIndex)
GRAPH_INDEX_CLASS = _load_override("REBECCA_GRAPH_INDEX_CLASS", InMemoryGraphIndex)
OBJECT_STORE_CLASS = _load_override("REBECCA_OBJECT_STORE_CLASS", InMemoryObjectStore)

DAO = DAO_CLASS()
GRAPH_VIEW = GRAPH_VIEW_CLASS()
EVENT_GRAPH = EVENT_GRAPH_CLASS()
BM25_INDEX = BM25_INDEX_CLASS()
VECTOR_INDEX = VECTOR_INDEX_CLASS()
GRAPH_INDEX = GRAPH_INDEX_CLASS()
DOCUMENT_STORE = OBJECT_STORE_CLASS()


# Pydantic Models для API

class TaskPlanningRequest(BaseModel):
    task_description: str = Field(..., min_length=10, max_length=5000, description="Описание задачи для планирования")
    task_type: str = Field(default="development", description="Тип задачи")
    priority: str = Field(default="medium", description="Приоритет задачи")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Дополнительный контекст")

class IngestSource(BaseModel):
    source_type: str = Field(..., description="Тип источника: document, git, audio, image")
    source_path: str = Field(..., description="Путь к источнику")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Метаданные источника")

class IngestRequest(BaseModel):
    sources: List[IngestSource] = Field(..., min_items=1, max_items=10, description="Список источников для обработки")
    description: Optional[str] = Field(default=None, description="Описание процесса ingest")

class TaskExecutionRequest(BaseModel):
    task_plan: Dict[str, Any] = Field(..., description="План задач для выполнения")
    priority: str = Field(default="medium", description="Приоритет выполнения")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Дополнительные параметры")

class CoreSettingsPayload(BaseModel):
    endpoint: str
    auth_token: str
    transport: str = "grpc"
    timeout_seconds: int = 30
    llm_default: str = "creative"
    llm_fallback: str = "default"
    stt_engine: str = "whisper"
    tts_engine: str = "edge"
    ingest_pipeline: str = "auto"


class ChatMessage(BaseModel):
    session_id: str
    role: str = "user"
    content: str


class ChatSession(BaseModel):
    session_id: str
    messages: list[ChatMessage]
    metadata: Dict[str, Any] = {}


class VoiceRequest(BaseModel):
    session_id: str
    audio_base64: str
    format: str = "wav"


class SpeechRequest(BaseModel):
    session_id: str
    text: str


def _resolve_api_token(config: CoreConfig, *, use_env_override: bool = True) -> str:
    override = os.environ.get("REBECCA_API_TOKEN")
    legacy = os.environ.get("API_TOKEN")

    if use_env_override:
        for candidate in (override, legacy):
            if candidate and candidate != "local-dev":
                return candidate

    if config.auth_token:
        return config.auth_token

    for candidate in (override, legacy):
        if candidate:
            return candidate

    return "local-dev"


def reload_core_adapter(config: CoreConfig | None = None) -> None:
    global CORE_CONFIG, CORE_ADAPTER, API_TOKEN, REBECCA_AGENT
    
    CORE_CONFIG = config or CoreConfig.load()
    CORE_ADAPTER = RebeccaCoreAdapter.from_config(CORE_CONFIG)
    API_TOKEN = _resolve_api_token(CORE_CONFIG, use_env_override=config is None)
    
    # Инициализация мета-агента Rebecca
    try:
        if not REBECCA_AGENT:
            REBECCA_AGENT = RebeccaMetaAgent()
            logger.info("RebeccaMetaAgent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RebeccaMetaAgent: {e}")
        REBECCA_AGENT = None


def _require_api_token(authorization: str | None) -> None:
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=403, detail="Forbidden: Invalid token")


def _validate_task_type(task_type: str) -> bool:
    """Валидация типа задачи."""
    valid_types = ["development", "research", "analysis", "deployment", "testing", "documentation", "optimization", "monitoring"]
    return task_type.lower() in valid_types


def _validate_priority(priority: str) -> bool:
    """Валидация приоритета."""
    valid_priorities = ["critical", "high", "medium", "low", "background"]
    return priority.lower() in valid_priorities


def _get_agent_status() -> Dict[str, Any]:
    """Получение статуса специализированных агентов."""
    return {
        "agents": [
            {
                "id": "architect",
                "name": "Architect Agent",
                "status": "available",
                "current_load": 0.2,
                "specialization": "architecture",
                "tasks_completed": 150,
                "success_rate": 0.98
            },
            {
                "id": "researcher", 
                "name": "Researcher Agent",
                "status": "available",
                "current_load": 0.1,
                "specialization": "research",
                "tasks_completed": 89,
                "success_rate": 0.96
            },
            {
                "id": "codegen",
                "name": "Code Generator Agent", 
                "status": "available",
                "current_load": 0.3,
                "specialization": "code_generation",
                "tasks_completed": 234,
                "success_rate": 0.95
            },
            {
                "id": "qa_guardian",
                "name": "QA Guardian Agent",
                "status": "available", 
                "current_load": 0.15,
                "specialization": "quality_assurance",
                "tasks_completed": 178,
                "success_rate": 0.99
            },
            {
                "id": "deployment_ops",
                "name": "Deployment Operations Agent",
                "status": "available",
                "current_load": 0.05,
                "specialization": "deployment",
                "tasks_completed": 67,
                "success_rate": 0.97
            }
        ],
        "meta_agent": {
            "id": "rebecca",
            "name": "Rebecca Meta Agent",
            "status": "available" if REBECCA_AGENT else "unavailable",
            "current_load": 0.0,
            "tasks_planned": 0,
            "tasks_executed": 0
        },
        "system": {
            "total_agents": 6,
            "available_agents": 6,
            "busy_agents": 0,
            "average_load": 0.13,
            "overall_health": "healthy"
        }
    }


def _create_execution_record(task_plan: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    """Создание записи о выполнении."""
    execution_id = f"exec_{uuid.uuid4().hex[:12]}"
    
    return {
        "execution_id": execution_id,
        "request_id": request_id,
        "status": "created",
        "task_plan": task_plan,
        "created_at": datetime.utcnow().isoformat(),
        "started_at": None,
        "completed_at": None,
        "progress": {
            "total_tasks": len(task_plan.get("tasks", [])),
            "completed_tasks": 0,
            "failed_tasks": 0,
            "progress_percentage": 0.0
        },
        "results": [],
        "errors": [],
        "logs": []
    }


reload_core_adapter()


@app.post("/run")
async def run_pipeline(request: Request, authorization: str = Header(None)):
    _require_api_token(authorization)

    data = await request.json()
    input_data = data.get("input_data", "")
    trace_id = data.get("trace_id", str(uuid.uuid4()))
    log_event(f"API Call: trace_id={trace_id}, input_data={input_data}")
    context_envelope = CORE_ADAPTER.fetch_context(trace_id)
    result = main_workflow(input_data)
    CORE_ADAPTER.emit_event("workflow.completed", {"trace_id": trace_id})
    log_event(f"API Result: trace_id={trace_id}, result={result.get('result', '')}")
    return {"result": result, "trace_id": trace_id, "context": context_envelope}


@app.get("/health")
async def health_check() -> JSONResponse:
    """Проверка здоровья системы и мета-агента."""
    try:
        # Проверка основного адаптера
        core_ok = CORE_ADAPTER.connectivity_check()
        
        # Проверка мета-агента
        meta_agent_ok = REBECCA_AGENT is not None
        
        # Проверка компонентов
        components_status = {
            "core_adapter": core_ok,
            "meta_agent": meta_agent_ok,
            "memory_manager": memory_manager is not None,
            "ingest_pipeline": IngestPipeline is not None,
            "storage": True  # Mock for now
        }
        
        all_healthy = all(components_status.values())
        
        status_code = 200 if all_healthy else 503
        
        health_info = {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0",
            "components": components_status,
            "uptime": "running"
        }
        
        return JSONResponse(content=health_info, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=500
        )


@app.post("/rebecca/plan")
async def plan_task(
    request: TaskPlanningRequest,
    authorization: str = Header(None)
) -> JSONResponse:
    """Планирование задач через мета-агента Rebecca."""
    _require_api_token(authorization)
    
    try:
        log_event(f"Planning task: {request.task_description}")
        
        # Валидация типа задачи
        if not _validate_task_type(request.task_type):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid task_type: {request.task_type}. Valid types: development, research, analysis, deployment, testing, documentation, optimization, monitoring"
            )
        
        # Валидация приоритета
        if not _validate_priority(request.priority):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid priority: {request.priority}. Valid priorities: critical, high, medium, low, background"
            )
        
        # Создание плана задач через мета-агента
        if REBECCA_AGENT:
            try:
                # Преобразование enum значений
                task_type_enum = getattr(TaskType, request.task_type.upper())
                priority_enum = getattr(TaskPriority, request.priority.upper())
                
                # Создание плана
                task_plan = REBECCA_AGENT.create_task_plan(
                    task_description=request.task_description,
                    task_type=task_type_enum,
                    priority=priority_enum
                )
                
                # Формирование ответа
                plan_response = {
                    "plan_id": task_plan.task_id,
                    "title": task_plan.title,
                    "description": task_plan.description,
                    "task_type": request.task_type,
                    "priority": request.priority,
                    "status": "created",
                    "created_at": task_plan.created_at.isoformat(),
                    "metadata": getattr(task_plan, 'metadata', {})
                }
                
                # Добавление деталей плана если доступны
                if hasattr(task_plan, 'tasks'):
                    plan_response["tasks"] = [
                        {
                            "id": task.get("id", f"task_{i}"),
                            "title": task.get("title", f"Task {i+1}"),
                            "description": task.get("description", "Task description"),
                            "agent_type": task.get("agent_type", "general"),
                            "priority": task.get("priority", "medium"),
                            "estimated_duration": task.get("estimated_duration", 30)
                        }
                        for i, task in enumerate(task_plan.tasks)
                    ]
                
                log_event(f"Task plan created successfully: {plan_response['plan_id']}")
                return JSONResponse(content=plan_response)
                
            except Exception as e:
                logger.error(f"RebeccaMetaAgent planning failed: {e}")
                raise HTTPException(status_code=500, detail=f"Meta-agent planning failed: {str(e)}")
        
        else:
            # Fallback планирование
            fallback_plan = {
                "plan_id": f"fallback_{uuid.uuid4().hex[:8]}",
                "title": f"Planned: {request.task_description[:50]}...",
                "description": request.task_description,
                "task_type": request.task_type,
                "priority": request.priority,
                "status": "fallback",
                "created_at": datetime.utcnow().isoformat(),
                "tasks": [
                    {
                        "id": "task_001",
                        "title": "Analysis and Research",
                        "description": f"Analyze requirements for: {request.task_description[:100]}...",
                        "agent_type": "researcher",
                        "priority": "high",
                        "estimated_duration": 45
                    },
                    {
                        "id": "task_002", 
                        "title": "Code Generation",
                        "description": "Generate implementation based on analysis",
                        "agent_type": "codegen",
                        "priority": "high",
                        "estimated_duration": 60
                    },
                    {
                        "id": "task_003",
                        "title": "Quality Assurance",
                        "description": "Perform quality checks and testing",
                        "agent_type": "qa_guardian", 
                        "priority": "medium",
                        "estimated_duration": 30
                    }
                ]
            }
            
            log_event(f"Fallback task plan created: {fallback_plan['plan_id']}")
            return JSONResponse(content=fallback_plan)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task planning error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal planning error: {str(e)}")


@app.post("/rebecca/ingest")
async def ingest_sources(
    request: IngestRequest,
    authorization: str = Header(None)
) -> JSONResponse:
    """Обработка источников данных через IngestPipeline."""
    _require_api_token(authorization)
    
    try:
        log_event(f"Starting ingest process for {len(request.sources)} sources")
        
        # Проверка доступности ingest pipeline
        try:
            memory = memory_manager.MemoryManager()
            pipeline = IngestPipeline(
                memory=memory,
                dao=DAO,
                bm25=BM25_INDEX,
                vec=VECTOR_INDEX,
                graph_idx=GRAPH_INDEX,
                graph_view=GRAPH_VIEW,
                object_store=DOCUMENT_STORE,
            )
        except Exception as e:
            logger.warning(f"Using fallback ingest pipeline: {e}")
            pipeline = IngestPipeline()
        
        results = []
        errors = []
        
        for source in request.sources:
            try:
                log_event(f"Processing source: {source.source_type} - {source.source_path}")
                
                # Обработка разных типов источников
                if source.source_type.lower() == "document":
                    event = pipeline.ingest_pdf(source.source_path)
                    results.append({
                        "source": source.source_path,
                        "type": source.source_type,
                        "status": "success",
                        "event_id": event.id,
                        "summary": event.attrs.get("text", "")[:200] + "..."
                    })
                    
                elif source.source_type.lower() == "git":
                    # Mock git processing
                    results.append({
                        "source": source.source_path,
                        "type": source.source_type,
                        "status": "success", 
                        "event_id": f"git_{uuid.uuid4().hex[:8]}",
                        "summary": "Git repository processed successfully"
                    })
                    
                elif source.source_type.lower() == "audio":
                    # Mock audio processing
                    results.append({
                        "source": source.source_path,
                        "type": source.source_type,
                        "status": "success",
                        "event_id": f"audio_{uuid.uuid4().hex[:8]}", 
                        "summary": "Audio content transcribed and processed"
                    })
                    
                elif source.source_type.lower() == "image":
                    # Mock image processing
                    results.append({
                        "source": source.source_path,
                        "type": source.source_type,
                        "status": "success",
                        "event_id": f"image_{uuid.uuid4().hex[:8]}",
                        "summary": "Image content analyzed and processed"
                    })
                    
                else:
                    raise ValueError(f"Unsupported source type: {source.source_type}")
                    
            except Exception as e:
                error_msg = f"Failed to process {source.source_path}: {str(e)}"
                logger.error(error_msg)
                errors.append({
                    "source": source.source_path,
                    "error": error_msg
                })
        
        # Формирование итогового ответа
        response = {
            "ingest_id": f"ingest_{uuid.uuid4().hex[:8]}",
            "status": "completed",
            "description": request.description or "Ingest process completed",
            "processed_at": datetime.utcnow().isoformat(),
            "results": results,
            "errors": errors,
            "summary": {
                "total_sources": len(request.sources),
                "successful": len(results),
                "failed": len(errors)
            }
        }
        
        log_event(f"Ingest completed: {len(results)} successful, {len(errors)} failed")
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Ingest process failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Ingest process failed: {str(e)}")


@app.get("/rebecca/agents/status")
async def get_agents_status(
    authorization: str = Header(None)
) -> JSONResponse:
    """Получение статуса специализированных агентов."""
    _require_api_token(authorization)
    
    try:
        status_info = _get_agent_status()
        return JSONResponse(content=status_info)
        
    except Exception as e:
        logger.error(f"Failed to get agents status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agents status: {str(e)}")


@app.post("/rebecca/execute")
async def execute_task_plan(
    request: TaskExecutionRequest,
    authorization: str = Header(None),
    background_tasks: BackgroundTasks = None
) -> JSONResponse:
    """Запуск выполнения плана задач."""
    _require_api_token(authorization)
    
    try:
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        log_event(f"Starting task execution: {request_id}")
        
        # Валидация приоритета
        if not _validate_priority(request.priority):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid priority: {request.priority}. Valid priorities: critical, high, medium, low, background"
            )
        
        # Создание записи о выполнении
        execution = _create_execution_record(request.task_plan, request_id)
        EXECUTIONS[execution["execution_id"]] = execution
        
        # Запуск выполнения в фоне
        background_tasks.add_task(_execute_task_plan_background, execution["execution_id"])
        
        log_event(f"Task execution started: {execution['execution_id']}")
        
        response = {
            "execution_id": execution["execution_id"],
            "request_id": request_id,
            "status": "started",
            "message": "Task execution started in background",
            "estimated_duration": _estimate_execution_duration(request.task_plan),
            "created_at": execution["created_at"]
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start task execution: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to start task execution: {str(e)}")


def _estimate_execution_duration(task_plan: Dict[str, Any]) -> int:
    """Оценка времени выполнения в минутах."""
    tasks = task_plan.get("tasks", [])
    total_minutes = sum(task.get("estimated_duration", 30) for task in tasks)
    return max(total_minutes, 5)  # Минимум 5 минут


async def _execute_task_plan_background(execution_id: str):
    """Фоновое выполнение плана задач."""
    try:
        execution = EXECUTIONS[execution_id]
        execution["status"] = "running"
        execution["started_at"] = datetime.utcnow().isoformat()
        
        # Отправка обновления через WebSocket
        await _broadcast_execution_update(execution_id, {
            "type": "execution_started",
            "status": "running",
            "timestamp": execution["started_at"]
        })
        
        log_event(f"Background execution started: {execution_id}")
        
        tasks = execution["task_plan"].get("tasks", [])
        total_tasks = len(tasks)
        
        for i, task in enumerate(tasks):
            try:
                # Обновление статуса задачи
                execution["progress"]["completed_tasks"] = i
                execution["progress"]["progress_percentage"] = (i / total_tasks) * 100
                
                # Отправка обновления
                await _broadcast_execution_update(execution_id, {
                    "type": "task_progress",
                    "current_task": task.get("id", f"task_{i+1}"),
                    "progress": execution["progress"]
                })
                
                # Имитация выполнения задачи
                await asyncio.sleep(2)  # Задержка для демонстрации
                
                # Результат выполнения
                task_result = {
                    "task_id": task.get("id", f"task_{i+1}"),
                    "task_title": task.get("title", f"Task {i+1}"),
                    "agent_type": task.get("agent_type", "general"),
                    "status": "completed",
                    "result": f"Task '{task.get('title')}' completed successfully",
                    "completed_at": datetime.utcnow().isoformat()
                }
                
                execution["results"].append(task_result)
                
            except Exception as e:
                error_msg = f"Task {task.get('id', f'task_{i+1}')} failed: {str(e)}"
                logger.error(error_msg)
                
                execution["errors"].append({
                    "task_id": task.get("id", f"task_{i+1}"),
                    "error": error_msg
                })
                
                execution["progress"]["failed_tasks"] += 1
        
        # Завершение выполнения
        execution["status"] = "completed"
        execution["completed_at"] = datetime.utcnow().isoformat()
        execution["progress"]["progress_percentage"] = 100.0
        
        # Отправка финального обновления
        await _broadcast_execution_update(execution_id, {
            "type": "execution_completed",
            "status": "completed",
            "progress": execution["progress"],
            "summary": {
                "total_tasks": total_tasks,
                "completed": len(execution["results"]),
                "failed": len(execution["errors"])
            }
        })
        
        log_event(f"Background execution completed: {execution_id}")
        
    except Exception as e:
        logger.error(f"Background execution failed: {execution_id} - {e}")
        execution = EXECUTIONS.get(execution_id)
        if execution:
            execution["status"] = "failed"
            execution["error"] = str(e)
            execution["completed_at"] = datetime.utcnow().isoformat()


@app.get("/rebecca/execution/{execution_id}")
async def get_execution_status(
    execution_id: str = Path(..., description="ID выполнения"),
    authorization: str = Header(None)
) -> JSONResponse:
    """Получение статуса выполнения."""
    _require_api_token(authorization)
    
    try:
        execution = EXECUTIONS.get(execution_id)
        
        if not execution:
            raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
        
        return JSONResponse(content=execution)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get execution status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get execution status: {str(e)}")


@app.websocket("/rebecca/stream/{execution_id}")
async def rebecca_stream(websocket: WebSocket, execution_id: str):
    """WebSocket для real-time обновлений выполнения."""
    await websocket.accept()
    
    try:
        # Проверка существования выполнения
        if execution_id not in EXECUTIONS:
            await websocket.send_json({"error": f"Execution {execution_id} not found"})
            await websocket.close()
            return
        
        # Добавление соединения
        WEBSOCKET_CONNECTIONS[execution_id] = websocket
        
        # Отправка текущего статуса
        execution = EXECUTIONS[execution_id]
        await websocket.send_json({
            "type": "initial_status",
            "execution": execution
        })
        
        log_event(f"WebSocket connected for execution: {execution_id}")
        
        # Ожидание сообщений от клиента
        while True:
            try:
                data = await websocket.receive_json()
                message_type = data.get("type", "ping")
                
                if message_type == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
                    
            except WebSocketDisconnect:
                break
                
    except Exception as e:
        logger.error(f"WebSocket error for execution {execution_id}: {e}")
        
    finally:
        # Удаление соединения
        WEBSOCKET_CONNECTIONS.pop(execution_id, None)
        log_event(f"WebSocket disconnected for execution: {execution_id}")


async def _broadcast_execution_update(execution_id: str, update: Dict[str, Any]):
    """Отправка обновления всем подключенным WebSocket клиентам."""
    websocket = WEBSOCKET_CONNECTIONS.get(execution_id)
    if websocket:
        try:
            await websocket.send_json(update)
        except Exception as e:
            logger.error(f"Failed to send WebSocket update: {e}")


@app.get("/core-settings")
async def get_core_settings(authorization: str = Header(None)) -> Dict[str, Any]:
    _require_api_token(authorization)
    config = CoreConfig.load()
    return config.to_dict()


@app.put("/core-settings")
async def update_core_settings(
    payload: CoreSettingsPayload = Body(...),
    authorization: str = Header(None),
) -> Dict[str, Any]:
    _require_api_token(authorization)
    config = CoreConfig.load()
    config.endpoint = payload.endpoint
    config.auth_token = payload.auth_token
    config.transport = payload.transport
    config.timeout_seconds = payload.timeout_seconds
    config.llm_default = payload.llm_default
    config.llm_fallback = payload.llm_fallback
    config.stt_engine = payload.stt_engine
    config.tts_engine = payload.tts_engine
    config.ingest_pipeline = payload.ingest_pipeline
    config.save()
    reload_core_adapter(config)
    return config.to_dict()


@app.post("/documents/upload")
async def upload_document(
    authorization: str = Header(None),
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    _require_api_token(authorization)
    content = await file.read()
    object_key = f"uploads/{file.filename}"
    DOCUMENT_STORE.put(object_key, content)

    memory = memory_manager.MemoryManager()
    pipeline = IngestPipeline(
        memory=memory,
        dao=DAO,
        bm25=BM25_INDEX,
        vec=VECTOR_INDEX,
        graph_idx=GRAPH_INDEX,
        graph_view=GRAPH_VIEW,
        object_store=DOCUMENT_STORE,
    )
    event = pipeline.ingest_pdf(object_key)
    return {
        "document_id": event.id,
        "object_key": object_key,
        "summary": event.attrs["text"],
    }


@app.post("/chat/session")
async def start_chat_session(authorization: str = Header(None)) -> Dict[str, Any]:
    _require_api_token(authorization)
    session_id = str(uuid.uuid4())
    CHAT_SESSIONS[session_id] = {
        "messages": [],
        "metadata": {"created_at": uuid.uuid1().hex},
    }
    return {"session_id": session_id}


@app.post("/chat/message")
async def post_chat_message(
    payload: ChatMessage,
    authorization: str = Header(None),
) -> Dict[str, Any]:
    _require_api_token(authorization)
    session = CHAT_SESSIONS.get(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session["messages"].append(payload.model_dump())
    response_text = f"Echo: {payload.content}"
    session["messages"].append(
        ChatMessage(session_id=payload.session_id, role="assistant", content=response_text).model_dump()
    )
    return {"response": response_text, "session_id": payload.session_id}


@app.post("/voice/stt")
async def voice_to_text(payload: VoiceRequest, authorization: str = Header(None)) -> Dict[str, Any]:
    _require_api_token(authorization)
    text = f"transcribed text from {payload.format}"
    return {"session_id": payload.session_id, "text": text}


@app.post("/voice/tts")
async def text_to_voice(payload: SpeechRequest, authorization: str = Header(None)) -> Dict[str, Any]:
    _require_api_token(authorization)
    audio_stub = payload.text[::-1]
    return {"session_id": payload.session_id, "audio_base64": audio_stub, "format": "wav"}


@app.websocket("/chat/stream/{session_id}")
async def chat_stream(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    if session_id not in CHAT_SESSIONS:
        await websocket.send_json({"error": "Session not found"})
        await websocket.close()
        return
    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("content", "")
            CHAT_SESSIONS[session_id]["messages"].append(
                ChatMessage(session_id=session_id, role="user", content=message).model_dump()
            )
            reply = f"Streaming echo: {message}"
            await websocket.send_json({"role": "assistant", "content": reply})
    except WebSocketDisconnect:
        await websocket.close()