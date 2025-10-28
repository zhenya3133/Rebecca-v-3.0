"""CodeGen agent scaffold for Rebecca-Platform.

Exports placeholders inspired by Trae-Agent's modular tools, OpenAI Codex's CLI
pipelines, and nanochat's lightweight iteration loops.
"""

from .code_writer import CodeWriter
from .code_reviewer import CodeReviewer
from .integration_manager import IntegrationManager
from .codegen_main import CodeGenAgent

__all__ = [
    "CodeWriter",
    "CodeReviewer",
    "IntegrationManager",
    "CodeGenAgent",
]
