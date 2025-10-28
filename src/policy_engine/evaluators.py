"""Placeholder evaluators for future ABAC/RBAC extensions."""

from typing import Callable

from schema.nodes import NodeBase


def allow_all(actor: str, node: NodeBase, action: str) -> bool:
    return True
