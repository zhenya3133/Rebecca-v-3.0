"""Adapters for connecting DROId to Rebecca core services."""

from .config import CoreConfig
from .bridge import RebeccaCoreAdapter

__all__ = ["CoreConfig", "RebeccaCoreAdapter"]
