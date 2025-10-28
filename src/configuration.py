"""Runtime configuration loader for Rebecca-Platform."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG_PATH = Path(os.environ.get("REBECCA_CONFIG", "config/config.yaml"))


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded."""


@lru_cache(maxsize=1)
def load_config(config_path: Path | None = None) -> Dict[str, Any]:
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def get_storage_config() -> Dict[str, Any]:
    return load_config().get("storage", {})


def get_llm_adapters() -> Dict[str, Any]:
    return load_config().get("llm_adapters", {})


def get_agent_config() -> Dict[str, Any]:
    return load_config().get("agents", {})


def get_ingest_config() -> Dict[str, Any]:
    return load_config().get("ingest", {})


def get_misc_config() -> Dict[str, Any]:
    return load_config().get("misc", {})
