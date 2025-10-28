"""Core configuration loader for Rebecca integration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class CoreConfig:
    endpoint: str
    auth_token: str
    transport: str
    timeout_seconds: int
    llm_default: str
    llm_fallback: str
    stt_engine: str
    tts_engine: str
    ingest_pipeline: str
    source_path: Path | None = None

    @classmethod
    def _default_path(cls) -> Path:
        env_path = os.environ.get("REBECCA_CORE_CONFIG")
        return Path(env_path) if env_path else Path("config/core.yaml")

    @classmethod
    def load(cls, path: Path | None = None) -> "CoreConfig":
        resolved_path = path or cls._default_path()
        if not resolved_path.exists():
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            default_payload = "\n".join(
                [
                    "core:",
                    "  endpoint: \"http://localhost:8000\"",
                    "  auth_token: \"supersecrettoken\"",
                    "  transport: \"grpc\"",
                    "  timeout_seconds: 30",
                    "llm:",
                    "  default: \"creative\"",
                    "  fallback: \"default\"",
                    "voice:",
                    "  stt: \"whisper\"",
                    "  tts: \"edge\"",
                    "documents:",
                    "  ingest_pipeline: \"auto\"",
                    "",
                ]
            )
            resolved_path.write_text(default_payload, encoding="utf-8")
        with resolved_path.open("r", encoding="utf-8") as file:
            raw: Dict[str, Any] = yaml.safe_load(file) or {}
        core = raw.get("core", {})
        llm = raw.get("llm", {})
        voice = raw.get("voice", {})
        documents = raw.get("documents", {})
        return cls(
            endpoint=core.get("endpoint", "http://localhost:8000"),
            auth_token=core.get("auth_token", ""),
            transport=core.get("transport", "grpc"),
            timeout_seconds=int(core.get("timeout_seconds", 30)),
            llm_default=llm.get("default", "creative"),
            llm_fallback=llm.get("fallback", "default"),
            stt_engine=voice.get("stt", "whisper"),
            tts_engine=voice.get("tts", "edge"),
            ingest_pipeline=documents.get("ingest_pipeline", "auto"),
            source_path=resolved_path,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "core": {
                "endpoint": self.endpoint,
                "auth_token": self.auth_token,
                "transport": self.transport,
                "timeout_seconds": self.timeout_seconds,
            },
            "llm": {
                "default": self.llm_default,
                "fallback": self.llm_fallback,
            },
            "voice": {
                "stt": self.stt_engine,
                "tts": self.tts_engine,
            },
            "documents": {
                "ingest_pipeline": self.ingest_pipeline,
            },
        }

    def save(self, path: Path | None = None) -> None:
        target = path or self.source_path or self._default_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as file:
            yaml.safe_dump(self.to_dict(), file, sort_keys=False)
        self.source_path = target
