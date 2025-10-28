"""Smoke tests for Rebecca core adapter configuration."""

from pathlib import Path

from core_adapter import CoreConfig, RebeccaCoreAdapter


def test_core_config_load(tmp_path):
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
core:
  endpoint: "http://example.com"
  auth_token: "token"
  transport: "rest"
  timeout_seconds: 10

llm:
  default: "creative"
  fallback: "default"

voice:
  stt: "whisper"
  tts: "edge"

documents:
  ingest_pipeline: "pdf"
""",
        encoding="utf-8",
    )
    config = CoreConfig.load(config_path)
    assert config.endpoint == "http://example.com"
    assert config.transport == "rest"
    assert config.timeout_seconds == 10
    assert config.ingest_pipeline == "pdf"


def test_adapter_connectivity(monkeypatch):
    # ensure default config exists
    default_path = Path("config/core.yaml")
    default_path.parent.mkdir(parents=True, exist_ok=True)
    default_path.write_text(
        """
core:
  endpoint: "http://localhost:8000"
""",
        encoding="utf-8",
    )
    config = CoreConfig.load(default_path)
    adapter = RebeccaCoreAdapter.from_config(config)
    assert adapter.connectivity_check() is True


def test_core_config_save(tmp_path):
    config_path = tmp_path / "core.yaml"
    config = CoreConfig.load(config_path)
    config.endpoint = "http://core"  # modify
    config.auth_token = "token"
    config.save()
    loaded_again = CoreConfig.load(config_path)
    assert loaded_again.endpoint == "http://core"
    assert loaded_again.auth_token == "token"
