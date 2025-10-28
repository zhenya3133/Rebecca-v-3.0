"""Ensures API layer exposes core context."""

import importlib
import json

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.smoke


def _fresh_api(monkeypatch, config_path=None):
    if config_path is not None:
        monkeypatch.setenv("REBECCA_CORE_CONFIG", str(config_path))
    import api as api_module

    return importlib.reload(api_module)


def test_run_pipeline_returns_context(monkeypatch):
    api_module = _fresh_api(monkeypatch)
    client = TestClient(api_module.app)

    payload = {"input_data": "hello"}
    response = client.post(
        "/run",
        headers={"Authorization": f"Bearer {api_module.API_TOKEN}"},
        json=payload,
    )
    assert response.status_code == 200
    body = response.json()
    assert "trace_id" in body
    assert "context" in body
    assert body["result"]["result"] == "General workflow processing completed"
    assert body["context"].get("metadata", {}).get("source") == "droid"


def test_health_endpoint(monkeypatch):
    api_module = _fresh_api(monkeypatch)
    client = TestClient(api_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_core_settings_roundtrip(tmp_path, monkeypatch):
    api_module = _fresh_api(monkeypatch, tmp_path / "core.yaml")
    client = TestClient(api_module.app)

    headers = {"Authorization": f"Bearer {api_module.API_TOKEN}"}

    get_response = client.get("/core-settings", headers=headers)
    assert get_response.status_code == 200
    original = get_response.json()
    assert original["core"]["endpoint"].startswith("http")

    update_payload = {
        "endpoint": "http://new-core",
        "auth_token": "secret",
        "transport": "rest",
        "timeout_seconds": 45,
        "llm_default": "creative",
        "llm_fallback": "default",
        "stt_engine": "whisper",
        "tts_engine": "edge",
        "ingest_pipeline": "auto",
    }
    put_response = client.put("/core-settings", json=update_payload, headers=headers)
    assert put_response.status_code == 200
    body = put_response.json()
    assert body["core"]["endpoint"] == "http://new-core"
    assert body["core"]["auth_token"] == "secret"

    refreshed_headers = {"Authorization": f"Bearer {body['core']['auth_token']}"}
    refreshed = client.get("/core-settings", headers=refreshed_headers).json()
    assert refreshed["core"]["endpoint"] == "http://new-core"


def test_document_upload(tmp_path, monkeypatch):
    api_module = _fresh_api(monkeypatch, tmp_path / "core.yaml")
    client = TestClient(api_module.app)
    headers = {"Authorization": f"Bearer {api_module.API_TOKEN}"}
    file_content = b"dummy pdf"
    response = client.post(
        "/documents/upload",
        headers=headers,
        files={"file": ("test.pdf", file_content, "application/pdf")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["document_id"].startswith("pdf::")
    assert "object_key" in body
    assert api_module.DOCUMENT_STORE.get(body["object_key"]) == file_content
    assert api_module.DAO.fetch_node(body["document_id"]) is not None


def test_chat_session(monkeypatch):
    api_module = _fresh_api(monkeypatch)
    client = TestClient(api_module.app)
    headers = {"Authorization": f"Bearer {api_module.API_TOKEN}"}
    session_resp = client.post("/chat/session", headers=headers)
    assert session_resp.status_code == 200
    session_id = session_resp.json()["session_id"]

    message_resp = client.post(
        "/chat/message",
        headers=headers,
        json={"session_id": session_id, "content": "Hello"},
    )
    assert message_resp.status_code == 200
    assert "Echo" in message_resp.json()["response"]


def test_run_requires_token(monkeypatch):
    api_module = _fresh_api(monkeypatch)
    client = TestClient(api_module.app)
    response = client.post("/run", json={"input_data": "hi"})
    assert response.status_code == 403


def test_voice_endpoints(monkeypatch):
    api_module = _fresh_api(monkeypatch)
    client = TestClient(api_module.app)
    headers = {"Authorization": f"Bearer {api_module.API_TOKEN}"}
    stt_resp = client.post(
        "/voice/stt",
        headers=headers,
        json={"session_id": "s1", "audio_base64": "abc"},
    )
    assert stt_resp.status_code == 200
    assert "text" in stt_resp.json()

    tts_resp = client.post(
        "/voice/tts",
        headers=headers,
        json={"session_id": "s1", "text": "hello"},
    )
    assert tts_resp.status_code == 200
    assert "audio_base64" in tts_resp.json()
